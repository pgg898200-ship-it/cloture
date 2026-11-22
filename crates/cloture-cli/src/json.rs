//! Minimal zero-dependency RFC-8259 JSON parser and serializer for `cloture-cli` and MCP.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(v)
                if v.is_finite()
                    && *v >= 0.0
                    && v.fract() == 0.0
                    && *v <= 9_007_199_254_740_991.0 =>
            {
                Some(*v as u64)
            }
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_obj()?.get(key)
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_to(&mut out);
        out
    }

    fn write_to(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(v) => {
                if v.is_finite() {
                    if *v == 0.0 {
                        out.push('0');
                    } else if v.fract() == 0.0 && v.abs() < 9.0e15 {
                        out.push_str(&format!("{:.0}", v));
                    } else {
                        out.push_str(&format!("{}", v));
                    }
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_escaped_str(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_to(out);
                }
                out.push(']');
            }
            Json::Obj(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped_str(k, out);
                    out.push(':');
                    v.write_to(out);
                }
                out.push('}');
            }
        }
    }
}

pub fn write_escaped_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

const MAX_JSON_DEPTH: usize = 128;

pub fn parse_json(input: &str) -> Result<Json, String> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    let val = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing bytes at offset {}", p.pos));
    }
    Ok(val)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'n') => self.expect_lit(b"null").map(|_| Json::Null),
            Some(b't') => self.expect_lit(b"true").map(|_| Json::Bool(true)),
            Some(b'f') => self.expect_lit(b"false").map(|_| Json::Bool(false)),
            Some(b'"') => self.parse_string().map(Json::Str),
            Some(b'[') => {
                if self.depth >= MAX_JSON_DEPTH {
                    return Err("maximum JSON nesting depth exceeded".to_string());
                }
                self.depth += 1;
                let res = self.parse_array();
                self.depth -= 1;
                res
            }
            Some(b'{') => {
                if self.depth >= MAX_JSON_DEPTH {
                    return Err("maximum JSON nesting depth exceeded".to_string());
                }
                self.depth += 1;
                let res = self.parse_object();
                self.depth -= 1;
                res
            }
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(c) => Err(format!("unexpected byte '{}' at {}", c as char, self.pos)),
            None => Err("unexpected EOF".to_string()),
        }
    }

    fn expect_lit(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(format!("expected literal at {}", self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at {}", self.pos));
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                if self.pos >= self.bytes.len() {
                    return Err("unterminated escape".to_string());
                }
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err("truncated \\u escape".to_string());
                        }
                        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                            .map_err(|_| "invalid utf8 in \\u".to_string())?;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|_| "invalid hex in \\u".to_string())?;
                        self.pos += 4;
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                    }
                    _ => return Err(format!("invalid escape '\\{}'", esc as char)),
                }
            } else {
                // Multi-byte UTF-8 handling
                let start = self.pos - 1;
                let width = if b < 0x80 {
                    1
                } else if (b & 0xE0) == 0xC0 {
                    2
                } else if (b & 0xF0) == 0xE0 {
                    3
                } else {
                    4
                };
                if start + width > self.bytes.len() {
                    return Err("truncated utf8".to_string());
                }
                let s = std::str::from_utf8(&self.bytes[start..start + width])
                    .map_err(|_| "invalid utf8".to_string())?;
                out.push_str(s);
                self.pos = start + width;
            }
        }
        Err("unterminated string".to_string())
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number utf8".to_string())?;
        let val: f64 = s.parse().map_err(|_| format!("invalid number '{}'", s))?;
        if !val.is_finite() {
            return Err(format!("number out of f64 range '{}'", s));
        }
        Ok(Json::Num(val))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at {}", self.pos));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(map));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }
}
