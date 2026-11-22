/**
 * Cloudflare Worker: Pure-Markdown Serving & Telemetry for AI Agent Crawlers
 *
 * Implements the empirically validated agent-discovery pattern:
 * - Serves clean, zero-chrome pure Markdown (.md / llms.txt / llms-full.txt) to
 *   AI assistants, coding agents, and LLM fetchers when `Accept: text/markdown`
 *   is requested or an AI user-agent is detected (never cloaking/deceiving:
 *   serves the exact same technical documentation and benchmark tables as the
 *   HTML docs, formatted cleanly in Markdown for optimal LLM parsing).
 * - Logs structured bot/agent telemetry (UA, ASN, path, format served) so you
 *   can track which pages AI assistants cite and optimize docs empirically.
 */

const AI_AGENT_PATTERNS = [
  /ClaudeBot/i,
  /Claude-User/i,
  /Claude-SearchBot/i,
  /Anthropic/i,
  /GPTBot/i,
  /OAI-SearchBot/i,
  /ChatGPT-User/i,
  /PerplexityBot/i,
  /Perplexity-User/i,
  /Google-Extended/i,
  /Gemini/i,
  /DuckAssistBot/i,
  /YouBot/i,
  /Meta-ExternalFetcher/i,
  /DeepSeek/i,
];

const MARKDOWN_ROUTES = new Map([
  ["/", "/README.md"],
  ["/llms.txt", "/llms.txt"],
  ["/llms-full.txt", "/llms-full.txt"],
  ["/benchmarks", "/docs/benchmarks.md"],
  ["/comparisons", "/docs/comparisons.md"],
  ["/integrations", "/docs/integrations.md"],
  ["/architecture", "/docs/ARCHITECTURE.md"],
  ["/skill", "/SKILL.md"],
]);

function wantsMarkdown(request) {
  const accept = request.headers.get("accept") || "";
  if (accept.includes("text/markdown") || accept.includes("text/plain")) {
    return true;
  }
  const ua = request.headers.get("user-agent") || "";
  return AI_AGENT_PATTERNS.some((re) => re.test(ua));
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const ua = request.headers.get("user-agent") || "";
    const asn = request.cf?.asn ?? 0;
    const isAgent = wantsMarkdown(request);

    const rawBase =
      env.RAW_DOCS_BASE ||
      "https://raw.githubusercontent.com/pgg898200-ship-it/cloture/main";

    if (isAgent || url.pathname.endsWith(".md") || url.pathname.startsWith("/llms")) {
      const targetPath =
        MARKDOWN_ROUTES.get(url.pathname) ||
        (url.pathname.endsWith(".md") ? url.pathname : "/llms.txt");

      const upstream = await fetch(`${rawBase}${targetPath}`, {
        cf: { cacheTtl: 300, cacheEverything: true },
      });

      if (upstream.ok) {
        const body = await upstream.text();
        if (env.AGENT_TELEMETRY) {
          ctx.waitUntil(
            env.AGENT_TELEMETRY.writeDataPoint({
              blobs: [url.pathname, targetPath, ua.slice(0, 120)],
              doubles: [asn, body.length],
              indexes: [isAgent ? "agent" : "direct_md"],
            })
          );
        }
        return new Response(body, {
          status: 200,
          headers: {
            "content-type": "text/markdown; charset=utf-8",
            "vary": "Accept, User-Agent",
            "x-content-format": "markdown",
            "cache-control": "public, max-age=300",
          },
        });
      }
    }

    // Human browser fallback: serve static site or redirect to docs.
    return fetch(request);
  },
};
