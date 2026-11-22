//! `cloture-core`: Sub-microsecond anytime-valid statistical inference,
//! consensus stopping, correlated-panel diagnostics, and active evaluation
//! for multi-agent AI systems.
//!
//! Zero external dependencies. Every hot-path statistic operates in log-space
//! with compensated arithmetic and finite-sample Ville/martingale guarantees.

pub mod betting;
pub mod cascade;
pub mod ffi;
pub mod fuse;
pub mod math;
pub mod mode;
pub mod panel;
pub mod ppi;
pub mod race;
pub mod sketch;
