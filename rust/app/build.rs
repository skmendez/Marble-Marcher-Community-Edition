//! Bakes the current commit hash into the binary as `MM_COMMIT`, for the
//! `?debug=1` overlay's last line.
//!
//! Exists because "is the build I am looking at the build I just pushed?"
//! is otherwise unanswerable from a phone, and answering it wrong costs a
//! debugging round trip: a bug report against a stale deploy looks exactly
//! like a fix that didn't work.
//!
//! Falls back to all-zeros rather than to a word like "unknown" so the
//! overlay line is always the same shape (pure hex), and so a build made
//! outside a git checkout is obvious rather than merely odd.

use std::path::Path;
use std::process::Command;

const UNKNOWN: &str = "00000000";

fn main() {
    let hash = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit()))
        .unwrap_or_else(|| UNKNOWN.to_owned());
    println!("cargo:rustc-env=MM_COMMIT={hash}");

    // Rebuild when HEAD moves (a commit, a checkout, a branch switch).
    // Emitted only for paths that exist: `rerun-if-changed` on a missing
    // path makes cargo rebuild this crate on *every* invocation.
    for candidate in ["../../.git/HEAD", "../.git/HEAD", ".git/HEAD"] {
        if Path::new(candidate).exists() {
            println!("cargo:rerun-if-changed={candidate}");
            break;
        }
    }
}
