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

use std::fs;
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

    // Rebuild when the commit changes. Watching `.git/HEAD` alone is not
    // enough and was wrong: committing on the branch you are already on
    // leaves HEAD's *contents* untouched (it still reads `ref:
    // refs/heads/master`) and moves the ref file it points at instead. So a
    // local build kept reporting a hash several commits stale -- which is
    // precisely the failure this whole file exists to make impossible, and
    // which had already cost one debugging round trip against a stale
    // deploy. Watch both: HEAD for checkouts and branch switches, the ref
    // it names for ordinary commits.
    //
    // Emitted only for paths that exist: `rerun-if-changed` on a missing
    // path makes cargo rebuild this crate on *every* invocation.
    let Some(git_dir) = ["../../.git", "../.git", ".git"]
        .into_iter()
        .map(Path::new)
        .find(|p| p.join("HEAD").exists())
    else {
        return;
    };
    let head = git_dir.join("HEAD");
    println!("cargo:rerun-if-changed={}", head.display());

    // `HEAD` is either `ref: refs/heads/<branch>` (on a branch) or a bare
    // hash (detached, where HEAD itself is the thing that moves and there is
    // no second file to watch).
    let Ok(contents) = fs::read_to_string(&head) else {
        return;
    };
    if let Some(reference) = contents.trim().strip_prefix("ref:") {
        let ref_path = git_dir.join(reference.trim());
        if ref_path.exists() {
            println!("cargo:rerun-if-changed={}", ref_path.display());
        } else {
            // Packed refs: a freshly cloned repo has no loose ref file until
            // the branch moves, so watch the pack that stands in for it.
            let packed = git_dir.join("packed-refs");
            if packed.exists() {
                println!("cargo:rerun-if-changed={}", packed.display());
            }
        }
    }
}
