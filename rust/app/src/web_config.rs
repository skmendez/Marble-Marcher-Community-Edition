//! Reads a config value from the page's URL query string on wasm, since
//! `std::env::var` (this codebase's native `MM_*` config convention) always
//! returns `Err` on wasm32-unknown-unknown -- there is no OS environment in
//! a browser. `?key=value` in the address bar is the equivalent for the
//! deployed web build. `cfg`-gated to a no-op on native so callers can ask
//! "what's key X" without caring which platform they're on; native callers
//! keep reading `std::env::var` directly, unchanged.

/// `None` on native (always -- callers should fall back to `std::env::var`
/// there) or if `key` isn't present in the URL's query string; `Some` with
/// the raw (already percent-decoded by `UrlSearchParams`) value otherwise.
#[cfg(target_arch = "wasm32")]
pub fn query_param(key: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    web_sys::UrlSearchParams::new_with_str(&search)
        .ok()?
        .get(key)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn query_param(_key: &str) -> Option<String> {
    None
}
