//! Opt-in one-shot screenshot: set `MM_SCREENSHOT=path.png` in the
//! environment to capture the primary window after a delay, then exit.
//! Useful for headless/CI verification and for confirming a change actually
//! renders without needing a human at the keyboard; a no-op (zero systems
//! added) when the env var is unset.
//!
//! `MM_SCREENSHOT_DELAY_SECS` (default 5) sets how long to wait before
//! capturing. This matters more than it sounds: an entity whose material's
//! render pipeline hasn't finished compiling yet is simply skipped for that
//! frame (not an error, just absent) — on a software (CPU) Vulkan fallback
//! like llvmpipe, compiling this ray marcher's shader can itself take
//! minutes, so a screenshot taken too early captures the window's plain
//! clear color with no indication anything is wrong.

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};

pub struct DebugScreenshotPlugin;

impl Plugin for DebugScreenshotPlugin {
    fn build(&self, app: &mut App) {
        let Ok(path) = std::env::var("MM_SCREENSHOT") else {
            return;
        };
        let delay_secs = std::env::var("MM_SCREENSHOT_DELAY_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5.0);
        app.insert_resource(ScreenshotConfig { path, delay_secs })
            .add_systems(Update, take_screenshot_once);
    }
}

#[derive(Resource)]
struct ScreenshotConfig {
    path: String,
    delay_secs: f32,
}

#[derive(Resource)]
struct AlreadyTaken;

fn take_screenshot_once(
    mut commands: Commands,
    config: Res<ScreenshotConfig>,
    time: Res<Time>,
    already_taken: Option<Res<AlreadyTaken>>,
) {
    if already_taken.is_some() || time.elapsed_secs() < config.delay_secs {
        return;
    }
    commands.insert_resource(AlreadyTaken);
    let path = config.path.clone();
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path))
        .observe(|_trigger: Trigger<bevy::render::view::screenshot::ScreenshotCaptured>,
                  mut exit: EventWriter<AppExit>| {
            exit.write(AppExit::Success);
        });
}
