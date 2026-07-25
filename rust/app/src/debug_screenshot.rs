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
        // A *sequence* of captures, `MM_SCREENSHOT_INTERVAL_SECS` apart,
        // written as `path.0.png`, `path.1.png`, ... when more than one is
        // asked for. One still says whether the renderer works; a filmstrip
        // is what actually shows a camera *following* something, which a
        // single frame cannot.
        let count: u32 = std::env::var("MM_SCREENSHOT_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let interval_secs = std::env::var("MM_SCREENSHOT_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3.0);
        app.insert_resource(ScreenshotConfig {
            path,
            delay_secs,
            count: count.max(1),
            interval_secs,
        })
        .insert_resource(ScreenshotProgress::default())
        .add_systems(Update, take_screenshot_once);
    }
}

#[derive(Resource)]
struct ScreenshotConfig {
    path: String,
    delay_secs: f32,
    count: u32,
    interval_secs: f32,
}

#[derive(Resource, Default)]
struct ScreenshotProgress {
    taken: u32,
}

fn take_screenshot_once(
    mut commands: Commands,
    config: Res<ScreenshotConfig>,
    time: Res<Time>,
    mut progress: ResMut<ScreenshotProgress>,
) {
    let due_at = config.delay_secs + progress.taken as f32 * config.interval_secs;
    if progress.taken >= config.count || time.elapsed_secs() < due_at {
        return;
    }
    let index = progress.taken;
    progress.taken += 1;
    let last = progress.taken >= config.count;
    let path = if config.count > 1 {
        match config.path.rsplit_once('.') {
            Some((stem, ext)) => format!("{stem}.{index}.{ext}"),
            None => format!("{}.{index}", config.path),
        }
    } else {
        config.path.clone()
    };
    let mut entity = commands.spawn(Screenshot::primary_window());
    entity.observe(save_to_disk(path));
    if last {
        entity.observe(
            |_trigger: Trigger<bevy::render::view::screenshot::ScreenshotCaptured>,
             mut exit: EventWriter<AppExit>| {
                exit.write(AppExit::Success);
            },
        );
    }
}
