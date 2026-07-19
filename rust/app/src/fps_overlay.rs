//! On-screen FPS counter overlay.
//!
//! A prerequisite for the upcoming mobile-adaptive-quality work: adaptive
//! quality decisions need an actual visible frame-time signal to test
//! against, so this makes Bevy's built-in frame-time diagnostic visible
//! rather than only queryable in code.
//!
//! Uses `bevy_diagnostic::FrameTimeDiagnosticsPlugin`/`DiagnosticsStore`
//! (already pulled in unconditionally by `bevy_internal` — see
//! `app/Cargo.toml`'s comment on the `bevy` dependency) plus a small
//! `bevy_ui`/`bevy_text` overlay (both newly enabled features; text
//! rendering has no lighter-weight built-in alternative in Bevy 0.16). Text
//! uses the `default_font` feature's bundled subset-of-FiraMono font, so no
//! font asset needs to be vendored or loaded.
//!
//! Displayed top-left, small, over a semi-transparent dark backing panel so
//! it stays legible against the fractal render behind it.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

pub struct FpsOverlayPlugin;

impl Plugin for FpsOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, spawn_fps_overlay)
            .add_systems(Update, update_fps_text);
    }
}

/// Marker for the `Text` entity showing the live FPS number.
#[derive(Component)]
struct FpsText;

fn spawn_fps_overlay(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(6.0),
                left: Val::Px(6.0),
                padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                ..default()
            },
            // Semi-transparent dark backing so the text reads over any part
            // of the (arbitrarily bright/colorful) fractal render.
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        ))
        .with_child((
            Text::new("FPS: --"),
            TextFont {
                font_size: 18.0,
                ..default()
            },
            // Bright, high-contrast against the dark backing panel.
            TextColor(Color::srgb(0.2, 1.0, 0.4)),
            FpsText,
        ));
}

fn update_fps_text(diagnostics: Res<DiagnosticsStore>, mut text: Query<&mut Text, With<FpsText>>) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let Some(fps) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
    else {
        return;
    };
    text.0 = format!("FPS: {fps:.1}");
}
