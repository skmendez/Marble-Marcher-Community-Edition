//! Live params panel: exposes the current scene's runtime uniforms (the
//! `marble_csg::Params` slot table) in an on-screen, keyboard-driven UI so
//! they can be tweaked without an edit/rebuild/reload cycle.
//!
//! `P` toggles the panel; Up/Down select a row, Left/Right adjust it
//! (held-key repeat for float rows, one step per press for integer rows),
//! Shift multiplies the step by 10. Every exposed parameter is
//! `*Value::Param`-driven in the scene tree, so an edit is purely a
//! storage-buffer value change -- no shader regeneration or pipeline
//! recompile, which is exactly the property the `Params` table exists to
//! provide (`marble_csg`'s `GLSLVariable` port, `lib.rs`'s module doc).
//!
//! What an edit has to touch, and why (the three consumers of a param):
//!  1. `mp.sim`'s owned `Scene` (`RollbackSim::params_mut`) -- the
//!     authoritative copy; physics collides against it, and it's what a
//!     future join-time scene sync would serialize.
//!  2. `MarcherFrameData::params` -- the GPU-bound copy
//!     (`gpu::write_marcher_buffers` uploads it every frame); rewritten
//!     here immediately so the edit renders on this same frame even for
//!     scenes with no animations (where `render::update_frame_data`
//!     deliberately skips the per-frame params re-copy).
//!  3. `SceneState::bounding_sphere` -- computed once at setup on the
//!     assumption that params are static thereafter (its doc); a `scale`/
//!     `shift`/`iters` edit can genuinely move the scene's outer extent, and
//!     a stale too-small bound makes `ray_sphere_clip` silently cull real
//!     geometry, so it's recomputed on every edit (a cheap tree walk).
//!
//! Editing is **solo-only** (`MultiplayerSession::is_solo`): a param edit is
//! local state outside the rollback input log, so a connected peer would
//! never see it and every subsequent checksum exchange would flag a
//! divergence (`RollbackSim::params_mut`'s doc). The panel still *shows*
//! values while connected -- only the adjust keys are gated off, with the
//! hint line saying so.
//!
//! The entry list is built from [`crate::render::SceneHandles`] -- the typed
//! per-scene handle bundles `setup` already keeps -- rather than from the
//! anonymous slot table itself: a raw `Vec4` slot has no name, no scalar-vs-
//! vec3-vs-mat2 kind, and no sensible range, all of which the UI needs.
//! Handle-less params stay unexposed by design; notably
//! `MengerOscillatingSphere`'s bite radius, which is `Expr`-animated -- the
//! physics tick overwrites it every tick, so a hand edit would just be
//! fought and immediately lost (its scene exposes the sponge's own
//! depth/color instead).

use bevy::prelude::*;
use marble_csg::scenes::{rotation_mat2, ClassicHandles, HollowDonutHandles, MengerHandles};
use marble_csg::{IntParam, Mat2Param, Params, ScalarParam, Vec3Param};

use crate::gpu::MarcherFrameData;
use crate::physics_sys::MultiplayerSession;
use crate::render::{pack_bounding_sphere, SceneHandles, SceneState};

/// How a [`ParamEntry`]'s scalar UI value maps onto the `Params` table.
#[derive(Clone, Copy, Debug)]
pub enum ParamBinding {
    Scalar(ScalarParam),
    /// Adjusted in whole steps, stepped once per key *press* (no held-key
    /// repeat -- an int sweeping 60 values/second is uncontrollable).
    Int(IntParam),
    /// A rotation angle in radians, written as its `rotation_mat2` matrix --
    /// the UI-side inverse of how `set_fractal_params` turns `ang1`/`ang2`
    /// into `Mat2Param`s.
    Angle(Mat2Param),
    /// One component (`0`/`1`/`2`) of a `Vec3Param` -- a vec3 exposes as
    /// three consecutive rows, each editing its own lane.
    Vec3Component(Vec3Param, usize),
}

impl ParamBinding {
    /// Reads this binding's current scalar value out of `params` -- used
    /// once at panel build time to seed [`ParamEntry::value`]; edits flow
    /// the other way (`apply`) from then on.
    fn read(&self, params: &Params) -> f32 {
        match *self {
            ParamBinding::Scalar(h) => params.scalar(h),
            ParamBinding::Int(h) => params.int(h) as f32,
            // `rotation_mat2(a)` has columns `(cos, -sin)` / `(sin, cos)`,
            // so the angle recovers as `atan2(sin, cos)` from the two
            // column heads. Only exact for matrices that really are
            // `rotation_mat2` products, which every `Mat2Param` in this
            // codebase's scenes is.
            ParamBinding::Angle(h) => {
                let m = params.mat2(h);
                m.y_axis.x.atan2(m.x_axis.x)
            }
            ParamBinding::Vec3Component(h, i) => params.vec3(h)[i],
        }
    }

    /// Writes `value` through to the live `Params` table.
    fn apply(&self, params: &mut Params, value: f32) {
        match *self {
            ParamBinding::Scalar(h) => params.set_scalar(h, value),
            ParamBinding::Int(h) => params.set_int(h, value.round() as i32),
            ParamBinding::Angle(h) => params.set_mat2(h, rotation_mat2(value)),
            ParamBinding::Vec3Component(h, i) => {
                let mut v = params.vec3(h);
                v[i] = value;
                params.set_vec3(h, v);
            }
        }
    }

    fn is_int(&self) -> bool {
        matches!(self, ParamBinding::Int(_))
    }
}

/// One row of the panel: a named, ranged scalar view onto one param (or one
/// lane of one). `value` is the UI's own copy, seeded from the live table at
/// build time and written through on every edit -- the UI owns it from then
/// on rather than re-deriving it per frame, since nothing else writes these
/// particular params post-setup (the one animated param is deliberately not
/// exposed -- module doc).
pub struct ParamEntry {
    pub name: &'static str,
    pub binding: ParamBinding,
    pub value: f32,
    pub min: f32,
    pub max: f32,
    /// Per-adjustment step: applied once per press for `Int` rows, once per
    /// held frame for everything else; Shift multiplies by 10.
    pub step: f32,
}

impl ParamEntry {
    fn new(
        name: &'static str,
        binding: ParamBinding,
        params: &Params,
        min: f32,
        max: f32,
        step: f32,
    ) -> Self {
        let value = binding.read(params);
        Self { name, binding, value, min, max, step }
    }
}

/// The panel's state: the entry list built for the current scene, cursor,
/// and visibility. Inserted by [`spawn_param_panel`] at startup.
#[derive(Resource)]
pub struct ParamUi {
    pub entries: Vec<ParamEntry>,
    pub selected: usize,
    pub visible: bool,
}

/// Builds the exposed-entry list for a scene's handles. Ranges/steps are
/// hand-picked per param: wide enough to explore well past the stock values
/// (`beware_of_bumps`, `MENGER_DEPTH`), narrow enough that the far ends
/// still render/behave sanely (e.g. `iters`/`depth` are also CPU `de` loop
/// counts -- physics cost scales with them too, not just GPU cost).
fn build_entries(handles: &SceneHandles, params: &Params) -> Vec<ParamEntry> {
    use std::f32::consts::TAU;
    let mut entries = Vec::new();
    let classic = |entries: &mut Vec<ParamEntry>, h: &ClassicHandles| {
        entries.push(ParamEntry::new("scale", ParamBinding::Scalar(h.scale), params, 1.0, 3.0, 0.002));
        entries.push(ParamEntry::new("ang1", ParamBinding::Angle(h.rot1), params, -TAU, TAU, 0.002));
        entries.push(ParamEntry::new("ang2", ParamBinding::Angle(h.rot2), params, -TAU, TAU, 0.002));
        for (name, i) in [("shift.x", 0), ("shift.y", 1), ("shift.z", 2)] {
            entries.push(ParamEntry::new(name, ParamBinding::Vec3Component(h.shift, i), params, -8.0, 8.0, 0.01));
        }
        for (name, i) in [("color.r", 0), ("color.g", 1), ("color.b", 2)] {
            entries.push(ParamEntry::new(name, ParamBinding::Vec3Component(h.color, i), params, 0.0, 1.0, 0.005));
        }
        entries.push(ParamEntry::new("iters", ParamBinding::Int(h.iters), params, 1.0, 24.0, 1.0));
    };
    let menger = |entries: &mut Vec<ParamEntry>, h: &MengerHandles| {
        entries.push(ParamEntry::new("depth", ParamBinding::Int(h.depth), params, 1.0, 14.0, 1.0));
        for (name, i) in [("color.r", 0), ("color.g", 1), ("color.b", 2)] {
            entries.push(ParamEntry::new(name, ParamBinding::Vec3Component(h.color, i), params, 0.0, 1.0, 0.005));
        }
    };
    let donut = |entries: &mut Vec<ParamEntry>, h: &HollowDonutHandles| {
        // `minor`'s floor stays above `thickness`'s ceiling plus the marble
        // radius (0.15), so no slider position can close the tube's free
        // interior around the marble entirely.
        entries.push(ParamEntry::new("major", ParamBinding::Scalar(h.major), params, 1.5, 6.0, 0.01));
        entries.push(ParamEntry::new("minor", ParamBinding::Scalar(h.minor), params, 0.7, 2.0, 0.005));
        entries.push(ParamEntry::new("thick", ParamBinding::Scalar(h.thickness), params, 0.05, 0.4, 0.002));
    };
    match handles {
        SceneHandles::Classic(h) => classic(&mut entries, h),
        SceneHandles::Menger(h) => menger(&mut entries, h),
        // The bite radius is Expr-animated (module doc) -- only the sponge's
        // own params are editable here.
        SceneHandles::MengerOscillatingSphere(h) => menger(&mut entries, &h.menger),
        SceneHandles::HollowDonut(h) => donut(&mut entries, h),
    }
    entries
}

/// Marker for the panel's root node (visibility toggling).
#[derive(Component)]
pub struct ParamPanel;

/// Marker for the panel's single multi-line `Text` (rebuilt on change).
#[derive(Component)]
pub struct ParamPanelText;

/// Startup system (chained after `render::setup` -- needs `SceneState`'s
/// handles and the sim's live params): builds the entry list and spawns the
/// (initially hidden) panel. Spawned unconditionally, not gated on
/// `config.debug_enabled` -- unlike the FPS overlay's always-on readouts,
/// this draws nothing and costs nothing until `P` actually opens it.
pub fn spawn_param_panel(
    mut commands: Commands,
    scene_state: Res<SceneState>,
    mp: Res<MultiplayerSession>,
) {
    let entries = build_entries(&scene_state.handles, &mp.sim.scene().params);
    commands.insert_resource(ParamUi { entries, selected: 0, visible: false });
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(6.0),
                right: Val::Px(6.0),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(5.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.65)),
            Visibility::Hidden,
            ParamPanel,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::srgb(0.9, 0.9, 0.7)),
                ParamPanelText,
            ));
        });
}

/// Update system (must run before `render::update_frame_data`, so an edit
/// reaches this frame's uniforms/bounding-sphere write rather than landing a
/// frame late): `P` toggle, row selection, and value adjustment -- see the
/// module doc for the three places an edit writes through to.
pub fn param_ui_input(
    keys: Res<ButtonInput<KeyCode>>,
    ui: Option<ResMut<ParamUi>>,
    mut mp: ResMut<MultiplayerSession>,
    mut scene_state: ResMut<SceneState>,
    mut frame: ResMut<MarcherFrameData>,
) {
    let Some(mut ui) = ui else { return };
    if keys.just_pressed(KeyCode::KeyP) {
        ui.visible = !ui.visible;
    }
    if !ui.visible || ui.entries.is_empty() {
        return;
    }

    let rows = ui.entries.len();
    if keys.just_pressed(KeyCode::ArrowUp) {
        ui.selected = (ui.selected + rows - 1) % rows;
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        ui.selected = (ui.selected + 1) % rows;
    }

    // Edits are solo-only (module doc); navigation above is always allowed.
    if !mp.is_solo() {
        return;
    }

    let selected = ui.selected;
    let entry = &ui.entries[selected];
    let held = |key| keys.pressed(key);
    let tapped = |key| keys.just_pressed(key);
    // Ints step once per press; floats repeat every held frame (one step
    // per frame at the display rate is a comfortable slew for the chosen
    // step sizes).
    let (dec, inc) = if entry.binding.is_int() {
        (tapped(KeyCode::ArrowLeft), tapped(KeyCode::ArrowRight))
    } else {
        (held(KeyCode::ArrowLeft), held(KeyCode::ArrowRight))
    };
    let direction = (inc as i32 - dec as i32) as f32;
    if direction == 0.0 {
        return;
    }
    let boost = if held(KeyCode::ShiftLeft) || held(KeyCode::ShiftRight) { 10.0 } else { 1.0 };

    let entry = &mut ui.entries[selected];
    entry.value = (entry.value + direction * entry.step * boost).clamp(entry.min, entry.max);
    entry.binding.apply(mp.sim.params_mut(), entry.value);

    // Write-through to the GPU copy and the bounding sphere (module doc's
    // consumers 2 and 3; consumer 1 is the `params_mut` line above).
    let scene = mp.sim.scene();
    frame.params.clear();
    frame.params.extend_from_slice(scene.params.slots());
    scene_state.bounding_sphere = pack_bounding_sphere(&scene.object, &scene.params);
}

/// Update system: syncs the panel's visibility and rebuilds its text.
/// Follows this codebase's overlay convention (`fps_overlay.rs`) of
/// comparing before writing so `Text` change detection stays quiet on the
/// (vast majority of) frames where nothing changed.
pub fn update_param_panel_text(
    ui: Option<Res<ParamUi>>,
    mp: Res<MultiplayerSession>,
    mut panels: Query<&mut Visibility, With<ParamPanel>>,
    mut texts: Query<&mut Text, With<ParamPanelText>>,
) {
    let Some(ui) = ui else { return };
    let Ok(mut visibility) = panels.single_mut() else { return };
    let desired = if ui.visible { Visibility::Visible } else { Visibility::Hidden };
    if *visibility != desired {
        *visibility = desired;
    }
    if !ui.visible {
        return;
    }
    let Ok(mut text) = texts.single_mut() else { return };

    let mut lines = vec![
        if mp.is_solo() {
            "params  [P] close  [up/down] select  [left/right] adjust  [shift] x10".to_string()
        } else {
            "params  [P] close  (read-only while connected)".to_string()
        },
    ];
    for (i, entry) in ui.entries.iter().enumerate() {
        let cursor = if i == ui.selected { ">" } else { " " };
        let value = if entry.binding.is_int() {
            format!("{}", entry.value.round() as i32)
        } else {
            format!("{:+.3}", entry.value)
        };
        lines.push(format!("{cursor} {:<8} {value}", entry.name));
    }
    let new_text = lines.join("\n");
    if text.0 != new_text {
        text.0 = new_text;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;
    use marble_csg::scenes::{
        beware_of_bumps, classic, menger_sponge, set_fractal_params, set_menger_params,
    };

    fn classic_setup() -> (SceneHandles, Params) {
        let mut params = Params::new();
        let (_object, handles) = classic(&mut params);
        set_fractal_params(
            &mut params,
            &handles,
            beware_of_bumps::SCALE,
            beware_of_bumps::ANG1,
            beware_of_bumps::ANG2,
            beware_of_bumps::SHIFT,
            beware_of_bumps::COLOR,
            beware_of_bumps::ITERS,
        );
        (SceneHandles::Classic(handles), params)
    }

    #[test]
    fn classic_entries_seed_from_the_live_params() {
        let (handles, params) = classic_setup();
        let entries = build_entries(&handles, &params);
        let by_name = |name: &str| {
            entries
                .iter()
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("missing entry {name}"))
        };
        assert!((by_name("scale").value - beware_of_bumps::SCALE).abs() < 1e-6);
        // Angle recovery: the mat2 was built by rotation_mat2(ANG1); the
        // entry must read the same angle back out of the matrix.
        assert!((by_name("ang1").value - beware_of_bumps::ANG1).abs() < 1e-5);
        assert!((by_name("ang2").value - beware_of_bumps::ANG2).abs() < 1e-5);
        assert!((by_name("shift.y").value - beware_of_bumps::SHIFT.y).abs() < 1e-6);
        assert!((by_name("color.b").value - beware_of_bumps::COLOR.z).abs() < 1e-6);
        assert_eq!(by_name("iters").value, beware_of_bumps::ITERS as f32);
    }

    #[test]
    fn menger_entries_seed_from_the_live_params() {
        let mut params = Params::new();
        let (_object, handles) = menger_sponge(&mut params);
        set_menger_params(&mut params, &handles, 7, Vec3::new(0.9, 0.65, 0.15));
        let entries = build_entries(&SceneHandles::Menger(handles), &params);
        assert_eq!(entries[0].name, "depth");
        assert_eq!(entries[0].value, 7.0);
        assert!((entries[1].value - 0.9).abs() < 1e-6);
    }

    #[test]
    fn apply_writes_every_binding_kind_through_to_params() {
        let (handles, mut params) = classic_setup();
        let entries = build_entries(&handles, &params);
        let SceneHandles::Classic(h) = &handles else { unreachable!() };

        for entry in &entries {
            entry.binding.apply(&mut params, entry.value + 0.0); // no-op write must not corrupt
        }
        // Scalar.
        entries[0].binding.apply(&mut params, 2.5);
        assert!((params.scalar(h.scale) - 2.5).abs() < 1e-6);
        // Angle: write 0.7, matrix must be rotation_mat2(0.7), and reading
        // the binding back must recover 0.7 (full round trip through mat2).
        entries[1].binding.apply(&mut params, 0.7);
        let m = params.mat2(h.rot1);
        let expected = rotation_mat2(0.7);
        assert!((m.x_axis - expected.x_axis).length() < 1e-6);
        assert!((entries[1].binding.read(&params) - 0.7).abs() < 1e-5);
        // Vec3 lane: writing shift.y must not disturb x/z.
        let before = params.vec3(h.shift);
        entries[4].binding.apply(&mut params, 1.25);
        let after = params.vec3(h.shift);
        assert_eq!(after.y, 1.25);
        assert_eq!(after.x, before.x);
        assert_eq!(after.z, before.z);
        // Int: rounds rather than truncates.
        entries[9].binding.apply(&mut params, 11.7);
        assert_eq!(params.int(h.iters), 12);
    }

    #[test]
    fn oscillating_scene_exposes_the_sponge_but_not_the_animated_radius() {
        let mut params = Params::new();
        let (_object, handles) = marble_csg::scenes::menger_oscillating_sphere(&mut params);
        // Sentinel through the real radius handle, so "no entry touches it"
        // is checkable against the slot table itself rather than by
        // trusting entry names.
        params.set_scalar(handles.radius, 123.456);
        let entries =
            build_entries(&SceneHandles::MengerOscillatingSphere(handles), &params);
        assert!(!entries.is_empty(), "the sponge's own params must still be exposed");
        let mut probe = params.clone();
        for entry in &entries {
            entry.binding.apply(&mut probe, entry.max);
        }
        assert!(
            probe.slots().iter().any(|v| (v.x - 123.456).abs() < 1e-3),
            "an entry overwrote the animated radius slot"
        );
    }
}
