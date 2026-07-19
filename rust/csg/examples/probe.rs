use glam::Vec4;
use marble_csg::scenes::{beware_of_bumps, demo_scene, set_fractal_params};
use marble_csg::Params;

fn main() {
    let mut params = Params::new();
    let (object, handles) = demo_scene(&mut params);
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
    let start = beware_of_bumps::START;
    for i in 0..=68 {
        let y = start.y - (i as f32) * 0.1;
        let p = Vec4::new(start.x, y, start.z, 1.0);
        let de = object.de(p, &params);
        println!("y={:.3} de={:.5}", y, de);
    }
}
