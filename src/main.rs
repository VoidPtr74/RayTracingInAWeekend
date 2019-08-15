mod aabb;
mod camera;
mod hitable;
mod material;
mod ray;
mod rng;
mod vec3;

use camera::Camera;
use hitable::*;
use material::*;
use ray::Ray;
use rng::Random;
use std::f32;
use vec3::Vec3;

fn main() {
    let mut rnd = Random::create_with_seed(42);
    let nx = 3840;
    let ny = 2160;
    let nxd = nx as f32;
    let nyd = ny as f32;

    let look_from = Vec3::from(13.0, 2.0, 3.0);
    let look_at = Vec3::from(0.0, 0.0, 0.0);
    let aperture = 0.1;
    let dist_to_focus = 10.0;

    let camera = Camera::build(
        &look_from,
        &look_at,
        &Vec3::from(0.0, 1.0, 0.0),
        20.0,
        nxd / nyd,
        aperture,
        dist_to_focus,
    );

    let mut hitable_list = random_scene(&mut rnd);

    let bvh_tree = BvhTree::build(&mut hitable_list, &mut rnd);

    // let bvh_tree : Box<Hitable> = Box::new(hitable_list);
    let samples_per_pixel = 1024;
    let thread_count = 24;

    // let cols = render_single_thread(&camera, nx, ny, samples_per_pixel, &bvh_tree, &mut rnd);
    let cols = render_multi_thread(
        camera,
        nx,
        ny,
        samples_per_pixel,
        bvh_tree,
        &mut rnd,
        thread_count,
    );

    print!("P3\n{} {}\n255\n", nx, ny);
    for col in cols.iter() {
        let ir = (255.99 * col.r()) as i32;
        let ig = (255.99 * col.g()) as i32;
        let ib = (255.99 * col.b()) as i32;
        println!("{} {} {}", ir, ig, ib);
    }
}

fn render_multi_thread(
    camera: Camera,
    nx: usize,
    ny: usize,
    samples_per_pixel: i16,
    bvh_tree: BvhTree,
    _: &mut Random,
    thread_count: usize,
) -> Vec<Vec3> {
    let mut cols: Vec<Vec3> = Vec::with_capacity(nx * ny);
    let mut workers: Vec<std::thread::JoinHandle<std::vec::Vec<vec3::Vec3>>> =
        Vec::with_capacity(thread_count);
    let nxd = nx as f32;
    let nyd = ny as f32;

    let arc_tree = std::sync::Arc::new(bvh_tree);

    for thread_index in (0..thread_count).rev() {
        let local_bvh = arc_tree.clone();
        let thread_seed = 1234 * thread_index as u64;
        let (y0, y1) = get_segment(thread_count, thread_index, ny);

        let thd = std::thread::spawn(move || {
            let mut rnd = Random::create_with_seed(thread_seed);
            let mut cols: Vec<Vec3> = Vec::with_capacity(nx * ny);
            for y in (y0..y1).rev() {
                let yd = y as f32;
                for x in 0..nx {
                    let mut col = Vec3::from(0.0, 0.0, 0.0);
                    let xd = x as f32;
                    for _ in 0..samples_per_pixel {
                        let u = (xd + rnd.gen()) / nxd;
                        let v = (yd + rnd.gen()) / nyd;
                        let r = camera.get_ray(u, v, &mut rnd);
                        col += &colour(&r, local_bvh.as_ref(), &mut rnd, 1);
                    }

                    col /= f32::from(samples_per_pixel);
                    col = Vec3::from(col.x().sqrt(), col.y().sqrt(), col.z().sqrt());
                    cols.push(col);
                }
            }

            cols
        });

        workers.push(thd);
    }

    for waiter in workers {
        let mut result = waiter.join().unwrap();
        cols.append(&mut result);
    }

    cols
}

/*
fn render_single_thread(
    camera: &Camera,
    nx: usize,
    ny: usize,
    samples_per_pixel: i16,
    bvh_tree: &BvhTree,
    rnd: &mut Random,
) -> Vec<Vec3> {
    let nxd = nx as f32;
    let nyd = ny as f32;

    let mut cols: Vec<Vec3> = Vec::with_capacity(nx * ny);
    for y in (0..ny).rev() {
        let yd = y as f32;
        for x in 0..nx {
            let mut col = Vec3::from(0.0, 0.0, 0.0);
            let xd = x as f32;
            for _ in 0..samples_per_pixel {
                let u = (xd + rnd.gen()) / nxd;
                let v = (yd + rnd.gen()) / nyd;
                let r = camera.get_ray(u, v, rnd);
                col += &colour(&r, &bvh_tree, rnd, 1);
            }

            col /= f32::from(samples_per_pixel);
            col = Vec3::from(col.x().sqrt(), col.y().sqrt(), col.z().sqrt());
            cols.push(col);
        }
    }

    cols
}
*/ 

fn colour(ray: &Ray, world: &BvhTree, rnd: &mut Random, depth: i32) -> Vec3 {
    const MAX_THING: f32 = 1.0e10;
    let record = world.root.hit(ray, 0.001, MAX_THING);
    match record {
        None => {
            // Render "Sky"
            let direction = ray.direction.make_normalised();
            let t = 0.5 * (direction.y() + 1.0);

            (&Vec3::from(1.0, 1.0, 1.0) * (1.0 - t)) + (&Vec3::from(0.5, 0.7, 1.0) * t)
        }
        Some(rec) => {
            let mut scattered = Ray::default();
            let mut attenuation = Vec3::default();
            if depth < 20
                && rec
                    .material
                    .scatter(ray, &rec, rnd, &mut attenuation, &mut scattered)
            {
                attenuation.direct_product(&colour(&scattered, world, rnd, depth + 1))
            } else {
                Vec3::from(0.0, 0.0, 0.0)
            }
        }
    }
}

fn get_segment(thread_count: usize, thread_index: usize, ny: usize) -> (usize, usize) {
    let segment_size = ny / thread_count;
    let lower = segment_size * thread_index;
    let upper = if thread_index == thread_count - 1 {
        ny
    } else {
        segment_size * (thread_index + 1)
    };
    (lower, upper)
}

fn random_scene(rnd: &mut Random) -> Vec<Box<Hitable>> {
    let n = 500;
    let mut list: Vec<Box<Hitable>> = Vec::with_capacity(n + 1);
    list.push(build_sphere(
        Vec3::from(0.0, -1000.0, 0.0),
        1000.0,
        Box::new(Lambertian::with_albedo(Vec3::from(0.5, 0.5, 0.5))),
    ));
    for a in -11..11i16 {
        for b in -11..11i16 {
            let choose_mat = rnd.gen();
            let center = Vec3::from(
                f32::from(a) + 0.9 * rnd.gen(),
                0.2,
                f32::from(b) + 0.9 * rnd.gen(),
            );
            if (center - Vec3::from(4.0, 0.2, 0.0)).length() > 0.9 {
                let material: Box<Material> = match choose_mat {
                    // diffuse
                    x if x < 0.8 => Box::new(Lambertian::with_albedo(Vec3::from(
                        rnd.gen() * rnd.gen(),
                        rnd.gen() * rnd.gen(),
                        rnd.gen() * rnd.gen(),
                    ))),
                    x if x < 0.95 => Box::new(Metal::with_albedo(Vec3::from(
                        0.5 * (1.0 + rnd.gen()),
                        0.5 * (1.0 + rnd.gen()),
                        0.5 * (1.0 + rnd.gen()),
                    ))),
                    _ => Box::new(Dielectric::with_refraction_index(1.5)),
                };
                list.push(build_sphere(center, 0.2, material));
            }
        }
    }

    list.push(build_sphere(
        Vec3::from(0.0, 1.0, 0.0),
        1.0,
        Box::new(Dielectric::with_refraction_index(1.5)),
    ));
    list.push(build_sphere(
        Vec3::from(-4.0, 1.0, 0.0),
        1.0,
        Box::new(Lambertian::with_albedo(Vec3::from(0.4, 0.2, 0.1))),
    ));
    list.push(build_sphere(
        Vec3::from(4.0, 1.0, 0.0),
        1.0,
        Box::new(Metal::with_albedo(Vec3::from(0.7, 0.6, 0.5))),
    ));

    list
}

fn build_sphere(center: Vec3, radius: f32, material: Box<Material>) -> Box<Hitable> {
    let sphere = Sphere {
        center,
        radius,
        material,
    };
    Box::new(sphere)
}
