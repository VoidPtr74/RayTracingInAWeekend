use crate::ray::Ray;
use crate::rng::Random;
use crate::vec3::*;

use std::f32;

#[derive(Copy, Clone)]
pub struct Camera {
    pub origin: Vec3,
    pub lower_left_corner: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn build(
        look_from: &Vec3,
        look_at: &Vec3,
        camera_up: &Vec3,
        vertical_fov_degrees: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_distance: f32,
    ) -> Self {
        let lens_radius = aperture / 2.0;
        let theta = vertical_fov_degrees * f32::consts::PI / 180.0; // switch to radians
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        let w = (look_from - look_at).make_normalised();
        let u = cross(camera_up, &w).make_normalised();
        let v = cross(&w, &u);
        let lower_left_corner = look_from
            - &(&u * (focus_distance * half_width))
            - &v * (focus_distance * half_height)
            - &w * focus_distance;
        let horizontal = &u * (2.0 * half_width * focus_distance);
        let vertical = &v * (2.0 * half_height * focus_distance);

        Camera {
            origin: *look_from,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius,
        }
    }

    pub fn get_ray(&self, s: f32, t: f32, rnd: &mut Random) -> Ray {
        let rd = &random_in_unit_disk(rnd) * self.lens_radius;
        let offset = &self.u * *rd.x() + &self.v * *rd.y();
        Ray {
            origin: self.origin + offset,
            direction: self.lower_left_corner + &self.horizontal * s + &self.vertical * t
                - self.origin
                - offset,
        }
    }
}

fn random_in_unit_disk(rnd: &mut Random) -> Vec3 {
    loop {
        let p = &Vec3::from(rnd.gen(), rnd.gen(), 0.0) * 2.0 - Vec3::from(1.0, 1.0, 0.0);
        if p.square_length() <= 1.0 {
            break p;
        }
    }
}
