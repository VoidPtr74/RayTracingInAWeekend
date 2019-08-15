use crate::ray::Ray;
use crate::vec3::Vec3;
// tasty single-instruction versions.
// rust's f32::max/min use 3 instructions.
fn ffmin(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

fn ffmax(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

#[derive(Copy, Clone)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn build(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn surrounding_box(left: &Aabb, right: &Aabb) -> Self {
        Self {
            min: left.min.min(&right.min),
            max: left.max.max(&right.max),
        }
    }

    // Shamelessly stolen from GPSnoopy's implementation
    pub fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> bool {
        let inv_d = ray.direction.invert_elems();
        let t0 = (self.min - ray.origin).direct_product(&inv_d);
        let t1 = (self.max - ray.origin).direct_product(&inv_d);

        let t_min = ffmax(tmin, t0.min(&t1).max_elem());
        let t_max = ffmin(tmax, t0.max(&t1).min_elem());
        t_max > t_min
    }
}
