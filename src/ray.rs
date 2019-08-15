use crate::material::Material;
use crate::vec3::Vec3;

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn default() -> Ray {
        Ray {
            origin: Vec3::default(),
            direction: Vec3::default(),
        }
    }

    pub fn point_at_parameter(&self, t: f32) -> Vec3 {
        self.origin + (&self.direction * t)
    }
}

pub struct HitRecord<'a> {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub material: &'a Material,
}
