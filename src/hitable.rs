use crate::aabb::Aabb;
use crate::material::Material;
use crate::ray::*;
use crate::rng::Random;
use crate::vec3::*;

use std::cmp::Ordering;
use std::vec::Vec;

pub trait Hitable: Send + Sync {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    fn bounding_box(&self) -> Aabb;
}

pub struct BvhTree {
    pub root: Box<Hitable>,
}

pub struct BvhNode {
    pub bounding_box: Aabb,
    pub left: Box<Hitable>,
    pub right: Box<Hitable>,
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Box<Material>,
}

impl BvhTree {
    pub fn build(hitables: &mut Vec<Box<Hitable>>, rnd: &mut Random) -> Self {
        BvhTree {
            root: BvhNode::build_bvh_tree(hitables, rnd),
        }
    }
}

impl BvhNode {
    fn build_bvh_tree(hitables: &mut Vec<Box<Hitable>>, rnd: &mut Random) -> Box<Hitable> {
        match hitables.len() {
            1 => return hitables.remove(0),
            2 => {
                let left = hitables.remove(0);
                let right = hitables.remove(0);
                return Box::new(Self::create(left, right));
            }
            _ => {}
        };

        let axis = (rnd.gen() * 3.0) as usize;
        hitables.sort_by(|left, right| {
            let bb_left = *left.bounding_box().min.get(axis);
            let bb_right = *right.bounding_box().min.get(axis);
            if bb_left < bb_right {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        let mut split = hitables.split_off(hitables.len() / 2);

        let left = Self::build_bvh_tree(hitables, rnd);
        let right = Self::build_bvh_tree(&mut split, rnd);
        Box::new(Self::create(left, right))
    }

    fn create(left: Box<Hitable>, right: Box<Hitable>) -> BvhNode {
        let bounding_box = Aabb::surrounding_box(&left.bounding_box(), &right.bounding_box());
        BvhNode {
            bounding_box,
            left,
            right,
        }
    }
}

impl Hitable for BvhNode {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if self.bounding_box.hit(ray, t_min, t_max) {
            let hit_left = self.left.hit(ray, t_min, t_max);
            let hit_right = self.right.hit(ray, t_min, t_max);
            return match (&hit_left, &hit_right) {
                (Some(left), Some(right)) => {
                    if left.t < right.t {
                        hit_left
                    } else {
                        hit_right
                    }
                }
                (Some(_), None) => hit_left,
                (None, Some(_)) => hit_right,
                _ => Option::None,
            };
        }

        Option::None
    }

    fn bounding_box(&self) -> Aabb {
        self.bounding_box
    }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.square_length();
        let b = dot(&oc, &ray.direction);
        let c = oc.square_length() - self.radius * self.radius;
        let discriminant = b * b - a * c;
        if discriminant > 0.0 {
            let tmp = (-b - discriminant.sqrt()) / a;
            if tmp < t_max && tmp > t_min {
                let hit_point = ray.point_at_parameter(tmp);
                let record = HitRecord {
                    t: tmp,
                    p: hit_point,
                    normal: &(hit_point - self.center) / self.radius,
                    material: &*self.material,
                };
                return Option::Some(record);
            }

            let tmp = (-b + discriminant.sqrt()) / a;
            if tmp < t_max && tmp > t_min {
                let hit_point = ray.point_at_parameter(tmp);
                let record = HitRecord {
                    t: tmp,
                    p: hit_point,
                    normal: &(hit_point - self.center) / self.radius,
                    material: &*self.material,
                };
                return Option::Some(record);
            }
        }

        Option::None
    }

    fn bounding_box(&self) -> Aabb {
        let radial_length = Vec3::from(self.radius, self.radius, self.radius);
        Aabb::build(self.center - radial_length, self.center + radial_length)
    }
}
