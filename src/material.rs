use crate::ray::*;
use crate::rng::Random;
use crate::vec3::*;

pub trait Material: Send + Sync {
    fn scatter(
        &self,
        ray: &Ray,
        rec: &HitRecord,
        rnd: &mut Random,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> bool;
}

pub struct Lambertian {
    albedo: Vec3,
}

pub struct Metal {
    albedo: Vec3,
}

pub struct Dielectric {
    refraction_index: f32,
}

impl Lambertian {
    pub fn with_albedo(albedo: Vec3) -> Lambertian {
        Lambertian { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        _ray: &Ray,
        rec: &HitRecord,
        rnd: &mut Random,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> bool {
        let target = rec.p + rec.normal + random_in_unit_sphere(rnd);
        scattered.origin = rec.p;
        scattered.direction = target - rec.p;
        attenuation.set(&self.albedo);
        true
    }
}

impl Metal {
    pub fn with_albedo(albedo: Vec3) -> Metal {
        Metal { albedo }
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        ray: &Ray,
        rec: &HitRecord,
        _: &mut Random,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> bool {
        let reflected = reflect(&ray.direction.make_normalised(), &rec.normal);
        scattered.origin = rec.p;
        scattered.direction = reflected;
        attenuation.set(&self.albedo);
        dot(&scattered.direction, &rec.normal) > 0.0
    }
}

impl Dielectric {
    pub fn with_refraction_index(refraction_index: f32) -> Dielectric {
        Dielectric { refraction_index }
    }
}

fn schlick(cosine: f32, refraction_index: f32) -> f32 {
    let r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

impl Material for Dielectric {
    fn scatter(
        &self,
        ray: &Ray,
        rec: &HitRecord,
        rnd: &mut Random,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> bool {
        let reflected = reflect(&ray.direction, &rec.normal);
        attenuation.set(&Vec3::from(1.0, 1.0, 1.0));
        let outward_normal: Vec3;
        let ni_over_nt: f32;
        let incident_dot_normal = dot(&ray.direction, &rec.normal);
        let cosine = if incident_dot_normal > 0.0 {
            outward_normal = &rec.normal * -1.0;
            ni_over_nt = self.refraction_index;
            self.refraction_index * incident_dot_normal / ray.direction.length()
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / self.refraction_index;
            -incident_dot_normal / ray.direction.length()
        };

        scattered.origin = rec.p;
        let refracted_maybe = refract(&ray.direction, &outward_normal, ni_over_nt);
        match &refracted_maybe {
            None => {
                scattered.direction = reflected;
            }
            Some(refracted) => {
                let reflect_prob = schlick(cosine, self.refraction_index);
                if rnd.gen() < reflect_prob {
                    scattered.direction = reflected
                } else {
                    scattered.direction.set(refracted)
                }
            }
        };

        true
    }
}
