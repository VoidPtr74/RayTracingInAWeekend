use std::arch::x86_64::*;
use std::f32;
use std::fmt;
use std::ops;

use crate::rng::Random;

#[derive(Copy, Clone)]
pub union Vec3 {
    array: [f32; 4],
    sse: __m128,
}

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe { write!(f, "{:?}", self.array) }
    }
}

impl Vec3 {
    pub fn from4(x: f32, y: f32, z: f32, w: f32) -> Self {
        unsafe {
            Self {
                sse: _mm_set_ps(w, z, y, x),
            }
        }
    }

    pub fn from(x: f32, y: f32, z: f32) -> Self {
        Self::from4(x, y, z, 0.0)
    }

    pub fn default() -> Self {
        unsafe {
            Self {
                sse: _mm_set1_ps(0.0),
            }
        }
    }

    pub fn set(&mut self, other: &Vec3) {
        unsafe {
            self.sse = other.sse;
        }
    }

    // "Public" interface
    pub fn x(&self) -> &f32 {
        self.get(0)
    }

    pub fn y(&self) -> &f32 {
        self.get(1)
    }

    pub fn z(&self) -> &f32 {
        self.get(2)
    }

    pub fn r(&self) -> &f32 {
        self.get(0)
    }

    pub fn g(&self) -> &f32 {
        self.get(1)
    }

    pub fn b(&self) -> &f32 {
        self.get(2)
    }

    pub fn length(&self) -> f32 {
        self.square_length().sqrt()
    }

    pub fn square_length(&self) -> f32 {
        unsafe {
            let v = &self.array;
            v[0].mul_add(v[0], v[1].mul_add(v[1], v[2].mul_add(v[2], 0.0)))
        }
    }

    pub fn normalise(&mut self) {
        *self = self.direct_product(&self.inv_len())
    }

    pub fn make_normalised(&self) -> Self {
        self.direct_product(&self.inv_len())
    }

    pub fn direct_product(&self, other: &Self) -> Self {
        unsafe {
            Self {
                sse: _mm_mul_ps(self.sse, other.sse),
            }
        }
    }

    pub fn invert_elems(&self) -> Self {
        unsafe {
            let numerators = _mm_set1_ps(1.0);
            Self {
                sse: _mm_div_ps(numerators, self.sse),
            }
        }
    }

    pub fn max_elem(&self) -> f32 {
        unsafe { ffmax(self.array[0], ffmax(self.array[1], self.array[2])) }
    }

    pub fn min_elem(&self) -> f32 {
        unsafe { ffmin(self.array[0], ffmin(self.array[1], self.array[2])) }
    }

    pub fn max(&self, other: &Vec3) -> Vec3 {
        unsafe {
            Vec3 {
                sse: _mm_max_ps(self.sse, other.sse),
            }
        }
    }

    pub fn min(&self, other: &Vec3) -> Vec3 {
        unsafe {
            Vec3 {
                sse: _mm_min_ps(self.sse, other.sse),
            }
        }
    }

    pub fn get(&self, index: usize) -> &f32 {
        unsafe { &self.array[index] }
    }
}

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

impl ops::Add<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn add(self, other: &Vec3) -> Vec3 {
        unsafe { self.add_sse(other) }
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        unsafe { self.add_sse(&other) }
    }
}

impl ops::AddAssign<&Vec3> for Vec3 {
    fn add_assign(&mut self, other: &Vec3) {
        unsafe { self.add_self(other) }
    }
}

impl ops::Sub<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, other: &Vec3) -> Vec3 {
        unsafe { self.sub_sse(other) }
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        unsafe { self.sub_sse(&other) }
    }
}

impl ops::SubAssign<&Vec3> for Vec3 {
    fn sub_assign(&mut self, other: &Vec3) {
        unsafe { self.sub_self(other) }
    }
}

impl ops::Mul<f32> for &Vec3 {
    type Output = Vec3;

    fn mul(self, other: f32) -> Vec3 {
        unsafe { self.mul_sse(other) }
    }
}

impl ops::MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, other: f32) {
        unsafe { self.mul_self(other) }
    }
}

impl ops::Div<f32> for &Vec3 {
    type Output = Vec3;

    fn div(self, other: f32) -> Vec3 {
        unsafe { self.div_sse(other) }
    }
}

impl ops::DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, other: f32) {
        unsafe { self.div_self(other) }
    }
}

// "Private" implementation details
impl Vec3 {
    fn inv_len(&self) -> Self {
        unsafe {
            let length = _mm_dp_ps(self.sse, self.sse, 0x7f);
            Self {
                sse: _mm_rsqrt_ps(length),
            }
        }
    }

    unsafe fn add_self(&mut self, other: &Self) {
        self.sse = _mm_add_ps(self.sse, other.sse);
    }

    unsafe fn add_sse(&self, other: &Vec3) -> Self {
        Self {
            sse: _mm_add_ps(self.sse, other.sse),
        }
    }

    unsafe fn sub_self(&mut self, other: &Self) {
        self.sse = _mm_sub_ps(self.sse, other.sse);
    }

    unsafe fn sub_sse(&self, other: &Self) -> Self {
        Self {
            sse: _mm_sub_ps(self.sse, other.sse),
        }
    }

    unsafe fn mul_self(&mut self, other: f32) {
        let t = _mm_set1_ps(other);
        self.sse = _mm_mul_ps(self.sse, t);
    }

    unsafe fn mul_sse(&self, other: f32) -> Self {
        let t = _mm_set1_ps(other);
        Self {
            sse: _mm_mul_ps(self.sse, t),
        }
    }

    unsafe fn div_self(&mut self, other: f32) {
        let t = _mm_set1_ps(other);
        self.sse = _mm_div_ps(self.sse, t);
    }

    unsafe fn div_sse(&self, other: f32) -> Self {
        let t = _mm_set1_ps(other);
        Self {
            sse: _mm_div_ps(self.sse, t),
        }
    }
}

pub fn random_in_unit_sphere(rnd: &mut Random) -> Vec3 {
    loop {
        let p = Vec3::from(rnd.gen(), rnd.gen(), rnd.gen());
        if p.square_length() <= 1.0 {
            break p;
        }
    }
}

pub fn cross(v1: &Vec3, v2: &Vec3) -> Vec3 {
    Vec3::from(
        v1.y() * v2.z() - v1.z() * v2.y(),
        -v1.x() * v2.z() + v1.z() * v2.x(),
        v1.x() * v2.y() - v1.y() * v2.x(),
    )
}

pub fn dot(v1: &Vec3, v2: &Vec3) -> f32 {
    unsafe {
        let v = v1.array;
        let w = v2.array;
        v[0].mul_add(w[0], v[1].mul_add(w[1], v[2].mul_add(w[2], 0.0)))
    }
}

pub fn reflect(incident: &Vec3, normal: &Vec3) -> Vec3 {
    incident - &(normal * (2.0 * dot(incident, normal)))
}

pub fn refract(incident: &Vec3, normal: &Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let uv = incident.make_normalised();
    let dt = dot(&uv, normal);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);

    if discriminant > 0.0 {
        return Option::Some(&(uv - (normal * dt)) * ni_over_nt - normal * discriminant.sqrt());
    }

    Option::None
}
