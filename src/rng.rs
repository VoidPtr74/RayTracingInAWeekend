extern crate rand;

use rand::distributions::*;
use rand::rngs::*;
use rand::*;

pub struct Random {
    rng: SmallRng,
    dist: Uniform<f32>,
}

impl Random {
    pub fn create_with_seed(seed: u64) -> Random {
        Random {
            rng: SmallRng::seed_from_u64(seed),
            dist: Uniform::new(0.0f32, 1.0f32),
        }
    }

    pub fn gen(&mut self) -> f32 {
        self.dist.sample(&mut self.rng)
    }
}
