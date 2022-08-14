use std::{collections::HashMap, path::Path};

use super::{ldm::latent_diffusion, Model};
use crate::result::{Error, Result};

#[derive(Debug)]
pub struct DiffusionScheduleParam {
    pub timestep: usize,
    pub alpha_cumprod: f64,
    pub alpha_cumprod_prev: f64,
    pub sigma: f64,
}

pub trait Diffusion: Model {
    fn make_schedule(&self, num_steps: usize) -> Vec<DiffusionScheduleParam>;
    fn make_noise(&self, b: usize, w: usize, h: usize) -> ndarray::ArrayD<f32>;
    fn execute(
        &mut self,
        x: &ndarray::ArrayD<f32>,
        t: &DiffusionScheduleParam,
        conditions: &HashMap<String, ndarray::ArrayD<f32>>,
    ) -> Result<ndarray::ArrayD<f32>>;
}

pub fn load_diffusion(kind: impl AsRef<str>, path: impl AsRef<Path>) -> Result<Box<dyn Diffusion>> {
    match kind.as_ref() {
        "ldm/ldm" => Ok(Box::new(latent_diffusion::LatentDiffusion::new(
            path.as_ref(),
        )?)),

        k => Err(Error::UnsupportedModel("diffuse".to_string(), k.to_owned())),
    }
}
