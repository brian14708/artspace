use std::path::PathBuf;

use itertools::Itertools;
use ndarray::Array;
use ndarray_rand::{rand_distr::Normal, RandomExt};

use crate::{
    model::{Diffusion, DiffusionScheduleParam, Model},
    ort::Session,
    result::Result,
};

pub struct LatentDiffusion {
    path: PathBuf,
    session: Option<Session>,
}

impl Diffusion for LatentDiffusion {
    fn make_schedule(&self, num_steps: usize) -> Vec<DiffusionScheduleParam> {
        if num_steps == 0 {
            return Vec::new();
        }

        // TODO store in json
        const BETA_START: f64 = 0.00085;
        const BETA_END: f64 = 0.012;
        const NUM_STEPS: usize = 1000;

        let betas = (0..NUM_STEPS).map(|i| {
            (BETA_START.sqrt()
                + (BETA_END.sqrt() - BETA_START.sqrt()) * i as f64 / (NUM_STEPS - 1) as f64)
                .powi(2)
        });

        let alphas_cumprod: Vec<_> = Some(1.)
            .into_iter()
            .chain(betas.map(|beta| 1. - beta).scan(1., |prod, alpha| {
                *prod *= alpha;
                Some(*prod)
            }))
            .chain(Some(0.))
            .collect();

        let timesteps: Vec<_> = if true {
            // original
            let num_steps = num_steps.min(NUM_STEPS);
            (0..NUM_STEPS)
                .step_by(NUM_STEPS / num_steps)
                .map(|x| x + 1)
                .collect()
        } else {
            // uniform
            (0..num_steps)
                .map(|i| {
                    1 + ((NUM_STEPS - 1) as f64 * (i as f64) / (num_steps as f64)).round() as usize
                })
                .dedup()
                .collect()
        };

        let ddim_alphas: Vec<_> = timesteps
            .iter()
            .rev()
            .map(|&i| alphas_cumprod[i + 1])
            .collect();
        let ddim_alphas_prev: Vec<_> = timesteps
            .iter()
            .take(timesteps.len() - 1)
            .rev()
            .map(|&i| alphas_cumprod[i + 1])
            .chain(Some(alphas_cumprod[1]))
            .collect();
        let ddim_sigmas: Vec<_> = ddim_alphas_prev
            .iter()
            .zip(ddim_alphas.iter())
            .map(|(&prev, &alpha)| ((1. - prev) / (1. - alpha) * (1. - alpha / prev)).sqrt())
            .collect();

        timesteps
            .iter()
            .rev()
            .enumerate()
            .map(|(i, &t)| DiffusionScheduleParam {
                timestep: t,
                alpha_cumprod: ddim_alphas[i],
                alpha_cumprod_prev: ddim_alphas_prev[i],
                sigma: ddim_sigmas[i],
            })
            .collect()
    }

    fn make_noise(&self, b: usize, w: usize, h: usize) -> ndarray::ArrayD<f32> {
        Array::random((b, 4, h / 8, w / 8), Normal::new(0.0, 1.0).unwrap()).into_dyn()
    }

    fn execute(
        &mut self,
        x: &ndarray::ArrayD<f32>,
        t: &DiffusionScheduleParam,
        conditions: &std::collections::HashMap<String, ndarray::ArrayD<f32>>,
    ) -> Result<ndarray::ArrayD<f32>> {
        if x.is_empty() {
            return Ok(ndarray::ArrayD::zeros([].as_slice()));
        }

        let session = if let Some(session) = &self.session {
            session
        } else {
            self.session.insert(Session::load(&self.path, "ldm.onnx")?)
        };

        let mut tt = ndarray::Array1::<i64>::zeros((x.shape()[0],));
        tt.fill(t.timestep as i64);
        let mut run = session.prepare();
        run.set_input("t", &tt)?;
        run.set_input("x", x)?;
        for (k, v) in conditions {
            run.set_input(k, v)?;
        }
        let ret = run.exec()?;
        let y = ret.get_output_idx::<f32, ndarray::Ix4>(0)?;
        Ok(y.into_dyn().to_owned())
    }
}

impl Model for LatentDiffusion {
    fn unload_model(&mut self) {
        self.session = None;
    }
}

impl LatentDiffusion {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            path: path.into(),
            session: None,
        })
    }
}
