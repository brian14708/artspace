use std::{collections::HashMap, io::Read, path::PathBuf};

use ndarray::{Array, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use serde::Deserialize;
use smallvec::SmallVec;

use crate::{
    model::{Diffusion, DiffusionScheduleParam, Model},
    ort::{DataType, Session, TensorInfo},
    result::{Error, Result},
};

pub struct LatentDiffusion {
    metadata: Metadata,
    path: PathBuf,
    session: Option<Session>,
    input_types: HashMap<String, TensorInfo>,
}

#[derive(Deserialize)]
struct Metadata {
    beta: BetaParam,
    normalize_condition: Option<bool>,
    num_channels: Option<usize>,
    image_scale: Option<usize>,
    timesteps: usize,
}

#[derive(Deserialize)]
struct BetaParam {
    start: f64,
    end: f64,
    schedule: String,
}

impl Diffusion for LatentDiffusion {
    fn make_schedule(&self, num_steps: usize) -> Vec<DiffusionScheduleParam> {
        if num_steps == 0 {
            return Vec::new();
        }

        assert_eq!(self.metadata.beta.schedule, "linear");

        let betas = (0..self.metadata.timesteps).map(|i| {
            (self.metadata.beta.start.sqrt()
                + (self.metadata.beta.end.sqrt() - self.metadata.beta.start.sqrt()) * i as f64
                    / (self.metadata.timesteps - 1) as f64)
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

        let timesteps: Vec<_> = if false {
            // original
            let num_steps = num_steps.min(self.metadata.timesteps);
            (0..self.metadata.timesteps)
                .step_by(self.metadata.timesteps / num_steps)
                .map(|x| x + 1)
                .collect()
        } else {
            // power function
            (0..num_steps)
                .map(|i| {
                    (((i as f64) / ((num_steps - 1) as f64)).powf(1.2)
                        * (self.metadata.timesteps as f64 - 2.))
                        .round() as usize
                        + 1
                })
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
        Array::random(
            (
                b,
                self.metadata.num_channels.unwrap_or(4),
                h / self.metadata.image_scale.unwrap_or(8),
                w / self.metadata.image_scale.unwrap_or(8),
            ),
            Normal::new(0.0, 1.0).unwrap(),
        )
        .into_dyn()
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
            let s = self
                .session
                .insert(Session::load(&self.path, "ldm.onnx", false)?);
            self.input_types = s.inputs()?;
            s
        };

        enum TimeInput {
            F32(ndarray::Array1<f32>),
            I64(ndarray::Array1<i64>),
        }
        let ti: TimeInput = match self.input_types["t"].elem_type {
            DataType::Float32 => {
                let mut tt = ndarray::Array1::<f32>::zeros((x.shape()[0],));
                tt.fill(t.timestep as f32);
                TimeInput::F32(tt)
            }
            DataType::Int64 => {
                let mut tt = ndarray::Array1::<i64>::zeros((x.shape()[0],));
                tt.fill(t.timestep as i64);
                TimeInput::I64(tt)
            }
            t => {
                return Err(Error::InvalidInput(format!(
                    "Unsupported time type: {:?}",
                    t
                )))
            }
        };

        let mut temp: HashMap<String, ndarray::ArrayD<f32>> = HashMap::new();

        let mut run = session.prepare();
        run.set_input("x", x)?;
        match &ti {
            TimeInput::F32(t) => run.set_input("t", t)?,
            TimeInput::I64(t) => run.set_input("t", t)?,
        }

        if self.input_types.get("img").is_some() {
            let zero = ndarray::ArrayD::<f32>::zeros(x.shape());
            temp.insert("_img".to_string(), zero);
        }

        if let Some(true) = self.metadata.normalize_condition {
            for (k, v) in conditions {
                let mut norm = v.mapv(|v| v * v);
                while norm.shape().len() > 1 {
                    norm = norm.sum_axis(Axis(norm.ndim() - 1));
                }
                norm.map_inplace(|v| *v = v.sqrt());

                let shape: SmallVec<[usize; 4]> = v
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| if i == 0 { s } else { 1 })
                    .collect();
                let norm = norm.into_shape(shape.as_slice()).unwrap();
                temp.insert(k.clone(), v / norm);
            }
        }
        for (k, v) in conditions {
            if let Some(t) = self.input_types.get(k) {
                let d = t.shape.len() as isize - v.shape().len() as isize;
                match d.cmp(&0) {
                    std::cmp::Ordering::Less => {
                        return Err(Error::InvalidInput(format!(
                            "Condition {:?} has {} dimensions, but model expects {}",
                            k,
                            v.shape().len(),
                            t.shape.len()
                        )));
                    }
                    std::cmp::Ordering::Equal => {}
                    std::cmp::Ordering::Greater => {
                        let mut v = temp.remove(k).unwrap_or_else(|| v.clone());
                        for _ in 0..d {
                            v = v.insert_axis(Axis(1));
                        }
                        temp.insert(k.clone(), v);
                    }
                }
            }
        }
        for (k, v) in conditions {
            if self.input_types.get(k).is_some() {
                if let Some(vv) = temp.get(k) {
                    run.set_input(k, vv)?;
                } else {
                    run.set_input(k, v)?;
                }
            }
        }

        for (k, v) in &temp {
            if let Some(k) = k.strip_prefix('_') {
                run.set_input(k, v)?;
            }
        }

        let ret = run.exec(t.timestep == 1)?;
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
        let path = path.into();
        let metadata_json = if path.is_dir() {
            std::fs::read_to_string(path.join("metadata.json"))?
        } else {
            let mut ar = tsar::Archive::new(std::fs::File::open(&path)?)?;
            let mut buf = String::new();
            ar.file_by_name("metadata.json")?.read_to_string(&mut buf)?;
            buf
        };

        let metadata: Metadata = serde_json::from_str(&metadata_json)?;

        Ok(Self {
            metadata,
            path,
            session: None,
            input_types: HashMap::new(),
        })
    }
}
