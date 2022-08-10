use std::path::PathBuf;

use super::{Model, SuperResolution};
use crate::{ort::Session, result::Result};

pub struct Esrgan {
    path: PathBuf,
    session: Option<Session>,
}

impl SuperResolution for Esrgan {
    fn execute(&mut self, x: &ndarray::ArrayD<f32>) -> crate::result::Result<ndarray::ArrayD<f32>> {
        if x.is_empty() {
            return Ok(ndarray::ArrayD::zeros([].as_slice()));
        }

        let session = if let Some(session) = &self.session {
            session
        } else {
            self.session.insert(Session::load(&self.path, "sr.onnx")?)
        };

        let mut run = session.prepare();
        run.set_input("input", x)?;
        let out = run.exec()?;
        let out = out.get_output_idx::<f32, ndarray::Ix4>(0)?;
        let out = out.mapv(|v| f32::min(f32::max(v, 0.0), 1.0));
        Ok(out.into_dyn().to_owned())
    }
}

impl Model for Esrgan {}

impl Esrgan {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            path: path.into(),
            session: None,
        })
    }
}
