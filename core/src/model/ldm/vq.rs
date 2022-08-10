use std::path::PathBuf;

use crate::{
    model::{AutoEncoder, Model},
    ort::Session,
    result::Result,
};

pub struct Vq {
    path: PathBuf,
    session: Option<Session>,
}

impl AutoEncoder for Vq {
    fn decode(&mut self, x: &ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>> {
        if x.is_empty() {
            return Ok(ndarray::ArrayD::zeros([].as_slice()));
        }

        let session = if let Some(session) = &self.session {
            session
        } else {
            self.session
                .insert(Session::load(&self.path, "decoder.onnx")?)
        };
        let x = (1. / 0.18215) as f32 * x;

        let mut run = session.prepare();
        run.set_input("z", &x)?;
        let out = run.exec()?;
        let out = out.get_output_idx::<f32, ndarray::Ix4>(0)?;
        let out = ((out.to_owned() + 1.0) / 2.0).mapv(|v| f32::min(f32::max(v, 0.0), 1.0));
        Ok(out.into_dyn().to_owned())
    }
}

impl Model for Vq {
    fn unload_model(&mut self) {
        self.session = None;
    }
}

impl Vq {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            path: path.into(),
            session: None,
        })
    }
}
