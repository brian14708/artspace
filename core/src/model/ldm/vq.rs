use std::{io::Read, path::PathBuf};

use serde::Deserialize;

use crate::{
    model::{AutoEncoder, Model},
    ort::Session,
    result::Result,
};

pub struct Vq {
    metadata: Metadata,
    path: PathBuf,
    encoder_session: Option<Session>,
    decoder_session: Option<Session>,
}

#[derive(Deserialize)]
struct Metadata {
    scale_factor: f64,
}

impl AutoEncoder for Vq {
    fn encode(&mut self, x: &ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>> {
        if x.is_empty() {
            return Ok(ndarray::ArrayD::zeros([].as_slice()));
        }

        let session = if let Some(session) = &self.encoder_session {
            session
        } else {
            self.encoder_session
                .insert(Session::load(&self.path, "encoder.onnx", true)?)
        };
        let x = x * 2.0 - 1.0;

        let mut run = session.prepare();
        run.set_input("img", &x)?;
        let out = run.exec(true)?;
        let out = out.get_output_idx::<f32, ndarray::Ix4>(0)?;
        let ch = out.shape()[1];
        let mean = out.slice_axis(ndarray::Axis(1), ndarray::Slice::from(0..ch / 2));
        Ok(mean.to_owned().into_dyn() * self.metadata.scale_factor as f32)
    }

    fn decode(&mut self, x: &ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>> {
        if x.is_empty() {
            return Ok(ndarray::ArrayD::zeros([].as_slice()));
        }

        let session = if let Some(session) = &self.decoder_session {
            session
        } else {
            self.decoder_session
                .insert(Session::load(&self.path, "decoder.onnx", true)?)
        };
        let x = (1. / self.metadata.scale_factor) as f32 * x;

        let mut run = session.prepare();
        run.set_input("z", &x)?;
        let out = run.exec(true)?;
        let out = out.get_output_idx::<f32, ndarray::Ix4>(0)?;
        let out = ((out.to_owned() + 1.0) / 2.0).mapv(|v| v.clamp(0.0, 1.0));
        Ok(out.into_dyn().to_owned())
    }
}

impl Model for Vq {
    fn unload_model(&mut self) {
        self.encoder_session = None;
        self.decoder_session = None;
    }
}

impl Vq {
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
            encoder_session: None,
            decoder_session: None,
        })
    }
}
