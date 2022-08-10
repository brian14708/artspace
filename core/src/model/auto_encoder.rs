use std::path::Path;

use super::{ldm::vq, Model};
use crate::result::{Error, Result};

pub trait AutoEncoder: Model {
    fn decode(&mut self, x: &ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>>;
}

pub fn load_auto_encoder(
    kind: impl AsRef<str>,
    path: impl AsRef<Path>,
) -> Result<Box<dyn AutoEncoder>> {
    match kind.as_ref() {
        "ldm/vq" => Ok(Box::new(vq::Vq::new(path.as_ref())?)),

        k => Err(Error::UnsupportedModel("diffuse".to_string(), k.to_owned())),
    }
}
