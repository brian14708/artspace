use std::path::Path;

use super::{esrgan::Esrgan, swinir::SwinIR, Model};
use crate::result::{Error, Result};

pub trait SuperResolution: Model {
    fn execute(&mut self, x: &ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>>;
}

pub fn load_super_resolution(
    kind: impl AsRef<str>,
    path: impl AsRef<Path>,
) -> Result<Box<dyn SuperResolution>> {
    match kind.as_ref() {
        "swinir" => Ok(Box::new(SwinIR::new(path.as_ref())?)),
        "esrgan" => Ok(Box::new(Esrgan::new(path.as_ref())?)),

        k => Err(Error::UnsupportedModel(
            "text encoder".to_string(),
            k.to_owned(),
        )),
    }
}
