use std::path::Path;

use crate::{
    ldm::{bert, clip},
    result::{Error, Result},
};

pub trait TextEncoder {
    fn encode(&mut self, inp: &[String]) -> Result<ndarray::ArrayD<f32>>;
}

pub fn load(kind: impl AsRef<str>, path: impl AsRef<Path>) -> Result<Box<dyn TextEncoder>> {
    match kind.as_ref() {
        "ldm/bert" => Ok(Box::new(bert::BertEncoder::new(path.as_ref())?)),
        "ldm/clip" => Ok(Box::new(clip::ClipEncoder::new(path.as_ref())?)),
        k => Err(Error::UnsupportedModel(
            "text encoder".to_string(),
            k.to_owned(),
        )),
    }
}
