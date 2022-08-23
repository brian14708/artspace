use std::path::Path;

use super::{clip, ldm::bert, Model};
use crate::result::{Error, Result};

pub trait TextEncoder: Model {
    fn tokenize(&mut self, inp: &str) -> Result<tokenizers::Encoding>;
    fn encode(&mut self, enc: &[tokenizers::Encoding]) -> Result<ndarray::ArrayD<f32>>;
}

pub fn load_text_encoder(
    kind: impl AsRef<str>,
    path: impl AsRef<Path>,
) -> Result<Box<dyn TextEncoder>> {
    match kind.as_ref() {
        "ldm/bert" => Ok(Box::new(bert::BertEncoder::new(path.as_ref())?)),
        "clip-pos" => Ok(Box::new(clip::ClipEncoder::new(path.as_ref(), 1)?)),
        "clip" => Ok(Box::new(clip::ClipEncoder::new(path.as_ref(), 0)?)),
        k => Err(Error::UnsupportedModel(
            "text encoder".to_string(),
            k.to_owned(),
        )),
    }
}
