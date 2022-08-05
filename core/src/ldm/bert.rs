use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use tokenizers::Tokenizer;

use crate::ort::Session;
use crate::result::Error;
use crate::result::Result;
use crate::text_encoder::TextEncoder;

pub struct BertEncoder {
    tokenizer: Tokenizer,
    session: Session,
}

impl TextEncoder for BertEncoder {
    fn encode(&mut self, inp: &[String]) -> Result<ndarray::ArrayD<f32>> {
        let enc = self
            .tokenizer
            .encode_batch(inp.iter().map(|s| s.as_str()).collect(), true)
            .map_err(|e| Error::Tokenizer(e))?;
        if enc.is_empty() {
            return Ok(ndarray::ArrayD::zeros([0].as_slice()));
        }
        let enc = ndarray::Array2::from_shape_vec(
            (enc.len(), enc[0].len()),
            enc.iter()
                .flat_map(|e| e.get_ids())
                .map(|&v| v as i64)
                .collect(),
        )
        .unwrap();

        let mut run = self.session.prepare();
        run.set_input("x", &enc)?;
        let out = run.exec()?;
        let out = out.get_output_idx::<f32, ndarray::Ix3>(0)?;
        Ok(out.into_dyn().to_owned())
    }
}

impl BertEncoder {
    pub fn new(path: &Path) -> Result<Self> {
        let mut ar = tsar::Archive::new(std::fs::File::open(path)?)?;
        let mut buf = String::new();
        ar.file_by_name("tokenizer.json")?
            .read_to_string(&mut buf)?;
        let mut tokenizer = Tokenizer::from_str(&buf).map_err(|e| Error::Tokenizer(e))?;
        tokenizer
            .with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(77),
                ..Default::default()
            }))
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 77,
                ..Default::default()
            }));

        let session = Session::load(&mut ar, "bert.onnx")?;
        Ok(Self { tokenizer, session })
    }
}
