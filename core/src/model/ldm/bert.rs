use std::{io::Read, path::PathBuf, str::FromStr};

use tokenizers::Tokenizer;

use crate::{
    model::{text_encoder::TextEncoder, Model},
    ort::Session,
    result::{Error, Result},
};

pub struct BertEncoder {
    tokenizer: Tokenizer,
    path: PathBuf,
    session: Option<Session>,
}

impl TextEncoder for BertEncoder {
    fn tokenize(&mut self, inp: &str) -> Result<tokenizers::Encoding> {
        self.tokenizer
            .encode(inp, true)
            .map_err(|e| Error::Tokenizer(e))
    }

    fn encode(&mut self, enc: &[tokenizers::Encoding]) -> Result<ndarray::ArrayD<f32>> {
        if enc.is_empty() {
            return Ok(ndarray::ArrayD::zeros([].as_slice()));
        }
        assert!(enc.iter().all(|e| e.len() == enc[0].len()));

        let session = if let Some(session) = &self.session {
            session
        } else {
            self.session.insert(Session::load(&self.path, "bert.onnx")?)
        };

        let enc = ndarray::Array2::from_shape_vec(
            (enc.len(), enc[0].len()),
            enc.iter()
                .flat_map(|e| e.get_ids())
                .map(|&v| v as i64)
                .collect(),
        )
        .unwrap();

        let mut run = session.prepare();
        run.set_input("x", &enc)?;
        let out = run.exec()?;
        let out = out.get_output_idx::<f32, ndarray::Ix3>(0)?;
        Ok(out.into_dyn().to_owned())
    }
}

impl Model for BertEncoder {
    fn unload_model(&mut self) {
        self.session = None;
    }
}

impl BertEncoder {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let tokenizer_json = if path.is_dir() {
            std::fs::read_to_string(path.join("tokenizer.json"))?
        } else {
            let mut ar = tsar::Archive::new(std::fs::File::open(&path)?)?;
            let mut buf = String::new();
            ar.file_by_name("tokenizer.json")?
                .read_to_string(&mut buf)?;
            buf
        };

        let mut tokenizer =
            Tokenizer::from_str(&tokenizer_json).map_err(|e| Error::Tokenizer(e))?;
        tokenizer
            .with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(77),
                ..Default::default()
            }))
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 77,
                ..Default::default()
            }));

        Ok(Self {
            tokenizer,
            path,
            session: None,
        })
    }
}
