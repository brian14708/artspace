use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use artspace_core::ddim_sampler::DdimSampler;
use ndarray::{Axis, Slice};

use crate::model_manager::ModelManager;

pub struct Pipeline {
    text_encoders: Vec<Arc<Mutex<Box<dyn artspace_core::model::TextEncoder>>>>,
    diffuse: Box<dyn artspace_core::model::Diffusion>,
    autoencoder: Box<dyn artspace_core::model::AutoEncoder>,
    steps: usize,
    sr: Option<Box<dyn artspace_core::model::SuperResolution>>,

    text_embedding: Option<Vec<ndarray::ArrayD<f32>>>,
}

impl Pipeline {
    pub async fn new(kind: &str, mm: &ModelManager, progress: impl Fn(String)) -> Result<Self> {
        if kind == "small" {
            Ok(Self {
                text_encoders: vec![
                    Arc::new(Mutex::new(artspace_core::model::load_text_encoder(
                        "clip",
                        mm.download("clip/vit-l-14.int8.tsar", &progress).await?,
                    )?)),
                    Arc::new(Mutex::new(artspace_core::model::load_text_encoder(
                        "ldm/bert",
                        mm.download("ldm/text2img-large/bert.int8.tsar", &progress)
                            .await?,
                    )?)),
                ],
                diffuse: artspace_core::model::load_diffusion(
                    "ldm/ldm-clip",
                    mm.download("ldm/glid-3-xl/ldm.int8.tsar", &progress)
                        .await?,
                )?,
                autoencoder: artspace_core::model::load_auto_encoder(
                    "ldm/vq",
                    mm.download("ldm/text2img-large/vq.tsar", &progress).await?,
                )?,
                steps: 45,
                sr: Some(artspace_core::model::load_super_resolution(
                    "esrgan",
                    mm.download("esrgan/x4plus.tsar", &progress).await?,
                )?),

                text_embedding: None,
            })
        } else {
            Err(anyhow::anyhow!("unknown pipeline kind: {}", kind))
        }
    }

    pub async fn step_text(&mut self, s: &str) -> Result<()> {
        let f = futures::future::join_all(self.text_encoders.iter_mut().map(|e| {
            let s = s.to_owned();
            let e = e.clone();
            async_std::task::spawn_blocking(move || -> Result<ndarray::ArrayD<f32>> {
                let mut e = e.lock().unwrap();
                let enc = e.tokenize(&s)?;
                let uncond_enc = e.tokenize("")?;
                Ok(e.encode(&[uncond_enc, enc])?)
            })
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

        self.text_embedding = Some(f);
        Ok(())
    }

    pub async fn step_diffuse(
        &mut self,
        w: f32,
        h: f32,
        progress: impl Fn(String),
    ) -> Result<ndarray::ArrayD<f32>> {
        self.text_encoders
            .iter_mut()
            .for_each(|e| e.lock().unwrap().unload_model());

        let min = w.min(h);

        let noise = self.diffuse.make_noise(
            1,
            (w / min * 256.).round() as usize / 32 * 32,
            (h / min * 256.).round() as usize / 32 * 32,
        );
        let sched = self.diffuse.make_schedule(self.steps);

        let cond: HashMap<_, _> = [
            (
                "clip".to_owned(),
                self.text_embedding.as_ref().unwrap()[0]
                    .slice_axis(Axis(0), Slice::from(1usize..))
                    .to_owned(),
            ),
            (
                "c".to_owned(),
                self.text_embedding.as_ref().unwrap()[1]
                    .slice_axis(Axis(0), Slice::from(1usize..))
                    .to_owned(),
            ),
        ]
        .into_iter()
        .collect();
        let uncond: HashMap<_, _> = [
            (
                "clip".to_owned(),
                self.text_embedding.as_ref().unwrap()[0]
                    .slice_axis(Axis(0), Slice::from(..1usize))
                    .to_owned(),
            ),
            (
                "c".to_owned(),
                self.text_embedding.as_ref().unwrap()[1]
                    .slice_axis(Axis(0), Slice::from(..1usize))
                    .to_owned(),
            ),
        ]
        .into_iter()
        .collect();

        let d = {
            let mut d = DdimSampler::new(self.diffuse.as_mut(), cond, uncond, noise);
            for (i, s) in sched.iter().enumerate() {
                progress(format!("Diffusion step {}/{}", i + 1, sched.len()));
                d.next(s);
            }
            d.seed
        };

        progress("Decoding image...".to_string());
        Ok(self.autoencoder.decode(&d)?)
    }

    pub async fn step_post_process(
        &mut self,
        image: &ndarray::ArrayD<f32>,
    ) -> Result<ndarray::ArrayD<f32>> {
        self.diffuse.unload_model();
        Ok(if let Some(sr) = &mut self.sr {
            sr.execute(image)?
        } else {
            image.to_owned()
        })
    }

    pub fn get_png(&self, image: &ndarray::ArrayD<f32>) -> Vec<u8> {
        let image = image
            .mapv(|f| (f * 255.0) as u8)
            .permuted_axes([0, 2, 3, 1].as_slice());
        let image = image.as_standard_layout();

        let mut out = vec![];
        let mut encoder =
            png::Encoder::new(&mut out, image.shape()[2] as u32, image.shape()[1] as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(image.index_axis(Axis(0), 0).as_slice().unwrap())
            .unwrap();
        out
    }
}
