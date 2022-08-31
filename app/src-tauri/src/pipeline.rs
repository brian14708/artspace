use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use artspace_core::sampler::LmsSampler;
use ndarray::{Axis, Slice};
use nshare::ToNdarray3;

use crate::model_manager::ModelManager;

struct TextEncoder {
    model: Arc<Mutex<Box<dyn artspace_core::model::TextEncoder>>>,
    key: String,
}

pub struct Pipeline {
    text_encoders: Vec<TextEncoder>,
    diffuse: Box<dyn artspace_core::model::Diffusion>,
    autoencoder: Box<dyn artspace_core::model::AutoEncoder>,
    diffuse_output_size: (usize, usize),
    steps: usize,
    sr: Option<Box<dyn artspace_core::model::SuperResolution>>,

    text_embedding: Option<Vec<ndarray::ArrayD<f32>>>,
}

impl Pipeline {
    pub async fn new(kind: &str, mm: &ModelManager, progress: impl Fn(String)) -> Result<Self> {
        if kind == "small" {
            Ok(Self {
                text_encoders: vec![
                    TextEncoder {
                        model: Arc::new(Mutex::new(artspace_core::model::load_text_encoder(
                            "clip",
                            mm.download("clip/vit-l-14.tsar", &progress).await?,
                        )?)),
                        key: "clip".to_string(),
                    },
                    TextEncoder {
                        model: Arc::new(Mutex::new(artspace_core::model::load_text_encoder(
                            "ldm/bert",
                            mm.download("ldm/text2img-large/bert.int8.tsar", &progress)
                                .await?,
                        )?)),
                        key: "c".to_string(),
                    },
                ],
                diffuse: artspace_core::model::load_diffusion(
                    "ldm/ldm",
                    if std::env::var("USE_FP16").is_ok() {
                        mm.download("ldm/glid-3-xl/ldm.fp16.tsar", &progress)
                            .await?
                    } else {
                        mm.download("ldm/glid-3-xl/ldm.int8.tsar", &progress)
                            .await?
                    },
                )?,
                diffuse_output_size: (256, 32),
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
        } else if kind == "large" {
            Ok(Self {
                text_encoders: vec![TextEncoder {
                    model: Arc::new(Mutex::new(artspace_core::model::load_text_encoder(
                        "clip-pos",
                        mm.download("clip/vit-l-14.tsar", &progress).await?,
                    )?)),
                    key: "c".to_string(),
                }],
                diffuse: artspace_core::model::load_diffusion(
                    "ldm/ldm",
                    if std::env::var("USE_FP16").is_ok() {
                        mm.download("stable-diffusion/unet.fp16.tsar", &progress)
                            .await?
                    } else {
                        mm.download("stable-diffusion/unet.int8.tsar", &progress)
                            .await?
                    },
                )?,
                diffuse_output_size: (512, 64),
                autoencoder: artspace_core::model::load_auto_encoder(
                    "ldm/vq",
                    mm.download("stable-diffusion/vae.tsar", &progress).await?,
                )?,
                steps: 45,
                sr: None,

                text_embedding: None,
            })
        } else {
            Err(anyhow::anyhow!("unknown pipeline kind: {}", kind))
        }
    }

    pub fn list() -> Vec<String> {
        ["small", "large"].iter().map(|s| s.to_string()).collect()
    }

    pub async fn step_text(&mut self, s: &str) -> Result<()> {
        let f = futures::future::join_all(self.text_encoders.iter_mut().map(|e| {
            let s = s.to_owned();
            let model = e.model.clone();
            async_std::task::spawn_blocking(move || -> Result<ndarray::ArrayD<f32>> {
                let mut e = model.lock().unwrap();
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
        seed: Option<(ndarray::ArrayD<f32>, f32)>,
        progress: impl Fn(String),
    ) -> Result<ndarray::ArrayD<f32>> {
        self.text_encoders
            .iter_mut()
            .for_each(|e| e.model.lock().unwrap().unload_model());

        let min = w.min(h);

        let (noise, sched) = if let Some((seed, strength)) = seed {
            let sched = self.diffuse.make_schedule(self.steps);
            let init_t = ((sched.len() - 1) as f32 * strength) as usize;
            let sched = sched.into_iter().skip(init_t).collect::<Vec<_>>();
            let seed_shape = seed.shape();

            progress("Encoding seed...".to_string());
            let img = self.autoencoder.encode(&seed)?;
            progress("Encoding seed done".to_string());
            let noise = self.diffuse.make_noise(1, seed_shape[3], seed_shape[2]);
            (LmsSampler::add_noise(&sched[0], &img, &noise), sched)
        } else {
            (
                self.diffuse.make_noise(
                    1,
                    (w / min * self.diffuse_output_size.0 as f32).round() as usize
                        / self.diffuse_output_size.1
                        * self.diffuse_output_size.1,
                    (h / min * self.diffuse_output_size.0 as f32).round() as usize
                        / self.diffuse_output_size.1
                        * self.diffuse_output_size.1,
                ),
                self.diffuse.make_schedule(self.steps),
            )
        };

        let mut cond = HashMap::<String, ndarray::ArrayD<f32>>::new();
        let mut uncond = HashMap::<String, ndarray::ArrayD<f32>>::new();

        for (i, e) in self.text_encoders.iter().enumerate() {
            uncond.insert(
                e.key.clone(),
                self.text_embedding.as_ref().unwrap()[i]
                    .slice_axis(Axis(0), Slice::from(..1usize))
                    .to_owned(),
            );
            cond.insert(
                e.key.clone(),
                self.text_embedding.as_ref().unwrap()[i]
                    .slice_axis(Axis(0), Slice::from(1usize..))
                    .to_owned(),
            );
        }

        let d = {
            let mut d = LmsSampler::new(self.diffuse.as_mut(), &sched, cond, uncond, noise);
            for (i, _) in sched.iter().enumerate() {
                progress(format!("Diffusion step {}/{}", i + 1, sched.len()));
                d.next(i);
            }
            d.seed
        };

        progress("Decoding image...".to_string());
        let r = self.autoencoder.decode(&d)?;
        progress("Decoding image done".to_string());
        Ok(r)
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

    pub fn open_seed(&self, p: impl AsRef<Path>) -> Result<ndarray::ArrayD<f32>> {
        let img = image::io::Reader::open(p)?.decode()?.into_rgb8();
        let w = img.width();
        let h = img.height();
        let scale = self.diffuse_output_size.0 as f32 / w.min(h) as f32;
        let w = (scale * w as f32).round() as u32;
        let h = (scale * h as f32).round() as u32;
        let mut img = image::imageops::resize(&img, w, h, image::imageops::FilterType::CatmullRom);

        let align = self.diffuse_output_size.0 as u32;
        let nw = w / align * align;
        let nh = h / align * align;
        Ok(
            image::imageops::crop(&mut img, (w - nw) / 2, (h - nh) / 2, nw, nh)
                .to_image()
                .into_ndarray3()
                .insert_axis(ndarray::Axis(0))
                .mapv(|x| f32::from(x) / 255.)
                .as_standard_layout()
                .to_owned()
                .into_dyn(),
        )
    }

    pub fn get_png(image: &ndarray::ArrayD<f32>) -> Vec<u8> {
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
