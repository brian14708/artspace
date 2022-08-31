use std::{
    collections::HashMap,
    io::{BufWriter, Write},
    path::PathBuf,
};

use artspace_core::sampler::DdimSampler;
use clap::{Parser, Subcommand};
use ndarray::{Axis, Slice};
use nshare::ToNdarray3;
use tauri::api::path;

use crate::{model_manager::ModelManager, pipeline::Pipeline};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    TextEncode {
        /// Kind of text encoder
        kind: String,

        /// Path of the model to use
        path: PathBuf,

        text: Vec<String>,

        #[clap(short, long)]
        repeat: Option<i32>,

        #[clap(short, long)]
        output: Option<PathBuf>,
    },
    Diffuse {
        kind: String,
        path: PathBuf,

        decoder_kind: String,
        decoder_path: PathBuf,

        sr_kind: String,
        sr_path: PathBuf,

        cond_path: PathBuf,
        clip_cond_path: PathBuf,

        iter: usize,

        output: Option<PathBuf>,

        #[clap(long)]
        width: Option<usize>,
        #[clap(long)]
        height: Option<usize>,
    },
    Pipeline {
        kind: String,
        text: String,
        output: PathBuf,

        #[clap(long)]
        width: Option<f32>,
        #[clap(long)]
        height: Option<f32>,

        #[clap(long)]
        seed: Option<PathBuf>,
        #[clap(long)]
        seed_strength: Option<f32>,
    },
    AutoEncoder {
        kind: String,
        path: PathBuf,
        input: PathBuf,
        output: PathBuf,
    },
}

pub async fn exec() -> bool {
    let cli = Cli::parse();
    match &cli.command {
        Some(Commands::TextEncode {
            kind,
            path,
            repeat,
            text,
            output,
        }) => {
            let mut m = artspace_core::model::load_text_encoder(kind, path).unwrap();
            let out = {
                let enc = text
                    .iter()
                    .map(|s| m.tokenize(s.as_str()))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                m.encode(&enc).unwrap()
            };
            if let Some(path) = output {
                ndarray_npy::write_npy(path, &out).unwrap();
            }
            if let Some(n) = repeat {
                use std::time::Instant;
                let now = Instant::now();
                for _ in 0..*n {
                    let enc = text
                        .iter()
                        .map(|s| m.tokenize(s.as_str()))
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    m.encode(&enc).unwrap();
                }
                let elapsed = now.elapsed();
                println!(
                    "Execution time (per iteration): {:.2?}",
                    elapsed / (*n as u32)
                );
            }
        }
        Some(Commands::Diffuse {
            kind,
            path,
            cond_path,
            clip_cond_path,
            iter,
            output,
            decoder_kind,
            decoder_path,
            sr_kind,
            sr_path,
            width,
            height,
        }) => {
            let mut sr = artspace_core::model::load_super_resolution(sr_kind, sr_path).unwrap();

            let mut m = artspace_core::model::load_diffusion(kind, path).unwrap();
            let cond: ndarray::ArrayD<f32> = ndarray_npy::read_npy(cond_path).unwrap();
            let clip_cond: ndarray::ArrayD<f32> = ndarray_npy::read_npy(clip_cond_path).unwrap();
            let noise = m.make_noise(
                cond.shape()[0] - 1,
                width.unwrap_or(256),
                height.unwrap_or(256),
            );
            let sched = m.make_schedule(*iter);
            let cd: HashMap<_, _> = [
                (
                    "c".to_owned(),
                    cond.slice_axis(Axis(0), Slice::from(1usize..)).to_owned(),
                ),
                (
                    "clip".to_owned(),
                    clip_cond
                        .slice_axis(Axis(0), Slice::from(1usize..))
                        .to_owned(),
                ),
            ]
            .into_iter()
            .collect();
            let ud: HashMap<_, _> = [
                (
                    "c".to_owned(),
                    cond.slice_axis(Axis(0), Slice::from(..1usize)).to_owned(),
                ),
                (
                    "clip".to_owned(),
                    clip_cond
                        .slice_axis(Axis(0), Slice::from(..1usize))
                        .to_owned(),
                ),
            ]
            .into_iter()
            .collect();
            let d = {
                let mut d = DdimSampler::new(m.as_mut(), &sched, cd, ud, noise);
                for (i, _) in sched.iter().enumerate() {
                    println!("{}/{}", i, sched.len());
                    d.next(i);
                }
                d.seed
            };

            let image = {
                let mut m =
                    artspace_core::model::load_auto_encoder(decoder_kind, decoder_path).unwrap();
                m.decode(&d).unwrap()
            };
            let image = { sr.execute(&image).unwrap() };

            let image = image
                .mapv(|f| (f * 255.0) as u8)
                .permuted_axes([0, 2, 3, 1].as_slice());
            let image = image.as_standard_layout();

            if let Some(path) = output {
                for i in 0..image.shape()[0] {
                    let writer = BufWriter::new(
                        std::fs::File::create(path.join(format!("{:03}.png", i))).unwrap(),
                    );

                    let mut encoder =
                        png::Encoder::new(writer, image.shape()[2] as u32, image.shape()[1] as u32);
                    encoder.set_color(png::ColorType::Rgb);
                    encoder.set_depth(png::BitDepth::Eight);
                    let mut writer = encoder.write_header().unwrap();

                    writer
                        .write_image_data(image.index_axis(Axis(0), i).as_slice().unwrap())
                        .unwrap();
                }
            }
        }
        Some(Commands::AutoEncoder {
            kind,
            path,
            input,
            output,
        }) => {
            let img = image::io::Reader::open(input)
                .unwrap()
                .decode()
                .unwrap()
                .into_rgb8();
            let w = img.width();
            let h = img.height();
            let scale = 512. / w.min(h) as f32;
            let w = (scale * w as f32).round() as u32;
            let h = (scale * h as f32).round() as u32;
            let mut img =
                image::imageops::resize(&img, w, h, image::imageops::FilterType::CatmullRom);

            let nw = w / 64 * 64;
            let nh = h / 64 * 64;
            let img = image::imageops::crop(&mut img, (w - nw) / 2, (h - nh) / 2, nw, nh)
                .to_image()
                .into_ndarray3()
                .insert_axis(ndarray::Axis(0))
                .mapv(|x| f32::from(x) / 255.)
                .as_standard_layout()
                .to_owned()
                .into_dyn();

            let mut ae = artspace_core::model::load_auto_encoder(kind, path).unwrap();

            let e = ae.encode(&img).unwrap();
            let img = ae.decode(&e).unwrap();

            let mut out = std::fs::File::create(output).unwrap();
            out.write_all(&Pipeline::get_png(&img)).unwrap();
        }
        Some(Commands::Pipeline {
            kind,
            text,
            output,
            width,
            height,

            seed,
            seed_strength,
        }) => {
            let mm = ModelManager::new(
                path::data_dir()
                    .unwrap_or_else(|| "./".into())
                    .join("artspace/models"),
            )
            .unwrap();

            let mut p = Pipeline::new(kind, &mm, |p| println!("{}", p))
                .await
                .unwrap();
            p.step_text(text).await.unwrap();

            let seed = seed
                .as_ref()
                .map(|f| (p.open_seed(f).unwrap(), seed_strength.unwrap_or(0.5)));

            let img = p
                .step_diffuse(width.unwrap_or(1.), height.unwrap_or(1.), seed, |p| {
                    println!("{}", p)
                })
                .await
                .unwrap();

            let mut out = std::fs::File::create(output).unwrap();
            out.write_all(&Pipeline::get_png(&img)).unwrap();
        }
        None => return false,
    }

    true
}
