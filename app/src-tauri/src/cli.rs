use std::{collections::HashMap, io::BufWriter, path::PathBuf};

use artspace_core::ddim_sampler::DdimSampler;
use clap::{Parser, Subcommand};
use ndarray::{Axis, Slice};

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
}

pub fn exec() {
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

            let m = artspace_core::model::load_diffusion(kind, path).unwrap();
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
                let mut d = DdimSampler::new(m, cd, ud, noise);
                for i in sched {
                    d.next(&i);
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
        None => return,
    }
    std::process::exit(0)
}
