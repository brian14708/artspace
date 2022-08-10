use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
        output: Option<String>,
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
            let mut m = artspace_core::text_encoder::load(kind, path).unwrap();
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
        None => return,
    }
    std::process::exit(0)
}