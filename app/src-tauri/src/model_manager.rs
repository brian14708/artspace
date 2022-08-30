use std::{
    collections::{HashMap, HashSet},
    io::{self, Read, Seek, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use rayon::prelude::*;
use serde::Deserialize;

pub struct ModelManager {
    data_dir: PathBuf,
    manifest: Manifest,
}

#[derive(Deserialize)]
struct Manifest {
    models: HashMap<String, ModelManifest>,
}

#[derive(Deserialize)]
struct ModelManifest {
    url: String,
    sha256: String,
}

impl ModelManager {
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        if !data_dir.is_dir() {
            std::fs::create_dir_all(&data_dir)?;
        }

        let manifest: Manifest = serde_json::from_slice(include_bytes!("../manifest.json"))?;

        Ok(Self { data_dir, manifest })
    }

    pub async fn cleanup(&self) -> Result<()> {
        let sha256 = self
            .manifest
            .models
            .iter()
            .map(|(_, m)| m.sha256.clone())
            .collect::<HashSet<_>>();

        let mut to_delete = Vec::new();
        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap().to_str().unwrap();
                if !sha256.contains(name) {
                    to_delete.push(path);
                }
            }
        }

        for path in to_delete {
            async_std::fs::remove_dir_all(&path).await?;
        }

        Ok(())
    }

    pub async fn download(&self, name: &str, progress: impl Fn(String)) -> Result<PathBuf> {
        let model = self
            .manifest
            .models
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model {} not found", name))?;
        let path = self.data_dir.join(&model.sha256);
        if path.is_dir() {
            return Ok(path);
        }

        let mut resp = reqwest::get(&model.url).await?;
        let total_size = resp.content_length();
        let mut downloaded = 0;
        let mut temp_out = tempfile::tempfile_in(&self.data_dir)?;

        let mut prev = 0;
        while let Some(chunk) = resp.chunk().await? {
            temp_out.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            if downloaded > prev + total_size.unwrap_or(0) / 1000 {
                prev = downloaded;
                progress(format!(
                    "Downloading {} ({}/{})...",
                    name,
                    downloaded,
                    total_size.unwrap_or(0)
                ));
            }
        }
        progress(format!("Download {} finished", name,));

        let temp_extract = tempfile::tempdir_in(&self.data_dir)?;
        let mut ar = tsar::Archive::new(temp_out).unwrap();

        let files = ar.file_names().map(|s| s.to_owned()).collect::<Vec<_>>();
        for f in files.iter() {
            let outpath = temp_extract.path().join(f);
            ensure_dir(&outpath)?;
            let mut dst = std::fs::File::create(outpath)?;
            let mut src = ar.file_by_name(f)?;
            std::io::copy(&mut src, &mut dst)?;
        }

        {
            let blobs = ar
                .blob_names()
                .map(|s| s.to_owned())
                .collect::<Vec<_>>()
                .into_iter()
                .map(|b| ar.blob_by_name(b))
                .collect::<tsar::Result<Vec<_>>>()?;

            let mut m = HashMap::<&str, u64>::new();
            for b in blobs.iter() {
                if let Some((k, offset)) = b.target_file() {
                    let vv = m.entry(k).or_default();
                    *vv = (*vv).max(offset + b.byte_len().unwrap_or_default() as u64);
                }
            }
            for (k, v) in m.into_iter() {
                let outpath = temp_extract.path().join(k);
                ensure_dir(&outpath)?;
                let dst = async_std::fs::File::create(outpath).await?;
                dst.set_len(v).await?;
            }

            let lk = std::sync::Arc::new(std::sync::Mutex::new(()));
            blobs.into_par_iter().for_each(|mut b| {
                if let Some((target_file, offset)) = b.target_file() {
                    let outpath = temp_extract.path().join(target_file);
                    let lk = lk.clone();
                    {
                        let mut dst = std::fs::OpenOptions::new()
                            .write(true)
                            .open(outpath)
                            .unwrap();
                        if offset > 0 {
                            dst.seek(io::SeekFrom::Start(offset)).unwrap();
                        }

                        let mut buf = vec![0; 1024 * 1024];
                        loop {
                            match b.read(&mut buf).unwrap() {
                                0 => break,
                                n => {
                                    let _lk = lk.lock();
                                    dst.write_all(&buf[..n]).unwrap()
                                }
                            }
                        }
                    }
                }
            });
        }

        async_std::fs::rename(temp_extract.path(), &path).await?;

        Ok(path)
    }
}

fn ensure_dir(outpath: &Path) -> std::io::Result<()> {
    if let Some(p) = outpath.parent() {
        if !p.exists() {
            std::fs::create_dir_all(&p)?;
        }
    }
    Ok(())
}
