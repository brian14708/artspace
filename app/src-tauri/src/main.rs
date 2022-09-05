#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::{io::Write, sync::Mutex};

use artspace_core::ort;
use async_std::path::PathBuf;
use lazy_static::lazy_static;
use tauri::api::path;

use crate::pipeline::Pipeline;

mod cli;
mod model_manager;
mod pipeline;

lazy_static! {
    static ref CURRENT_STATUS: Mutex<String> = Mutex::new(String::from(""));
    static ref PIPELINE: async_std::sync::Mutex<Option<Pipeline>> =
        async_std::sync::Mutex::new(None);
    static ref RESULTS: async_std::sync::Mutex<Vec<ndarray::ArrayD<f32>>> =
        async_std::sync::Mutex::new(vec![]);
}

#[tauri::command]
fn get_status() -> String {
    CURRENT_STATUS.lock().unwrap().clone()
}

fn set_error<T, E>(e: std::result::Result<T, E>) -> Option<T>
where
    E: std::fmt::Debug,
{
    match e {
        Ok(v) => {
            CURRENT_STATUS.lock().unwrap().clear();
            Some(v)
        }
        Err(e) => {
            *CURRENT_STATUS.lock().unwrap() = format!("{:?}", e);
            None
        }
    }
}

#[tauri::command]
fn list_pipeline() -> Vec<String> {
    Pipeline::list()
}

#[tauri::command]
async fn init(kind: String) -> Option<bool> {
    let mut p = PIPELINE.lock().await;
    if p.is_none() {
        let mm = model_manager::ModelManager::new(
            path::data_dir()
                .unwrap_or_else(|| "./".into())
                .join("artspace/models"),
        )
        .unwrap();

        let _ = mm.cleanup().await;

        let e = (Pipeline::new(&kind, &mm, log)).await;
        *p = Some(set_error(e)?);
    }
    Some(true)
}

#[tauri::command]
async fn step_text(text: String) -> Option<bool> {
    log("Processing text...");
    RESULTS.lock().await.clear();
    let mut p = PIPELINE.lock().await;
    set_error(p.as_mut().unwrap().step_text(&text).await)?;
    Some(true)
}

#[tauri::command]
async fn step_diffuse(w: f32, h: f32, idx: usize) -> Option<Vec<u8>> {
    let mut p = PIPELINE.lock().await;
    let img = set_error(p.as_mut().unwrap().step_diffuse(w, h, None, log).await)?;
    let png = Pipeline::get_png(&img);
    let mut result = RESULTS.lock().await;
    if result.len() <= idx {
        result.resize(idx + 1, ndarray::ArrayD::zeros(ndarray::IxDyn(&[])));
    }
    result[idx] = img;

    Some(png)
}

#[tauri::command]
async fn step_post(idx: usize, path: String) -> Option<()> {
    log("Processing image...");
    let mut p = PIPELINE.lock().await;
    let img = set_error(
        p.as_mut()
            .unwrap()
            .step_post_process(&RESULTS.lock().await[idx])
            .await,
    )?;

    log("Saving image...");
    let mut out = std::fs::File::create(PathBuf::from(path)).unwrap();
    set_error(out.write_all(&Pipeline::get_png(&img)))?;
    Some(())
}

fn log(s: impl Into<String>) {
    *CURRENT_STATUS.lock().unwrap() = s.into();
}

#[tokio::main]
async fn main() {
    if !cli::exec().await {
        #[allow(unused_imports)]
        tauri::Builder::default()
            .invoke_handler(tauri::generate_handler![
                list_pipeline,
                get_status,
                init,
                step_text,
                step_diffuse,
                step_post
            ])
            .run(tauri::tauri_build_context!())
            .expect("error while running tauri application");
    }
    ort::deinit();
}
