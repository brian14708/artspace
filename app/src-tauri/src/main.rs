#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::api::path;

mod cli;
mod model_manager;
mod pipeline;

#[tauri::command]
fn greet(name: &str) -> String {
    format!(
        "Hello, {}! ORT: {} {:?}",
        name,
        artspace_core::ort::version(),
        artspace_core::ort::list_providers().unwrap(),
    )
}

#[tokio::main]
async fn main() {
    let model_manager = model_manager::ModelManager::new(
        path::data_dir()
            .unwrap_or_else(|| "./".into())
            .join("artspace/models"),
    )
    .unwrap();

    cli::exec(&model_manager).await;

    #[allow(unused_imports)]
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::tauri_build_context!())
        .expect("error while running tauri application");
}
