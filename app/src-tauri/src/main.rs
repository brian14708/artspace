#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod cli;

#[tauri::command]
fn greet(name: &str) -> String {
    format!(
        "Hello, {}! ORT: {} {:?}",
        name,
        artspace_core::ort::version(),
        artspace_core::ort::list_providers().unwrap(),
    )
}

fn main() {
    cli::exec();

    #[allow(unused_imports)]
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::tauri_build_context!())
        .expect("error while running tauri application");
}
