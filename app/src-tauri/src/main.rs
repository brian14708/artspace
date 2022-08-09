#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

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
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        // blocked by https://github.com/tauri-apps/tauri/pull/4894
        // .run(tauri::tauri_build_context!())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
