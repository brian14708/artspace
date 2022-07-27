fn main() {
    std::fs::create_dir_all("../build").unwrap();
    tauri_build::build()
}
