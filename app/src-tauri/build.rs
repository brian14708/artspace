fn main() {
    std::fs::create_dir_all("../build").unwrap();
    if !std::path::PathBuf::from("../build/index.html").is_file() {
        std::fs::File::create("../build/index.html").unwrap();
    }
    println!("cargo:rerun-if-changed=../build/index.html");

    let mut codegen = tauri_build::CodegenContext::new();
    if !cfg!(feature = "custom-protocol") {
        codegen = codegen.dev();
    }
    codegen.build();
    tauri_build::build()
}
