fn main() {
    std::fs::create_dir_all("../build").unwrap();

    let mut codegen = tauri_build::CodegenContext::new();
    if !cfg!(feature = "custom-protocol") {
        codegen = codegen.dev();
    }
    codegen.build();
    tauri_build::build()
}
