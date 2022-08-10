fn main() {
    std::fs::create_dir_all("../build").unwrap();

    let mut codegen = tauri_build::CodegenContext::new();
    if !cfg!(feature = "custom-protocol") {
        codegen = codegen.dev();
    }
    let p = codegen.build();
    // workaround for https://github.com/tauri-apps/tauri/pull/4894
    {
        std::fs::write(
            &p,
            std::fs::read_to_string(&p)
                .unwrap()
                .replace(":: tauri :: Context ::", ":: tauri ::"),
        )
        .unwrap();
    }
    tauri_build::build()
}
