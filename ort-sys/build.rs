use std::{env, path::PathBuf};

fn main() {
    let install_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("..")
        .join("vendor")
        .join(env::var("TARGET").unwrap());

    let include_dir = install_dir.join("include");
    let lib_dir = install_dir.join("lib");

    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&[
            format!("-I{}", include_dir.display()),
            format!(
                "-I{}",
                include_dir
                    .join("onnxruntime")
                    .join("core")
                    .join("session")
                    .display()
            ),
        ])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
