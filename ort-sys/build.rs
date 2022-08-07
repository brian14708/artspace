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

    {
        let (stub_src, compile_shared) = match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
            "linux" => ("src/cuda_stub_linux.c", true),
            "windows" => ("src/cuda_stub_windows.c", false),
            _ => ("src/cuda_stub.c", false),
        };

        println!("cargo:rerun-if-changed=src/cuda_stub.h");
        println!("cargo:rerun-if-changed={}", stub_src);
        if compile_shared {
            let stublib = lib_dir.join("libcuda_stub.so");
            let t = cc::Build::new()
                .debug(false)
                .flag("-O3")
                .flag("-s")
                .shared_flag(true)
                .get_compiler();
            let mut t = t.to_command();
            t.args(&[stub_src, "-o", stublib.display().to_string().as_str()]);
            if !t.status().unwrap().success() {
                panic!("fail to build");
            }
            println!("cargo:rustc-link-lib=dylib=cuda_stub");
        } else {
            cc::Build::new().file(stub_src).compile("cuda_stub");
        }
    }

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
