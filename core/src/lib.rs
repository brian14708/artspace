pub fn version() -> String {
    unsafe {
        std::ffi::CStr::from_ptr(ort_sys::OrtGetApiBase()
            .as_ref()
            .unwrap()
            .GetVersionString
            .unwrap()())
        .to_str()
        .unwrap()
        .to_owned()
    }
}

#[test]
fn test_version() {
    assert_ne!(version(), "");
}
