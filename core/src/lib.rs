#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate scopeguard;

mod ldm;
pub mod ort;
mod result;
pub mod text_encoder;

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

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        assert_eq!(super::version(), "1.12.1");
    }
}
