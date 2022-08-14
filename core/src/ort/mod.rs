use std::sync::atomic::{AtomicPtr, Ordering};

use ort_sys as sys;

mod cuda;
mod env;
mod session;
pub use env::{list_providers, version};
pub use session::{DataType, Session, TensorInfo};

pub struct Error(sys::OrtErrorCode, String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ORT({}): {}", self.0, self.1)
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ORT({}): {}", self.0, self.1)
    }
}

impl std::error::Error for Error {}

fn status(status: *mut sys::OrtStatus) -> Result<(), Error> {
    if status.is_null() {
        Ok(())
    } else {
        let api = crate::ort::get_api();
        let code = unsafe { api.GetErrorCode.unwrap()(status) };
        if code == sys::OrtErrorCode_ORT_OK {
            unsafe {
                api.ReleaseStatus.unwrap()(status);
            }
            Ok(())
        } else {
            let msg = unsafe { std::ffi::CStr::from_ptr(api.GetErrorMessage.unwrap()(status)) }
                .to_string_lossy()
                .into_owned();
            unsafe {
                api.ReleaseStatus.unwrap()(status);
            }
            Err(crate::ort::Error(code, msg))
        }
    }
}

macro_rules! ort_call {
    ($api:expr, $( $args:expr ),* $(,)?) => {{
        crate::ort::status(unsafe {
            ($api).unwrap()($($args),*)
        })
    }
    };
}
pub(self) use ort_call;

lazy_static! {
    static ref G_ORT: AtomicPtr<sys::OrtApi> = {
        let base: *const sys::OrtApiBase = unsafe { sys::OrtGetApiBase() };
        assert_ne!(base, std::ptr::null());
        let api = unsafe { base.as_ref().unwrap().GetApi.unwrap()(sys::ORT_API_VERSION) }
            as *mut sys::OrtApi;
        assert_ne!(api, std::ptr::null_mut());

        AtomicPtr::new(api)
    };
}

fn get_api() -> &'static sys::OrtApi {
    unsafe { &*G_ORT.load(Ordering::Relaxed) }
}
