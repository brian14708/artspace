use std::{
    ffi::CString,
    sync::atomic::{AtomicPtr, Ordering},
};

use ort_sys as sys;

use crate::ort::{get_api, ort_call};

lazy_static! {
    static ref G_ENV: AtomicPtr<sys::OrtEnv> = init();
    static ref G_CPU_MEM_INFO: AtomicPtr<sys::OrtMemoryInfo> = {
        let mut mem_info: AtomicPtr<sys::OrtMemoryInfo> = AtomicPtr::new(std::ptr::null_mut());
        ort_call!(
            get_api().CreateCpuMemoryInfo,
            sys::OrtAllocatorType_OrtDeviceAllocator,
            sys::OrtMemType_OrtMemTypeDefault,
            mem_info.get_mut()
        )
        .unwrap();
        mem_info
    };
}

fn init() -> AtomicPtr<sys::OrtEnv> {
    let api = get_api();
    let mut topt: *mut sys::OrtThreadingOptions = std::ptr::null_mut();
    ort_call!(api.CreateThreadingOptions, &mut topt).unwrap();
    defer! {
        unsafe { api.ReleaseThreadingOptions.unwrap()(topt); }
    }

    ort_call!(api.SetGlobalDenormalAsZero, topt).unwrap();
    let lid = CString::new("ort").unwrap();

    let mut env: *mut sys::OrtEnv = std::ptr::null_mut();
    ort_call!(
        api.CreateEnvWithGlobalThreadPools,
        sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
        lid.as_ptr(),
        topt,
        &mut env,
    )
    .unwrap();
    ort_call!(api.DisableTelemetryEvents, env).unwrap();

    let mut mem_info: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
    ort_call!(
        api.CreateCpuMemoryInfo,
        sys::OrtAllocatorType_OrtArenaAllocator,
        sys::OrtMemType_OrtMemTypeDefault,
        &mut mem_info
    )
    .unwrap();
    defer! {
        unsafe { api.ReleaseMemoryInfo.unwrap()(mem_info); }
    }

    let mut arena_info: *mut sys::OrtArenaCfg = std::ptr::null_mut();
    ort_call!(api.CreateArenaCfg, 0, -1, -1, -1, &mut arena_info).unwrap();
    defer! {
        unsafe { api.ReleaseArenaCfg.unwrap()(arena_info); }
    }
    ort_call!(api.CreateAndRegisterAllocator, env, mem_info, arena_info).unwrap();

    AtomicPtr::new(env)
}

pub fn get_env() -> *const sys::OrtEnv {
    G_ENV.load(Ordering::Relaxed) as *const sys::OrtEnv
}

pub fn get_cpu_mem_info() -> *const sys::OrtMemoryInfo {
    G_CPU_MEM_INFO.load(Ordering::Relaxed) as *const sys::OrtMemoryInfo
}
