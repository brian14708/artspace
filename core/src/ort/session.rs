use std::{
    collections::HashMap,
    ffi::{c_void, CStr, CString},
    io::{Read, Seek},
    path::Path,
    sync::Mutex,
};

use ort_sys as sys;
use rayon::prelude::*;
use serde::Deserialize;
use smallvec::SmallVec;

use super::ort_call;
use crate::{
    ort::{
        env::{get_cpu_mem_info, get_env},
        get_api,
    },
    result::{Error, Result},
};

pub struct Session {
    session: *mut sys::OrtSession,
    run_option: *mut sys::OrtRunOptions,
    outputs: Vec<CString>,
}

impl Session {
    pub fn load(base: impl AsRef<Path>, path: impl AsRef<str>) -> Result<Self> {
        let base = base.as_ref();
        let api = get_api();
        let mut session_options: *mut sys::OrtSessionOptions = std::ptr::null_mut();
        ort_call!(api.CreateSessionOptions, &mut session_options)?;
        defer! {
            unsafe { api.ReleaseSessionOptions.unwrap()(session_options); }
        }
        ort_call!(api.DisablePerSessionThreads, session_options)?;
        ort_call!(
            api.SetSessionGraphOptimizationLevel,
            session_options,
            sys::GraphOptimizationLevel_ORT_ENABLE_ALL,
        )?;
        ort_call!(
            api.AddSessionConfigEntry,
            session_options,
            sys::kOrtSessionOptionsConfigUseEnvAllocators as *const _ as *const i8,
            "1\0".as_ptr() as *const i8,
        )?;
        ort_call!(api.EnableMemPattern, session_options)?;

        #[cfg(target_os = "macos")]
        {
            let coreml: u32 = 0;
            super::status(unsafe {
                sys::OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml)
            })
            .unwrap();
        }
        #[cfg(all(
            target_arch = "x86_64",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            let mut cuda_options: *mut sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
            if ort_call!(api.CreateCUDAProviderOptions, &mut cuda_options).is_ok() {
                defer! {
                    unsafe { api.ReleaseCUDAProviderOptions.unwrap()(cuda_options); }
                }

                _ = ort_call!(
                    api.SessionOptionsAppendExecutionProvider_CUDA_V2,
                    session_options,
                    cuda_options,
                );
            }
        }

        struct OrtWeight {
            name: CString,
            _data: Vec<u8>,
            ptr: *mut sys::OrtValue,
        }
        unsafe impl Send for OrtWeight {}

        let ort_blobs: Mutex<Vec<OrtWeight>> = Mutex::new(Vec::new());
        defer! {
            ort_blobs.lock().unwrap().iter().for_each(|o| {
                unsafe { api.ReleaseValue.unwrap()(o.ptr); }
            });
        }

        let mut result = Self {
            session: std::ptr::null_mut(),
            run_option: std::ptr::null_mut(),
            outputs: Vec::new(),
        };

        if base.is_file() {
            let mut t = tsar::Archive::new(std::fs::File::open(base)?)?;
            load_blobs(&mut t, &format!(".{}.json", path.as_ref()))?
                .into_par_iter()
                .try_for_each(|(k, mut v)| -> Result<()> {
                    let shape: smallvec::SmallVec<[_; 4]> =
                        v.shape().into_iter().map(|s| s as i64).collect();

                    let mut tmp: Vec<u8> = Vec::new();
                    v.read_to_end(&mut tmp)?;

                    let mut blob: *mut sys::OrtValue = std::ptr::null_mut();
                    ort_call!(
                        api.CreateTensorWithDataAsOrtValue,
                        get_cpu_mem_info(),
                        tmp.as_mut_ptr() as *mut _,
                        tmp.len() as _,
                        shape.as_ptr() as *const _,
                        shape.len() as _,
                        datatype_to_onnx(v.data_type()),
                        &mut blob
                    )?;

                    ort_blobs.lock().unwrap().push(OrtWeight {
                        name: CString::new(k).expect("CString::new failed"),
                        _data: tmp,
                        ptr: blob,
                    });
                    Ok(())
                })?;

            ort_blobs.lock().unwrap().iter().try_for_each(|o| {
                ort_call!(
                    api.AddExternalInitializers,
                    session_options,
                    &o.name.as_ptr(),
                    &(o.ptr as *const _),
                    1,
                )
            })?;

            let onnx = {
                let mut data = Vec::new();
                t.file_by_name(path)?.read_to_end(&mut data)?;
                data
            };
            ort_call!(
                api.CreateSessionFromArray,
                get_env(),
                onnx.as_ptr() as *const _,
                onnx.len() as _,
                session_options,
                &mut result.session,
            )?;
        } else {
            let onnx = path_to_cstring(path.as_ref());
            ort_call!(
                api.CreateSession,
                get_env(),
                onnx.as_ptr(),
                session_options,
                &mut result.session,
            )?;
        }

        ort_call!(api.CreateRunOptions, &mut result.run_option)?;
        ort_call!(
            api.AddRunConfigEntry,
            result.run_option,
            sys::kOrtRunOptionsConfigEnableMemoryArenaShrinkage as *const _ as *const i8,
            "cpu:0\0".as_ptr() as *const i8,
        )?;

        let mut out_len = 0;
        ort_call!(api.SessionGetOutputCount, result.session, &mut out_len)?;

        let mut alloc: *mut sys::OrtAllocator = std::ptr::null_mut();
        ort_call!(api.GetAllocatorWithDefaultOptions, &mut alloc)?;
        for i in 0..out_len {
            let mut name: *mut i8 = std::ptr::null_mut();
            ort_call!(
                api.SessionGetOutputName,
                result.session,
                i,
                alloc,
                &mut name,
            )?;
            result
                .outputs
                .push(unsafe { CStr::from_ptr(name) }.to_owned());
            unsafe {
                (*alloc).Free.unwrap()(alloc, name as *mut _);
            }
        }

        Ok(result)
    }

    pub fn prepare(&self) -> SessionRun<'_> {
        SessionRun {
            sess: self,
            inputs: SmallVec::new(),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let api = get_api();
        if !self.session.is_null() {
            unsafe {
                api.ReleaseSession.unwrap()(self.session);
            }
        }
    }
}

pub struct SessionRun<'s> {
    sess: &'s Session,
    inputs: SmallVec<[(CString, *mut sys::OrtValue); 4]>,
}

impl<'s> SessionRun<'s> {
    pub fn set_input<S, D>(
        &mut self,
        name: impl AsRef<str>,
        data: &'s ndarray::ArrayBase<S, D>,
    ) -> Result<()>
    where
        S: ndarray::RawData,
        S::Elem: AsOnnxDataType,
        D: ndarray::Dimension,
    {
        let shape: smallvec::SmallVec<[_; 4]> = data.shape().iter().map(|s| *s as i64).collect();
        if !data.is_standard_layout() {
            return Err(Error::InvalidInput(
                "ndarray is not standard layout".to_string(),
            ));
        }

        let mut blob: *mut sys::OrtValue = std::ptr::null_mut();
        ort_call!(
            get_api().CreateTensorWithDataAsOrtValue,
            get_cpu_mem_info(),
            data.as_ptr() as *mut _,
            (data.raw_dim().size() * std::mem::size_of::<S::Elem>()) as _,
            shape.as_ptr() as *const _,
            shape.len() as _,
            S::Elem::as_onnx_data_type(),
            &mut blob
        )?;

        self.inputs.push((
            CString::new(name.as_ref()).expect("CString::new failed"),
            blob,
        ));

        Ok(())
    }

    pub fn exec(&mut self) -> Result<SessionRunResult<'_>> {
        let input_names: SmallVec<[*const i8; 4]> =
            self.inputs.iter().map(|(n, _)| n.as_ptr()).collect();
        let input_values: SmallVec<[*const sys::OrtValue; 4]> =
            self.inputs.iter().map(|(_, v)| *v as *const _).collect();
        let output_names: SmallVec<[*const i8; 4]> =
            self.sess.outputs.iter().map(|n| n.as_ptr()).collect();
        let api = get_api();

        let mut outputs: SmallVec<[*mut sys::OrtValue; 4]> = SmallVec::new();
        outputs.resize(self.sess.outputs.len(), std::ptr::null_mut());

        ort_call!(
            api.Run,
            self.sess.session,
            self.sess.run_option,
            input_names.as_ptr(),
            input_values.as_ptr(),
            self.inputs.len() as _,
            output_names.as_ptr(),
            self.sess.outputs.len() as _,
            outputs.as_mut_ptr(),
        )?;

        Ok(SessionRunResult { run: self, outputs })
    }
}

impl Drop for SessionRun<'_> {
    fn drop(&mut self) {
        let api = get_api();
        for inp in &self.inputs {
            unsafe {
                api.ReleaseValue.unwrap()(inp.1);
            }
        }
    }
}

pub struct SessionRunResult<'s> {
    run: &'s SessionRun<'s>,
    outputs: SmallVec<[*mut sys::OrtValue; 4]>,
}

impl<'s> SessionRunResult<'s> {
    pub fn get_output<A, D>(&self, name: impl AsRef<str>) -> Result<ndarray::ArrayView<A, D>>
    where
        A: AsOnnxDataType,
        D: ndarray::Dimension,
    {
        let name = name.as_ref();
        let id = self
            .run
            .sess
            .outputs
            .iter()
            .enumerate()
            .find(|(_, c)| c.as_bytes() == name.as_bytes());
        if let Some((id, _)) = id {
            self.get_output_idx(id)
        } else {
            Err(Error::InvalidInput(format!(
                "output name {} not found",
                name
            )))
        }
    }

    pub fn get_output_idx<A, D>(&self, idx: usize) -> Result<ndarray::ArrayView<A, D>>
    where
        A: AsOnnxDataType,
        D: ndarray::Dimension,
    {
        if idx >= self.outputs.len() {
            return Err(Error::InvalidInput(format!(
                "output index {} out of range",
                idx
            )));
        }
        let out = self.outputs[idx];

        let api = get_api();
        let mut data: *mut c_void = std::ptr::null_mut();
        ort_call!(api.GetTensorMutableData, out, &mut data)?;

        let mut info: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        ort_call!(api.GetTensorTypeAndShape, out, &mut info)?;
        defer! {
            unsafe { api.ReleaseTensorTypeAndShapeInfo.unwrap()(info); }
        }

        let mut dims = 0;
        ort_call!(api.GetDimensionsCount, info, &mut dims)?;
        if let Some(d) = D::NDIM {
            if (dims as usize) != d {
                return Err(Error::InvalidInput(format!(
                    "invalid output dimension for {}",
                    idx,
                )));
            }
        }

        let mut shape: SmallVec<[i64; 4]> = SmallVec::new();
        shape.resize(dims as usize, 0);
        ort_call!(api.GetDimensions, info, shape.as_mut_ptr() as *mut _, dims)?;

        let mut ty = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_call!(api.GetValueType, out, &mut ty)?;
        if A::as_onnx_data_type() != ty {
            return Err(Error::InvalidInput(format!(
                "invalid output type for {}",
                idx,
            )));
        }

        let mut d = D::zeros(shape.len());
        let mut v = d.as_array_view_mut();
        for (i, o) in shape.iter().enumerate() {
            v[i] = *o as usize;
        }
        Ok(unsafe { ndarray::ArrayView::<A, D>::from_shape_ptr(d, data as *mut _) })
    }
}

impl Drop for SessionRunResult<'_> {
    fn drop(&mut self) {
        let api = get_api();
        for &out in &self.outputs {
            unsafe {
                api.ReleaseValue.unwrap()(out);
            }
        }
    }
}

fn load_blobs<R>(t: &mut tsar::Archive<R>, p: &str) -> Result<HashMap<String, tsar::Blob>>
where
    R: Read + Seek,
{
    #[derive(Deserialize)]
    struct OnnxMeta {
        blobs: HashMap<String, String>,
    }
    let meta: OnnxMeta = {
        let rdr = t.file_by_name(p)?;
        serde_json::from_reader(rdr)?
    };
    meta.blobs
        .into_iter()
        .map(|(k, v)| Ok((k, t.blob_by_name(v)?)))
        .collect()
}

fn datatype_to_onnx(dt: Option<tsar::DataType>) -> sys::ONNXTensorElementDataType {
    match dt {
        None | Some(tsar::DataType::Byte) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        }
        Some(tsar::DataType::Float32) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        }
        Some(tsar::DataType::Float64) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
        }
        Some(tsar::DataType::Float16) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        }
        Some(tsar::DataType::Bfloat16) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
        }
        Some(tsar::DataType::Int8) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        }
        Some(tsar::DataType::Uint8) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        }
        Some(tsar::DataType::Int16) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        }
        Some(tsar::DataType::Uint16) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        }
        Some(tsar::DataType::Int32) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        }
        Some(tsar::DataType::Uint32) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
        }
        Some(tsar::DataType::Int64) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        }
        Some(tsar::DataType::Uint64) => {
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
        }
    }
}

pub trait AsOnnxDataType {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType;
}

impl AsOnnxDataType for i32 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    }
}

impl AsOnnxDataType for u32 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    }
}

impl AsOnnxDataType for i64 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    }
}

impl AsOnnxDataType for u64 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    }
}

impl AsOnnxDataType for i8 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    }
}

impl AsOnnxDataType for u8 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    }
}

impl AsOnnxDataType for u16 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    }
}

impl AsOnnxDataType for i16 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    }
}

impl AsOnnxDataType for f32 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    }
}

impl AsOnnxDataType for f64 {
    fn as_onnx_data_type() -> sys::ONNXTensorElementDataType {
        sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    }
}

#[cfg(unix)]
fn path_to_cstring<P: AsRef<Path>>(path: P) -> CString {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_ref().as_os_str().as_bytes()).unwrap()
}

#[cfg(not(unix))]
fn path_to_cstring<P: AsRef<Path>>(path: P) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    path.as_ref()
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>()
}
