//! Raw FFI for the Lumen IREE shim. The C contract is documented in
//! `csrc/lumen_iree.h`; use the safe `lumen-iree` crate instead of this one.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_void};

#[repr(C)]
pub struct lumen_iree_error_t {
    _private: [u8; 0],
}
#[repr(C)]
pub struct lumen_iree_runtime_t {
    _private: [u8; 0],
}
#[repr(C)]
pub struct lumen_iree_model_t {
    _private: [u8; 0],
}
#[repr(C)]
pub struct lumen_iree_outputs_t {
    _private: [u8; 0],
}

pub const LUMEN_IREE_ELEMENT_F32: i32 = 1;
pub const LUMEN_IREE_ELEMENT_I64: i32 = 2;
pub const LUMEN_IREE_ELEMENT_I32: i32 = 3;

pub const LUMEN_IREE_PARAMS_MMAP: i32 = 0;
pub const LUMEN_IREE_PARAMS_READ: i32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lumen_iree_tensor_t {
    pub element_type: i32,
    pub rank: usize,
    pub dims: *const i64,
    pub data: *const c_void,
    pub byte_length: usize,
}

unsafe extern "C" {
    pub fn lumen_iree_error_message(error: *const lumen_iree_error_t) -> *const c_char;
    pub fn lumen_iree_error_free(error: *mut lumen_iree_error_t);

    pub fn lumen_iree_runtime_create(
        driver: *const c_char,
        out_runtime: *mut *mut lumen_iree_runtime_t,
    ) -> *mut lumen_iree_error_t;
    pub fn lumen_iree_runtime_release(runtime: *mut lumen_iree_runtime_t);

    pub fn lumen_iree_model_load(
        runtime: *mut lumen_iree_runtime_t,
        vmfb_path: *const c_char,
        irpa_path: *const c_char,
        param_scope: *const c_char,
        params_mode: i32,
        out_model: *mut *mut lumen_iree_model_t,
    ) -> *mut lumen_iree_error_t;
    pub fn lumen_iree_model_release(model: *mut lumen_iree_model_t);
    pub fn lumen_iree_model_has_function(
        model: *mut lumen_iree_model_t,
        function: *const c_char,
    ) -> i32;
    pub fn lumen_iree_model_invoke(
        model: *mut lumen_iree_model_t,
        function: *const c_char,
        inputs: *const lumen_iree_tensor_t,
        input_count: usize,
        out_outputs: *mut *mut lumen_iree_outputs_t,
    ) -> *mut lumen_iree_error_t;
    pub fn lumen_iree_model_trim(model: *mut lumen_iree_model_t) -> *mut lumen_iree_error_t;

    pub fn lumen_iree_outputs_count(outputs: *const lumen_iree_outputs_t) -> usize;
    pub fn lumen_iree_outputs_describe(
        outputs: *const lumen_iree_outputs_t,
        index: usize,
        out_element_type: *mut i32,
        out_rank: *mut usize,
        dims: *mut i64,
        dims_capacity: usize,
        out_byte_length: *mut usize,
    ) -> *mut lumen_iree_error_t;
    pub fn lumen_iree_outputs_read(
        outputs: *const lumen_iree_outputs_t,
        index: usize,
        dst: *mut c_void,
        dst_byte_length: usize,
    ) -> *mut lumen_iree_error_t;
    pub fn lumen_iree_outputs_release(outputs: *mut lumen_iree_outputs_t);
}
