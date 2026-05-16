#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![doc = "Raw FFI bindings for the small ncnn C API surface used by Lumen."]

use std::ffi::{c_char, c_int, c_uchar, c_void};

pub type size_t = usize;

#[repr(C)]
pub struct __ncnn_allocator_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct __ncnn_option_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct __ncnn_mat_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct __ncnn_net_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct __ncnn_extractor_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct __ncnn_datareader_t {
    _private: [u8; 0],
}

pub type ncnn_allocator_t = *mut __ncnn_allocator_t;
pub type ncnn_option_t = *mut __ncnn_option_t;
pub type ncnn_mat_t = *mut __ncnn_mat_t;
pub type ncnn_net_t = *mut __ncnn_net_t;
pub type ncnn_extractor_t = *mut __ncnn_extractor_t;
pub type ncnn_datareader_t = *mut __ncnn_datareader_t;

pub const NCNN_BORDER_CONSTANT: c_int = 0;
pub const NCNN_BORDER_REPLICATE: c_int = 1;
pub const NCNN_BORDER_REFLECT: c_int = 2;
pub const NCNN_BORDER_TRANSPARENT: c_int = -233;

unsafe extern "C" {
    pub fn ncnn_version() -> *const c_char;
    pub fn ncnn_version_number() -> c_int;

    pub fn ncnn_allocator_create_pool_allocator() -> ncnn_allocator_t;
    pub fn ncnn_allocator_create_unlocked_pool_allocator() -> ncnn_allocator_t;
    pub fn ncnn_allocator_destroy(allocator: ncnn_allocator_t);

    pub fn ncnn_option_create() -> ncnn_option_t;
    pub fn ncnn_option_destroy(opt: ncnn_option_t);
    pub fn ncnn_option_get_num_threads(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_num_threads(opt: ncnn_option_t, num_threads: c_int);
    pub fn ncnn_option_set_blob_allocator(opt: ncnn_option_t, allocator: ncnn_allocator_t);
    pub fn ncnn_option_set_workspace_allocator(opt: ncnn_option_t, allocator: ncnn_allocator_t);
    pub fn ncnn_option_get_use_vulkan_compute(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_use_vulkan_compute(opt: ncnn_option_t, enable: c_int);
    pub fn ncnn_option_get_use_packing_layout(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_use_packing_layout(opt: ncnn_option_t, enable: c_int);
    pub fn ncnn_option_get_use_fp16_packed(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_use_fp16_packed(opt: ncnn_option_t, enable: c_int);
    pub fn ncnn_option_get_use_fp16_storage(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_use_fp16_storage(opt: ncnn_option_t, enable: c_int);
    pub fn ncnn_option_get_use_fp16_arithmetic(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_use_fp16_arithmetic(opt: ncnn_option_t, enable: c_int);
    pub fn ncnn_option_get_use_int8_storage(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_use_int8_storage(opt: ncnn_option_t, enable: c_int);
    pub fn ncnn_option_get_use_int8_arithmetic(opt: ncnn_option_t) -> c_int;
    pub fn ncnn_option_set_use_int8_arithmetic(opt: ncnn_option_t, enable: c_int);

    pub fn ncnn_mat_create() -> ncnn_mat_t;
    pub fn ncnn_mat_create_1d(w: c_int, allocator: ncnn_allocator_t) -> ncnn_mat_t;
    pub fn ncnn_mat_create_2d(w: c_int, h: c_int, allocator: ncnn_allocator_t) -> ncnn_mat_t;
    pub fn ncnn_mat_create_3d(
        w: c_int,
        h: c_int,
        c: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_4d(
        w: c_int,
        h: c_int,
        d: c_int,
        c: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_1d(
        w: c_int,
        data: *mut c_void,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_2d(
        w: c_int,
        h: c_int,
        data: *mut c_void,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_3d(
        w: c_int,
        h: c_int,
        c: c_int,
        data: *mut c_void,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_4d(
        w: c_int,
        h: c_int,
        d: c_int,
        c: c_int,
        data: *mut c_void,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_1d_elem(
        w: c_int,
        data: *mut c_void,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_2d_elem(
        w: c_int,
        h: c_int,
        data: *mut c_void,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_3d_elem(
        w: c_int,
        h: c_int,
        c: c_int,
        data: *mut c_void,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_external_4d_elem(
        w: c_int,
        h: c_int,
        d: c_int,
        c: c_int,
        data: *mut c_void,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_1d_elem(
        w: c_int,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_2d_elem(
        w: c_int,
        h: c_int,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_3d_elem(
        w: c_int,
        h: c_int,
        c: c_int,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_create_4d_elem(
        w: c_int,
        h: c_int,
        d: c_int,
        c: c_int,
        elemsize: size_t,
        elempack: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_destroy(mat: ncnn_mat_t);
    pub fn ncnn_mat_fill_float(mat: ncnn_mat_t, v: f32);
    pub fn ncnn_mat_clone(mat: ncnn_mat_t, allocator: ncnn_allocator_t) -> ncnn_mat_t;
    pub fn ncnn_mat_reshape_1d(
        mat: ncnn_mat_t,
        w: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_reshape_2d(
        mat: ncnn_mat_t,
        w: c_int,
        h: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_reshape_3d(
        mat: ncnn_mat_t,
        w: c_int,
        h: c_int,
        c: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_reshape_4d(
        mat: ncnn_mat_t,
        w: c_int,
        h: c_int,
        d: c_int,
        c: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_get_dims(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_w(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_h(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_d(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_c(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_elemsize(mat: ncnn_mat_t) -> size_t;
    pub fn ncnn_mat_get_elempack(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_cstep(mat: ncnn_mat_t) -> size_t;
    pub fn ncnn_mat_get_data(mat: ncnn_mat_t) -> *mut c_void;
    pub fn ncnn_mat_get_channel_data(mat: ncnn_mat_t, c: c_int) -> *mut c_void;
    pub fn ncnn_mat_substract_mean_normalize(
        mat: ncnn_mat_t,
        mean_vals: *const f32,
        norm_vals: *const f32,
    );
    pub fn ncnn_convert_packing(
        src: ncnn_mat_t,
        dst: *mut ncnn_mat_t,
        elempack: c_int,
        opt: ncnn_option_t,
    );
    pub fn ncnn_flatten(src: ncnn_mat_t, dst: *mut ncnn_mat_t, opt: ncnn_option_t);

    pub fn ncnn_net_create() -> ncnn_net_t;
    pub fn ncnn_net_destroy(net: ncnn_net_t);
    pub fn ncnn_net_get_option(net: ncnn_net_t) -> ncnn_option_t;
    pub fn ncnn_net_set_option(net: ncnn_net_t, opt: ncnn_option_t);
    #[cfg(feature = "vulkan")]
    pub fn ncnn_net_set_vulkan_device(net: ncnn_net_t, device_index: c_int);
    pub fn ncnn_net_load_param(net: ncnn_net_t, path: *const c_char) -> c_int;
    pub fn ncnn_net_load_param_bin(net: ncnn_net_t, path: *const c_char) -> c_int;
    pub fn ncnn_net_load_model(net: ncnn_net_t, path: *const c_char) -> c_int;
    #[cfg(windows)]
    pub fn ncnn_net_load_param_w(net: ncnn_net_t, path: *const u16) -> c_int;
    #[cfg(windows)]
    pub fn ncnn_net_load_param_bin_w(net: ncnn_net_t, path: *const u16) -> c_int;
    #[cfg(windows)]
    pub fn ncnn_net_load_model_w(net: ncnn_net_t, path: *const u16) -> c_int;
    pub fn ncnn_net_load_param_memory(net: ncnn_net_t, mem: *const c_char) -> c_int;
    pub fn ncnn_net_load_param_bin_memory(net: ncnn_net_t, mem: *const c_uchar) -> size_t;
    pub fn ncnn_net_load_model_memory(net: ncnn_net_t, mem: *const c_uchar) -> size_t;
    pub fn ncnn_net_load_param_datareader(net: ncnn_net_t, dr: ncnn_datareader_t) -> c_int;
    pub fn ncnn_net_load_param_bin_datareader(net: ncnn_net_t, dr: ncnn_datareader_t) -> c_int;
    pub fn ncnn_net_load_model_datareader(net: ncnn_net_t, dr: ncnn_datareader_t) -> c_int;
    pub fn ncnn_net_clear(net: ncnn_net_t);
    pub fn ncnn_net_get_input_count(net: ncnn_net_t) -> c_int;
    pub fn ncnn_net_get_output_count(net: ncnn_net_t) -> c_int;
    pub fn ncnn_net_get_input_name(net: ncnn_net_t, i: c_int) -> *const c_char;
    pub fn ncnn_net_get_output_name(net: ncnn_net_t, i: c_int) -> *const c_char;
    pub fn ncnn_net_get_input_index(net: ncnn_net_t, i: c_int) -> c_int;
    pub fn ncnn_net_get_output_index(net: ncnn_net_t, i: c_int) -> c_int;

    pub fn ncnn_datareader_create_from_memory(mem: *mut *const c_uchar) -> ncnn_datareader_t;
    pub fn ncnn_datareader_destroy(dr: ncnn_datareader_t);

    pub fn ncnn_extractor_create(net: ncnn_net_t) -> ncnn_extractor_t;
    pub fn ncnn_extractor_destroy(ex: ncnn_extractor_t);
    pub fn ncnn_extractor_set_option(ex: ncnn_extractor_t, opt: ncnn_option_t);
    pub fn ncnn_extractor_input(
        ex: ncnn_extractor_t,
        name: *const c_char,
        mat: ncnn_mat_t,
    ) -> c_int;
    pub fn ncnn_extractor_extract(
        ex: ncnn_extractor_t,
        name: *const c_char,
        mat: *mut ncnn_mat_t,
    ) -> c_int;
    pub fn ncnn_extractor_input_index(ex: ncnn_extractor_t, index: c_int, mat: ncnn_mat_t)
    -> c_int;
    pub fn ncnn_extractor_extract_index(
        ex: ncnn_extractor_t,
        index: c_int,
        mat: *mut ncnn_mat_t,
    ) -> c_int;
}
