// Lumen IREE shim implementation. See lumen_iree.h for the contract.
#include "lumen_iree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/parser_registry.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/runtime/api.h"

//===----------------------------------------------------------------------===//
// Errors
//===----------------------------------------------------------------------===//

struct lumen_iree_error_t {
  char* message;
};

static lumen_iree_error_t* lumen_error_from_cstr(const char* text) {
  lumen_iree_error_t* error = (lumen_iree_error_t*)malloc(sizeof(*error));
  if (!error) abort();
  size_t len = strlen(text);
  error->message = (char*)malloc(len + 1);
  if (!error->message) abort();
  memcpy(error->message, text, len + 1);
  return error;
}

// Consumes |status|. Returns NULL for OK.
static lumen_iree_error_t* lumen_error_from_status(iree_status_t status) {
  if (iree_status_is_ok(status)) return NULL;
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = NULL;
  iree_host_size_t length = 0;
  lumen_iree_error_t* error = NULL;
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    error = lumen_error_from_cstr(buffer);
    iree_allocator_free(allocator, buffer);
  } else {
    error = lumen_error_from_cstr(
        iree_status_code_string(iree_status_code(status)));
  }
  iree_status_free(status);
  return error;
}

const char* lumen_iree_error_message(const lumen_iree_error_t* error) {
  return error ? error->message : "";
}

void lumen_iree_error_free(lumen_iree_error_t* error) {
  if (!error) return;
  free(error->message);
  free(error);
}

//===----------------------------------------------------------------------===//
// Runtime (instance + device)
//===----------------------------------------------------------------------===//

struct lumen_iree_runtime_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
};

lumen_iree_error_t* lumen_iree_runtime_create(const char* driver,
                                              lumen_iree_runtime_t** out_runtime) {
  if (!driver || !out_runtime) {
    return lumen_error_from_cstr("lumen_iree_runtime_create: null argument");
  }
  *out_runtime = NULL;
  lumen_iree_runtime_t* runtime =
      (lumen_iree_runtime_t*)calloc(1, sizeof(*runtime));
  if (!runtime) abort();

  iree_runtime_instance_options_t options;
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
  iree_status_t status = iree_runtime_instance_create(
      &options, iree_allocator_system(), &runtime->instance);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        runtime->instance, iree_make_cstring_view(driver), &runtime->device);
  }
  if (!iree_status_is_ok(status)) {
    lumen_iree_runtime_release(runtime);
    return lumen_error_from_status(status);
  }
  *out_runtime = runtime;
  return NULL;
}

void lumen_iree_runtime_release(lumen_iree_runtime_t* runtime) {
  if (!runtime) return;
  iree_hal_device_release(runtime->device);
  iree_runtime_instance_release(runtime->instance);
  free(runtime);
}

//===----------------------------------------------------------------------===//
// Model (session + io_parameters module + bytecode module)
//===----------------------------------------------------------------------===//

struct lumen_iree_model_t {
  lumen_iree_runtime_t* runtime;  // unowned; caller keeps runtime alive
  iree_runtime_session_t* session;
};

static void lumen_release_mapping(void* user_data,
                                  iree_io_file_handle_primitive_t primitive) {
  (void)primitive;
  iree_io_file_mapping_release((iree_io_file_mapping_t*)user_data);
}

// Opens |irpa_path| as a file handle according to |params_mode|.
static iree_status_t lumen_open_params(const char* irpa_path, int32_t params_mode,
                                       iree_allocator_t host_allocator,
                                       iree_io_file_handle_t** out_handle) {
  *out_handle = NULL;
  iree_io_file_handle_t* file_handle = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_SHARE_READ,
      iree_make_cstring_view(irpa_path), host_allocator, &file_handle));
  if (params_mode == LUMEN_IREE_PARAMS_READ) {
    *out_handle = file_handle;
    return iree_ok_status();
  }
  // LUMEN_IREE_PARAMS_MMAP: map the whole file read-only and expose the mapping
  // as a host allocation so devices that can import host memory alias it.
  iree_io_file_mapping_t* mapping = NULL;
  iree_status_t status = iree_io_file_map_view(
      file_handle, IREE_IO_FILE_ACCESS_READ, 0, IREE_HOST_SIZE_MAX,
      IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS, host_allocator, &mapping);
  iree_io_file_handle_release(file_handle);  // mapping retains the handle
  IREE_RETURN_IF_ERROR(status);
  iree_const_byte_span_t contents = iree_io_file_mapping_contents_ro(mapping);
  iree_io_file_handle_release_callback_t release = {
      .fn = lumen_release_mapping,
      .user_data = mapping,
  };
  status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span((void*)contents.data, contents.data_length), release,
      host_allocator, out_handle);
  if (!iree_status_is_ok(status)) iree_io_file_mapping_release(mapping);
  return status;
}

static iree_status_t lumen_create_params_module(iree_runtime_session_t* session,
                                                const char* irpa_path,
                                                const char* param_scope,
                                                int32_t params_mode,
                                                iree_vm_module_t** out_module) {
  iree_allocator_t host_allocator = iree_runtime_session_host_allocator(session);
  iree_io_parameter_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(iree_io_parameter_index_create(host_allocator, &index));

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status =
      lumen_open_params(irpa_path, params_mode, host_allocator, &file_handle);
  if (iree_status_is_ok(status)) {
    // The path is only used for format detection (by extension) and logging.
    status = iree_io_parse_file_index(iree_make_cstring_view(irpa_path),
                                      file_handle, index, host_allocator);
  }
  iree_io_file_handle_release(file_handle);  // the index retains it

  iree_io_parameter_provider_t* provider = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        iree_make_cstring_view(param_scope), index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        host_allocator, &provider);
  }
  iree_io_parameter_index_release(index);  // the provider retains it

  if (iree_status_is_ok(status)) {
    status = iree_io_parameters_module_create(
        iree_runtime_instance_vm_instance(iree_runtime_session_instance(session)),
        /*provider_count=*/1, &provider, host_allocator, out_module);
  }
  iree_io_parameter_provider_release(provider);  // the module retains it
  return status;
}

lumen_iree_error_t* lumen_iree_model_load(lumen_iree_runtime_t* runtime,
                                          const char* vmfb_path,
                                          const char* irpa_path,
                                          const char* param_scope,
                                          int32_t params_mode,
                                          lumen_iree_model_t** out_model) {
  if (!runtime || !vmfb_path || !irpa_path || !param_scope || !out_model) {
    return lumen_error_from_cstr("lumen_iree_model_load: null argument");
  }
  if (params_mode != LUMEN_IREE_PARAMS_MMAP &&
      params_mode != LUMEN_IREE_PARAMS_READ) {
    return lumen_error_from_cstr("lumen_iree_model_load: invalid params_mode");
  }
  *out_model = NULL;
  lumen_iree_model_t* model = (lumen_iree_model_t*)calloc(1, sizeof(*model));
  if (!model) abort();
  model->runtime = runtime;

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_status_t status = iree_runtime_session_create_with_device(
      runtime->instance, &session_options, runtime->device,
      iree_runtime_instance_host_allocator(runtime->instance), &model->session);

  // Module registration order matters: dependencies before dependents.
  // The session already registered the `hal` module.
  iree_vm_module_t* params_module = NULL;
  if (iree_status_is_ok(status)) {
    status = lumen_create_params_module(model->session, irpa_path, param_scope,
                                        params_mode, &params_module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(model->session, params_module);
  }
  iree_vm_module_release(params_module);

  if (iree_status_is_ok(status)) {
    iree_io_file_contents_t* contents = NULL;
    status = iree_io_file_contents_map(
        iree_make_cstring_view(vmfb_path), IREE_IO_FILE_ACCESS_READ,
        iree_runtime_session_host_allocator(model->session), &contents);
    if (iree_status_is_ok(status)) {
      // Ownership of |contents| transfers to the module (freed on failure too).
      status = iree_runtime_session_append_bytecode_module_from_memory(
          model->session, contents->const_buffer,
          iree_io_file_contents_deallocator(contents));
    }
  }

  if (!iree_status_is_ok(status)) {
    lumen_iree_model_release(model);
    return lumen_error_from_status(status);
  }
  *out_model = model;
  return NULL;
}

void lumen_iree_model_release(lumen_iree_model_t* model) {
  if (!model) return;
  iree_runtime_session_release(model->session);
  free(model);
}

static iree_status_t lumen_full_name(const char* function, char* buffer,
                                     size_t capacity) {
  int n = snprintf(buffer, capacity, "module.%s", function);
  if (n < 0 || (size_t)n >= capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function name too long");
  }
  return iree_ok_status();
}

int32_t lumen_iree_model_has_function(lumen_iree_model_t* model,
                                      const char* function) {
  if (!model || !function) return 0;
  char name[256];
  iree_status_t status = lumen_full_name(function, name, sizeof(name));
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }
  iree_vm_function_t fn;
  status = iree_runtime_session_lookup_function(
      model->session, iree_make_cstring_view(name), &fn);
  int32_t found = iree_status_is_ok(status) ? 1 : 0;
  iree_status_ignore(status);
  return found;
}

lumen_iree_error_t* lumen_iree_model_trim(lumen_iree_model_t* model) {
  if (!model) return lumen_error_from_cstr("lumen_iree_model_trim: null model");
  return lumen_error_from_status(iree_runtime_session_trim(model->session));
}

//===----------------------------------------------------------------------===//
// Invocation
//===----------------------------------------------------------------------===//

struct lumen_iree_outputs_t {
  iree_hal_device_t* device;  // retained
  iree_host_size_t count;
  iree_hal_buffer_view_t** views;  // retained
};

static iree_status_t lumen_element_type(int32_t element_type,
                                        iree_hal_element_type_t* out_type,
                                        size_t* out_size) {
  switch (element_type) {
    case LUMEN_IREE_ELEMENT_F32:
      *out_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      *out_size = 4;
      return iree_ok_status();
    case LUMEN_IREE_ELEMENT_I64:
      *out_type = IREE_HAL_ELEMENT_TYPE_INT_64;
      *out_size = 8;
      return iree_ok_status();
    case LUMEN_IREE_ELEMENT_I32:
      *out_type = IREE_HAL_ELEMENT_TYPE_INT_32;
      *out_size = 4;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported element type %d", element_type);
  }
}

static int32_t lumen_element_type_from_hal(iree_hal_element_type_t type) {
  switch (type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return LUMEN_IREE_ELEMENT_F32;
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return LUMEN_IREE_ELEMENT_I64;
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return LUMEN_IREE_ELEMENT_I32;
    default:
      return 0;
  }
}

static iree_status_t lumen_push_input(iree_runtime_session_t* session,
                                      iree_runtime_call_t* call,
                                      const lumen_iree_tensor_t* tensor) {
  iree_hal_element_type_t element_type;
  size_t element_size = 0;
  IREE_RETURN_IF_ERROR(
      lumen_element_type(tensor->element_type, &element_type, &element_size));
  if (tensor->rank > 8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "rank %zu > 8",
                            tensor->rank);
  }
  iree_hal_dim_t shape[8];
  size_t element_count = 1;
  for (size_t i = 0; i < tensor->rank; ++i) {
    if (tensor->dims[i] < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "negative dim");
    }
    shape[i] = (iree_hal_dim_t)tensor->dims[i];
    element_count *= (size_t)tensor->dims[i];
  }
  if (element_count * element_size != tensor->byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "byte_length %zu does not match shape (%zu bytes)",
                            tensor->byte_length, element_count * element_size);
  }
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  iree_hal_buffer_view_t* view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(session),
      iree_runtime_session_device_allocator(session), tensor->rank, shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(tensor->data, tensor->byte_length), &view));
  iree_status_t status = iree_runtime_call_inputs_push_back_buffer_view(call, view);
  iree_hal_buffer_view_release(view);
  return status;
}

lumen_iree_error_t* lumen_iree_model_invoke(lumen_iree_model_t* model,
                                            const char* function,
                                            const lumen_iree_tensor_t* inputs,
                                            size_t input_count,
                                            lumen_iree_outputs_t** out_outputs) {
  if (!model || !function || (!inputs && input_count) || !out_outputs) {
    return lumen_error_from_cstr("lumen_iree_model_invoke: null argument");
  }
  *out_outputs = NULL;
  char name[256];
  iree_status_t status = lumen_full_name(function, name, sizeof(name));
  if (!iree_status_is_ok(status)) return lumen_error_from_status(status);

  iree_runtime_call_t call;
  status = iree_runtime_call_initialize_by_name(
      model->session, iree_make_cstring_view(name), &call);
  if (!iree_status_is_ok(status)) return lumen_error_from_status(status);

  for (size_t i = 0; i < input_count && iree_status_is_ok(status); ++i) {
    status = lumen_push_input(model->session, &call, &inputs[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  lumen_iree_outputs_t* outputs = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t count = iree_vm_list_size(iree_runtime_call_outputs(&call));
    outputs = (lumen_iree_outputs_t*)calloc(1, sizeof(*outputs));
    if (!outputs) abort();
    outputs->views = (iree_hal_buffer_view_t**)calloc(
        count ? count : 1, sizeof(iree_hal_buffer_view_t*));
    if (!outputs->views) abort();
    outputs->device = iree_runtime_session_device(model->session);
    iree_hal_device_retain(outputs->device);
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      status = iree_runtime_call_outputs_pop_front_buffer_view(
          &call, &outputs->views[i]);
      if (iree_status_is_ok(status)) outputs->count = i + 1;
    }
  }
  iree_runtime_call_deinitialize(&call);

  if (!iree_status_is_ok(status)) {
    lumen_iree_outputs_release(outputs);
    return lumen_error_from_status(status);
  }
  *out_outputs = outputs;
  return NULL;
}

size_t lumen_iree_outputs_count(const lumen_iree_outputs_t* outputs) {
  return outputs ? outputs->count : 0;
}

lumen_iree_error_t* lumen_iree_outputs_describe(const lumen_iree_outputs_t* outputs,
                                                size_t index, int32_t* out_element_type,
                                                size_t* out_rank, int64_t* dims,
                                                size_t dims_capacity,
                                                size_t* out_byte_length) {
  if (!outputs || index >= outputs->count || !out_element_type || !out_rank ||
      !out_byte_length) {
    return lumen_error_from_cstr("lumen_iree_outputs_describe: bad argument");
  }
  iree_hal_buffer_view_t* view = outputs->views[index];
  int32_t element_type =
      lumen_element_type_from_hal(iree_hal_buffer_view_element_type(view));
  if (element_type == 0) {
    return lumen_error_from_cstr("output has an unsupported element type");
  }
  size_t rank = iree_hal_buffer_view_shape_rank(view);
  *out_element_type = element_type;
  *out_rank = rank;
  *out_byte_length = (size_t)iree_hal_buffer_view_byte_length(view);
  if (dims) {
    if (dims_capacity < rank) {
      return lumen_error_from_cstr("lumen_iree_outputs_describe: dims too small");
    }
    for (size_t i = 0; i < rank; ++i) {
      dims[i] = (int64_t)iree_hal_buffer_view_shape_dim(view, i);
    }
  }
  return NULL;
}

lumen_iree_error_t* lumen_iree_outputs_read(const lumen_iree_outputs_t* outputs,
                                            size_t index, void* dst,
                                            size_t dst_byte_length) {
  if (!outputs || index >= outputs->count || (!dst && dst_byte_length)) {
    return lumen_error_from_cstr("lumen_iree_outputs_read: bad argument");
  }
  iree_hal_buffer_view_t* view = outputs->views[index];
  iree_device_size_t length = iree_hal_buffer_view_byte_length(view);
  if ((size_t)length != dst_byte_length) {
    return lumen_error_from_cstr("lumen_iree_outputs_read: length mismatch");
  }
  return lumen_error_from_status(iree_hal_device_transfer_d2h(
      outputs->device, iree_hal_buffer_view_buffer(view), 0, dst, length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
}

void lumen_iree_outputs_release(lumen_iree_outputs_t* outputs) {
  if (!outputs) return;
  for (iree_host_size_t i = 0; i < outputs->count; ++i) {
    iree_hal_buffer_view_release(outputs->views[i]);
  }
  free(outputs->views);
  iree_hal_device_release(outputs->device);
  free(outputs);
}
