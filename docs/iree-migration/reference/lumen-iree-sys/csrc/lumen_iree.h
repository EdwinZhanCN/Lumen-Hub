// Lumen IREE shim: the complete C ABI that lumen-iree-sys exposes to Rust.
//
// The shim is compiled against the pinned IREE runtime headers and hides every
// IREE macro, inline function and struct layout behind opaque handles and
// plain-C argument types, so the Rust side never mirrors IREE internals.
//
// Conventions
//   * Every function that can fail returns NULL on success or an owned
//     lumen_iree_error_t* that the caller releases with lumen_iree_error_free.
//   * Out-parameters are written only on success; on failure they are NULL/0.
//   * Handles are released exactly once with their *_release function.
//   * lumen_iree_runtime_t is thread-safe. A lumen_iree_model_t must not be
//     used from two threads at the same time (the Rust wrapper serializes it).
//   * A model keeps no reference to its runtime: the caller must release all
//     models (and outputs) of a runtime before releasing the runtime.
#ifndef LUMEN_IREE_H_
#define LUMEN_IREE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lumen_iree_error_t lumen_iree_error_t;
typedef struct lumen_iree_runtime_t lumen_iree_runtime_t;
typedef struct lumen_iree_model_t lumen_iree_model_t;
typedef struct lumen_iree_outputs_t lumen_iree_outputs_t;

// Element types accepted and produced at the ABI. Values are stable ABI.
enum {
  LUMEN_IREE_ELEMENT_F32 = 1,
  LUMEN_IREE_ELEMENT_I64 = 2,
  LUMEN_IREE_ELEMENT_I32 = 3,
};

// Parameter (.irpa) residency. Values are stable ABI.
enum {
  // Map the archive read-only. Devices that can import host memory (the CPU
  // local-task device) alias the mapping: weights stay page cache, not heap.
  LUMEN_IREE_PARAMS_MMAP = 0,
  // Read parameters through file I/O into device buffers at load time.
  LUMEN_IREE_PARAMS_READ = 1,
};

typedef struct lumen_iree_tensor_t {
  int32_t element_type;  // LUMEN_IREE_ELEMENT_*
  size_t rank;           // <= 8
  const int64_t* dims;   // rank entries, each >= 0
  const void* data;      // dense, row-major
  size_t byte_length;    // == product(dims) * element size
} lumen_iree_tensor_t;

// Returns the NUL-terminated message of |error| ("" for NULL).
const char* lumen_iree_error_message(const lumen_iree_error_t* error);
void lumen_iree_error_free(lumen_iree_error_t* error);

// Creates an IREE instance with every driver compiled into the library and the
// default device of |driver| ("local-task", "cuda" or "metal").
lumen_iree_error_t* lumen_iree_runtime_create(const char* driver,
                                              lumen_iree_runtime_t** out_runtime);
void lumen_iree_runtime_release(lumen_iree_runtime_t* runtime);

// Loads one compiled module (.vmfb) with its parameter archive (.irpa) whose
// parameters live in scope |param_scope| ("model" for every Lumen artifact).
lumen_iree_error_t* lumen_iree_model_load(lumen_iree_runtime_t* runtime,
                                          const char* vmfb_path,
                                          const char* irpa_path,
                                          const char* param_scope,
                                          int32_t params_mode,
                                          lumen_iree_model_t** out_model);
void lumen_iree_model_release(lumen_iree_model_t* model);

// Returns 1 if the module exports |function| (looked up as "module.<function>").
int32_t lumen_iree_model_has_function(lumen_iree_model_t* model,
                                      const char* function);

// Synchronously invokes "module.<function>" with |input_count| inputs and
// returns all results.
lumen_iree_error_t* lumen_iree_model_invoke(lumen_iree_model_t* model,
                                            const char* function,
                                            const lumen_iree_tensor_t* inputs,
                                            size_t input_count,
                                            lumen_iree_outputs_t** out_outputs);

// Releases cached device memory of the model's session
// (iree_runtime_session_trim).
lumen_iree_error_t* lumen_iree_model_trim(lumen_iree_model_t* model);

size_t lumen_iree_outputs_count(const lumen_iree_outputs_t* outputs);

// Describes output |index|. |dims| may be NULL to query only the rank; when
// non-NULL it must hold at least rank entries (|dims_capacity|).
lumen_iree_error_t* lumen_iree_outputs_describe(
    const lumen_iree_outputs_t* outputs, size_t index, int32_t* out_element_type,
    size_t* out_rank, int64_t* dims, size_t dims_capacity,
    size_t* out_byte_length);

// Copies output |index| into |dst|; |dst_byte_length| must equal its byte
// length as reported by lumen_iree_outputs_describe.
lumen_iree_error_t* lumen_iree_outputs_read(const lumen_iree_outputs_t* outputs,
                                            size_t index, void* dst,
                                            size_t dst_byte_length);
void lumen_iree_outputs_release(lumen_iree_outputs_t* outputs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LUMEN_IREE_H_
