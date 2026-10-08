# Builds the IREE runtime (no compiler) plus the Lumen shim into ONE static
# archive, liblumen_iree.a / lumen_iree.lib, driven by lumen-iree-sys/build.rs.
#
#   cmake -G Ninja -S csrc -B <build> -DCMAKE_BUILD_TYPE=Release \
#         -DIREE_SOURCE_DIR=<pinned IREE checkout> \
#         -DLUMEN_IREE_DRIVERS="local-task;cuda"      # or "local-task;metal"
#   cmake --build <build> --target lumen_iree
cmake_minimum_required(VERSION 3.21)
project(lumen_iree C CXX)

set(CMAKE_C_STANDARD 11)
set(IREE_SOURCE_DIR "" CACHE PATH "Pinned IREE source checkout")
set(LUMEN_IREE_DRIVERS "local-task" CACHE STRING "Semicolon list of HAL drivers: local-task, cuda, metal")
if(NOT EXISTS "${IREE_SOURCE_DIR}/runtime/src/iree/runtime/api.h")
  message(FATAL_ERROR "IREE_SOURCE_DIR=${IREE_SOURCE_DIR} is not an IREE checkout (run `cargo xtask iree-fetch`)")
endif()

# Runtime-only IREE.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_LIBBACKTRACE OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_WERROR_FLAG OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_RUNTIME_TRACING OFF CACHE BOOL "" FORCE)

# Exactly the requested HAL drivers.
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
foreach(_driver LOCAL_SYNC LOCAL_TASK VULKAN CUDA HIP METAL AMDGPU NULL)
  set(IREE_HAL_DRIVER_${_driver} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(_d IN LISTS LUMEN_IREE_DRIVERS)
  string(TOUPPER "${_d}" _D)
  string(REPLACE "-" "_" _D "${_D}")
  if(NOT _D MATCHES "^(LOCAL_TASK|CUDA|METAL)$")
    message(FATAL_ERROR "unsupported driver ${_d}")
  endif()
  set(IREE_HAL_DRIVER_${_D} ON CACHE BOOL "" FORCE)
endforeach()

# CPU executables: embedded ELF only (the format iree-compile emits by default).
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF ON CACHE BOOL "" FORCE)

add_subdirectory(${IREE_SOURCE_DIR} iree EXCLUDE_FROM_ALL)

set(_roots
  iree::runtime::impl
  iree::io::formats::parser_registry
  iree::io::parameter_index_provider
  iree::modules::io::parameters)

add_library(lumen_iree_shim OBJECT lumen_iree.c)
target_link_libraries(lumen_iree_shim PRIVATE ${_roots})

# Fold the shim and every transitive IREE object into one archive.
set(_objs "")
set(_libs "")
foreach(_r IN LISTS _roots)
  list(APPEND _objs "$<GENEX_EVAL:$<TARGET_PROPERTY:${_r},INTERFACE_IREE_TRANSITIVE_OBJECTS>>")
  list(APPEND _libs "$<GENEX_EVAL:$<TARGET_PROPERTY:${_r},INTERFACE_IREE_TRANSITIVE_OBJECT_LIBS>>")
endforeach()
# flatcc::parsing (the VM bytecode verifier) is a plain static library, not an
# IREE object library, so its objects are folded in explicitly.
add_library(lumen_iree STATIC
  $<TARGET_OBJECTS:lumen_iree_shim>
  "$<REMOVE_DUPLICATES:${_objs}>"
  $<TARGET_OBJECTS:flatcc_parsing>)

# Non-object link dependencies for build.rs (system libraries, frameworks).
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/lumen_iree_link_libs.txt"
     CONTENT "$<REMOVE_DUPLICATES:${_libs}>\n")
