# lumnn-ncnn-sys

Minimal raw ncnn C API bindings for Lumen.

This crate intentionally hand-writes the small FFI surface Lumen needs instead
of using bindgen. That keeps the Lumen Hub one-click path free from libclang and
large generated binding files.

## Default build

Default features are:

```text
vendored + shared + vulkan
```

Vendored mode expects an ncnn checkout at:

```text
crates/lumnn-ncnn-sys/third_party/ncnn
```

or a custom path:

```bash
export NCNN_SOURCE_DIR=/absolute/path/to/ncnn
```

## macOS Vulkan

macOS Vulkan goes through MoltenVK. Point `VULKAN_SDK` to the SDK's `macOS`
directory:

```bash
export VULKAN_SDK="$HOME/VulkanSDK/1.4.x.x/macOS"
```

The build script passes these CMake values:

```text
Vulkan_INCLUDE_DIR=$VULKAN_SDK/include
Vulkan_LIBRARY=$VULKAN_SDK/lib/libMoltenVK.dylib
```

For release packaging, copy both `libncnn.dylib` and `libMoltenVK.dylib` next to
the Lumen Hub distribution's `lib/` directory.

## System mode

For local debugging with an already-installed ncnn:

```bash
NCNN_DIR=/absolute/path/to/ncnn/install \
  cargo check -p lumnn-ncnn-sys --no-default-features --features system
```
