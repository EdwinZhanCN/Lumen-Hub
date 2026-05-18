use crate::core::packet::{MLPacketDataType, MLPacketDescriptor};

/// Converts an MNN shape slice into an [`MLPacketDescriptor`].
///
/// `halide_code` is the packed halide type: lower 16 bits = code (0=int,1=uint,2=float),
/// upper 16 bits = bits (8/16/32/64).
pub(crate) fn mnn_shape_to_descriptor(shape: &[usize], packed_type: i32) -> MLPacketDescriptor {
    let (concrete_shape, dynamic_axes): (Vec<usize>, Vec<bool>) = shape
        .iter()
        .map(|&d| {
            if d > 100_000 {
                // -1 dimension → dynamic axis, use 1 as placeholder
                (1usize, true)
            } else {
                (d, false)
            }
        })
        .unzip();

    let dynamic_batch = dynamic_axes.first().copied().unwrap_or(false);

    let dtype = {
        let code = packed_type & 0xFFFF;
        let bits = (packed_type >> 16) & 0xFFFF;
        match (code, bits) {
            (2, 32) => MLPacketDataType::Float32, // halide_type_float, 32
            (0, 64) => MLPacketDataType::Int64,   // halide_type_int, 64
            (0, 32) => MLPacketDataType::Int32,   // halide_type_int, 32
            _ => MLPacketDataType::Float32,       // fallback
        }
    };

    MLPacketDescriptor {
        dtype,
        shape: concrete_shape,
        dynamic_batch,
        dynamic_axes,
    }
}
