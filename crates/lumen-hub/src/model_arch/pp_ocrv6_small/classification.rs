// Generated from ONNX "onnx/pp-ocrv6-small/classification.prepared.onnx" by burn-onnx
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    constant37: burn::module::Param<Tensor<B, 1>>,
    constant38: burn::module::Param<Tensor<B, 2>>,
    constant39: burn::module::Param<Tensor<B, 1>>,
    constant40: burn::module::Param<Tensor<B, 1>>,
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    globalaveragepool1: AdaptiveAvgPool2d,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    globalaveragepool2: AdaptiveAvgPool2d,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    globalaveragepool3: AdaptiveAvgPool2d,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    globalaveragepool4: AdaptiveAvgPool2d,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    globalaveragepool5: AdaptiveAvgPool2d,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    globalaveragepool6: AdaptiveAvgPool2d,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    globalaveragepool7: AdaptiveAvgPool2d,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    globalaveragepool8: AdaptiveAvgPool2d,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    globalaveragepool9: AdaptiveAvgPool2d,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    globalaveragepool10: AdaptiveAvgPool2d,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/Volumes/CodeBase/Projects/Lumen-Hub/target/debug/build/lumen-convert-517457e5909be238/out/pp_ocrv6_small/classification/classification.bpk",
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant37: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([2], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [2].into(),
        );
        let constant38: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2>::zeros([200, 2], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [200, 2].into(),
        );
        let constant39: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([6f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant40: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([3f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d1 = Conv2dConfig::new([3, 8], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([8, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([8, 8], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(8)
            .with_bias(true)
            .init(device);
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d4 = Conv2dConfig::new([8, 2], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([2, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([8, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([8, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(24)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([24, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([8, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([32, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([8, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let globalaveragepool2 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d15 = Conv2dConfig::new([32, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([8, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([32, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([16, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([88, 88], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(88)
            .with_bias(true)
            .init(device);
        let globalaveragepool3 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d20 = Conv2dConfig::new([88, 22], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([22, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([88, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([16, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([88, 88], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(88)
            .with_bias(true)
            .init(device);
        let globalaveragepool4 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d25 = Conv2dConfig::new([88, 22], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([22, 88], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([88, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([16, 40], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([40, 40], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(40)
            .with_bias(true)
            .init(device);
        let globalaveragepool5 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d30 = Conv2dConfig::new([40, 10], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([10, 40], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([40, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([16, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([48, 48], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(48)
            .with_bias(true)
            .init(device);
        let globalaveragepool6 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d35 = Conv2dConfig::new([48, 12], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([12, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([48, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([16, 104], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([104, 104], [5, 5])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(104)
            .with_bias(true)
            .init(device);
        let globalaveragepool7 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d40 = Conv2dConfig::new([104, 26], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([26, 104], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([104, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([32, 200], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([200, 200], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(200)
            .with_bias(true)
            .init(device);
        let globalaveragepool8 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d45 = Conv2dConfig::new([200, 50], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([50, 200], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([200, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([32, 200], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([200, 200], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(200)
            .with_bias(true)
            .init(device);
        let globalaveragepool9 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d50 = Conv2dConfig::new([200, 50], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([50, 200], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([200, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([32, 200], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let globalaveragepool10 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        Self {
            constant37,
            constant38,
            constant39,
            constant40,
            conv2d1,
            conv2d2,
            conv2d3,
            globalaveragepool1,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            globalaveragepool2,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            globalaveragepool3,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            globalaveragepool4,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            globalaveragepool5,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            globalaveragepool6,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            globalaveragepool7,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            globalaveragepool8,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            globalaveragepool9,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            maxpool2d1,
            globalaveragepool10,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let constant37_out1 = self.constant37.val();
        let constant38_out1 = self.constant38.val();
        let constant39_out1 = self.constant39.val();
        let constant40_out1 = self.constant40.val();
        let conv2d1_out1 = crate::model_arch::conv_fwd(&self.conv2d1, x);
        let add1_out1 = conv2d1_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip1_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add1_out1.clamp(__clip_min, __clip_max)
        };
        let mul1_out1 = conv2d1_out1.mul(clip1_out1);
        let div1_out1 =
            mul1_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d2_out1 = crate::model_arch::conv_fwd(&self.conv2d2, div1_out1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = crate::model_arch::conv_fwd(&self.conv2d3, relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(relu2_out1.clone());
        let conv2d4_out1 = crate::model_arch::conv_fwd(&self.conv2d4, globalaveragepool1_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = crate::model_arch::conv_fwd(&self.conv2d5, relu3_out1);
        let hardsigmoid1_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d5_out1, 0.20000000298023224, 0.5);
        let mul2_out1 = relu2_out1.mul(hardsigmoid1_out1);
        let conv2d6_out1 = crate::model_arch::conv_fwd(&self.conv2d6, mul2_out1);
        let conv2d7_out1 = crate::model_arch::conv_fwd(&self.conv2d7, conv2d6_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = crate::model_arch::conv_fwd(&self.conv2d8, relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = crate::model_arch::conv_fwd(&self.conv2d9, relu5_out1);
        let conv2d10_out1 = crate::model_arch::conv_fwd(&self.conv2d10, conv2d9_out1.clone());
        let relu6_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = crate::model_arch::conv_fwd(&self.conv2d11, relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = crate::model_arch::conv_fwd(&self.conv2d12, relu7_out1);
        let add2_out1 = conv2d9_out1.add(conv2d12_out1);
        let conv2d13_out1 = crate::model_arch::conv_fwd(&self.conv2d13, add2_out1);
        let add3_out1 = conv2d13_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip2_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add3_out1.clamp(__clip_min, __clip_max)
        };
        let mul3_out1 = conv2d13_out1.mul(clip2_out1);
        let div2_out1 =
            mul3_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d14_out1 = crate::model_arch::conv_fwd(&self.conv2d14, div2_out1);
        let add4_out1 = conv2d14_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip3_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add4_out1.clamp(__clip_min, __clip_max)
        };
        let mul4_out1 = conv2d14_out1.mul(clip3_out1);
        let div3_out1 =
            mul4_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool2_out1 = self.globalaveragepool2.forward(div3_out1.clone());
        let conv2d15_out1 = crate::model_arch::conv_fwd(&self.conv2d15, globalaveragepool2_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = crate::model_arch::conv_fwd(&self.conv2d16, relu8_out1);
        let hardsigmoid2_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d16_out1, 0.20000000298023224, 0.5);
        let mul5_out1 = div3_out1.mul(hardsigmoid2_out1);
        let conv2d17_out1 = crate::model_arch::conv_fwd(&self.conv2d17, mul5_out1);
        let conv2d18_out1 = crate::model_arch::conv_fwd(&self.conv2d18, conv2d17_out1.clone());
        let add5_out1 = conv2d18_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip4_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add5_out1.clamp(__clip_min, __clip_max)
        };
        let mul6_out1 = conv2d18_out1.mul(clip4_out1);
        let div4_out1 =
            mul6_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d19_out1 = crate::model_arch::conv_fwd(&self.conv2d19, div4_out1);
        let add6_out1 = conv2d19_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip5_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add6_out1.clamp(__clip_min, __clip_max)
        };
        let mul7_out1 = conv2d19_out1.mul(clip5_out1);
        let div5_out1 =
            mul7_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool3_out1 = self.globalaveragepool3.forward(div5_out1.clone());
        let conv2d20_out1 = crate::model_arch::conv_fwd(&self.conv2d20, globalaveragepool3_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = crate::model_arch::conv_fwd(&self.conv2d21, relu9_out1);
        let hardsigmoid3_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d21_out1, 0.20000000298023224, 0.5);
        let mul8_out1 = div5_out1.mul(hardsigmoid3_out1);
        let conv2d22_out1 = crate::model_arch::conv_fwd(&self.conv2d22, mul8_out1);
        let add7_out1 = conv2d17_out1.add(conv2d22_out1);
        let conv2d23_out1 = crate::model_arch::conv_fwd(&self.conv2d23, add7_out1.clone());
        let add8_out1 = conv2d23_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip6_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add8_out1.clamp(__clip_min, __clip_max)
        };
        let mul9_out1 = conv2d23_out1.mul(clip6_out1);
        let div6_out1 =
            mul9_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d24_out1 = crate::model_arch::conv_fwd(&self.conv2d24, div6_out1);
        let add9_out1 = conv2d24_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip7_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add9_out1.clamp(__clip_min, __clip_max)
        };
        let mul10_out1 = conv2d24_out1.mul(clip7_out1);
        let div7_out1 =
            mul10_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool4_out1 = self.globalaveragepool4.forward(div7_out1.clone());
        let conv2d25_out1 = crate::model_arch::conv_fwd(&self.conv2d25, globalaveragepool4_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = crate::model_arch::conv_fwd(&self.conv2d26, relu10_out1);
        let hardsigmoid4_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d26_out1, 0.20000000298023224, 0.5);
        let mul11_out1 = div7_out1.mul(hardsigmoid4_out1);
        let conv2d27_out1 = crate::model_arch::conv_fwd(&self.conv2d27, mul11_out1);
        let add10_out1 = add7_out1.add(conv2d27_out1);
        let conv2d28_out1 = crate::model_arch::conv_fwd(&self.conv2d28, add10_out1.clone());
        let add11_out1 = conv2d28_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip8_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add11_out1.clamp(__clip_min, __clip_max)
        };
        let mul12_out1 = conv2d28_out1.mul(clip8_out1);
        let div8_out1 =
            mul12_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d29_out1 = crate::model_arch::conv_fwd(&self.conv2d29, div8_out1);
        let add12_out1 = conv2d29_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip9_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add12_out1.clamp(__clip_min, __clip_max)
        };
        let mul13_out1 = conv2d29_out1.mul(clip9_out1);
        let div9_out1 =
            mul13_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool5_out1 = self.globalaveragepool5.forward(div9_out1.clone());
        let conv2d30_out1 = crate::model_arch::conv_fwd(&self.conv2d30, globalaveragepool5_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = crate::model_arch::conv_fwd(&self.conv2d31, relu11_out1);
        let hardsigmoid5_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d31_out1, 0.20000000298023224, 0.5);
        let mul14_out1 = div9_out1.mul(hardsigmoid5_out1);
        let conv2d32_out1 = crate::model_arch::conv_fwd(&self.conv2d32, mul14_out1);
        let add13_out1 = add10_out1.add(conv2d32_out1);
        let conv2d33_out1 = crate::model_arch::conv_fwd(&self.conv2d33, add13_out1.clone());
        let add14_out1 = conv2d33_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip10_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add14_out1.clamp(__clip_min, __clip_max)
        };
        let mul15_out1 = conv2d33_out1.mul(clip10_out1);
        let div10_out1 =
            mul15_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d34_out1 = crate::model_arch::conv_fwd(&self.conv2d34, div10_out1);
        let add15_out1 = conv2d34_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip11_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add15_out1.clamp(__clip_min, __clip_max)
        };
        let mul16_out1 = conv2d34_out1.mul(clip11_out1);
        let div11_out1 =
            mul16_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool6_out1 = self.globalaveragepool6.forward(div11_out1.clone());
        let conv2d35_out1 = crate::model_arch::conv_fwd(&self.conv2d35, globalaveragepool6_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = crate::model_arch::conv_fwd(&self.conv2d36, relu12_out1);
        let hardsigmoid6_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d36_out1, 0.20000000298023224, 0.5);
        let mul17_out1 = div11_out1.mul(hardsigmoid6_out1);
        let conv2d37_out1 = crate::model_arch::conv_fwd(&self.conv2d37, mul17_out1);
        let add16_out1 = add13_out1.add(conv2d37_out1);
        let conv2d38_out1 = crate::model_arch::conv_fwd(&self.conv2d38, add16_out1);
        let add17_out1 = conv2d38_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip12_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add17_out1.clamp(__clip_min, __clip_max)
        };
        let mul18_out1 = conv2d38_out1.mul(clip12_out1);
        let div12_out1 =
            mul18_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d39_out1 = crate::model_arch::conv_fwd(&self.conv2d39, div12_out1);
        let add18_out1 = conv2d39_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip13_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add18_out1.clamp(__clip_min, __clip_max)
        };
        let mul19_out1 = conv2d39_out1.mul(clip13_out1);
        let div13_out1 =
            mul19_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool7_out1 = self.globalaveragepool7.forward(div13_out1.clone());
        let conv2d40_out1 = crate::model_arch::conv_fwd(&self.conv2d40, globalaveragepool7_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let conv2d41_out1 = crate::model_arch::conv_fwd(&self.conv2d41, relu13_out1);
        let hardsigmoid7_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d41_out1, 0.20000000298023224, 0.5);
        let mul20_out1 = div13_out1.mul(hardsigmoid7_out1);
        let conv2d42_out1 = crate::model_arch::conv_fwd(&self.conv2d42, mul20_out1);
        let conv2d43_out1 = crate::model_arch::conv_fwd(&self.conv2d43, conv2d42_out1.clone());
        let add19_out1 = conv2d43_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip14_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add19_out1.clamp(__clip_min, __clip_max)
        };
        let mul21_out1 = conv2d43_out1.mul(clip14_out1);
        let div14_out1 =
            mul21_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d44_out1 = crate::model_arch::conv_fwd(&self.conv2d44, div14_out1);
        let add20_out1 = conv2d44_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip15_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add20_out1.clamp(__clip_min, __clip_max)
        };
        let mul22_out1 = conv2d44_out1.mul(clip15_out1);
        let div15_out1 =
            mul22_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool8_out1 = self.globalaveragepool8.forward(div15_out1.clone());
        let conv2d45_out1 = crate::model_arch::conv_fwd(&self.conv2d45, globalaveragepool8_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let conv2d46_out1 = crate::model_arch::conv_fwd(&self.conv2d46, relu14_out1);
        let hardsigmoid8_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d46_out1, 0.20000000298023224, 0.5);
        let mul23_out1 = div15_out1.mul(hardsigmoid8_out1);
        let conv2d47_out1 = crate::model_arch::conv_fwd(&self.conv2d47, mul23_out1);
        let add21_out1 = conv2d42_out1.add(conv2d47_out1);
        let conv2d48_out1 = crate::model_arch::conv_fwd(&self.conv2d48, add21_out1.clone());
        let add22_out1 = conv2d48_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip16_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add22_out1.clamp(__clip_min, __clip_max)
        };
        let mul24_out1 = conv2d48_out1.mul(clip16_out1);
        let div16_out1 =
            mul24_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d49_out1 = crate::model_arch::conv_fwd(&self.conv2d49, div16_out1);
        let add23_out1 = conv2d49_out1
            .clone()
            .add((constant40_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip17_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add23_out1.clamp(__clip_min, __clip_max)
        };
        let mul25_out1 = conv2d49_out1.mul(clip17_out1);
        let div17_out1 =
            mul25_out1.div((constant39_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool9_out1 = self.globalaveragepool9.forward(div17_out1.clone());
        let conv2d50_out1 = crate::model_arch::conv_fwd(&self.conv2d50, globalaveragepool9_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d50_out1);
        let conv2d51_out1 = crate::model_arch::conv_fwd(&self.conv2d51, relu15_out1);
        let hardsigmoid9_out1 =
            burn::tensor::activation::hard_sigmoid(conv2d51_out1, 0.20000000298023224, 0.5);
        let mul26_out1 = div17_out1.mul(hardsigmoid9_out1);
        let conv2d52_out1 = crate::model_arch::conv_fwd(&self.conv2d52, mul26_out1);
        let add24_out1 = add21_out1.add(conv2d52_out1);
        let conv2d53_out1 = crate::model_arch::conv_fwd(&self.conv2d53, add24_out1);
        let add25_out1 = conv2d53_out1
            .clone()
            .add((constant40_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let clip18_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 6f64;
            add25_out1.clamp(__clip_min, __clip_max)
        };
        let mul27_out1 = conv2d53_out1.mul(clip18_out1);
        let div18_out1 =
            mul27_out1.div((constant39_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let maxpool2d1_out1 = self.maxpool2d1.forward(div18_out1);
        let globalaveragepool10_out1 = self.globalaveragepool10.forward(maxpool2d1_out1);
        let reshape1_out1 = globalaveragepool10_out1.reshape([1, 200]);
        let gemm1_out1 = reshape1_out1.matmul(constant38_out1) + constant37_out1.unsqueeze();
        let softmax1_out1 = burn::tensor::activation::softmax(gemm1_out1, 1);
        softmax1_out1
    }
}
