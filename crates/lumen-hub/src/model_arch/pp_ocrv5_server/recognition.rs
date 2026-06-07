// Generated from ONNX "onnx/pp-ocrv5-server/recognition.prepared.onnx" by burn-onnx
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv2d1: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
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
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_ceil_mode(true)
            .init();
        let conv2d2 = Conv2dConfig::new([32, 16], [2, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([16, 32], [2, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([64, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(48)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([336, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([128, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([704, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([512, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            maxpool2d1,
            conv2d2,
            conv2d3,
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
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = crate::model_arch::conv_fwd(&self.conv2d1, x);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu1_out1.clone());
        let conv2d2_out1 = crate::model_arch::conv_fwd(&self.conv2d2, relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = crate::model_arch::conv_fwd(&self.conv2d3, relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([maxpool2d1_out1, relu3_out1].into(), 1);
        let conv2d4_out1 = crate::model_arch::conv_fwd(&self.conv2d4, concat1_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = crate::model_arch::conv_fwd(&self.conv2d5, relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = crate::model_arch::conv_fwd(&self.conv2d6, relu5_out1);
        let conv2d7_out1 = crate::model_arch::conv_fwd(&self.conv2d7, conv2d6_out1.clone());
        let relu6_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = crate::model_arch::conv_fwd(&self.conv2d8, relu6_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = crate::model_arch::conv_fwd(&self.conv2d9, relu7_out1.clone());
        let relu8_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = crate::model_arch::conv_fwd(&self.conv2d10, relu8_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = crate::model_arch::conv_fwd(&self.conv2d11, relu9_out1.clone());
        let relu10_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = crate::model_arch::conv_fwd(&self.conv2d12, relu10_out1.clone());
        let relu11_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [
                conv2d6_out1,
                relu6_out1,
                relu7_out1,
                relu8_out1,
                relu9_out1,
                relu10_out1,
                relu11_out1,
            ]
            .into(),
            1,
        );
        let conv2d13_out1 = crate::model_arch::conv_fwd(&self.conv2d13, concat2_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = crate::model_arch::conv_fwd(&self.conv2d14, relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = crate::model_arch::conv_fwd(&self.conv2d15, relu13_out1);
        let conv2d16_out1 = crate::model_arch::conv_fwd(&self.conv2d16, conv2d15_out1.clone());
        let relu14_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = crate::model_arch::conv_fwd(&self.conv2d17, relu14_out1.clone());
        let relu15_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = crate::model_arch::conv_fwd(&self.conv2d18, relu15_out1.clone());
        let relu16_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let conv2d19_out1 = crate::model_arch::conv_fwd(&self.conv2d19, relu16_out1.clone());
        let relu17_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = crate::model_arch::conv_fwd(&self.conv2d20, relu17_out1.clone());
        let relu18_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = crate::model_arch::conv_fwd(&self.conv2d21, relu18_out1.clone());
        let relu19_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [
                conv2d15_out1,
                relu14_out1,
                relu15_out1,
                relu16_out1,
                relu17_out1,
                relu18_out1,
                relu19_out1,
            ]
            .into(),
            1,
        );
        let conv2d22_out1 = crate::model_arch::conv_fwd(&self.conv2d22, concat3_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d22_out1);
        let conv2d23_out1 = crate::model_arch::conv_fwd(&self.conv2d23, relu20_out1);
        let relu21_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = crate::model_arch::conv_fwd(&self.conv2d24, relu21_out1);
        let conv2d25_out1 = crate::model_arch::conv_fwd(&self.conv2d25, conv2d24_out1.clone());
        let conv2d26_out1 = crate::model_arch::conv_fwd(&self.conv2d26, conv2d25_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let conv2d27_out1 = crate::model_arch::conv_fwd(&self.conv2d27, relu22_out1.clone());
        let conv2d28_out1 = crate::model_arch::conv_fwd(&self.conv2d28, conv2d27_out1);
        let relu23_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = crate::model_arch::conv_fwd(&self.conv2d29, relu23_out1.clone());
        let conv2d30_out1 = crate::model_arch::conv_fwd(&self.conv2d30, conv2d29_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = crate::model_arch::conv_fwd(&self.conv2d31, relu24_out1.clone());
        let conv2d32_out1 = crate::model_arch::conv_fwd(&self.conv2d32, conv2d31_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv2d32_out1);
        let conv2d33_out1 = crate::model_arch::conv_fwd(&self.conv2d33, relu25_out1.clone());
        let conv2d34_out1 = crate::model_arch::conv_fwd(&self.conv2d34, conv2d33_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d34_out1);
        let conv2d35_out1 = crate::model_arch::conv_fwd(&self.conv2d35, relu26_out1.clone());
        let conv2d36_out1 = crate::model_arch::conv_fwd(&self.conv2d36, conv2d35_out1);
        let relu27_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [
                conv2d24_out1,
                relu22_out1,
                relu23_out1,
                relu24_out1,
                relu25_out1,
                relu26_out1,
                relu27_out1,
            ]
            .into(),
            1,
        );
        concat4_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    conv2d58: Conv2d<B>,
    conv2d59: Conv2d<B>,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    conv2d62: Conv2d<B>,
    conv2d63: Conv2d<B>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    conv2d72: Conv2d<B>,
    conv2d73: Conv2d<B>,
    conv2d74: Conv2d<B>,
    conv2d75: Conv2d<B>,
    conv2d76: Conv2d<B>,
    conv2d77: Conv2d<B>,
    conv2d78: Conv2d<B>,
    conv2d79: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d37 = Conv2dConfig::new([1664, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([1024, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([2176, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([1024, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([2176, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([1024, 1024], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1024)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([1024, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        let conv2d74 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d75 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        let conv2d78 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        Self {
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            conv2d74,
            conv2d75,
            conv2d76,
            conv2d77,
            conv2d78,
            conv2d79,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, concat4_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d37_out1 = crate::model_arch::conv_fwd(&self.conv2d37, concat4_out1);
        let relu28_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = crate::model_arch::conv_fwd(&self.conv2d38, relu28_out1);
        let relu29_out1 = burn::tensor::activation::relu(conv2d38_out1);
        let conv2d39_out1 = crate::model_arch::conv_fwd(&self.conv2d39, relu29_out1.clone());
        let conv2d40_out1 = crate::model_arch::conv_fwd(&self.conv2d40, conv2d39_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let conv2d41_out1 = crate::model_arch::conv_fwd(&self.conv2d41, relu30_out1.clone());
        let conv2d42_out1 = crate::model_arch::conv_fwd(&self.conv2d42, conv2d41_out1);
        let relu31_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = crate::model_arch::conv_fwd(&self.conv2d43, relu31_out1.clone());
        let conv2d44_out1 = crate::model_arch::conv_fwd(&self.conv2d44, conv2d43_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = crate::model_arch::conv_fwd(&self.conv2d45, relu32_out1.clone());
        let conv2d46_out1 = crate::model_arch::conv_fwd(&self.conv2d46, conv2d45_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d46_out1);
        let conv2d47_out1 = crate::model_arch::conv_fwd(&self.conv2d47, relu33_out1.clone());
        let conv2d48_out1 = crate::model_arch::conv_fwd(&self.conv2d48, conv2d47_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = crate::model_arch::conv_fwd(&self.conv2d49, relu34_out1.clone());
        let conv2d50_out1 = crate::model_arch::conv_fwd(&self.conv2d50, conv2d49_out1);
        let relu35_out1 = burn::tensor::activation::relu(conv2d50_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [
                relu29_out1.clone(),
                relu30_out1,
                relu31_out1,
                relu32_out1,
                relu33_out1,
                relu34_out1,
                relu35_out1,
            ]
            .into(),
            1,
        );
        let conv2d51_out1 = crate::model_arch::conv_fwd(&self.conv2d51, concat5_out1);
        let relu36_out1 = burn::tensor::activation::relu(conv2d51_out1);
        let conv2d52_out1 = crate::model_arch::conv_fwd(&self.conv2d52, relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let add1_out1 = relu37_out1.add(relu29_out1);
        let conv2d53_out1 = crate::model_arch::conv_fwd(&self.conv2d53, add1_out1.clone());
        let conv2d54_out1 = crate::model_arch::conv_fwd(&self.conv2d54, conv2d53_out1);
        let relu38_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let conv2d55_out1 = crate::model_arch::conv_fwd(&self.conv2d55, relu38_out1.clone());
        let conv2d56_out1 = crate::model_arch::conv_fwd(&self.conv2d56, conv2d55_out1);
        let relu39_out1 = burn::tensor::activation::relu(conv2d56_out1);
        let conv2d57_out1 = crate::model_arch::conv_fwd(&self.conv2d57, relu39_out1.clone());
        let conv2d58_out1 = crate::model_arch::conv_fwd(&self.conv2d58, conv2d57_out1);
        let relu40_out1 = burn::tensor::activation::relu(conv2d58_out1);
        let conv2d59_out1 = crate::model_arch::conv_fwd(&self.conv2d59, relu40_out1.clone());
        let conv2d60_out1 = crate::model_arch::conv_fwd(&self.conv2d60, conv2d59_out1);
        let relu41_out1 = burn::tensor::activation::relu(conv2d60_out1);
        let conv2d61_out1 = crate::model_arch::conv_fwd(&self.conv2d61, relu41_out1.clone());
        let conv2d62_out1 = crate::model_arch::conv_fwd(&self.conv2d62, conv2d61_out1);
        let relu42_out1 = burn::tensor::activation::relu(conv2d62_out1);
        let conv2d63_out1 = crate::model_arch::conv_fwd(&self.conv2d63, relu42_out1.clone());
        let conv2d64_out1 = crate::model_arch::conv_fwd(&self.conv2d64, conv2d63_out1);
        let relu43_out1 = burn::tensor::activation::relu(conv2d64_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [
                add1_out1.clone(),
                relu38_out1,
                relu39_out1,
                relu40_out1,
                relu41_out1,
                relu42_out1,
                relu43_out1,
            ]
            .into(),
            1,
        );
        let conv2d65_out1 = crate::model_arch::conv_fwd(&self.conv2d65, concat6_out1);
        let relu44_out1 = burn::tensor::activation::relu(conv2d65_out1);
        let conv2d66_out1 = crate::model_arch::conv_fwd(&self.conv2d66, relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(conv2d66_out1);
        let add2_out1 = relu45_out1.add(add1_out1);
        let conv2d67_out1 = crate::model_arch::conv_fwd(&self.conv2d67, add2_out1);
        let conv2d68_out1 = crate::model_arch::conv_fwd(&self.conv2d68, conv2d67_out1.clone());
        let conv2d69_out1 = crate::model_arch::conv_fwd(&self.conv2d69, conv2d68_out1);
        let relu46_out1 = burn::tensor::activation::relu(conv2d69_out1);
        let conv2d70_out1 = crate::model_arch::conv_fwd(&self.conv2d70, relu46_out1.clone());
        let conv2d71_out1 = crate::model_arch::conv_fwd(&self.conv2d71, conv2d70_out1);
        let relu47_out1 = burn::tensor::activation::relu(conv2d71_out1);
        let conv2d72_out1 = crate::model_arch::conv_fwd(&self.conv2d72, relu47_out1.clone());
        let conv2d73_out1 = crate::model_arch::conv_fwd(&self.conv2d73, conv2d72_out1);
        let relu48_out1 = burn::tensor::activation::relu(conv2d73_out1);
        let conv2d74_out1 = crate::model_arch::conv_fwd(&self.conv2d74, relu48_out1.clone());
        let conv2d75_out1 = crate::model_arch::conv_fwd(&self.conv2d75, conv2d74_out1);
        let relu49_out1 = burn::tensor::activation::relu(conv2d75_out1);
        let conv2d76_out1 = crate::model_arch::conv_fwd(&self.conv2d76, relu49_out1.clone());
        let conv2d77_out1 = crate::model_arch::conv_fwd(&self.conv2d77, conv2d76_out1);
        let relu50_out1 = burn::tensor::activation::relu(conv2d77_out1);
        let conv2d78_out1 = crate::model_arch::conv_fwd(&self.conv2d78, relu50_out1.clone());
        let conv2d79_out1 = crate::model_arch::conv_fwd(&self.conv2d79, conv2d78_out1);
        let relu51_out1 = burn::tensor::activation::relu(conv2d79_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [
                conv2d67_out1,
                relu46_out1,
                relu47_out1,
                relu48_out1,
                relu49_out1,
                relu50_out1,
                relu51_out1,
            ]
            .into(),
            1,
        );
        concat7_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    averagepool2d1: AvgPool2d,
    conv2d82: Conv2d<B>,
    conv2d83: Conv2d<B>,
    constant174: burn::module::Param<Tensor<B, 1>>,
    constant175: burn::module::Param<Tensor<B, 1>>,
    constant176: burn::module::Param<Tensor<B, 1>>,
    constant177: burn::module::Param<Tensor<B, 1>>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    constant185: burn::module::Param<Tensor<B, 1>>,
    constant186: burn::module::Param<Tensor<B, 1>>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d80 = Conv2dConfig::new([3328, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([1024, 2048], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d1 = AvgPool2dConfig::new([3, 2])
            .with_strides([3, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv2d82 = Conv2dConfig::new([2048, 256], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([256, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant174: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([2f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant175: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([0.000009999999747378752f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant176: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let constant177: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let linear1 = LinearConfig::new(120, 360).with_bias(true).init(device);
        let linear2 = LinearConfig::new(120, 120).with_bias(true).init(device);
        let constant185: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let constant186: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let linear3 = LinearConfig::new(120, 240).with_bias(true).init(device);
        let linear4 = LinearConfig::new(240, 120).with_bias(true).init(device);
        Self {
            conv2d80,
            conv2d81,
            averagepool2d1,
            conv2d82,
            conv2d83,
            constant174,
            constant175,
            constant176,
            constant177,
            linear1,
            linear2,
            constant185,
            constant186,
            linear3,
            linear4,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        concat7_out1: Tensor<B, 4>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        [i64; 4],
        Tensor<B, 4>,
    ) {
        let conv2d80_out1 = crate::model_arch::conv_fwd(&self.conv2d80, concat7_out1);
        let relu52_out1 = burn::tensor::activation::relu(conv2d80_out1);
        let conv2d81_out1 = crate::model_arch::conv_fwd(&self.conv2d81, relu52_out1);
        let relu53_out1 = burn::tensor::activation::relu(conv2d81_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(relu53_out1);
        let unsqueeze1_out1: Tensor<B, 5> = averagepool2d1_out1.unsqueeze_dims::<5>(&[0]);
        let squeeze1_out1 = unsqueeze1_out1.squeeze_dims::<4>(&[0]);
        let conv2d82_out1 = crate::model_arch::conv_fwd(&self.conv2d82, squeeze1_out1.clone());
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d82_out1.clone());
        let mul1_out1 = conv2d82_out1.mul(sigmoid1_out1);
        let conv2d83_out1 = crate::model_arch::conv_fwd(&self.conv2d83, mul1_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv2d83_out1.clone());
        let mul2_out1 = conv2d83_out1.mul(sigmoid2_out1);
        let shape1_out1: [i64; 4] = {
            let axes = &mul2_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 2] = shape1_out1[0..2].try_into().unwrap();
        let slice2_out1: [i64; 1] = shape1_out1[3..4].try_into().unwrap();
        let constant172_out1: [i64; 1] = [-1i64];
        let concat8_out1: [i64; 3usize] = [&slice1_out1[..], &constant172_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let squeeze2_out1 = slice2_out1[0] as i64;
        let reshape1_out1 = mul2_out1.reshape(concat8_out1);
        let unsqueeze2_out1 = [squeeze2_out1 as i64];
        let transpose1_out1 = reshape1_out1.permute([0, 2, 1]);
        let constant163_out1: [i64; 1] = [0i64];
        let constant171_out1: [i64; 1] = [1i64];
        let constant173_out1: [i64; 1] = [120i64];
        let concat9_out1: [i64; 4usize] = [
            &constant163_out1[..],
            &constant171_out1[..],
            &unsqueeze2_out1[..],
            &constant173_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reducemean1_out1 = { transpose1_out1.clone().mean_dim(2usize) };
        let sub1_out1 = transpose1_out1.clone().sub(reducemean1_out1);
        let constant174_out1 = self.constant174.val();
        let pow1_out1 = sub1_out1
            .clone()
            .powf((constant174_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean2_out1 = { pow1_out1.mean_dim(2usize) };
        let constant175_out1 = self.constant175.val();
        let add3_out1 =
            reducemean2_out1.add((constant175_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt1_out1 = add3_out1.sqrt();
        let div1_out1 = sub1_out1.div(sqrt1_out1);
        let constant176_out1 = self.constant176.val();
        let mul3_out1 = div1_out1.mul((constant176_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant177_out1 = self.constant177.val();
        let add4_out1 = mul3_out1.add((constant177_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear1_out1 = self.linear1.forward(add4_out1);
        let reshape2_out1 = linear1_out1.reshape([0, -1, 3, 8, 15]);
        let transpose2_out1 = reshape2_out1.permute([2, 0, 3, 1, 4]);
        let slice3_out1 = transpose2_out1.clone().slice(s![0..1, .., .., .., ..]);
        let slice4_out1 = transpose2_out1.clone().slice(s![1..2, .., .., .., ..]);
        let slice5_out1 = transpose2_out1.slice(s![2..3, .., .., .., ..]);
        let squeeze3_out1 = slice3_out1.squeeze_dims::<4>(&[0]);
        let squeeze4_out1 = slice4_out1.squeeze_dims::<4>(&[0]);
        let squeeze5_out1 = slice5_out1.squeeze_dims::<4>(&[0]);
        let (matmul3_out1,) = {
            let q = squeeze3_out1;
            let k = squeeze4_out1;
            let v = squeeze5_out1;
            let matmul3_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.25819888710975647f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul3_out1,)
        };
        let transpose4_out1 = matmul3_out1.permute([0, 2, 1, 3]);
        let reshape3_out1 = transpose4_out1.reshape([0, -1, 120]);
        let linear2_out1 = self.linear2.forward(reshape3_out1);
        let add5_out1 = transpose1_out1.add(linear2_out1);
        let reducemean3_out1 = { add5_out1.clone().mean_dim(2usize) };
        let sub2_out1 = add5_out1.clone().sub(reducemean3_out1);
        let pow2_out1 = sub2_out1
            .clone()
            .powf((constant174_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean4_out1 = { pow2_out1.mean_dim(2usize) };
        let add6_out1 =
            reducemean4_out1.add((constant175_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt2_out1 = add6_out1.sqrt();
        let div2_out1 = sub2_out1.div(sqrt2_out1);
        let constant185_out1 = self.constant185.val();
        let mul5_out1 = div2_out1.mul((constant185_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant186_out1 = self.constant186.val();
        let add7_out1 = mul5_out1.add((constant186_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear3_out1 = self.linear3.forward(add7_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(linear3_out1.clone());
        let mul6_out1 = linear3_out1.mul(sigmoid3_out1);
        let linear4_out1 = self.linear4.forward(mul6_out1);
        let add8_out1 = add5_out1.add(linear4_out1);
        (
            add8_out1,
            constant174_out1,
            constant175_out1,
            concat9_out1,
            squeeze1_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    constant191: burn::module::Param<Tensor<B, 1>>,
    constant192: burn::module::Param<Tensor<B, 1>>,
    linear5: Linear<B>,
    linear6: Linear<B>,
    constant197: burn::module::Param<Tensor<B, 1>>,
    constant198: burn::module::Param<Tensor<B, 1>>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    constant203: burn::module::Param<Tensor<B, 1>>,
    constant204: burn::module::Param<Tensor<B, 1>>,
    constant205: burn::module::Param<Tensor<B, 1>>,
    conv2d84: Conv2d<B>,
    conv2d85: Conv2d<B>,
    conv2d86: Conv2d<B>,
    linear9: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant191: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let constant192: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let linear5 = LinearConfig::new(120, 360).with_bias(true).init(device);
        let linear6 = LinearConfig::new(120, 120).with_bias(true).init(device);
        let constant197: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let constant198: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let linear7 = LinearConfig::new(120, 240).with_bias(true).init(device);
        let linear8 = LinearConfig::new(240, 120).with_bias(true).init(device);
        let constant203: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([0.0000009999999974752427f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant204: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let constant205: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let conv2d84 = Conv2dConfig::new([120, 2048], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d85 = Conv2dConfig::new([4096, 256], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d86 = Conv2dConfig::new([256, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear9 = LinearConfig::new(120, 18385).with_bias(true).init(device);
        Self {
            constant191,
            constant192,
            linear5,
            linear6,
            constant197,
            constant198,
            linear7,
            linear8,
            constant203,
            constant204,
            constant205,
            conv2d84,
            conv2d85,
            conv2d86,
            linear9,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add8_out1: Tensor<B, 3>,
        constant174_out1: Tensor<B, 1>,
        constant175_out1: Tensor<B, 1>,
        concat9_out1: [i64; 4],
        squeeze1_out1: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let reducemean5_out1 = { add8_out1.clone().mean_dim(2usize) };
        let sub3_out1 = add8_out1.clone().sub(reducemean5_out1);
        let pow3_out1 = sub3_out1
            .clone()
            .powf((constant174_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean6_out1 = { pow3_out1.mean_dim(2usize) };
        let add9_out1 =
            reducemean6_out1.add((constant175_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt3_out1 = add9_out1.sqrt();
        let div3_out1 = sub3_out1.div(sqrt3_out1);
        let constant191_out1 = self.constant191.val();
        let mul7_out1 = div3_out1.mul((constant191_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant192_out1 = self.constant192.val();
        let add10_out1 = mul7_out1.add((constant192_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear5_out1 = self.linear5.forward(add10_out1);
        let reshape4_out1 = linear5_out1.reshape([0, -1, 3, 8, 15]);
        let transpose5_out1 = reshape4_out1.permute([2, 0, 3, 1, 4]);
        let slice6_out1 = transpose5_out1.clone().slice(s![0..1, .., .., .., ..]);
        let slice7_out1 = transpose5_out1.clone().slice(s![1..2, .., .., .., ..]);
        let slice8_out1 = transpose5_out1.slice(s![2..3, .., .., .., ..]);
        let squeeze6_out1 = slice6_out1.squeeze_dims::<4>(&[0]);
        let squeeze7_out1 = slice7_out1.squeeze_dims::<4>(&[0]);
        let squeeze8_out1 = slice8_out1.squeeze_dims::<4>(&[0]);
        let (matmul9_out1,) = {
            let q = squeeze6_out1;
            let k = squeeze7_out1;
            let v = squeeze8_out1;
            let matmul9_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.25819888710975647f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul9_out1,)
        };
        let transpose7_out1 = matmul9_out1.permute([0, 2, 1, 3]);
        let reshape5_out1 = transpose7_out1.reshape([0, -1, 120]);
        let linear6_out1 = self.linear6.forward(reshape5_out1);
        let add11_out1 = add8_out1.add(linear6_out1);
        let reducemean7_out1 = { add11_out1.clone().mean_dim(2usize) };
        let sub4_out1 = add11_out1.clone().sub(reducemean7_out1);
        let pow4_out1 = sub4_out1
            .clone()
            .powf((constant174_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean8_out1 = { pow4_out1.mean_dim(2usize) };
        let add12_out1 = reducemean8_out1.add((constant175_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt4_out1 = add12_out1.sqrt();
        let div4_out1 = sub4_out1.div(sqrt4_out1);
        let constant197_out1 = self.constant197.val();
        let mul9_out1 = div4_out1.mul((constant197_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant198_out1 = self.constant198.val();
        let add13_out1 = mul9_out1.add((constant198_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear7_out1 = self.linear7.forward(add13_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(linear7_out1.clone());
        let mul10_out1 = linear7_out1.mul(sigmoid4_out1);
        let linear8_out1 = self.linear8.forward(mul10_out1);
        let add14_out1 = add11_out1.add(linear8_out1);
        let reducemean9_out1 = { add14_out1.clone().mean_dim(2usize) };
        let sub5_out1 = add14_out1.sub(reducemean9_out1);
        let pow5_out1 = sub5_out1
            .clone()
            .powf((constant174_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean10_out1 = { pow5_out1.mean_dim(2usize) };
        let constant203_out1 = self.constant203.val();
        let add15_out1 =
            reducemean10_out1.add((constant203_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt5_out1 = add15_out1.sqrt();
        let div5_out1 = sub5_out1.div(sqrt5_out1);
        let constant204_out1 = self.constant204.val();
        let mul11_out1 = div5_out1.mul((constant204_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant205_out1 = self.constant205.val();
        let add16_out1 = mul11_out1.add((constant205_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reshape6_out1 = add16_out1.reshape(concat9_out1);
        let transpose8_out1 = reshape6_out1.permute([0, 3, 1, 2]);
        let conv2d84_out1 = crate::model_arch::conv_fwd(&self.conv2d84, transpose8_out1);
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(conv2d84_out1.clone());
        let mul12_out1 = conv2d84_out1.mul(sigmoid5_out1);
        let concat10_out1 = burn::tensor::Tensor::cat([squeeze1_out1, mul12_out1].into(), 1);
        let conv2d85_out1 = crate::model_arch::conv_fwd(&self.conv2d85, concat10_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(conv2d85_out1.clone());
        let mul13_out1 = conv2d85_out1.mul(sigmoid6_out1);
        let conv2d86_out1 = crate::model_arch::conv_fwd(&self.conv2d86, mul13_out1);
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(conv2d86_out1.clone());
        let mul14_out1 = conv2d86_out1.mul(sigmoid7_out1);
        let squeeze9_out1 = mul14_out1.squeeze_dims::<3>(&[2]);
        let transpose9_out1 = squeeze9_out1.permute([0, 2, 1]);
        let linear9_out1 = self.linear9.forward(transpose9_out1);
        let softmax3_out1 = burn::tensor::activation::softmax(linear9_out1, 2);
        softmax3_out1
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    submodule1: Submodule1<B>,
    submodule2: Submodule2<B>,
    submodule3: Submodule3<B>,
    submodule4: Submodule4<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/Volumes/CodeBase/Projects/Lumen-Hub/target/release/build/lumen-convert-0036f7b8de134cd6/out/pp_ocrv5_server/recognition/recognition.bpk",
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
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let concat4_out1 = self.submodule1.forward(x);
        let concat7_out1 = self.submodule2.forward(concat4_out1);
        let (add8_out1, constant174_out1, constant175_out1, concat9_out1, squeeze1_out1) =
            self.submodule3.forward(concat7_out1);
        let softmax3_out1 = self.submodule4.forward(
            add8_out1,
            constant174_out1,
            constant175_out1,
            concat9_out1,
            squeeze1_out1,
        );
        softmax3_out1
    }
}
