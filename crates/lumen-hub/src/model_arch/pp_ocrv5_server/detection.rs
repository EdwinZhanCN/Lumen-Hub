// Generated from ONNX "onnx/pp-ocrv5-server/detection.prepared.onnx" by burn-onnx
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;
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
    conv2d37: Conv2d<B>,
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
            .with_stride([2, 2])
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
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
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
        let conv2d12 = Conv2dConfig::new([336, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d15 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
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
        let conv2d24 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d25 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([512, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([192, 192], [5, 5])
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
            conv2d37,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
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
        let conv2d6_out1 = crate::model_arch::conv_fwd(&self.conv2d6, relu5_out1.clone());
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = crate::model_arch::conv_fwd(&self.conv2d7, relu6_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = crate::model_arch::conv_fwd(&self.conv2d8, relu7_out1.clone());
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = crate::model_arch::conv_fwd(&self.conv2d9, relu8_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = crate::model_arch::conv_fwd(&self.conv2d10, relu9_out1.clone());
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = crate::model_arch::conv_fwd(&self.conv2d11, relu10_out1.clone());
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [
                relu5_out1,
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
        let conv2d12_out1 = crate::model_arch::conv_fwd(&self.conv2d12, concat2_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = crate::model_arch::conv_fwd(&self.conv2d13, relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = crate::model_arch::conv_fwd(&self.conv2d14, relu13_out1.clone());
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
        let conv2d24_out1 = crate::model_arch::conv_fwd(&self.conv2d24, relu21_out1.clone());
        let conv2d25_out1 = crate::model_arch::conv_fwd(&self.conv2d25, relu21_out1);
        let conv2d26_out1 = crate::model_arch::conv_fwd(&self.conv2d26, conv2d25_out1.clone());
        let conv2d27_out1 = crate::model_arch::conv_fwd(&self.conv2d27, conv2d26_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv2d27_out1);
        let conv2d28_out1 = crate::model_arch::conv_fwd(&self.conv2d28, relu22_out1.clone());
        let conv2d29_out1 = crate::model_arch::conv_fwd(&self.conv2d29, conv2d28_out1);
        let relu23_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = crate::model_arch::conv_fwd(&self.conv2d30, relu23_out1.clone());
        let conv2d31_out1 = crate::model_arch::conv_fwd(&self.conv2d31, conv2d30_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = crate::model_arch::conv_fwd(&self.conv2d32, relu24_out1.clone());
        let conv2d33_out1 = crate::model_arch::conv_fwd(&self.conv2d33, conv2d32_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let conv2d34_out1 = crate::model_arch::conv_fwd(&self.conv2d34, relu25_out1.clone());
        let conv2d35_out1 = crate::model_arch::conv_fwd(&self.conv2d35, conv2d34_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = crate::model_arch::conv_fwd(&self.conv2d36, relu26_out1.clone());
        let conv2d37_out1 = crate::model_arch::conv_fwd(&self.conv2d37, conv2d36_out1);
        let relu27_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [
                conv2d25_out1,
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
        (concat4_out1, conv2d24_out1, conv2d14_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
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
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d38 = Conv2dConfig::new([1664, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([1024, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([2176, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([1024, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([2176, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([1024, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d69 = Conv2dConfig::new([1024, 1024], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1024)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([1024, 384], [1, 1])
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
        let conv2d80 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        Self {
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
            conv2d80,
            conv2d81,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, concat4_out1: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let conv2d38_out1 = crate::model_arch::conv_fwd(&self.conv2d38, concat4_out1);
        let relu28_out1 = burn::tensor::activation::relu(conv2d38_out1);
        let conv2d39_out1 = crate::model_arch::conv_fwd(&self.conv2d39, relu28_out1);
        let relu29_out1 = burn::tensor::activation::relu(conv2d39_out1);
        let conv2d40_out1 = crate::model_arch::conv_fwd(&self.conv2d40, relu29_out1.clone());
        let conv2d41_out1 = crate::model_arch::conv_fwd(&self.conv2d41, conv2d40_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d41_out1);
        let conv2d42_out1 = crate::model_arch::conv_fwd(&self.conv2d42, relu30_out1.clone());
        let conv2d43_out1 = crate::model_arch::conv_fwd(&self.conv2d43, conv2d42_out1);
        let relu31_out1 = burn::tensor::activation::relu(conv2d43_out1);
        let conv2d44_out1 = crate::model_arch::conv_fwd(&self.conv2d44, relu31_out1.clone());
        let conv2d45_out1 = crate::model_arch::conv_fwd(&self.conv2d45, conv2d44_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let conv2d46_out1 = crate::model_arch::conv_fwd(&self.conv2d46, relu32_out1.clone());
        let conv2d47_out1 = crate::model_arch::conv_fwd(&self.conv2d47, conv2d46_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d47_out1);
        let conv2d48_out1 = crate::model_arch::conv_fwd(&self.conv2d48, relu33_out1.clone());
        let conv2d49_out1 = crate::model_arch::conv_fwd(&self.conv2d49, conv2d48_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv2d49_out1);
        let conv2d50_out1 = crate::model_arch::conv_fwd(&self.conv2d50, relu34_out1.clone());
        let conv2d51_out1 = crate::model_arch::conv_fwd(&self.conv2d51, conv2d50_out1);
        let relu35_out1 = burn::tensor::activation::relu(conv2d51_out1);
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
        let conv2d52_out1 = crate::model_arch::conv_fwd(&self.conv2d52, concat5_out1);
        let relu36_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let conv2d53_out1 = crate::model_arch::conv_fwd(&self.conv2d53, relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(conv2d53_out1);
        let add1_out1 = relu37_out1.add(relu29_out1);
        let conv2d54_out1 = crate::model_arch::conv_fwd(&self.conv2d54, add1_out1.clone());
        let conv2d55_out1 = crate::model_arch::conv_fwd(&self.conv2d55, conv2d54_out1);
        let relu38_out1 = burn::tensor::activation::relu(conv2d55_out1);
        let conv2d56_out1 = crate::model_arch::conv_fwd(&self.conv2d56, relu38_out1.clone());
        let conv2d57_out1 = crate::model_arch::conv_fwd(&self.conv2d57, conv2d56_out1);
        let relu39_out1 = burn::tensor::activation::relu(conv2d57_out1);
        let conv2d58_out1 = crate::model_arch::conv_fwd(&self.conv2d58, relu39_out1.clone());
        let conv2d59_out1 = crate::model_arch::conv_fwd(&self.conv2d59, conv2d58_out1);
        let relu40_out1 = burn::tensor::activation::relu(conv2d59_out1);
        let conv2d60_out1 = crate::model_arch::conv_fwd(&self.conv2d60, relu40_out1.clone());
        let conv2d61_out1 = crate::model_arch::conv_fwd(&self.conv2d61, conv2d60_out1);
        let relu41_out1 = burn::tensor::activation::relu(conv2d61_out1);
        let conv2d62_out1 = crate::model_arch::conv_fwd(&self.conv2d62, relu41_out1.clone());
        let conv2d63_out1 = crate::model_arch::conv_fwd(&self.conv2d63, conv2d62_out1);
        let relu42_out1 = burn::tensor::activation::relu(conv2d63_out1);
        let conv2d64_out1 = crate::model_arch::conv_fwd(&self.conv2d64, relu42_out1.clone());
        let conv2d65_out1 = crate::model_arch::conv_fwd(&self.conv2d65, conv2d64_out1);
        let relu43_out1 = burn::tensor::activation::relu(conv2d65_out1);
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
        let conv2d66_out1 = crate::model_arch::conv_fwd(&self.conv2d66, concat6_out1);
        let relu44_out1 = burn::tensor::activation::relu(conv2d66_out1);
        let conv2d67_out1 = crate::model_arch::conv_fwd(&self.conv2d67, relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(conv2d67_out1);
        let add2_out1 = relu45_out1.add(add1_out1);
        let conv2d68_out1 = crate::model_arch::conv_fwd(&self.conv2d68, add2_out1.clone());
        let conv2d69_out1 = crate::model_arch::conv_fwd(&self.conv2d69, add2_out1);
        let conv2d70_out1 = crate::model_arch::conv_fwd(&self.conv2d70, conv2d69_out1.clone());
        let conv2d71_out1 = crate::model_arch::conv_fwd(&self.conv2d71, conv2d70_out1);
        let relu46_out1 = burn::tensor::activation::relu(conv2d71_out1);
        let conv2d72_out1 = crate::model_arch::conv_fwd(&self.conv2d72, relu46_out1.clone());
        let conv2d73_out1 = crate::model_arch::conv_fwd(&self.conv2d73, conv2d72_out1);
        let relu47_out1 = burn::tensor::activation::relu(conv2d73_out1);
        let conv2d74_out1 = crate::model_arch::conv_fwd(&self.conv2d74, relu47_out1.clone());
        let conv2d75_out1 = crate::model_arch::conv_fwd(&self.conv2d75, conv2d74_out1);
        let relu48_out1 = burn::tensor::activation::relu(conv2d75_out1);
        let conv2d76_out1 = crate::model_arch::conv_fwd(&self.conv2d76, relu48_out1.clone());
        let conv2d77_out1 = crate::model_arch::conv_fwd(&self.conv2d77, conv2d76_out1);
        let relu49_out1 = burn::tensor::activation::relu(conv2d77_out1);
        let conv2d78_out1 = crate::model_arch::conv_fwd(&self.conv2d78, relu49_out1.clone());
        let conv2d79_out1 = crate::model_arch::conv_fwd(&self.conv2d79, conv2d78_out1);
        let relu50_out1 = burn::tensor::activation::relu(conv2d79_out1);
        let conv2d80_out1 = crate::model_arch::conv_fwd(&self.conv2d80, relu50_out1.clone());
        let conv2d81_out1 = crate::model_arch::conv_fwd(&self.conv2d81, conv2d80_out1);
        let relu51_out1 = burn::tensor::activation::relu(conv2d81_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [
                conv2d69_out1,
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
        (concat7_out1, conv2d68_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    conv2d82: Conv2d<B>,
    conv2d83: Conv2d<B>,
    conv2d84: Conv2d<B>,
    resize1: burn::nn::interpolate::Interpolate2d,
    conv2d85: Conv2d<B>,
    resize2: burn::nn::interpolate::Interpolate2d,
    conv2d86: Conv2d<B>,
    resize3: burn::nn::interpolate::Interpolate2d,
    conv2d87: Conv2d<B>,
    conv2d88: Conv2d<B>,
    conv2d89: Conv2d<B>,
    conv2d90: Conv2d<B>,
    conv2d91: Conv2d<B>,
    conv2d92: Conv2d<B>,
    conv2d93: Conv2d<B>,
    conv2d94: Conv2d<B>,
    conv2d95: Conv2d<B>,
    conv2d96: Conv2d<B>,
    conv2d97: Conv2d<B>,
    conv2d98: Conv2d<B>,
    conv2d99: Conv2d<B>,
    conv2d100: Conv2d<B>,
    conv2d101: Conv2d<B>,
    conv2d102: Conv2d<B>,
    conv2d103: Conv2d<B>,
    conv2d104: Conv2d<B>,
    conv2d105: Conv2d<B>,
    conv2d106: Conv2d<B>,
    conv2d107: Conv2d<B>,
    conv2d108: Conv2d<B>,
    conv2d109: Conv2d<B>,
    conv2d110: Conv2d<B>,
    conv2d111: Conv2d<B>,
    conv2d112: Conv2d<B>,
    conv2d113: Conv2d<B>,
    conv2d114: Conv2d<B>,
    conv2d115: Conv2d<B>,
    conv2d116: Conv2d<B>,
    conv2d117: Conv2d<B>,
    conv2d118: Conv2d<B>,
    conv2d119: Conv2d<B>,
    conv2d120: Conv2d<B>,
    conv2d121: Conv2d<B>,
    conv2d122: Conv2d<B>,
    conv2d123: Conv2d<B>,
    conv2d124: Conv2d<B>,
    conv2d125: Conv2d<B>,
    conv2d126: Conv2d<B>,
    conv2d127: Conv2d<B>,
    conv2d128: Conv2d<B>,
    conv2d129: Conv2d<B>,
    conv2d130: Conv2d<B>,
    conv2d131: Conv2d<B>,
    conv2d132: Conv2d<B>,
    conv2d133: Conv2d<B>,
    conv2d134: Conv2d<B>,
    conv2d135: Conv2d<B>,
    conv2d136: Conv2d<B>,
    conv2d137: Conv2d<B>,
    conv2d138: Conv2d<B>,
    resize4: burn::nn::interpolate::Interpolate2d,
    conv2d139: Conv2d<B>,
    resize5: burn::nn::interpolate::Interpolate2d,
    resize6: burn::nn::interpolate::Interpolate2d,
    conv2d140: Conv2d<B>,
    convtranspose2d1: ConvTranspose2d<B>,
    constant260: burn::module::Param<Tensor<B, 4>>,
    batchnormalization1: BatchNorm<B>,
    resize7: burn::nn::interpolate::Interpolate2d,
    convtranspose2d2: ConvTranspose2d<B>,
    constant266: burn::module::Param<Tensor<B, 4>>,
    conv2d141: Conv2d<B>,
    conv2d142: Conv2d<B>,
    constant270: burn::module::Param<Tensor<B, 4>>,
    constant271: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d82 = Conv2dConfig::new([3328, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([1024, 2048], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d84 = Conv2dConfig::new([2048, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let resize1 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d85 = Conv2dConfig::new([256, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let resize2 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d86 = Conv2dConfig::new([256, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let resize3 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d87 = Conv2dConfig::new([256, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d88 = Conv2dConfig::new([256, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d89 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d90 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d91 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d93 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d94 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d95 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d96 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d97 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d98 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d99 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d100 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d101 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d102 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d103 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d104 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d105 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d106 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d107 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d108 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d109 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d110 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d111 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d112 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d113 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d114 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d115 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d116 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d117 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d118 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d119 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d120 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d121 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d122 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d123 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d124 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d125 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d126 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d127 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d128 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d129 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d130 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d131 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d132 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d133 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d134 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d135 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d136 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d137 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d138 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize4 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d139 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize5 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([4.0, 4.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize6 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([8.0, 8.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d140 = Conv2dConfig::new([256, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d1 = ConvTranspose2dConfig::new([64, 64], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant260: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 64, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 64, 1, 1].into(),
        );
        let batchnormalization1 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let resize7 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let convtranspose2d2 = ConvTranspose2dConfig::new([64, 1], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant266: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 1, 1].into(),
        );
        let conv2d141 = Conv2dConfig::new([65, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d142 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant270: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 1, 1].into(),
        );
        let constant271: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            conv2d82,
            conv2d83,
            conv2d84,
            resize1,
            conv2d85,
            resize2,
            conv2d86,
            resize3,
            conv2d87,
            conv2d88,
            conv2d89,
            conv2d90,
            conv2d91,
            conv2d92,
            conv2d93,
            conv2d94,
            conv2d95,
            conv2d96,
            conv2d97,
            conv2d98,
            conv2d99,
            conv2d100,
            conv2d101,
            conv2d102,
            conv2d103,
            conv2d104,
            conv2d105,
            conv2d106,
            conv2d107,
            conv2d108,
            conv2d109,
            conv2d110,
            conv2d111,
            conv2d112,
            conv2d113,
            conv2d114,
            conv2d115,
            conv2d116,
            conv2d117,
            conv2d118,
            conv2d119,
            conv2d120,
            conv2d121,
            conv2d122,
            conv2d123,
            conv2d124,
            conv2d125,
            conv2d126,
            conv2d127,
            conv2d128,
            conv2d129,
            conv2d130,
            conv2d131,
            conv2d132,
            conv2d133,
            conv2d134,
            conv2d135,
            conv2d136,
            conv2d137,
            conv2d138,
            resize4,
            conv2d139,
            resize5,
            resize6,
            conv2d140,
            convtranspose2d1,
            constant260,
            batchnormalization1,
            resize7,
            convtranspose2d2,
            constant266,
            conv2d141,
            conv2d142,
            constant270,
            constant271,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        concat7_out1: Tensor<B, 4>,
        conv2d68_out1: Tensor<B, 4>,
        conv2d24_out1: Tensor<B, 4>,
        conv2d14_out1: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let conv2d82_out1 = crate::model_arch::conv_fwd(&self.conv2d82, concat7_out1);
        let relu52_out1 = burn::tensor::activation::relu(conv2d82_out1);
        let conv2d83_out1 = crate::model_arch::conv_fwd(&self.conv2d83, relu52_out1);
        let relu53_out1 = burn::tensor::activation::relu(conv2d83_out1);
        let conv2d84_out1 = crate::model_arch::conv_fwd(&self.conv2d84, relu53_out1);
        let resize1_out1 = self.resize1.forward(conv2d84_out1.clone());
        let conv2d85_out1 = crate::model_arch::conv_fwd(&self.conv2d85, conv2d84_out1);
        let add3_out1 = conv2d68_out1.add(resize1_out1);
        let resize2_out1 = self.resize2.forward(add3_out1.clone());
        let conv2d86_out1 = crate::model_arch::conv_fwd(&self.conv2d86, add3_out1);
        let add4_out1 = conv2d24_out1.add(resize2_out1);
        let resize3_out1 = self.resize3.forward(add4_out1.clone());
        let conv2d87_out1 = crate::model_arch::conv_fwd(&self.conv2d87, add4_out1);
        let add5_out1 = conv2d14_out1.add(resize3_out1);
        let conv2d88_out1 = crate::model_arch::conv_fwd(&self.conv2d88, add5_out1);
        let conv2d89_out1 = crate::model_arch::conv_fwd(&self.conv2d89, conv2d88_out1.clone());
        let conv2d90_out1 = crate::model_arch::conv_fwd(&self.conv2d90, conv2d88_out1);
        let add6_out1 = conv2d87_out1.add(conv2d89_out1);
        let conv2d91_out1 = crate::model_arch::conv_fwd(&self.conv2d91, conv2d90_out1.clone());
        let conv2d92_out1 = crate::model_arch::conv_fwd(&self.conv2d92, add6_out1.clone());
        let conv2d93_out1 = crate::model_arch::conv_fwd(&self.conv2d93, add6_out1);
        let conv2d94_out1 = crate::model_arch::conv_fwd(&self.conv2d94, conv2d91_out1.clone());
        let conv2d95_out1 = crate::model_arch::conv_fwd(&self.conv2d95, conv2d91_out1.clone());
        let conv2d96_out1 = crate::model_arch::conv_fwd(&self.conv2d96, conv2d91_out1);
        let add7_out1 = conv2d86_out1.add(conv2d92_out1);
        let conv2d97_out1 = crate::model_arch::conv_fwd(&self.conv2d97, conv2d93_out1.clone());
        let add8_out1 = conv2d94_out1.add(conv2d95_out1);
        let conv2d98_out1 = crate::model_arch::conv_fwd(&self.conv2d98, add7_out1.clone());
        let conv2d99_out1 = crate::model_arch::conv_fwd(&self.conv2d99, add7_out1);
        let conv2d100_out1 = crate::model_arch::conv_fwd(&self.conv2d100, conv2d97_out1.clone());
        let conv2d101_out1 = crate::model_arch::conv_fwd(&self.conv2d101, conv2d97_out1.clone());
        let conv2d102_out1 = crate::model_arch::conv_fwd(&self.conv2d102, conv2d97_out1);
        let add9_out1 = add8_out1.add(conv2d96_out1);
        let add10_out1 = conv2d85_out1.add(conv2d98_out1);
        let conv2d103_out1 = crate::model_arch::conv_fwd(&self.conv2d103, conv2d99_out1.clone());
        let add11_out1 = conv2d100_out1.add(conv2d101_out1);
        let conv2d104_out1 = crate::model_arch::conv_fwd(&self.conv2d104, add9_out1.clone());
        let conv2d105_out1 = crate::model_arch::conv_fwd(&self.conv2d105, add9_out1.clone());
        let conv2d106_out1 = crate::model_arch::conv_fwd(&self.conv2d106, add9_out1);
        let conv2d107_out1 = crate::model_arch::conv_fwd(&self.conv2d107, add10_out1);
        let conv2d108_out1 = crate::model_arch::conv_fwd(&self.conv2d108, conv2d103_out1.clone());
        let conv2d109_out1 = crate::model_arch::conv_fwd(&self.conv2d109, conv2d103_out1.clone());
        let conv2d110_out1 = crate::model_arch::conv_fwd(&self.conv2d110, conv2d103_out1);
        let add12_out1 = add11_out1.add(conv2d102_out1);
        let add13_out1 = conv2d104_out1.add(conv2d105_out1);
        let conv2d111_out1 = crate::model_arch::conv_fwd(&self.conv2d111, conv2d107_out1.clone());
        let add14_out1 = conv2d108_out1.add(conv2d109_out1);
        let conv2d112_out1 = crate::model_arch::conv_fwd(&self.conv2d112, add12_out1.clone());
        let conv2d113_out1 = crate::model_arch::conv_fwd(&self.conv2d113, add12_out1.clone());
        let conv2d114_out1 = crate::model_arch::conv_fwd(&self.conv2d114, add12_out1);
        let add15_out1 = add13_out1.add(conv2d106_out1);
        let conv2d115_out1 = crate::model_arch::conv_fwd(&self.conv2d115, conv2d111_out1.clone());
        let conv2d116_out1 = crate::model_arch::conv_fwd(&self.conv2d116, conv2d111_out1.clone());
        let conv2d117_out1 = crate::model_arch::conv_fwd(&self.conv2d117, conv2d111_out1);
        let add16_out1 = add14_out1.add(conv2d110_out1);
        let add17_out1 = conv2d112_out1.add(conv2d113_out1);
        let conv2d118_out1 = crate::model_arch::conv_fwd(&self.conv2d118, add15_out1.clone());
        let conv2d119_out1 = crate::model_arch::conv_fwd(&self.conv2d119, add15_out1.clone());
        let conv2d120_out1 = crate::model_arch::conv_fwd(&self.conv2d120, add15_out1);
        let add18_out1 = conv2d115_out1.add(conv2d116_out1);
        let conv2d121_out1 = crate::model_arch::conv_fwd(&self.conv2d121, add16_out1.clone());
        let conv2d122_out1 = crate::model_arch::conv_fwd(&self.conv2d122, add16_out1.clone());
        let conv2d123_out1 = crate::model_arch::conv_fwd(&self.conv2d123, add16_out1);
        let add19_out1 = add17_out1.add(conv2d114_out1);
        let add20_out1 = conv2d118_out1.add(conv2d119_out1);
        let add21_out1 = add18_out1.add(conv2d117_out1);
        let add22_out1 = conv2d121_out1.add(conv2d122_out1);
        let conv2d124_out1 = crate::model_arch::conv_fwd(&self.conv2d124, add19_out1.clone());
        let conv2d125_out1 = crate::model_arch::conv_fwd(&self.conv2d125, add19_out1.clone());
        let conv2d126_out1 = crate::model_arch::conv_fwd(&self.conv2d126, add19_out1);
        let add23_out1 = add20_out1.add(conv2d120_out1);
        let conv2d127_out1 = crate::model_arch::conv_fwd(&self.conv2d127, add21_out1.clone());
        let conv2d128_out1 = crate::model_arch::conv_fwd(&self.conv2d128, add21_out1.clone());
        let conv2d129_out1 = crate::model_arch::conv_fwd(&self.conv2d129, add21_out1);
        let add24_out1 = add22_out1.add(conv2d123_out1);
        let add25_out1 = conv2d124_out1.add(conv2d125_out1);
        let conv2d130_out1 = crate::model_arch::conv_fwd(&self.conv2d130, add23_out1);
        let add26_out1 = conv2d127_out1.add(conv2d128_out1);
        let conv2d131_out1 = crate::model_arch::conv_fwd(&self.conv2d131, add24_out1.clone());
        let conv2d132_out1 = crate::model_arch::conv_fwd(&self.conv2d132, add24_out1.clone());
        let conv2d133_out1 = crate::model_arch::conv_fwd(&self.conv2d133, add24_out1);
        let add27_out1 = add25_out1.add(conv2d126_out1);
        let relu54_out1 = burn::tensor::activation::relu(conv2d130_out1);
        let add28_out1 = add26_out1.add(conv2d129_out1);
        let add29_out1 = conv2d131_out1.add(conv2d132_out1);
        let add30_out1 = conv2d90_out1.add(relu54_out1);
        let conv2d134_out1 = crate::model_arch::conv_fwd(&self.conv2d134, add27_out1);
        let conv2d135_out1 = crate::model_arch::conv_fwd(&self.conv2d135, add28_out1.clone());
        let conv2d136_out1 = crate::model_arch::conv_fwd(&self.conv2d136, add28_out1.clone());
        let conv2d137_out1 = crate::model_arch::conv_fwd(&self.conv2d137, add28_out1);
        let add31_out1 = add29_out1.add(conv2d133_out1);
        let relu55_out1 = burn::tensor::activation::relu(conv2d134_out1);
        let add32_out1 = conv2d135_out1.add(conv2d136_out1);
        let add33_out1 = conv2d93_out1.add(relu55_out1);
        let conv2d138_out1 = crate::model_arch::conv_fwd(&self.conv2d138, add31_out1);
        let add34_out1 = add32_out1.add(conv2d137_out1);
        let relu56_out1 = burn::tensor::activation::relu(conv2d138_out1);
        let resize4_out1 = self.resize4.forward(add33_out1);
        let add35_out1 = conv2d99_out1.add(relu56_out1);
        let conv2d139_out1 = crate::model_arch::conv_fwd(&self.conv2d139, add34_out1);
        let relu57_out1 = burn::tensor::activation::relu(conv2d139_out1);
        let resize5_out1 = self.resize5.forward(add35_out1);
        let add36_out1 = conv2d107_out1.add(relu57_out1);
        let resize6_out1 = self.resize6.forward(add36_out1);
        let concat8_out1 = burn::tensor::Tensor::cat(
            [resize6_out1, resize5_out1, resize4_out1, add30_out1].into(),
            1,
        );
        let conv2d140_out1 = crate::model_arch::conv_fwd(&self.conv2d140, concat8_out1);
        let relu58_out1 = burn::tensor::activation::relu(conv2d140_out1);
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(relu58_out1);
        let constant260_out1 = self.constant260.val();
        let add37_out1 = convtranspose2d1_out1.add(constant260_out1);
        let batchnormalization1_out1 = self.batchnormalization1.forward(add37_out1);
        let relu59_out1 = burn::tensor::activation::relu(batchnormalization1_out1);
        let resize7_out1 = self.resize7.forward(relu59_out1.clone());
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(relu59_out1);
        let constant266_out1 = self.constant266.val();
        let add38_out1 = convtranspose2d2_out1.add(constant266_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(add38_out1);
        let concat9_out1 =
            burn::tensor::Tensor::cat([sigmoid1_out1.clone(), resize7_out1].into(), 1);
        let conv2d141_out1 = crate::model_arch::conv_fwd(&self.conv2d141, concat9_out1);
        let relu60_out1 = burn::tensor::activation::relu(conv2d141_out1);
        let conv2d142_out1 = crate::model_arch::conv_fwd(&self.conv2d142, relu60_out1);
        let constant270_out1 = self.constant270.val();
        let add39_out1 = conv2d142_out1.add(constant270_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(add39_out1);
        let add40_out1 = sigmoid1_out1.add(sigmoid2_out1);
        let constant271_out1 = self.constant271.val();
        let mul1_out1 =
            add40_out1.mul((constant271_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        mul1_out1
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    submodule1: Submodule1<B>,
    submodule2: Submodule2<B>,
    submodule3: Submodule3<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/Volumes/CodeBase/Projects/Lumen-Hub/target/release/build/lumen-convert-0036f7b8de134cd6/out/pp_ocrv5_server/detection/detection.bpk",
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
        Self {
            submodule1,
            submodule2,
            submodule3,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (concat4_out1, conv2d24_out1, conv2d14_out1) = self.submodule1.forward(x);
        let (concat7_out1, conv2d68_out1) = self.submodule2.forward(concat4_out1);
        let mul1_out1 =
            self.submodule3
                .forward(concat7_out1, conv2d68_out1, conv2d24_out1, conv2d14_out1);
        mul1_out1
    }
}
