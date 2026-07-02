// Generated from ONNX "src/antelopev2/recognition.fp32.onnx" by burn-onnx
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::nn::PRelu;
use burn::nn::PReluConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv2d1: Conv2d<B>,
    prelu1: PRelu<B>,
    batchnormalization1: BatchNorm<B>,
    conv2d2: Conv2d<B>,
    prelu2: PRelu<B>,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    batchnormalization2: BatchNorm<B>,
    conv2d5: Conv2d<B>,
    prelu3: PRelu<B>,
    conv2d6: Conv2d<B>,
    batchnormalization3: BatchNorm<B>,
    conv2d7: Conv2d<B>,
    prelu4: PRelu<B>,
    conv2d8: Conv2d<B>,
    batchnormalization4: BatchNorm<B>,
    conv2d9: Conv2d<B>,
    prelu5: PRelu<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    batchnormalization5: BatchNorm<B>,
    conv2d12: Conv2d<B>,
    prelu6: PRelu<B>,
    conv2d13: Conv2d<B>,
    batchnormalization6: BatchNorm<B>,
    conv2d14: Conv2d<B>,
    prelu7: PRelu<B>,
    conv2d15: Conv2d<B>,
    batchnormalization7: BatchNorm<B>,
    conv2d16: Conv2d<B>,
    prelu8: PRelu<B>,
    conv2d17: Conv2d<B>,
    batchnormalization8: BatchNorm<B>,
    conv2d18: Conv2d<B>,
    prelu9: PRelu<B>,
    conv2d19: Conv2d<B>,
    batchnormalization9: BatchNorm<B>,
    conv2d20: Conv2d<B>,
    prelu10: PRelu<B>,
    conv2d21: Conv2d<B>,
    batchnormalization10: BatchNorm<B>,
    conv2d22: Conv2d<B>,
    prelu11: PRelu<B>,
    conv2d23: Conv2d<B>,
    batchnormalization11: BatchNorm<B>,
    conv2d24: Conv2d<B>,
    prelu12: PRelu<B>,
    conv2d25: Conv2d<B>,
    batchnormalization12: BatchNorm<B>,
    conv2d26: Conv2d<B>,
    prelu13: PRelu<B>,
    conv2d27: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu1 = PReluConfig::new().with_num_parameters(64).init(device);
        let batchnormalization1 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu2 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d3 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d5 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu3 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d6 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d7 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu4 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d8 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d9 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu5 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d10 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d12 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu6 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d13 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d14 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu7 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d15 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization7 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d16 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu8 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d17 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization8 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d18 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu9 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d19 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization9 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d20 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu10 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d21 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization10 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d22 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu11 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d23 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization11 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d24 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu12 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d25 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization12 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d26 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu13 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d27 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            prelu1,
            batchnormalization1,
            conv2d2,
            prelu2,
            conv2d3,
            conv2d4,
            batchnormalization2,
            conv2d5,
            prelu3,
            conv2d6,
            batchnormalization3,
            conv2d7,
            prelu4,
            conv2d8,
            batchnormalization4,
            conv2d9,
            prelu5,
            conv2d10,
            conv2d11,
            batchnormalization5,
            conv2d12,
            prelu6,
            conv2d13,
            batchnormalization6,
            conv2d14,
            prelu7,
            conv2d15,
            batchnormalization7,
            conv2d16,
            prelu8,
            conv2d17,
            batchnormalization8,
            conv2d18,
            prelu9,
            conv2d19,
            batchnormalization9,
            conv2d20,
            prelu10,
            conv2d21,
            batchnormalization10,
            conv2d22,
            prelu11,
            conv2d23,
            batchnormalization11,
            conv2d24,
            prelu12,
            conv2d25,
            batchnormalization12,
            conv2d26,
            prelu13,
            conv2d27,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input_1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = crate::model_arch::conv_fwd(&self.conv2d1, input_1);
        let prelu1_out1 = self.prelu1.forward(conv2d1_out1);
        let batchnormalization1_out1 = self.batchnormalization1.forward(prelu1_out1.clone());
        let conv2d2_out1 = crate::model_arch::conv_fwd(&self.conv2d2, batchnormalization1_out1);
        let prelu2_out1 = self.prelu2.forward(conv2d2_out1);
        let conv2d3_out1 = crate::model_arch::conv_fwd(&self.conv2d3, prelu2_out1);
        let conv2d4_out1 = crate::model_arch::conv_fwd(&self.conv2d4, prelu1_out1);
        let add1_out1 = conv2d3_out1.add(conv2d4_out1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(add1_out1.clone());
        let conv2d5_out1 = crate::model_arch::conv_fwd(&self.conv2d5, batchnormalization2_out1);
        let prelu3_out1 = self.prelu3.forward(conv2d5_out1);
        let conv2d6_out1 = crate::model_arch::conv_fwd(&self.conv2d6, prelu3_out1);
        let add2_out1 = conv2d6_out1.add(add1_out1);
        let batchnormalization3_out1 = self.batchnormalization3.forward(add2_out1.clone());
        let conv2d7_out1 = crate::model_arch::conv_fwd(&self.conv2d7, batchnormalization3_out1);
        let prelu4_out1 = self.prelu4.forward(conv2d7_out1);
        let conv2d8_out1 = crate::model_arch::conv_fwd(&self.conv2d8, prelu4_out1);
        let add3_out1 = conv2d8_out1.add(add2_out1);
        let batchnormalization4_out1 = self.batchnormalization4.forward(add3_out1.clone());
        let conv2d9_out1 = crate::model_arch::conv_fwd(&self.conv2d9, batchnormalization4_out1);
        let prelu5_out1 = self.prelu5.forward(conv2d9_out1);
        let conv2d10_out1 = crate::model_arch::conv_fwd(&self.conv2d10, prelu5_out1);
        let conv2d11_out1 = crate::model_arch::conv_fwd(&self.conv2d11, add3_out1);
        let add4_out1 = conv2d10_out1.add(conv2d11_out1);
        let batchnormalization5_out1 = self.batchnormalization5.forward(add4_out1.clone());
        let conv2d12_out1 = crate::model_arch::conv_fwd(&self.conv2d12, batchnormalization5_out1);
        let prelu6_out1 = self.prelu6.forward(conv2d12_out1);
        let conv2d13_out1 = crate::model_arch::conv_fwd(&self.conv2d13, prelu6_out1);
        let add5_out1 = conv2d13_out1.add(add4_out1);
        let batchnormalization6_out1 = self.batchnormalization6.forward(add5_out1.clone());
        let conv2d14_out1 = crate::model_arch::conv_fwd(&self.conv2d14, batchnormalization6_out1);
        let prelu7_out1 = self.prelu7.forward(conv2d14_out1);
        let conv2d15_out1 = crate::model_arch::conv_fwd(&self.conv2d15, prelu7_out1);
        let add6_out1 = conv2d15_out1.add(add5_out1);
        let batchnormalization7_out1 = self.batchnormalization7.forward(add6_out1.clone());
        let conv2d16_out1 = crate::model_arch::conv_fwd(&self.conv2d16, batchnormalization7_out1);
        let prelu8_out1 = self.prelu8.forward(conv2d16_out1);
        let conv2d17_out1 = crate::model_arch::conv_fwd(&self.conv2d17, prelu8_out1);
        let add7_out1 = conv2d17_out1.add(add6_out1);
        let batchnormalization8_out1 = self.batchnormalization8.forward(add7_out1.clone());
        let conv2d18_out1 = crate::model_arch::conv_fwd(&self.conv2d18, batchnormalization8_out1);
        let prelu9_out1 = self.prelu9.forward(conv2d18_out1);
        let conv2d19_out1 = crate::model_arch::conv_fwd(&self.conv2d19, prelu9_out1);
        let add8_out1 = conv2d19_out1.add(add7_out1);
        let batchnormalization9_out1 = self.batchnormalization9.forward(add8_out1.clone());
        let conv2d20_out1 = crate::model_arch::conv_fwd(&self.conv2d20, batchnormalization9_out1);
        let prelu10_out1 = self.prelu10.forward(conv2d20_out1);
        let conv2d21_out1 = crate::model_arch::conv_fwd(&self.conv2d21, prelu10_out1);
        let add9_out1 = conv2d21_out1.add(add8_out1);
        let batchnormalization10_out1 = self.batchnormalization10.forward(add9_out1.clone());
        let conv2d22_out1 = crate::model_arch::conv_fwd(&self.conv2d22, batchnormalization10_out1);
        let prelu11_out1 = self.prelu11.forward(conv2d22_out1);
        let conv2d23_out1 = crate::model_arch::conv_fwd(&self.conv2d23, prelu11_out1);
        let add10_out1 = conv2d23_out1.add(add9_out1);
        let batchnormalization11_out1 = self.batchnormalization11.forward(add10_out1.clone());
        let conv2d24_out1 = crate::model_arch::conv_fwd(&self.conv2d24, batchnormalization11_out1);
        let prelu12_out1 = self.prelu12.forward(conv2d24_out1);
        let conv2d25_out1 = crate::model_arch::conv_fwd(&self.conv2d25, prelu12_out1);
        let add11_out1 = conv2d25_out1.add(add10_out1);
        let batchnormalization12_out1 = self.batchnormalization12.forward(add11_out1.clone());
        let conv2d26_out1 = crate::model_arch::conv_fwd(&self.conv2d26, batchnormalization12_out1);
        let prelu13_out1 = self.prelu13.forward(conv2d26_out1);
        let conv2d27_out1 = crate::model_arch::conv_fwd(&self.conv2d27, prelu13_out1);
        let add12_out1 = conv2d27_out1.add(add11_out1);
        add12_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    batchnormalization13: BatchNorm<B>,
    conv2d28: Conv2d<B>,
    prelu14: PRelu<B>,
    conv2d29: Conv2d<B>,
    batchnormalization14: BatchNorm<B>,
    conv2d30: Conv2d<B>,
    prelu15: PRelu<B>,
    conv2d31: Conv2d<B>,
    batchnormalization15: BatchNorm<B>,
    conv2d32: Conv2d<B>,
    prelu16: PRelu<B>,
    conv2d33: Conv2d<B>,
    batchnormalization16: BatchNorm<B>,
    conv2d34: Conv2d<B>,
    prelu17: PRelu<B>,
    conv2d35: Conv2d<B>,
    batchnormalization17: BatchNorm<B>,
    conv2d36: Conv2d<B>,
    prelu18: PRelu<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    batchnormalization18: BatchNorm<B>,
    conv2d39: Conv2d<B>,
    prelu19: PRelu<B>,
    conv2d40: Conv2d<B>,
    batchnormalization19: BatchNorm<B>,
    conv2d41: Conv2d<B>,
    prelu20: PRelu<B>,
    conv2d42: Conv2d<B>,
    batchnormalization20: BatchNorm<B>,
    conv2d43: Conv2d<B>,
    prelu21: PRelu<B>,
    conv2d44: Conv2d<B>,
    batchnormalization21: BatchNorm<B>,
    conv2d45: Conv2d<B>,
    prelu22: PRelu<B>,
    conv2d46: Conv2d<B>,
    batchnormalization22: BatchNorm<B>,
    conv2d47: Conv2d<B>,
    prelu23: PRelu<B>,
    conv2d48: Conv2d<B>,
    batchnormalization23: BatchNorm<B>,
    conv2d49: Conv2d<B>,
    prelu24: PRelu<B>,
    conv2d50: Conv2d<B>,
    batchnormalization24: BatchNorm<B>,
    conv2d51: Conv2d<B>,
    prelu25: PRelu<B>,
    conv2d52: Conv2d<B>,
    batchnormalization25: BatchNorm<B>,
    conv2d53: Conv2d<B>,
    prelu26: PRelu<B>,
    conv2d54: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let batchnormalization13 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d28 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu14 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d29 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization14 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu15 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d31 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization15 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d32 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu16 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d33 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization16 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d34 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu17 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d35 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization17 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d36 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu18 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d37 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization18 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d39 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu19 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d40 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization19 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d41 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu20 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d42 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization20 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d43 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu21 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d44 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization21 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d45 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu22 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d46 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization22 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d47 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu23 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d48 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization23 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d49 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu24 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d50 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization24 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d51 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu25 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d52 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization25 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d53 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu26 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d54 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            batchnormalization13,
            conv2d28,
            prelu14,
            conv2d29,
            batchnormalization14,
            conv2d30,
            prelu15,
            conv2d31,
            batchnormalization15,
            conv2d32,
            prelu16,
            conv2d33,
            batchnormalization16,
            conv2d34,
            prelu17,
            conv2d35,
            batchnormalization17,
            conv2d36,
            prelu18,
            conv2d37,
            conv2d38,
            batchnormalization18,
            conv2d39,
            prelu19,
            conv2d40,
            batchnormalization19,
            conv2d41,
            prelu20,
            conv2d42,
            batchnormalization20,
            conv2d43,
            prelu21,
            conv2d44,
            batchnormalization21,
            conv2d45,
            prelu22,
            conv2d46,
            batchnormalization22,
            conv2d47,
            prelu23,
            conv2d48,
            batchnormalization23,
            conv2d49,
            prelu24,
            conv2d50,
            batchnormalization24,
            conv2d51,
            prelu25,
            conv2d52,
            batchnormalization25,
            conv2d53,
            prelu26,
            conv2d54,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add12_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let batchnormalization13_out1 = self.batchnormalization13.forward(add12_out1.clone());
        let conv2d28_out1 = crate::model_arch::conv_fwd(&self.conv2d28, batchnormalization13_out1);
        let prelu14_out1 = self.prelu14.forward(conv2d28_out1);
        let conv2d29_out1 = crate::model_arch::conv_fwd(&self.conv2d29, prelu14_out1);
        let add13_out1 = conv2d29_out1.add(add12_out1);
        let batchnormalization14_out1 = self.batchnormalization14.forward(add13_out1.clone());
        let conv2d30_out1 = crate::model_arch::conv_fwd(&self.conv2d30, batchnormalization14_out1);
        let prelu15_out1 = self.prelu15.forward(conv2d30_out1);
        let conv2d31_out1 = crate::model_arch::conv_fwd(&self.conv2d31, prelu15_out1);
        let add14_out1 = conv2d31_out1.add(add13_out1);
        let batchnormalization15_out1 = self.batchnormalization15.forward(add14_out1.clone());
        let conv2d32_out1 = crate::model_arch::conv_fwd(&self.conv2d32, batchnormalization15_out1);
        let prelu16_out1 = self.prelu16.forward(conv2d32_out1);
        let conv2d33_out1 = crate::model_arch::conv_fwd(&self.conv2d33, prelu16_out1);
        let add15_out1 = conv2d33_out1.add(add14_out1);
        let batchnormalization16_out1 = self.batchnormalization16.forward(add15_out1.clone());
        let conv2d34_out1 = crate::model_arch::conv_fwd(&self.conv2d34, batchnormalization16_out1);
        let prelu17_out1 = self.prelu17.forward(conv2d34_out1);
        let conv2d35_out1 = crate::model_arch::conv_fwd(&self.conv2d35, prelu17_out1);
        let add16_out1 = conv2d35_out1.add(add15_out1);
        let batchnormalization17_out1 = self.batchnormalization17.forward(add16_out1.clone());
        let conv2d36_out1 = crate::model_arch::conv_fwd(&self.conv2d36, batchnormalization17_out1);
        let prelu18_out1 = self.prelu18.forward(conv2d36_out1);
        let conv2d37_out1 = crate::model_arch::conv_fwd(&self.conv2d37, prelu18_out1);
        let conv2d38_out1 = crate::model_arch::conv_fwd(&self.conv2d38, add16_out1);
        let add17_out1 = conv2d37_out1.add(conv2d38_out1);
        let batchnormalization18_out1 = self.batchnormalization18.forward(add17_out1.clone());
        let conv2d39_out1 = crate::model_arch::conv_fwd(&self.conv2d39, batchnormalization18_out1);
        let prelu19_out1 = self.prelu19.forward(conv2d39_out1);
        let conv2d40_out1 = crate::model_arch::conv_fwd(&self.conv2d40, prelu19_out1);
        let add18_out1 = conv2d40_out1.add(add17_out1);
        let batchnormalization19_out1 = self.batchnormalization19.forward(add18_out1.clone());
        let conv2d41_out1 = crate::model_arch::conv_fwd(&self.conv2d41, batchnormalization19_out1);
        let prelu20_out1 = self.prelu20.forward(conv2d41_out1);
        let conv2d42_out1 = crate::model_arch::conv_fwd(&self.conv2d42, prelu20_out1);
        let add19_out1 = conv2d42_out1.add(add18_out1);
        let batchnormalization20_out1 = self.batchnormalization20.forward(add19_out1.clone());
        let conv2d43_out1 = crate::model_arch::conv_fwd(&self.conv2d43, batchnormalization20_out1);
        let prelu21_out1 = self.prelu21.forward(conv2d43_out1);
        let conv2d44_out1 = crate::model_arch::conv_fwd(&self.conv2d44, prelu21_out1);
        let add20_out1 = conv2d44_out1.add(add19_out1);
        let batchnormalization21_out1 = self.batchnormalization21.forward(add20_out1.clone());
        let conv2d45_out1 = crate::model_arch::conv_fwd(&self.conv2d45, batchnormalization21_out1);
        let prelu22_out1 = self.prelu22.forward(conv2d45_out1);
        let conv2d46_out1 = crate::model_arch::conv_fwd(&self.conv2d46, prelu22_out1);
        let add21_out1 = conv2d46_out1.add(add20_out1);
        let batchnormalization22_out1 = self.batchnormalization22.forward(add21_out1.clone());
        let conv2d47_out1 = crate::model_arch::conv_fwd(&self.conv2d47, batchnormalization22_out1);
        let prelu23_out1 = self.prelu23.forward(conv2d47_out1);
        let conv2d48_out1 = crate::model_arch::conv_fwd(&self.conv2d48, prelu23_out1);
        let add22_out1 = conv2d48_out1.add(add21_out1);
        let batchnormalization23_out1 = self.batchnormalization23.forward(add22_out1.clone());
        let conv2d49_out1 = crate::model_arch::conv_fwd(&self.conv2d49, batchnormalization23_out1);
        let prelu24_out1 = self.prelu24.forward(conv2d49_out1);
        let conv2d50_out1 = crate::model_arch::conv_fwd(&self.conv2d50, prelu24_out1);
        let add23_out1 = conv2d50_out1.add(add22_out1);
        let batchnormalization24_out1 = self.batchnormalization24.forward(add23_out1.clone());
        let conv2d51_out1 = crate::model_arch::conv_fwd(&self.conv2d51, batchnormalization24_out1);
        let prelu25_out1 = self.prelu25.forward(conv2d51_out1);
        let conv2d52_out1 = crate::model_arch::conv_fwd(&self.conv2d52, prelu25_out1);
        let add24_out1 = conv2d52_out1.add(add23_out1);
        let batchnormalization25_out1 = self.batchnormalization25.forward(add24_out1.clone());
        let conv2d53_out1 = crate::model_arch::conv_fwd(&self.conv2d53, batchnormalization25_out1);
        let prelu26_out1 = self.prelu26.forward(conv2d53_out1);
        let conv2d54_out1 = crate::model_arch::conv_fwd(&self.conv2d54, prelu26_out1);
        let add25_out1 = conv2d54_out1.add(add24_out1);
        add25_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    batchnormalization26: BatchNorm<B>,
    conv2d55: Conv2d<B>,
    prelu27: PRelu<B>,
    conv2d56: Conv2d<B>,
    batchnormalization27: BatchNorm<B>,
    conv2d57: Conv2d<B>,
    prelu28: PRelu<B>,
    conv2d58: Conv2d<B>,
    batchnormalization28: BatchNorm<B>,
    conv2d59: Conv2d<B>,
    prelu29: PRelu<B>,
    conv2d60: Conv2d<B>,
    batchnormalization29: BatchNorm<B>,
    conv2d61: Conv2d<B>,
    prelu30: PRelu<B>,
    conv2d62: Conv2d<B>,
    batchnormalization30: BatchNorm<B>,
    conv2d63: Conv2d<B>,
    prelu31: PRelu<B>,
    conv2d64: Conv2d<B>,
    batchnormalization31: BatchNorm<B>,
    conv2d65: Conv2d<B>,
    prelu32: PRelu<B>,
    conv2d66: Conv2d<B>,
    batchnormalization32: BatchNorm<B>,
    conv2d67: Conv2d<B>,
    prelu33: PRelu<B>,
    conv2d68: Conv2d<B>,
    batchnormalization33: BatchNorm<B>,
    conv2d69: Conv2d<B>,
    prelu34: PRelu<B>,
    conv2d70: Conv2d<B>,
    batchnormalization34: BatchNorm<B>,
    conv2d71: Conv2d<B>,
    prelu35: PRelu<B>,
    conv2d72: Conv2d<B>,
    batchnormalization35: BatchNorm<B>,
    conv2d73: Conv2d<B>,
    prelu36: PRelu<B>,
    conv2d74: Conv2d<B>,
    batchnormalization36: BatchNorm<B>,
    conv2d75: Conv2d<B>,
    prelu37: PRelu<B>,
    conv2d76: Conv2d<B>,
    batchnormalization37: BatchNorm<B>,
    conv2d77: Conv2d<B>,
    prelu38: PRelu<B>,
    conv2d78: Conv2d<B>,
    batchnormalization38: BatchNorm<B>,
    conv2d79: Conv2d<B>,
    prelu39: PRelu<B>,
    conv2d80: Conv2d<B>,
    batchnormalization39: BatchNorm<B>,
    conv2d81: Conv2d<B>,
    prelu40: PRelu<B>,
    conv2d82: Conv2d<B>,
    batchnormalization40: BatchNorm<B>,
    conv2d83: Conv2d<B>,
    prelu41: PRelu<B>,
    conv2d84: Conv2d<B>,
    batchnormalization41: BatchNorm<B>,
    conv2d85: Conv2d<B>,
    prelu42: PRelu<B>,
    conv2d86: Conv2d<B>,
    batchnormalization42: BatchNorm<B>,
    conv2d87: Conv2d<B>,
    prelu43: PRelu<B>,
    conv2d88: Conv2d<B>,
    batchnormalization43: BatchNorm<B>,
    conv2d89: Conv2d<B>,
    prelu44: PRelu<B>,
    conv2d90: Conv2d<B>,
    batchnormalization44: BatchNorm<B>,
    conv2d91: Conv2d<B>,
    prelu45: PRelu<B>,
    conv2d92: Conv2d<B>,
    batchnormalization45: BatchNorm<B>,
    conv2d93: Conv2d<B>,
    prelu46: PRelu<B>,
    conv2d94: Conv2d<B>,
    batchnormalization46: BatchNorm<B>,
    conv2d95: Conv2d<B>,
    prelu47: PRelu<B>,
    conv2d96: Conv2d<B>,
    batchnormalization47: BatchNorm<B>,
    conv2d97: Conv2d<B>,
    prelu48: PRelu<B>,
    conv2d98: Conv2d<B>,
    conv2d99: Conv2d<B>,
    batchnormalization48: BatchNorm<B>,
    conv2d100: Conv2d<B>,
    prelu49: PRelu<B>,
    conv2d101: Conv2d<B>,
    batchnormalization49: BatchNorm<B>,
    conv2d102: Conv2d<B>,
    prelu50: PRelu<B>,
    conv2d103: Conv2d<B>,
    batchnormalization50: BatchNorm<B>,
    linear1: Linear<B>,
    batchnormalization51: BatchNorm<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let batchnormalization26 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d55 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu27 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d56 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization27 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d57 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu28 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d58 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization28 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d59 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu29 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d60 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization29 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d61 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu30 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d62 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization30 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d63 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu31 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d64 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization31 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d65 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu32 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d66 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization32 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d67 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu33 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d68 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization33 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d69 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu34 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d70 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization34 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d71 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu35 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d72 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization35 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d73 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu36 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d74 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization36 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d75 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu37 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d76 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization37 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d77 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu38 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d78 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization38 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d79 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu39 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d80 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization39 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d81 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu40 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d82 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization40 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d83 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu41 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d84 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization41 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d85 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu42 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d86 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization42 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d87 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu43 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d88 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization43 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d89 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu44 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d90 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization44 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d91 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu45 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d92 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization45 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d93 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu46 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d94 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization46 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d95 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu47 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d96 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization47 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d97 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu48 = PReluConfig::new().with_num_parameters(512).init(device);
        let conv2d98 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d99 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization48 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d100 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu49 = PReluConfig::new().with_num_parameters(512).init(device);
        let conv2d101 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization49 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d102 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu50 = PReluConfig::new().with_num_parameters(512).init(device);
        let conv2d103 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization50 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let linear1 = LinearConfig::new(25088, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let batchnormalization51 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        Self {
            batchnormalization26,
            conv2d55,
            prelu27,
            conv2d56,
            batchnormalization27,
            conv2d57,
            prelu28,
            conv2d58,
            batchnormalization28,
            conv2d59,
            prelu29,
            conv2d60,
            batchnormalization29,
            conv2d61,
            prelu30,
            conv2d62,
            batchnormalization30,
            conv2d63,
            prelu31,
            conv2d64,
            batchnormalization31,
            conv2d65,
            prelu32,
            conv2d66,
            batchnormalization32,
            conv2d67,
            prelu33,
            conv2d68,
            batchnormalization33,
            conv2d69,
            prelu34,
            conv2d70,
            batchnormalization34,
            conv2d71,
            prelu35,
            conv2d72,
            batchnormalization35,
            conv2d73,
            prelu36,
            conv2d74,
            batchnormalization36,
            conv2d75,
            prelu37,
            conv2d76,
            batchnormalization37,
            conv2d77,
            prelu38,
            conv2d78,
            batchnormalization38,
            conv2d79,
            prelu39,
            conv2d80,
            batchnormalization39,
            conv2d81,
            prelu40,
            conv2d82,
            batchnormalization40,
            conv2d83,
            prelu41,
            conv2d84,
            batchnormalization41,
            conv2d85,
            prelu42,
            conv2d86,
            batchnormalization42,
            conv2d87,
            prelu43,
            conv2d88,
            batchnormalization43,
            conv2d89,
            prelu44,
            conv2d90,
            batchnormalization44,
            conv2d91,
            prelu45,
            conv2d92,
            batchnormalization45,
            conv2d93,
            prelu46,
            conv2d94,
            batchnormalization46,
            conv2d95,
            prelu47,
            conv2d96,
            batchnormalization47,
            conv2d97,
            prelu48,
            conv2d98,
            conv2d99,
            batchnormalization48,
            conv2d100,
            prelu49,
            conv2d101,
            batchnormalization49,
            conv2d102,
            prelu50,
            conv2d103,
            batchnormalization50,
            linear1,
            batchnormalization51,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add25_out1: Tensor<B, 4>) -> Tensor<B, 2> {
        let batchnormalization26_out1 = self.batchnormalization26.forward(add25_out1.clone());
        let conv2d55_out1 = crate::model_arch::conv_fwd(&self.conv2d55, batchnormalization26_out1);
        let prelu27_out1 = self.prelu27.forward(conv2d55_out1);
        let conv2d56_out1 = crate::model_arch::conv_fwd(&self.conv2d56, prelu27_out1);
        let add26_out1 = conv2d56_out1.add(add25_out1);
        let batchnormalization27_out1 = self.batchnormalization27.forward(add26_out1.clone());
        let conv2d57_out1 = crate::model_arch::conv_fwd(&self.conv2d57, batchnormalization27_out1);
        let prelu28_out1 = self.prelu28.forward(conv2d57_out1);
        let conv2d58_out1 = crate::model_arch::conv_fwd(&self.conv2d58, prelu28_out1);
        let add27_out1 = conv2d58_out1.add(add26_out1);
        let batchnormalization28_out1 = self.batchnormalization28.forward(add27_out1.clone());
        let conv2d59_out1 = crate::model_arch::conv_fwd(&self.conv2d59, batchnormalization28_out1);
        let prelu29_out1 = self.prelu29.forward(conv2d59_out1);
        let conv2d60_out1 = crate::model_arch::conv_fwd(&self.conv2d60, prelu29_out1);
        let add28_out1 = conv2d60_out1.add(add27_out1);
        let batchnormalization29_out1 = self.batchnormalization29.forward(add28_out1.clone());
        let conv2d61_out1 = crate::model_arch::conv_fwd(&self.conv2d61, batchnormalization29_out1);
        let prelu30_out1 = self.prelu30.forward(conv2d61_out1);
        let conv2d62_out1 = crate::model_arch::conv_fwd(&self.conv2d62, prelu30_out1);
        let add29_out1 = conv2d62_out1.add(add28_out1);
        let batchnormalization30_out1 = self.batchnormalization30.forward(add29_out1.clone());
        let conv2d63_out1 = crate::model_arch::conv_fwd(&self.conv2d63, batchnormalization30_out1);
        let prelu31_out1 = self.prelu31.forward(conv2d63_out1);
        let conv2d64_out1 = crate::model_arch::conv_fwd(&self.conv2d64, prelu31_out1);
        let add30_out1 = conv2d64_out1.add(add29_out1);
        let batchnormalization31_out1 = self.batchnormalization31.forward(add30_out1.clone());
        let conv2d65_out1 = crate::model_arch::conv_fwd(&self.conv2d65, batchnormalization31_out1);
        let prelu32_out1 = self.prelu32.forward(conv2d65_out1);
        let conv2d66_out1 = crate::model_arch::conv_fwd(&self.conv2d66, prelu32_out1);
        let add31_out1 = conv2d66_out1.add(add30_out1);
        let batchnormalization32_out1 = self.batchnormalization32.forward(add31_out1.clone());
        let conv2d67_out1 = crate::model_arch::conv_fwd(&self.conv2d67, batchnormalization32_out1);
        let prelu33_out1 = self.prelu33.forward(conv2d67_out1);
        let conv2d68_out1 = crate::model_arch::conv_fwd(&self.conv2d68, prelu33_out1);
        let add32_out1 = conv2d68_out1.add(add31_out1);
        let batchnormalization33_out1 = self.batchnormalization33.forward(add32_out1.clone());
        let conv2d69_out1 = crate::model_arch::conv_fwd(&self.conv2d69, batchnormalization33_out1);
        let prelu34_out1 = self.prelu34.forward(conv2d69_out1);
        let conv2d70_out1 = crate::model_arch::conv_fwd(&self.conv2d70, prelu34_out1);
        let add33_out1 = conv2d70_out1.add(add32_out1);
        let batchnormalization34_out1 = self.batchnormalization34.forward(add33_out1.clone());
        let conv2d71_out1 = crate::model_arch::conv_fwd(&self.conv2d71, batchnormalization34_out1);
        let prelu35_out1 = self.prelu35.forward(conv2d71_out1);
        let conv2d72_out1 = crate::model_arch::conv_fwd(&self.conv2d72, prelu35_out1);
        let add34_out1 = conv2d72_out1.add(add33_out1);
        let batchnormalization35_out1 = self.batchnormalization35.forward(add34_out1.clone());
        let conv2d73_out1 = crate::model_arch::conv_fwd(&self.conv2d73, batchnormalization35_out1);
        let prelu36_out1 = self.prelu36.forward(conv2d73_out1);
        let conv2d74_out1 = crate::model_arch::conv_fwd(&self.conv2d74, prelu36_out1);
        let add35_out1 = conv2d74_out1.add(add34_out1);
        let batchnormalization36_out1 = self.batchnormalization36.forward(add35_out1.clone());
        let conv2d75_out1 = crate::model_arch::conv_fwd(&self.conv2d75, batchnormalization36_out1);
        let prelu37_out1 = self.prelu37.forward(conv2d75_out1);
        let conv2d76_out1 = crate::model_arch::conv_fwd(&self.conv2d76, prelu37_out1);
        let add36_out1 = conv2d76_out1.add(add35_out1);
        let batchnormalization37_out1 = self.batchnormalization37.forward(add36_out1.clone());
        let conv2d77_out1 = crate::model_arch::conv_fwd(&self.conv2d77, batchnormalization37_out1);
        let prelu38_out1 = self.prelu38.forward(conv2d77_out1);
        let conv2d78_out1 = crate::model_arch::conv_fwd(&self.conv2d78, prelu38_out1);
        let add37_out1 = conv2d78_out1.add(add36_out1);
        let batchnormalization38_out1 = self.batchnormalization38.forward(add37_out1.clone());
        let conv2d79_out1 = crate::model_arch::conv_fwd(&self.conv2d79, batchnormalization38_out1);
        let prelu39_out1 = self.prelu39.forward(conv2d79_out1);
        let conv2d80_out1 = crate::model_arch::conv_fwd(&self.conv2d80, prelu39_out1);
        let add38_out1 = conv2d80_out1.add(add37_out1);
        let batchnormalization39_out1 = self.batchnormalization39.forward(add38_out1.clone());
        let conv2d81_out1 = crate::model_arch::conv_fwd(&self.conv2d81, batchnormalization39_out1);
        let prelu40_out1 = self.prelu40.forward(conv2d81_out1);
        let conv2d82_out1 = crate::model_arch::conv_fwd(&self.conv2d82, prelu40_out1);
        let add39_out1 = conv2d82_out1.add(add38_out1);
        let batchnormalization40_out1 = self.batchnormalization40.forward(add39_out1.clone());
        let conv2d83_out1 = crate::model_arch::conv_fwd(&self.conv2d83, batchnormalization40_out1);
        let prelu41_out1 = self.prelu41.forward(conv2d83_out1);
        let conv2d84_out1 = crate::model_arch::conv_fwd(&self.conv2d84, prelu41_out1);
        let add40_out1 = conv2d84_out1.add(add39_out1);
        let batchnormalization41_out1 = self.batchnormalization41.forward(add40_out1.clone());
        let conv2d85_out1 = crate::model_arch::conv_fwd(&self.conv2d85, batchnormalization41_out1);
        let prelu42_out1 = self.prelu42.forward(conv2d85_out1);
        let conv2d86_out1 = crate::model_arch::conv_fwd(&self.conv2d86, prelu42_out1);
        let add41_out1 = conv2d86_out1.add(add40_out1);
        let batchnormalization42_out1 = self.batchnormalization42.forward(add41_out1.clone());
        let conv2d87_out1 = crate::model_arch::conv_fwd(&self.conv2d87, batchnormalization42_out1);
        let prelu43_out1 = self.prelu43.forward(conv2d87_out1);
        let conv2d88_out1 = crate::model_arch::conv_fwd(&self.conv2d88, prelu43_out1);
        let add42_out1 = conv2d88_out1.add(add41_out1);
        let batchnormalization43_out1 = self.batchnormalization43.forward(add42_out1.clone());
        let conv2d89_out1 = crate::model_arch::conv_fwd(&self.conv2d89, batchnormalization43_out1);
        let prelu44_out1 = self.prelu44.forward(conv2d89_out1);
        let conv2d90_out1 = crate::model_arch::conv_fwd(&self.conv2d90, prelu44_out1);
        let add43_out1 = conv2d90_out1.add(add42_out1);
        let batchnormalization44_out1 = self.batchnormalization44.forward(add43_out1.clone());
        let conv2d91_out1 = crate::model_arch::conv_fwd(&self.conv2d91, batchnormalization44_out1);
        let prelu45_out1 = self.prelu45.forward(conv2d91_out1);
        let conv2d92_out1 = crate::model_arch::conv_fwd(&self.conv2d92, prelu45_out1);
        let add44_out1 = conv2d92_out1.add(add43_out1);
        let batchnormalization45_out1 = self.batchnormalization45.forward(add44_out1.clone());
        let conv2d93_out1 = crate::model_arch::conv_fwd(&self.conv2d93, batchnormalization45_out1);
        let prelu46_out1 = self.prelu46.forward(conv2d93_out1);
        let conv2d94_out1 = crate::model_arch::conv_fwd(&self.conv2d94, prelu46_out1);
        let add45_out1 = conv2d94_out1.add(add44_out1);
        let batchnormalization46_out1 = self.batchnormalization46.forward(add45_out1.clone());
        let conv2d95_out1 = crate::model_arch::conv_fwd(&self.conv2d95, batchnormalization46_out1);
        let prelu47_out1 = self.prelu47.forward(conv2d95_out1);
        let conv2d96_out1 = crate::model_arch::conv_fwd(&self.conv2d96, prelu47_out1);
        let add46_out1 = conv2d96_out1.add(add45_out1);
        let batchnormalization47_out1 = self.batchnormalization47.forward(add46_out1.clone());
        let conv2d97_out1 = crate::model_arch::conv_fwd(&self.conv2d97, batchnormalization47_out1);
        let prelu48_out1 = self.prelu48.forward(conv2d97_out1);
        let conv2d98_out1 = crate::model_arch::conv_fwd(&self.conv2d98, prelu48_out1);
        let conv2d99_out1 = crate::model_arch::conv_fwd(&self.conv2d99, add46_out1);
        let add47_out1 = conv2d98_out1.add(conv2d99_out1);
        let batchnormalization48_out1 = self.batchnormalization48.forward(add47_out1.clone());
        let conv2d100_out1 =
            crate::model_arch::conv_fwd(&self.conv2d100, batchnormalization48_out1);
        let prelu49_out1 = self.prelu49.forward(conv2d100_out1);
        let conv2d101_out1 = crate::model_arch::conv_fwd(&self.conv2d101, prelu49_out1);
        let add48_out1 = conv2d101_out1.add(add47_out1);
        let batchnormalization49_out1 = self.batchnormalization49.forward(add48_out1.clone());
        let conv2d102_out1 =
            crate::model_arch::conv_fwd(&self.conv2d102, batchnormalization49_out1);
        let prelu50_out1 = self.prelu50.forward(conv2d102_out1);
        let conv2d103_out1 = crate::model_arch::conv_fwd(&self.conv2d103, prelu50_out1);
        let add49_out1 = conv2d103_out1.add(add48_out1);
        let batchnormalization50_out1 = self.batchnormalization50.forward(add49_out1);
        let flatten1_out1 = {
            let leading_dim = batchnormalization50_out1.shape()[..1]
                .iter()
                .product::<usize>() as i32;
            batchnormalization50_out1.reshape::<2, _>([leading_dim, -1])
        };
        let linear1_out1 = self.linear1.forward(flatten1_out1);
        let batchnormalization51_out1 = self.batchnormalization51.forward(linear1_out1);
        batchnormalization51_out1
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
        panic!("model weights are not embedded; use Model::from_file or Model::from_bytes")
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
    pub fn forward(&self, input_1: Tensor<B, 4>) -> Tensor<B, 2> {
        let add12_out1 = self.submodule1.forward(input_1);
        let add25_out1 = self.submodule2.forward(add12_out1);
        let batchnormalization51_out1 = self.submodule3.forward(add25_out1);
        batchnormalization51_out1
    }
}
