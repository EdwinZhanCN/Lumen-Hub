// Generated from ONNX "src/ppocrv5/recognition.fp32.onnx" by burn-onnx
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv2d1: Conv2d<B>,
    batchnormalization1: BatchNorm<B>,
    conv2d2: Conv2d<B>,
    reshape1: burn::module::Param<Tensor<B, 4>>,
    constant105: burn::module::Param<Tensor<B, 1>>,
    constant106: burn::module::Param<Tensor<B, 1>>,
    constant107: burn::module::Param<Tensor<B, 1>>,
    constant108: burn::module::Param<Tensor<B, 1>>,
    conv2d3: Conv2d<B>,
    reshape2: burn::module::Param<Tensor<B, 4>>,
    constant129: burn::module::Param<Tensor<B, 1>>,
    constant130: burn::module::Param<Tensor<B, 1>>,
    constant151: burn::module::Param<Tensor<B, 1>>,
    constant152: burn::module::Param<Tensor<B, 1>>,
    conv2d4: Conv2d<B>,
    reshape3: burn::module::Param<Tensor<B, 4>>,
    constant173: burn::module::Param<Tensor<B, 1>>,
    constant174: burn::module::Param<Tensor<B, 1>>,
    constant195: burn::module::Param<Tensor<B, 1>>,
    constant196: burn::module::Param<Tensor<B, 1>>,
    conv2d5: Conv2d<B>,
    reshape4: burn::module::Param<Tensor<B, 4>>,
    constant209: burn::module::Param<Tensor<B, 1>>,
    constant210: burn::module::Param<Tensor<B, 1>>,
    constant211: burn::module::Param<Tensor<B, 1>>,
    constant212: burn::module::Param<Tensor<B, 1>>,
    conv2d6: Conv2d<B>,
    reshape5: burn::module::Param<Tensor<B, 4>>,
    constant213: burn::module::Param<Tensor<B, 1>>,
    constant214: burn::module::Param<Tensor<B, 1>>,
    constant215: burn::module::Param<Tensor<B, 1>>,
    constant216: burn::module::Param<Tensor<B, 1>>,
    conv2d7: Conv2d<B>,
    reshape6: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(16)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d2 = Conv2dConfig::new([16, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(false)
            .init(device);
        let reshape1: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 16, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 16, 1, 1].into(),
        );
        let constant105: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant106: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant107: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant108: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d3 = Conv2dConfig::new([16, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape2: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 32, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 32, 1, 1].into(),
        );
        let constant129: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant130: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant151: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant152: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d4 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(false)
            .init(device);
        let reshape3: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 32, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 32, 1, 1].into(),
        );
        let constant173: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant174: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant195: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant196: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d5 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape4: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 64, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 64, 1, 1].into(),
        );
        let constant209: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant210: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant211: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant212: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d6 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(false)
            .init(device);
        let reshape5: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 64, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 64, 1, 1].into(),
        );
        let constant213: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant214: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant215: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant216: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d7 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape6: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 64, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 64, 1, 1].into(),
        );
        Self {
            conv2d1,
            batchnormalization1,
            conv2d2,
            reshape1,
            constant105,
            constant106,
            constant107,
            constant108,
            conv2d3,
            reshape2,
            constant129,
            constant130,
            constant151,
            constant152,
            conv2d4,
            reshape3,
            constant173,
            constant174,
            constant195,
            constant196,
            conv2d5,
            reshape4,
            constant209,
            constant210,
            constant211,
            constant212,
            conv2d6,
            reshape5,
            constant213,
            constant214,
            constant215,
            constant216,
            conv2d7,
            reshape6,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = crate::model_arch::conv_fwd(&self.conv2d1, x);
        let batchnormalization1_out1 = self.batchnormalization1.forward(conv2d1_out1);
        let conv2d2_out1 = crate::model_arch::conv_fwd(&self.conv2d2, batchnormalization1_out1);
        let reshape1_out1 = self.reshape1.val();
        let add1_out1 = conv2d2_out1.add(reshape1_out1);
        let constant105_out1 = self.constant105.val();
        let mul1_out1 = (constant105_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add1_out1);
        let constant106_out1 = self.constant106.val();
        let add2_out1 = mul1_out1.add((constant106_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish1_out1 = burn::tensor::activation::hard_swish(add2_out1);
        let constant107_out1 = self.constant107.val();
        let mul2_out1 = (constant107_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish1_out1);
        let constant108_out1 = self.constant108.val();
        let add3_out1 = mul2_out1.add((constant108_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d3_out1 = crate::model_arch::conv_fwd(&self.conv2d3, add3_out1);
        let reshape2_out1 = self.reshape2.val();
        let add4_out1 = conv2d3_out1.add(reshape2_out1);
        let constant129_out1 = self.constant129.val();
        let mul3_out1 = (constant129_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add4_out1);
        let constant130_out1 = self.constant130.val();
        let add5_out1 = mul3_out1.add((constant130_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish2_out1 = burn::tensor::activation::hard_swish(add5_out1);
        let constant151_out1 = self.constant151.val();
        let mul4_out1 = (constant151_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish2_out1);
        let constant152_out1 = self.constant152.val();
        let add6_out1 = mul4_out1.add((constant152_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d4_out1 = crate::model_arch::conv_fwd(&self.conv2d4, add6_out1);
        let reshape3_out1 = self.reshape3.val();
        let add7_out1 = conv2d4_out1.add(reshape3_out1);
        let constant173_out1 = self.constant173.val();
        let mul5_out1 = (constant173_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add7_out1);
        let constant174_out1 = self.constant174.val();
        let add8_out1 = mul5_out1.add((constant174_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish3_out1 = burn::tensor::activation::hard_swish(add8_out1);
        let constant195_out1 = self.constant195.val();
        let mul6_out1 = (constant195_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish3_out1);
        let constant196_out1 = self.constant196.val();
        let add9_out1 = mul6_out1.add((constant196_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d5_out1 = crate::model_arch::conv_fwd(&self.conv2d5, add9_out1);
        let reshape4_out1 = self.reshape4.val();
        let add10_out1 = conv2d5_out1.add(reshape4_out1);
        let constant209_out1 = self.constant209.val();
        let mul7_out1 = (constant209_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add10_out1);
        let constant210_out1 = self.constant210.val();
        let add11_out1 =
            mul7_out1.add((constant210_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish4_out1 = burn::tensor::activation::hard_swish(add11_out1);
        let constant211_out1 = self.constant211.val();
        let mul8_out1 = (constant211_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish4_out1);
        let constant212_out1 = self.constant212.val();
        let add12_out1 =
            mul8_out1.add((constant212_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d6_out1 = crate::model_arch::conv_fwd(&self.conv2d6, add12_out1);
        let reshape5_out1 = self.reshape5.val();
        let add13_out1 = conv2d6_out1.add(reshape5_out1);
        let constant213_out1 = self.constant213.val();
        let mul9_out1 = (constant213_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add13_out1);
        let constant214_out1 = self.constant214.val();
        let add14_out1 =
            mul9_out1.add((constant214_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish5_out1 = burn::tensor::activation::hard_swish(add14_out1);
        let constant215_out1 = self.constant215.val();
        let mul10_out1 = (constant215_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish5_out1);
        let constant216_out1 = self.constant216.val();
        let add15_out1 =
            mul10_out1.add((constant216_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d7_out1 = crate::model_arch::conv_fwd(&self.conv2d7, add15_out1);
        let reshape6_out1 = self.reshape6.val();
        let add16_out1 = conv2d7_out1.add(reshape6_out1);
        add16_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    constant109: burn::module::Param<Tensor<B, 1>>,
    constant110: burn::module::Param<Tensor<B, 1>>,
    constant111: burn::module::Param<Tensor<B, 1>>,
    constant112: burn::module::Param<Tensor<B, 1>>,
    conv2d8: Conv2d<B>,
    reshape7: burn::module::Param<Tensor<B, 4>>,
    constant113: burn::module::Param<Tensor<B, 1>>,
    constant114: burn::module::Param<Tensor<B, 1>>,
    constant115: burn::module::Param<Tensor<B, 1>>,
    constant116: burn::module::Param<Tensor<B, 1>>,
    conv2d9: Conv2d<B>,
    reshape8: burn::module::Param<Tensor<B, 4>>,
    constant117: burn::module::Param<Tensor<B, 1>>,
    constant118: burn::module::Param<Tensor<B, 1>>,
    constant119: burn::module::Param<Tensor<B, 1>>,
    constant120: burn::module::Param<Tensor<B, 1>>,
    conv2d10: Conv2d<B>,
    reshape9: burn::module::Param<Tensor<B, 4>>,
    constant121: burn::module::Param<Tensor<B, 1>>,
    constant122: burn::module::Param<Tensor<B, 1>>,
    constant123: burn::module::Param<Tensor<B, 1>>,
    constant124: burn::module::Param<Tensor<B, 1>>,
    conv2d11: Conv2d<B>,
    reshape10: burn::module::Param<Tensor<B, 4>>,
    constant125: burn::module::Param<Tensor<B, 1>>,
    constant126: burn::module::Param<Tensor<B, 1>>,
    constant127: burn::module::Param<Tensor<B, 1>>,
    constant128: burn::module::Param<Tensor<B, 1>>,
    conv2d12: Conv2d<B>,
    reshape11: burn::module::Param<Tensor<B, 4>>,
    constant131: burn::module::Param<Tensor<B, 1>>,
    constant132: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant109: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant110: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant111: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant112: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d8 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(false)
            .init(device);
        let reshape7: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 64, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 64, 1, 1].into(),
        );
        let constant113: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant114: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant115: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant116: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d9 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape8: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 128, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 128, 1, 1].into(),
        );
        let constant117: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant118: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant119: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant120: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d10 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(false)
            .init(device);
        let reshape9: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 128, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 128, 1, 1].into(),
        );
        let constant121: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant122: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant123: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant124: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d11 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape10: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 128, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 128, 1, 1].into(),
        );
        let constant125: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant126: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant127: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant128: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d12 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(false)
            .init(device);
        let reshape11: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 128, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 128, 1, 1].into(),
        );
        let constant131: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant132: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            constant109,
            constant110,
            constant111,
            constant112,
            conv2d8,
            reshape7,
            constant113,
            constant114,
            constant115,
            constant116,
            conv2d9,
            reshape8,
            constant117,
            constant118,
            constant119,
            constant120,
            conv2d10,
            reshape9,
            constant121,
            constant122,
            constant123,
            constant124,
            conv2d11,
            reshape10,
            constant125,
            constant126,
            constant127,
            constant128,
            conv2d12,
            reshape11,
            constant131,
            constant132,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add16_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let constant109_out1 = self.constant109.val();
        let mul11_out1 = (constant109_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add16_out1);
        let constant110_out1 = self.constant110.val();
        let add17_out1 =
            mul11_out1.add((constant110_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish6_out1 = burn::tensor::activation::hard_swish(add17_out1);
        let constant111_out1 = self.constant111.val();
        let mul12_out1 = (constant111_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish6_out1);
        let constant112_out1 = self.constant112.val();
        let add18_out1 =
            mul12_out1.add((constant112_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d8_out1 = crate::model_arch::conv_fwd(&self.conv2d8, add18_out1);
        let reshape7_out1 = self.reshape7.val();
        let add19_out1 = conv2d8_out1.add(reshape7_out1);
        let constant113_out1 = self.constant113.val();
        let mul13_out1 = (constant113_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add19_out1);
        let constant114_out1 = self.constant114.val();
        let add20_out1 =
            mul13_out1.add((constant114_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish7_out1 = burn::tensor::activation::hard_swish(add20_out1);
        let constant115_out1 = self.constant115.val();
        let mul14_out1 = (constant115_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish7_out1);
        let constant116_out1 = self.constant116.val();
        let add21_out1 =
            mul14_out1.add((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d9_out1 = crate::model_arch::conv_fwd(&self.conv2d9, add21_out1);
        let reshape8_out1 = self.reshape8.val();
        let add22_out1 = conv2d9_out1.add(reshape8_out1);
        let constant117_out1 = self.constant117.val();
        let mul15_out1 = (constant117_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add22_out1);
        let constant118_out1 = self.constant118.val();
        let add23_out1 =
            mul15_out1.add((constant118_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish8_out1 = burn::tensor::activation::hard_swish(add23_out1);
        let constant119_out1 = self.constant119.val();
        let mul16_out1 = (constant119_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish8_out1);
        let constant120_out1 = self.constant120.val();
        let add24_out1 =
            mul16_out1.add((constant120_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d10_out1 = crate::model_arch::conv_fwd(&self.conv2d10, add24_out1);
        let reshape9_out1 = self.reshape9.val();
        let add25_out1 = conv2d10_out1.add(reshape9_out1);
        let constant121_out1 = self.constant121.val();
        let mul17_out1 = (constant121_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add25_out1);
        let constant122_out1 = self.constant122.val();
        let add26_out1 =
            mul17_out1.add((constant122_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish9_out1 = burn::tensor::activation::hard_swish(add26_out1);
        let constant123_out1 = self.constant123.val();
        let mul18_out1 = (constant123_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish9_out1);
        let constant124_out1 = self.constant124.val();
        let add27_out1 =
            mul18_out1.add((constant124_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d11_out1 = crate::model_arch::conv_fwd(&self.conv2d11, add27_out1);
        let reshape10_out1 = self.reshape10.val();
        let add28_out1 = conv2d11_out1.add(reshape10_out1);
        let constant125_out1 = self.constant125.val();
        let mul19_out1 = (constant125_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add28_out1);
        let constant126_out1 = self.constant126.val();
        let add29_out1 =
            mul19_out1.add((constant126_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish10_out1 = burn::tensor::activation::hard_swish(add29_out1);
        let constant127_out1 = self.constant127.val();
        let mul20_out1 = (constant127_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish10_out1);
        let constant128_out1 = self.constant128.val();
        let add30_out1 =
            mul20_out1.add((constant128_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d12_out1 = crate::model_arch::conv_fwd(&self.conv2d12, add30_out1);
        let reshape11_out1 = self.reshape11.val();
        let add31_out1 = conv2d12_out1.add(reshape11_out1);
        let constant131_out1 = self.constant131.val();
        let mul21_out1 = (constant131_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add31_out1);
        let constant132_out1 = self.constant132.val();
        let add32_out1 =
            mul21_out1.add((constant132_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        add32_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    constant133: burn::module::Param<Tensor<B, 1>>,
    constant134: burn::module::Param<Tensor<B, 1>>,
    conv2d13: Conv2d<B>,
    reshape12: burn::module::Param<Tensor<B, 4>>,
    constant135: burn::module::Param<Tensor<B, 1>>,
    constant136: burn::module::Param<Tensor<B, 1>>,
    constant137: burn::module::Param<Tensor<B, 1>>,
    constant138: burn::module::Param<Tensor<B, 1>>,
    conv2d14: Conv2d<B>,
    reshape13: burn::module::Param<Tensor<B, 4>>,
    constant139: burn::module::Param<Tensor<B, 1>>,
    constant140: burn::module::Param<Tensor<B, 1>>,
    constant141: burn::module::Param<Tensor<B, 1>>,
    constant142: burn::module::Param<Tensor<B, 1>>,
    conv2d15: Conv2d<B>,
    reshape14: burn::module::Param<Tensor<B, 4>>,
    constant143: burn::module::Param<Tensor<B, 1>>,
    constant144: burn::module::Param<Tensor<B, 1>>,
    constant145: burn::module::Param<Tensor<B, 1>>,
    constant146: burn::module::Param<Tensor<B, 1>>,
    conv2d16: Conv2d<B>,
    reshape15: burn::module::Param<Tensor<B, 4>>,
    constant147: burn::module::Param<Tensor<B, 1>>,
    constant148: burn::module::Param<Tensor<B, 1>>,
    constant149: burn::module::Param<Tensor<B, 1>>,
    constant150: burn::module::Param<Tensor<B, 1>>,
    conv2d17: Conv2d<B>,
    reshape16: burn::module::Param<Tensor<B, 4>>,
    constant153: burn::module::Param<Tensor<B, 1>>,
    constant154: burn::module::Param<Tensor<B, 1>>,
    constant155: burn::module::Param<Tensor<B, 1>>,
    constant156: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant133: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant134: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d13 = Conv2dConfig::new([128, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape12: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant135: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant136: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant137: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant138: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d14 = Conv2dConfig::new([240, 240], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(false)
            .init(device);
        let reshape13: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant139: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant140: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant141: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant142: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d15 = Conv2dConfig::new([240, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape14: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant143: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant144: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant145: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant146: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d16 = Conv2dConfig::new([240, 240], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(false)
            .init(device);
        let reshape15: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant147: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant148: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant149: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant150: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d17 = Conv2dConfig::new([240, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape16: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant153: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant154: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant155: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant156: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            constant133,
            constant134,
            conv2d13,
            reshape12,
            constant135,
            constant136,
            constant137,
            constant138,
            conv2d14,
            reshape13,
            constant139,
            constant140,
            constant141,
            constant142,
            conv2d15,
            reshape14,
            constant143,
            constant144,
            constant145,
            constant146,
            conv2d16,
            reshape15,
            constant147,
            constant148,
            constant149,
            constant150,
            conv2d17,
            reshape16,
            constant153,
            constant154,
            constant155,
            constant156,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add32_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let hardswish11_out1 = burn::tensor::activation::hard_swish(add32_out1);
        let constant133_out1 = self.constant133.val();
        let mul22_out1 = (constant133_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish11_out1);
        let constant134_out1 = self.constant134.val();
        let add33_out1 =
            mul22_out1.add((constant134_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d13_out1 = crate::model_arch::conv_fwd(&self.conv2d13, add33_out1);
        let reshape12_out1 = self.reshape12.val();
        let add34_out1 = conv2d13_out1.add(reshape12_out1);
        let constant135_out1 = self.constant135.val();
        let mul23_out1 = (constant135_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add34_out1);
        let constant136_out1 = self.constant136.val();
        let add35_out1 =
            mul23_out1.add((constant136_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish12_out1 = burn::tensor::activation::hard_swish(add35_out1);
        let constant137_out1 = self.constant137.val();
        let mul24_out1 = (constant137_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish12_out1);
        let constant138_out1 = self.constant138.val();
        let add36_out1 =
            mul24_out1.add((constant138_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d14_out1 = crate::model_arch::conv_fwd(&self.conv2d14, add36_out1);
        let reshape13_out1 = self.reshape13.val();
        let add37_out1 = conv2d14_out1.add(reshape13_out1);
        let constant139_out1 = self.constant139.val();
        let mul25_out1 = (constant139_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add37_out1);
        let constant140_out1 = self.constant140.val();
        let add38_out1 =
            mul25_out1.add((constant140_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish13_out1 = burn::tensor::activation::hard_swish(add38_out1);
        let constant141_out1 = self.constant141.val();
        let mul26_out1 = (constant141_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish13_out1);
        let constant142_out1 = self.constant142.val();
        let add39_out1 =
            mul26_out1.add((constant142_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d15_out1 = crate::model_arch::conv_fwd(&self.conv2d15, add39_out1);
        let reshape14_out1 = self.reshape14.val();
        let add40_out1 = conv2d15_out1.add(reshape14_out1);
        let constant143_out1 = self.constant143.val();
        let mul27_out1 = (constant143_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add40_out1);
        let constant144_out1 = self.constant144.val();
        let add41_out1 =
            mul27_out1.add((constant144_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish14_out1 = burn::tensor::activation::hard_swish(add41_out1);
        let constant145_out1 = self.constant145.val();
        let mul28_out1 = (constant145_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish14_out1);
        let constant146_out1 = self.constant146.val();
        let add42_out1 =
            mul28_out1.add((constant146_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d16_out1 = crate::model_arch::conv_fwd(&self.conv2d16, add42_out1);
        let reshape15_out1 = self.reshape15.val();
        let add43_out1 = conv2d16_out1.add(reshape15_out1);
        let constant147_out1 = self.constant147.val();
        let mul29_out1 = (constant147_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add43_out1);
        let constant148_out1 = self.constant148.val();
        let add44_out1 =
            mul29_out1.add((constant148_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish15_out1 = burn::tensor::activation::hard_swish(add44_out1);
        let constant149_out1 = self.constant149.val();
        let mul30_out1 = (constant149_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish15_out1);
        let constant150_out1 = self.constant150.val();
        let add45_out1 =
            mul30_out1.add((constant150_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d17_out1 = crate::model_arch::conv_fwd(&self.conv2d17, add45_out1);
        let reshape16_out1 = self.reshape16.val();
        let add46_out1 = conv2d17_out1.add(reshape16_out1);
        let constant153_out1 = self.constant153.val();
        let mul31_out1 = (constant153_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add46_out1);
        let constant154_out1 = self.constant154.val();
        let add47_out1 =
            mul31_out1.add((constant154_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish16_out1 = burn::tensor::activation::hard_swish(add47_out1);
        let constant155_out1 = self.constant155.val();
        let mul32_out1 = (constant155_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish16_out1);
        let constant156_out1 = self.constant156.val();
        let add48_out1 =
            mul32_out1.add((constant156_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        add48_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    conv2d18: Conv2d<B>,
    reshape17: burn::module::Param<Tensor<B, 4>>,
    constant157: burn::module::Param<Tensor<B, 1>>,
    constant158: burn::module::Param<Tensor<B, 1>>,
    constant159: burn::module::Param<Tensor<B, 1>>,
    constant160: burn::module::Param<Tensor<B, 1>>,
    conv2d19: Conv2d<B>,
    reshape18: burn::module::Param<Tensor<B, 4>>,
    constant161: burn::module::Param<Tensor<B, 1>>,
    constant162: burn::module::Param<Tensor<B, 1>>,
    constant163: burn::module::Param<Tensor<B, 1>>,
    constant164: burn::module::Param<Tensor<B, 1>>,
    conv2d20: Conv2d<B>,
    reshape19: burn::module::Param<Tensor<B, 4>>,
    constant165: burn::module::Param<Tensor<B, 1>>,
    constant166: burn::module::Param<Tensor<B, 1>>,
    constant167: burn::module::Param<Tensor<B, 1>>,
    constant168: burn::module::Param<Tensor<B, 1>>,
    conv2d21: Conv2d<B>,
    reshape20: burn::module::Param<Tensor<B, 4>>,
    constant169: burn::module::Param<Tensor<B, 1>>,
    constant170: burn::module::Param<Tensor<B, 1>>,
    constant171: burn::module::Param<Tensor<B, 1>>,
    constant172: burn::module::Param<Tensor<B, 1>>,
    conv2d22: Conv2d<B>,
    reshape21: burn::module::Param<Tensor<B, 4>>,
    constant175: burn::module::Param<Tensor<B, 1>>,
    constant176: burn::module::Param<Tensor<B, 1>>,
    constant177: burn::module::Param<Tensor<B, 1>>,
    constant178: burn::module::Param<Tensor<B, 1>>,
    globalaveragepool1: AdaptiveAvgPool2d,
    conv2d23: Conv2d<B>,
    reshape22: burn::module::Param<Tensor<B, 4>>,
    conv2d24: Conv2d<B>,
    reshape23: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d18 = Conv2dConfig::new([240, 240], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(false)
            .init(device);
        let reshape17: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant157: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant158: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant159: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant160: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d19 = Conv2dConfig::new([240, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape18: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant161: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant162: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant163: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant164: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d20 = Conv2dConfig::new([240, 240], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(false)
            .init(device);
        let reshape19: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant165: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant166: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant167: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant168: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d21 = Conv2dConfig::new([240, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape20: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant169: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant170: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant171: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant172: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d22 = Conv2dConfig::new([240, 240], [5, 5])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(false)
            .init(device);
        let reshape21: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        let constant175: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant176: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant177: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant178: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d23 = Conv2dConfig::new([240, 60], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape22: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 60, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 60, 1, 1].into(),
        );
        let conv2d24 = Conv2dConfig::new([60, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape23: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 240, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 240, 1, 1].into(),
        );
        Self {
            conv2d18,
            reshape17,
            constant157,
            constant158,
            constant159,
            constant160,
            conv2d19,
            reshape18,
            constant161,
            constant162,
            constant163,
            constant164,
            conv2d20,
            reshape19,
            constant165,
            constant166,
            constant167,
            constant168,
            conv2d21,
            reshape20,
            constant169,
            constant170,
            constant171,
            constant172,
            conv2d22,
            reshape21,
            constant175,
            constant176,
            constant177,
            constant178,
            globalaveragepool1,
            conv2d23,
            reshape22,
            conv2d24,
            reshape23,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add48_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d18_out1 = crate::model_arch::conv_fwd(&self.conv2d18, add48_out1);
        let reshape17_out1 = self.reshape17.val();
        let add49_out1 = conv2d18_out1.add(reshape17_out1);
        let constant157_out1 = self.constant157.val();
        let mul33_out1 = (constant157_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add49_out1);
        let constant158_out1 = self.constant158.val();
        let add50_out1 =
            mul33_out1.add((constant158_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish17_out1 = burn::tensor::activation::hard_swish(add50_out1);
        let constant159_out1 = self.constant159.val();
        let mul34_out1 = (constant159_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish17_out1);
        let constant160_out1 = self.constant160.val();
        let add51_out1 =
            mul34_out1.add((constant160_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d19_out1 = crate::model_arch::conv_fwd(&self.conv2d19, add51_out1);
        let reshape18_out1 = self.reshape18.val();
        let add52_out1 = conv2d19_out1.add(reshape18_out1);
        let constant161_out1 = self.constant161.val();
        let mul35_out1 = (constant161_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add52_out1);
        let constant162_out1 = self.constant162.val();
        let add53_out1 =
            mul35_out1.add((constant162_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish18_out1 = burn::tensor::activation::hard_swish(add53_out1);
        let constant163_out1 = self.constant163.val();
        let mul36_out1 = (constant163_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish18_out1);
        let constant164_out1 = self.constant164.val();
        let add54_out1 =
            mul36_out1.add((constant164_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d20_out1 = crate::model_arch::conv_fwd(&self.conv2d20, add54_out1);
        let reshape19_out1 = self.reshape19.val();
        let add55_out1 = conv2d20_out1.add(reshape19_out1);
        let constant165_out1 = self.constant165.val();
        let mul37_out1 = (constant165_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add55_out1);
        let constant166_out1 = self.constant166.val();
        let add56_out1 =
            mul37_out1.add((constant166_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish19_out1 = burn::tensor::activation::hard_swish(add56_out1);
        let constant167_out1 = self.constant167.val();
        let mul38_out1 = (constant167_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish19_out1);
        let constant168_out1 = self.constant168.val();
        let add57_out1 =
            mul38_out1.add((constant168_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d21_out1 = crate::model_arch::conv_fwd(&self.conv2d21, add57_out1);
        let reshape20_out1 = self.reshape20.val();
        let add58_out1 = conv2d21_out1.add(reshape20_out1);
        let constant169_out1 = self.constant169.val();
        let mul39_out1 = (constant169_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add58_out1);
        let constant170_out1 = self.constant170.val();
        let add59_out1 =
            mul39_out1.add((constant170_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish20_out1 = burn::tensor::activation::hard_swish(add59_out1);
        let constant171_out1 = self.constant171.val();
        let mul40_out1 = (constant171_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish20_out1);
        let constant172_out1 = self.constant172.val();
        let add60_out1 =
            mul40_out1.add((constant172_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d22_out1 = crate::model_arch::conv_fwd(&self.conv2d22, add60_out1);
        let reshape21_out1 = self.reshape21.val();
        let add61_out1 = conv2d22_out1.add(reshape21_out1);
        let constant175_out1 = self.constant175.val();
        let mul41_out1 = (constant175_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add61_out1);
        let constant176_out1 = self.constant176.val();
        let add62_out1 =
            mul41_out1.add((constant176_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish21_out1 = burn::tensor::activation::hard_swish(add62_out1);
        let constant177_out1 = self.constant177.val();
        let mul42_out1 = (constant177_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish21_out1);
        let constant178_out1 = self.constant178.val();
        let add63_out1 =
            mul42_out1.add((constant178_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(add63_out1.clone());
        let conv2d23_out1 = crate::model_arch::conv_fwd(&self.conv2d23, globalaveragepool1_out1);
        let reshape22_out1 = self.reshape22.val();
        let add64_out1 = conv2d23_out1.add(reshape22_out1);
        let relu1_out1 = burn::tensor::activation::relu(add64_out1);
        let conv2d24_out1 = crate::model_arch::conv_fwd(&self.conv2d24, relu1_out1);
        let reshape23_out1 = self.reshape23.val();
        let add65_out1 = conv2d24_out1.add(reshape23_out1);
        let hardsigmoid1_out1 =
            burn::tensor::activation::hard_sigmoid(add65_out1, 0.16666670143604279, 0.5);
        let mul43_out1 = add63_out1.mul(hardsigmoid1_out1);
        mul43_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5<B: Backend> {
    conv2d25: Conv2d<B>,
    reshape24: burn::module::Param<Tensor<B, 4>>,
    constant179: burn::module::Param<Tensor<B, 1>>,
    constant180: burn::module::Param<Tensor<B, 1>>,
    constant181: burn::module::Param<Tensor<B, 1>>,
    constant182: burn::module::Param<Tensor<B, 1>>,
    conv2d26: Conv2d<B>,
    reshape25: burn::module::Param<Tensor<B, 4>>,
    constant183: burn::module::Param<Tensor<B, 1>>,
    constant184: burn::module::Param<Tensor<B, 1>>,
    constant185: burn::module::Param<Tensor<B, 1>>,
    constant186: burn::module::Param<Tensor<B, 1>>,
    globalaveragepool2: AdaptiveAvgPool2d,
    conv2d27: Conv2d<B>,
    reshape26: burn::module::Param<Tensor<B, 4>>,
    conv2d28: Conv2d<B>,
    reshape27: burn::module::Param<Tensor<B, 4>>,
    conv2d29: Conv2d<B>,
    reshape28: burn::module::Param<Tensor<B, 4>>,
    constant187: burn::module::Param<Tensor<B, 1>>,
    constant188: burn::module::Param<Tensor<B, 1>>,
    constant189: burn::module::Param<Tensor<B, 1>>,
    constant190: burn::module::Param<Tensor<B, 1>>,
    conv2d30: Conv2d<B>,
    reshape29: burn::module::Param<Tensor<B, 4>>,
    constant191: burn::module::Param<Tensor<B, 1>>,
    constant192: burn::module::Param<Tensor<B, 1>>,
    constant193: burn::module::Param<Tensor<B, 1>>,
    constant194: burn::module::Param<Tensor<B, 1>>,
    conv2d31: Conv2d<B>,
    reshape30: burn::module::Param<Tensor<B, 4>>,
    constant197: burn::module::Param<Tensor<B, 1>>,
    constant198: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule5<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d25 = Conv2dConfig::new([240, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape24: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let constant179: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant180: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant181: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant182: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d26 = Conv2dConfig::new([480, 480], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(false)
            .init(device);
        let reshape25: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let constant183: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant184: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant185: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant186: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let globalaveragepool2 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d27 = Conv2dConfig::new([480, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape26: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 120, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 120, 1, 1].into(),
        );
        let conv2d28 = Conv2dConfig::new([120, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape27: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let conv2d29 = Conv2dConfig::new([480, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape28: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let constant187: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant188: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant189: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant190: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d30 = Conv2dConfig::new([480, 480], [5, 5])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(false)
            .init(device);
        let reshape29: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let constant191: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant192: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant193: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant194: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d31 = Conv2dConfig::new([480, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape30: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let constant197: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant198: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            conv2d25,
            reshape24,
            constant179,
            constant180,
            constant181,
            constant182,
            conv2d26,
            reshape25,
            constant183,
            constant184,
            constant185,
            constant186,
            globalaveragepool2,
            conv2d27,
            reshape26,
            conv2d28,
            reshape27,
            conv2d29,
            reshape28,
            constant187,
            constant188,
            constant189,
            constant190,
            conv2d30,
            reshape29,
            constant191,
            constant192,
            constant193,
            constant194,
            conv2d31,
            reshape30,
            constant197,
            constant198,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, mul43_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d25_out1 = crate::model_arch::conv_fwd(&self.conv2d25, mul43_out1);
        let reshape24_out1 = self.reshape24.val();
        let add66_out1 = conv2d25_out1.add(reshape24_out1);
        let constant179_out1 = self.constant179.val();
        let mul44_out1 = (constant179_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add66_out1);
        let constant180_out1 = self.constant180.val();
        let add67_out1 =
            mul44_out1.add((constant180_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish22_out1 = burn::tensor::activation::hard_swish(add67_out1);
        let constant181_out1 = self.constant181.val();
        let mul45_out1 = (constant181_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish22_out1);
        let constant182_out1 = self.constant182.val();
        let add68_out1 =
            mul45_out1.add((constant182_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d26_out1 = crate::model_arch::conv_fwd(&self.conv2d26, add68_out1);
        let reshape25_out1 = self.reshape25.val();
        let add69_out1 = conv2d26_out1.add(reshape25_out1);
        let constant183_out1 = self.constant183.val();
        let mul46_out1 = (constant183_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add69_out1);
        let constant184_out1 = self.constant184.val();
        let add70_out1 =
            mul46_out1.add((constant184_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish23_out1 = burn::tensor::activation::hard_swish(add70_out1);
        let constant185_out1 = self.constant185.val();
        let mul47_out1 = (constant185_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish23_out1);
        let constant186_out1 = self.constant186.val();
        let add71_out1 =
            mul47_out1.add((constant186_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool2_out1 = self.globalaveragepool2.forward(add71_out1.clone());
        let conv2d27_out1 = crate::model_arch::conv_fwd(&self.conv2d27, globalaveragepool2_out1);
        let reshape26_out1 = self.reshape26.val();
        let add72_out1 = conv2d27_out1.add(reshape26_out1);
        let relu2_out1 = burn::tensor::activation::relu(add72_out1);
        let conv2d28_out1 = crate::model_arch::conv_fwd(&self.conv2d28, relu2_out1);
        let reshape27_out1 = self.reshape27.val();
        let add73_out1 = conv2d28_out1.add(reshape27_out1);
        let hardsigmoid2_out1 =
            burn::tensor::activation::hard_sigmoid(add73_out1, 0.16666670143604279, 0.5);
        let mul48_out1 = add71_out1.mul(hardsigmoid2_out1);
        let conv2d29_out1 = crate::model_arch::conv_fwd(&self.conv2d29, mul48_out1);
        let reshape28_out1 = self.reshape28.val();
        let add74_out1 = conv2d29_out1.add(reshape28_out1);
        let constant187_out1 = self.constant187.val();
        let mul49_out1 = (constant187_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add74_out1);
        let constant188_out1 = self.constant188.val();
        let add75_out1 =
            mul49_out1.add((constant188_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish24_out1 = burn::tensor::activation::hard_swish(add75_out1);
        let constant189_out1 = self.constant189.val();
        let mul50_out1 = (constant189_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish24_out1);
        let constant190_out1 = self.constant190.val();
        let add76_out1 =
            mul50_out1.add((constant190_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d30_out1 = crate::model_arch::conv_fwd(&self.conv2d30, add76_out1);
        let reshape29_out1 = self.reshape29.val();
        let add77_out1 = conv2d30_out1.add(reshape29_out1);
        let constant191_out1 = self.constant191.val();
        let mul51_out1 = (constant191_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add77_out1);
        let constant192_out1 = self.constant192.val();
        let add78_out1 =
            mul51_out1.add((constant192_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish25_out1 = burn::tensor::activation::hard_swish(add78_out1);
        let constant193_out1 = self.constant193.val();
        let mul52_out1 = (constant193_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish25_out1);
        let constant194_out1 = self.constant194.val();
        let add79_out1 =
            mul52_out1.add((constant194_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d31_out1 = crate::model_arch::conv_fwd(&self.conv2d31, add79_out1);
        let reshape30_out1 = self.reshape30.val();
        let add80_out1 = conv2d31_out1.add(reshape30_out1);
        let constant197_out1 = self.constant197.val();
        let mul53_out1 = (constant197_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add80_out1);
        let constant198_out1 = self.constant198.val();
        let add81_out1 =
            mul53_out1.add((constant198_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        add81_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6<B: Backend> {
    constant199: burn::module::Param<Tensor<B, 1>>,
    constant200: burn::module::Param<Tensor<B, 1>>,
    conv2d32: Conv2d<B>,
    reshape31: burn::module::Param<Tensor<B, 4>>,
    constant201: burn::module::Param<Tensor<B, 1>>,
    constant202: burn::module::Param<Tensor<B, 1>>,
    constant203: burn::module::Param<Tensor<B, 1>>,
    constant204: burn::module::Param<Tensor<B, 1>>,
    conv2d33: Conv2d<B>,
    reshape32: burn::module::Param<Tensor<B, 4>>,
    constant205: burn::module::Param<Tensor<B, 1>>,
    constant206: burn::module::Param<Tensor<B, 1>>,
    constant207: burn::module::Param<Tensor<B, 1>>,
    constant208: burn::module::Param<Tensor<B, 1>>,
    averagepool2d1: AvgPool2d,
    conv2d34: Conv2d<B>,
    batchnormalization2: BatchNorm<B>,
    conv2d35: Conv2d<B>,
    batchnormalization3: BatchNorm<B>,
    constant280: burn::module::Param<Tensor<B, 1>>,
    constant279: burn::module::Param<Tensor<B, 1>>,
    reshape34: burn::module::Param<Tensor<B, 1>>,
    reshape35: burn::module::Param<Tensor<B, 1>>,
    linear1: Linear<B>,
    constant291: burn::module::Param<Tensor<B, 1>>,
    linear2: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule6<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant199: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant200: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d32 = Conv2dConfig::new([480, 480], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(false)
            .init(device);
        let reshape31: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let constant201: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant202: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant203: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant204: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d33 = Conv2dConfig::new([480, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape32: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 480, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 480, 1, 1].into(),
        );
        let constant205: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant206: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant207: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant208: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let averagepool2d1 = AvgPool2dConfig::new([3, 2])
            .with_strides([3, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv2d34 = Conv2dConfig::new([480, 60], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(60)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d35 = Conv2dConfig::new([60, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(120)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let constant280: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant279: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let reshape34: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let reshape35: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let linear1 = LinearConfig::new(120, 360).with_bias(true).init(device);
        let constant291: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let linear2 = LinearConfig::new(120, 120).with_bias(true).init(device);
        Self {
            constant199,
            constant200,
            conv2d32,
            reshape31,
            constant201,
            constant202,
            constant203,
            constant204,
            conv2d33,
            reshape32,
            constant205,
            constant206,
            constant207,
            constant208,
            averagepool2d1,
            conv2d34,
            batchnormalization2,
            conv2d35,
            batchnormalization3,
            constant280,
            constant279,
            reshape34,
            reshape35,
            linear1,
            constant291,
            linear2,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add81_out1: Tensor<B, 4>) -> (Tensor<B, 3>, i64, Tensor<B, 4>) {
        let hardswish26_out1 = burn::tensor::activation::hard_swish(add81_out1);
        let constant199_out1 = self.constant199.val();
        let mul54_out1 = (constant199_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish26_out1);
        let constant200_out1 = self.constant200.val();
        let add82_out1 =
            mul54_out1.add((constant200_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d32_out1 = crate::model_arch::conv_fwd(&self.conv2d32, add82_out1);
        let reshape31_out1 = self.reshape31.val();
        let add83_out1 = conv2d32_out1.add(reshape31_out1);
        let constant201_out1 = self.constant201.val();
        let mul55_out1 = (constant201_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add83_out1);
        let constant202_out1 = self.constant202.val();
        let add84_out1 =
            mul55_out1.add((constant202_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish27_out1 = burn::tensor::activation::hard_swish(add84_out1);
        let constant203_out1 = self.constant203.val();
        let mul56_out1 = (constant203_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish27_out1);
        let constant204_out1 = self.constant204.val();
        let add85_out1 =
            mul56_out1.add((constant204_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d33_out1 = crate::model_arch::conv_fwd(&self.conv2d33, add85_out1);
        let reshape32_out1 = self.reshape32.val();
        let add86_out1 = conv2d33_out1.add(reshape32_out1);
        let constant205_out1 = self.constant205.val();
        let mul57_out1 = (constant205_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add86_out1);
        let constant206_out1 = self.constant206.val();
        let add87_out1 =
            mul57_out1.add((constant206_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish28_out1 = burn::tensor::activation::hard_swish(add87_out1);
        let constant207_out1 = self.constant207.val();
        let mul58_out1 = (constant207_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish28_out1);
        let constant208_out1 = self.constant208.val();
        let add88_out1 =
            mul58_out1.add((constant208_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let averagepool2d1_out1 = self.averagepool2d1.forward(add88_out1);
        let unsqueeze1_out1: Tensor<B, 5> = averagepool2d1_out1.unsqueeze_dims::<5>(&[0]);
        let squeeze1_out1 = unsqueeze1_out1.squeeze_dims::<4>(&[0]);
        let conv2d34_out1 = crate::model_arch::conv_fwd(&self.conv2d34, squeeze1_out1.clone());
        let batchnormalization2_out1 = self.batchnormalization2.forward(conv2d34_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(batchnormalization2_out1.clone());
        let mul59_out1 = batchnormalization2_out1.mul(sigmoid1_out1);
        let conv2d35_out1 = crate::model_arch::conv_fwd(&self.conv2d35, mul59_out1);
        let batchnormalization3_out1 = self.batchnormalization3.forward(conv2d35_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(batchnormalization3_out1.clone());
        let mul60_out1 = batchnormalization3_out1.mul(sigmoid2_out1);
        let shape1_out1: [i64; 4] = {
            let axes = &mul60_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 1] = shape1_out1[3..4].try_into().unwrap();
        let squeeze2_out1 = slice1_out1[0] as i64;
        let slice2_out1: [i64; 2] = shape1_out1[0..2].try_into().unwrap();
        let constant275_out1: [i64; 1] = [-1i64];
        let concat1_out1: [i64; 3usize] = [&slice2_out1[..], &constant275_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape33_out1 = mul60_out1.reshape(concat1_out1);
        let transpose1_out1 = reshape33_out1.permute([0, 2, 1]);
        let reducemean1_out1 = { transpose1_out1.clone().mean_dim(2usize) };
        let sub1_out1 = transpose1_out1.clone().sub(reducemean1_out1);
        let constant280_out1 = self.constant280.val();
        let pow1_out1 = sub1_out1
            .clone()
            .powf((constant280_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean2_out1 = { pow1_out1.mean_dim(2usize) };
        let constant279_out1 = self.constant279.val();
        let add90_out1 = reducemean2_out1.add((constant279_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt1_out1 = add90_out1.sqrt();
        let div1_out1 = sub1_out1.div(sqrt1_out1);
        let reshape34_out1 = self.reshape34.val();
        let mul61_out1 = div1_out1.mul((reshape34_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reshape35_out1 = self.reshape35.val();
        let add91_out1 = mul61_out1.add((reshape35_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear1_out1 = self.linear1.forward(add91_out1);
        let reshape36_out1 = linear1_out1.reshape([0, -1, 3, 8, 15]);
        let transpose2_out1 = reshape36_out1.permute([2, 0, 3, 1, 4]);
        let slice3_out1 = transpose2_out1.clone().slice(s![0..1, .., .., .., ..]);
        let squeeze3_out1 = slice3_out1.squeeze_dims::<4>(&[0]);
        let constant291_out1 = self.constant291.val();
        let mul62_out1 = squeeze3_out1
            .clone()
            .mul((constant291_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let shape3_out1: [i64; 4] = {
            let axes = &squeeze3_out1.dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let reshape37_out1 = mul62_out1.reshape(shape3_out1);
        let slice4_out1 = transpose2_out1.clone().slice(s![1..2, .., .., .., ..]);
        let squeeze4_out1 = slice4_out1.squeeze_dims::<4>(&[0]);
        let slice5_out1 = transpose2_out1.slice(s![2..3, .., .., .., ..]);
        let squeeze5_out1 = slice5_out1.squeeze_dims::<4>(&[0]);
        let (matmul3_out1,) = {
            let q = reshape37_out1;
            let k = squeeze4_out1;
            let v = squeeze5_out1;
            let matmul3_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul3_out1,)
        };
        let transpose4_out1 = matmul3_out1.permute([0, 2, 1, 3]);
        let reshape38_out1 = transpose4_out1.reshape([0, -1, 120]);
        let linear2_out1 = self.linear2.forward(reshape38_out1);
        let add92_out1 = transpose1_out1.add(linear2_out1);
        (add92_out1, squeeze2_out1, squeeze1_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule7<B: Backend> {
    constant304: burn::module::Param<Tensor<B, 1>>,
    constant303: burn::module::Param<Tensor<B, 1>>,
    reshape39: burn::module::Param<Tensor<B, 1>>,
    reshape40: burn::module::Param<Tensor<B, 1>>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    constant310: burn::module::Param<Tensor<B, 1>>,
    constant309: burn::module::Param<Tensor<B, 1>>,
    reshape41: burn::module::Param<Tensor<B, 1>>,
    reshape42: burn::module::Param<Tensor<B, 1>>,
    linear5: Linear<B>,
    constant321: burn::module::Param<Tensor<B, 1>>,
    linear6: Linear<B>,
    constant334: burn::module::Param<Tensor<B, 1>>,
    constant333: burn::module::Param<Tensor<B, 1>>,
    reshape46: burn::module::Param<Tensor<B, 1>>,
    reshape47: burn::module::Param<Tensor<B, 1>>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    constant340: burn::module::Param<Tensor<B, 1>>,
    constant339: burn::module::Param<Tensor<B, 1>>,
    reshape48: burn::module::Param<Tensor<B, 1>>,
    reshape49: burn::module::Param<Tensor<B, 1>>,
    conv2d36: Conv2d<B>,
    batchnormalization4: BatchNorm<B>,
    conv2d37: Conv2d<B>,
    batchnormalization5: BatchNorm<B>,
    conv2d38: Conv2d<B>,
    batchnormalization6: BatchNorm<B>,
    linear9: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule7<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant304: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant303: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let reshape39: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let reshape40: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant310: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant309: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let reshape41: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let reshape42: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let linear5 = LinearConfig::new(120, 360).with_bias(true).init(device);
        let constant321: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let linear6 = LinearConfig::new(120, 120).with_bias(true).init(device);
        let constant334: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant333: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let reshape46: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let reshape47: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant340: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant339: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let reshape48: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let reshape49: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([120], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [120].into(),
        );
        let conv2d36 = Conv2dConfig::new([120, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(480)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d37 = Conv2dConfig::new([960, 60], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(60)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d38 = Conv2dConfig::new([60, 120], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(120)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let linear9 = LinearConfig::new(120, 18385).with_bias(true).init(device);
        Self {
            constant304,
            constant303,
            reshape39,
            reshape40,
            linear3,
            linear4,
            constant310,
            constant309,
            reshape41,
            reshape42,
            linear5,
            constant321,
            linear6,
            constant334,
            constant333,
            reshape46,
            reshape47,
            linear7,
            linear8,
            constant340,
            constant339,
            reshape48,
            reshape49,
            conv2d36,
            batchnormalization4,
            conv2d37,
            batchnormalization5,
            conv2d38,
            batchnormalization6,
            linear9,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add92_out1: Tensor<B, 3>,
        squeeze2_out1: i64,
        squeeze1_out1: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let reducemean3_out1 = { add92_out1.clone().mean_dim(2usize) };
        let sub2_out1 = add92_out1.clone().sub(reducemean3_out1);
        let constant304_out1 = self.constant304.val();
        let pow2_out1 = sub2_out1
            .clone()
            .powf((constant304_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean4_out1 = { pow2_out1.mean_dim(2usize) };
        let constant303_out1 = self.constant303.val();
        let add93_out1 = reducemean4_out1.add((constant303_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt2_out1 = add93_out1.sqrt();
        let div2_out1 = sub2_out1.div(sqrt2_out1);
        let reshape39_out1 = self.reshape39.val();
        let mul63_out1 = div2_out1.mul((reshape39_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reshape40_out1 = self.reshape40.val();
        let add94_out1 = mul63_out1.add((reshape40_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear3_out1 = self.linear3.forward(add94_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(linear3_out1.clone());
        let mul64_out1 = linear3_out1.mul(sigmoid3_out1);
        let linear4_out1 = self.linear4.forward(mul64_out1);
        let add95_out1 = add92_out1.add(linear4_out1);
        let reducemean5_out1 = { add95_out1.clone().mean_dim(2usize) };
        let sub3_out1 = add95_out1.clone().sub(reducemean5_out1);
        let constant310_out1 = self.constant310.val();
        let pow3_out1 = sub3_out1
            .clone()
            .powf((constant310_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean6_out1 = { pow3_out1.mean_dim(2usize) };
        let constant309_out1 = self.constant309.val();
        let add96_out1 = reducemean6_out1.add((constant309_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt3_out1 = add96_out1.sqrt();
        let div3_out1 = sub3_out1.div(sqrt3_out1);
        let reshape41_out1 = self.reshape41.val();
        let mul65_out1 = div3_out1.mul((reshape41_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reshape42_out1 = self.reshape42.val();
        let add97_out1 = mul65_out1.add((reshape42_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear5_out1 = self.linear5.forward(add97_out1);
        let reshape43_out1 = linear5_out1.reshape([0, -1, 3, 8, 15]);
        let transpose5_out1 = reshape43_out1.permute([2, 0, 3, 1, 4]);
        let slice6_out1 = transpose5_out1.clone().slice(s![0..1, .., .., .., ..]);
        let squeeze6_out1 = slice6_out1.squeeze_dims::<4>(&[0]);
        let constant321_out1 = self.constant321.val();
        let mul66_out1 = squeeze6_out1
            .clone()
            .mul((constant321_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let shape4_out1: [i64; 4] = {
            let axes = &squeeze6_out1.dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let reshape44_out1 = mul66_out1.reshape(shape4_out1);
        let slice7_out1 = transpose5_out1.clone().slice(s![1..2, .., .., .., ..]);
        let squeeze7_out1 = slice7_out1.squeeze_dims::<4>(&[0]);
        let slice8_out1 = transpose5_out1.slice(s![2..3, .., .., .., ..]);
        let squeeze8_out1 = slice8_out1.squeeze_dims::<4>(&[0]);
        let (matmul9_out1,) = {
            let q = reshape44_out1;
            let k = squeeze7_out1;
            let v = squeeze8_out1;
            let matmul9_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul9_out1,)
        };
        let transpose7_out1 = matmul9_out1.permute([0, 2, 1, 3]);
        let reshape45_out1 = transpose7_out1.reshape([0, -1, 120]);
        let linear6_out1 = self.linear6.forward(reshape45_out1);
        let add98_out1 = add95_out1.add(linear6_out1);
        let reducemean7_out1 = { add98_out1.clone().mean_dim(2usize) };
        let sub4_out1 = add98_out1.clone().sub(reducemean7_out1);
        let constant334_out1 = self.constant334.val();
        let pow4_out1 = sub4_out1
            .clone()
            .powf((constant334_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean8_out1 = { pow4_out1.mean_dim(2usize) };
        let constant333_out1 = self.constant333.val();
        let add99_out1 = reducemean8_out1.add((constant333_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt4_out1 = add99_out1.sqrt();
        let div4_out1 = sub4_out1.div(sqrt4_out1);
        let reshape46_out1 = self.reshape46.val();
        let mul67_out1 = div4_out1.mul((reshape46_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reshape47_out1 = self.reshape47.val();
        let add100_out1 = mul67_out1.add((reshape47_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear7_out1 = self.linear7.forward(add100_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(linear7_out1.clone());
        let mul68_out1 = linear7_out1.mul(sigmoid4_out1);
        let linear8_out1 = self.linear8.forward(mul68_out1);
        let add101_out1 = add98_out1.add(linear8_out1);
        let reducemean9_out1 = { add101_out1.clone().mean_dim(2usize) };
        let sub5_out1 = add101_out1.sub(reducemean9_out1);
        let constant340_out1 = self.constant340.val();
        let pow5_out1 = sub5_out1
            .clone()
            .powf((constant340_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean10_out1 = { pow5_out1.mean_dim(2usize) };
        let constant339_out1 = self.constant339.val();
        let add102_out1 =
            reducemean10_out1.add((constant339_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt5_out1 = add102_out1.sqrt();
        let div5_out1 = sub5_out1.div(sqrt5_out1);
        let reshape48_out1 = self.reshape48.val();
        let mul69_out1 = div5_out1.mul((reshape48_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reshape49_out1 = self.reshape49.val();
        let add103_out1 = mul69_out1.add((reshape49_out1).unsqueeze_dims(&[0isize, 1isize]));
        let unsqueeze4_out1 = [squeeze2_out1 as i64];
        let unsqueeze5_out1: [i64; 1] = [120i64];
        let unsqueeze2_out1: [i64; 1] = [0i64];
        let unsqueeze3_out1: [i64; 1] = [1i64];
        let concat2_out1: [i64; 4usize] = [
            &unsqueeze2_out1[..],
            &unsqueeze3_out1[..],
            &unsqueeze4_out1[..],
            &unsqueeze5_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape50_out1 = add103_out1.reshape(concat2_out1);
        let transpose8_out1 = reshape50_out1.permute([0, 3, 1, 2]);
        let conv2d36_out1 = crate::model_arch::conv_fwd(&self.conv2d36, transpose8_out1);
        let batchnormalization4_out1 = self.batchnormalization4.forward(conv2d36_out1);
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(batchnormalization4_out1.clone());
        let mul70_out1 = batchnormalization4_out1.mul(sigmoid5_out1);
        let concat3_out1 = burn::tensor::Tensor::cat([squeeze1_out1, mul70_out1].into(), 1);
        let conv2d37_out1 = crate::model_arch::conv_fwd(&self.conv2d37, concat3_out1);
        let batchnormalization5_out1 = self.batchnormalization5.forward(conv2d37_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(batchnormalization5_out1.clone());
        let mul71_out1 = batchnormalization5_out1.mul(sigmoid6_out1);
        let conv2d38_out1 = crate::model_arch::conv_fwd(&self.conv2d38, mul71_out1);
        let batchnormalization6_out1 = self.batchnormalization6.forward(conv2d38_out1);
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(batchnormalization6_out1.clone());
        let mul72_out1 = batchnormalization6_out1.mul(sigmoid7_out1);
        let squeeze9_out1 = mul72_out1.squeeze_dims::<3>(&[2]);
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
    submodule5: Submodule5<B>,
    submodule6: Submodule6<B>,
    submodule7: Submodule7<B>,
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
        let submodule4 = Submodule4::new(device);
        let submodule5 = Submodule5::new(device);
        let submodule6 = Submodule6::new(device);
        let submodule7 = Submodule7::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            submodule6,
            submodule7,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let add16_out1 = self.submodule1.forward(x);
        let add32_out1 = self.submodule2.forward(add16_out1);
        let add48_out1 = self.submodule3.forward(add32_out1);
        let mul43_out1 = self.submodule4.forward(add48_out1);
        let add81_out1 = self.submodule5.forward(mul43_out1);
        let (add92_out1, squeeze2_out1, squeeze1_out1) = self.submodule6.forward(add81_out1);
        let softmax3_out1 = self
            .submodule7
            .forward(add92_out1, squeeze2_out1, squeeze1_out1);
        softmax3_out1
    }
}
