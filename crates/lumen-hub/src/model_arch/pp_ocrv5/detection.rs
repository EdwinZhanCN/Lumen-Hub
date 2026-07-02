// Generated from ONNX "src/ppocrv5/detection.fp32.onnx" by burn-onnx
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
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
    constant131: burn::module::Param<Tensor<B, 1>>,
    constant132: burn::module::Param<Tensor<B, 1>>,
    constant133: burn::module::Param<Tensor<B, 1>>,
    constant134: burn::module::Param<Tensor<B, 1>>,
    conv2d3: Conv2d<B>,
    reshape2: burn::module::Param<Tensor<B, 4>>,
    constant153: burn::module::Param<Tensor<B, 1>>,
    constant154: burn::module::Param<Tensor<B, 1>>,
    constant173: burn::module::Param<Tensor<B, 1>>,
    constant174: burn::module::Param<Tensor<B, 1>>,
    conv2d4: Conv2d<B>,
    reshape3: burn::module::Param<Tensor<B, 4>>,
    constant195: burn::module::Param<Tensor<B, 1>>,
    constant196: burn::module::Param<Tensor<B, 1>>,
    conv2d5: Conv2d<B>,
    reshape4: burn::module::Param<Tensor<B, 4>>,
    constant227: burn::module::Param<Tensor<B, 1>>,
    constant228: burn::module::Param<Tensor<B, 1>>,
    constant229: burn::module::Param<Tensor<B, 1>>,
    constant230: burn::module::Param<Tensor<B, 1>>,
    conv2d6: Conv2d<B>,
    reshape5: burn::module::Param<Tensor<B, 4>>,
    constant231: burn::module::Param<Tensor<B, 1>>,
    constant232: burn::module::Param<Tensor<B, 1>>,
    constant233: burn::module::Param<Tensor<B, 1>>,
    constant234: burn::module::Param<Tensor<B, 1>>,
    conv2d7: Conv2d<B>,
    reshape6: burn::module::Param<Tensor<B, 4>>,
    constant135: burn::module::Param<Tensor<B, 1>>,
    constant136: burn::module::Param<Tensor<B, 1>>,
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
        let conv2d4 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([2, 2])
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
        let conv2d5 = Conv2dConfig::new([32, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape4: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 48, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 48, 1, 1].into(),
        );
        let constant227: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant228: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant229: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant230: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d6 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(48)
            .with_bias(false)
            .init(device);
        let reshape5: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 48, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 48, 1, 1].into(),
        );
        let constant231: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant232: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant233: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant234: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d7 = Conv2dConfig::new([48, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape6: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 48, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 48, 1, 1].into(),
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
        Self {
            conv2d1,
            batchnormalization1,
            conv2d2,
            reshape1,
            constant131,
            constant132,
            constant133,
            constant134,
            conv2d3,
            reshape2,
            constant153,
            constant154,
            constant173,
            constant174,
            conv2d4,
            reshape3,
            constant195,
            constant196,
            conv2d5,
            reshape4,
            constant227,
            constant228,
            constant229,
            constant230,
            conv2d6,
            reshape5,
            constant231,
            constant232,
            constant233,
            constant234,
            conv2d7,
            reshape6,
            constant135,
            constant136,
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
        let constant131_out1 = self.constant131.val();
        let mul1_out1 = (constant131_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add1_out1);
        let constant132_out1 = self.constant132.val();
        let add2_out1 = mul1_out1.add((constant132_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish1_out1 = burn::tensor::activation::hard_swish(add2_out1);
        let constant133_out1 = self.constant133.val();
        let mul2_out1 = (constant133_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish1_out1);
        let constant134_out1 = self.constant134.val();
        let add3_out1 = mul2_out1.add((constant134_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d3_out1 = crate::model_arch::conv_fwd(&self.conv2d3, add3_out1);
        let reshape2_out1 = self.reshape2.val();
        let add4_out1 = conv2d3_out1.add(reshape2_out1);
        let constant153_out1 = self.constant153.val();
        let mul3_out1 = (constant153_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add4_out1);
        let constant154_out1 = self.constant154.val();
        let add5_out1 = mul3_out1.add((constant154_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish2_out1 = burn::tensor::activation::hard_swish(add5_out1);
        let constant173_out1 = self.constant173.val();
        let mul4_out1 = (constant173_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish2_out1);
        let constant174_out1 = self.constant174.val();
        let add6_out1 = mul4_out1.add((constant174_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d4_out1 = crate::model_arch::conv_fwd(&self.conv2d4, add6_out1);
        let reshape3_out1 = self.reshape3.val();
        let add7_out1 = conv2d4_out1.add(reshape3_out1);
        let constant195_out1 = self.constant195.val();
        let mul5_out1 = (constant195_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add7_out1);
        let constant196_out1 = self.constant196.val();
        let add8_out1 = mul5_out1.add((constant196_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d5_out1 = crate::model_arch::conv_fwd(&self.conv2d5, add8_out1);
        let reshape4_out1 = self.reshape4.val();
        let add9_out1 = conv2d5_out1.add(reshape4_out1);
        let constant227_out1 = self.constant227.val();
        let mul6_out1 = (constant227_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add9_out1);
        let constant228_out1 = self.constant228.val();
        let add10_out1 =
            mul6_out1.add((constant228_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish3_out1 = burn::tensor::activation::hard_swish(add10_out1);
        let constant229_out1 = self.constant229.val();
        let mul7_out1 = (constant229_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish3_out1);
        let constant230_out1 = self.constant230.val();
        let add11_out1 =
            mul7_out1.add((constant230_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d6_out1 = crate::model_arch::conv_fwd(&self.conv2d6, add11_out1);
        let reshape5_out1 = self.reshape5.val();
        let add12_out1 = conv2d6_out1.add(reshape5_out1);
        let constant231_out1 = self.constant231.val();
        let mul8_out1 = (constant231_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add12_out1);
        let constant232_out1 = self.constant232.val();
        let add13_out1 =
            mul8_out1.add((constant232_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish4_out1 = burn::tensor::activation::hard_swish(add13_out1);
        let constant233_out1 = self.constant233.val();
        let mul9_out1 = (constant233_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish4_out1);
        let constant234_out1 = self.constant234.val();
        let add14_out1 =
            mul9_out1.add((constant234_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d7_out1 = crate::model_arch::conv_fwd(&self.conv2d7, add14_out1);
        let reshape6_out1 = self.reshape6.val();
        let add15_out1 = conv2d7_out1.add(reshape6_out1);
        let constant135_out1 = self.constant135.val();
        let mul10_out1 = (constant135_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add15_out1);
        let constant136_out1 = self.constant136.val();
        let add16_out1 =
            mul10_out1.add((constant136_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        add16_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    constant137: burn::module::Param<Tensor<B, 1>>,
    constant138: burn::module::Param<Tensor<B, 1>>,
    conv2d8: Conv2d<B>,
    reshape7: burn::module::Param<Tensor<B, 4>>,
    constant139: burn::module::Param<Tensor<B, 1>>,
    constant140: burn::module::Param<Tensor<B, 1>>,
    conv2d9: Conv2d<B>,
    reshape8: burn::module::Param<Tensor<B, 4>>,
    constant141: burn::module::Param<Tensor<B, 1>>,
    constant142: burn::module::Param<Tensor<B, 1>>,
    constant143: burn::module::Param<Tensor<B, 1>>,
    constant144: burn::module::Param<Tensor<B, 1>>,
    conv2d10: Conv2d<B>,
    reshape9: burn::module::Param<Tensor<B, 4>>,
    constant145: burn::module::Param<Tensor<B, 1>>,
    constant146: burn::module::Param<Tensor<B, 1>>,
    constant147: burn::module::Param<Tensor<B, 1>>,
    constant148: burn::module::Param<Tensor<B, 1>>,
    conv2d11: Conv2d<B>,
    reshape10: burn::module::Param<Tensor<B, 4>>,
    constant149: burn::module::Param<Tensor<B, 1>>,
    constant150: burn::module::Param<Tensor<B, 1>>,
    constant151: burn::module::Param<Tensor<B, 1>>,
    constant152: burn::module::Param<Tensor<B, 1>>,
    conv2d12: Conv2d<B>,
    reshape11: burn::module::Param<Tensor<B, 4>>,
    constant155: burn::module::Param<Tensor<B, 1>>,
    constant156: burn::module::Param<Tensor<B, 1>>,
    conv2d13: Conv2d<B>,
    reshape12: burn::module::Param<Tensor<B, 4>>,
    constant157: burn::module::Param<Tensor<B, 1>>,
    constant158: burn::module::Param<Tensor<B, 1>>,
    constant159: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
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
        let conv2d8 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(48)
            .with_bias(false)
            .init(device);
        let reshape7: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 48, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 48, 1, 1].into(),
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
        let conv2d9 = Conv2dConfig::new([48, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape8: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
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
        let conv2d10 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(false)
            .init(device);
        let reshape9: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
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
        let conv2d11 = Conv2dConfig::new([96, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape10: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
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
        let conv2d12 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(false)
            .init(device);
        let reshape11: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
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
        let conv2d13 = Conv2dConfig::new([96, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape12: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        Self {
            constant137,
            constant138,
            conv2d8,
            reshape7,
            constant139,
            constant140,
            conv2d9,
            reshape8,
            constant141,
            constant142,
            constant143,
            constant144,
            conv2d10,
            reshape9,
            constant145,
            constant146,
            constant147,
            constant148,
            conv2d11,
            reshape10,
            constant149,
            constant150,
            constant151,
            constant152,
            conv2d12,
            reshape11,
            constant155,
            constant156,
            conv2d13,
            reshape12,
            constant157,
            constant158,
            constant159,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add16_out1: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let hardswish5_out1 = burn::tensor::activation::hard_swish(add16_out1);
        let constant137_out1 = self.constant137.val();
        let mul11_out1 = (constant137_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish5_out1);
        let constant138_out1 = self.constant138.val();
        let add17_out1 =
            mul11_out1.add((constant138_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d8_out1 = crate::model_arch::conv_fwd(&self.conv2d8, add17_out1.clone());
        let reshape7_out1 = self.reshape7.val();
        let add18_out1 = conv2d8_out1.add(reshape7_out1);
        let constant139_out1 = self.constant139.val();
        let mul12_out1 = (constant139_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add18_out1);
        let constant140_out1 = self.constant140.val();
        let add19_out1 =
            mul12_out1.add((constant140_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d9_out1 = crate::model_arch::conv_fwd(&self.conv2d9, add19_out1);
        let reshape8_out1 = self.reshape8.val();
        let add20_out1 = conv2d9_out1.add(reshape8_out1);
        let constant141_out1 = self.constant141.val();
        let mul13_out1 = (constant141_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add20_out1);
        let constant142_out1 = self.constant142.val();
        let add21_out1 =
            mul13_out1.add((constant142_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish6_out1 = burn::tensor::activation::hard_swish(add21_out1);
        let constant143_out1 = self.constant143.val();
        let mul14_out1 = (constant143_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish6_out1);
        let constant144_out1 = self.constant144.val();
        let add22_out1 =
            mul14_out1.add((constant144_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d10_out1 = crate::model_arch::conv_fwd(&self.conv2d10, add22_out1);
        let reshape9_out1 = self.reshape9.val();
        let add23_out1 = conv2d10_out1.add(reshape9_out1);
        let constant145_out1 = self.constant145.val();
        let mul15_out1 = (constant145_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add23_out1);
        let constant146_out1 = self.constant146.val();
        let add24_out1 =
            mul15_out1.add((constant146_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish7_out1 = burn::tensor::activation::hard_swish(add24_out1);
        let constant147_out1 = self.constant147.val();
        let mul16_out1 = (constant147_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish7_out1);
        let constant148_out1 = self.constant148.val();
        let add25_out1 =
            mul16_out1.add((constant148_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d11_out1 = crate::model_arch::conv_fwd(&self.conv2d11, add25_out1);
        let reshape10_out1 = self.reshape10.val();
        let add26_out1 = conv2d11_out1.add(reshape10_out1);
        let constant149_out1 = self.constant149.val();
        let mul17_out1 = (constant149_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add26_out1);
        let constant150_out1 = self.constant150.val();
        let add27_out1 =
            mul17_out1.add((constant150_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish8_out1 = burn::tensor::activation::hard_swish(add27_out1);
        let constant151_out1 = self.constant151.val();
        let mul18_out1 = (constant151_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish8_out1);
        let constant152_out1 = self.constant152.val();
        let add28_out1 =
            mul18_out1.add((constant152_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d12_out1 = crate::model_arch::conv_fwd(&self.conv2d12, add28_out1.clone());
        let reshape11_out1 = self.reshape11.val();
        let add29_out1 = conv2d12_out1.add(reshape11_out1);
        let constant155_out1 = self.constant155.val();
        let mul19_out1 = (constant155_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add29_out1);
        let constant156_out1 = self.constant156.val();
        let add30_out1 =
            mul19_out1.add((constant156_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d13_out1 = crate::model_arch::conv_fwd(&self.conv2d13, add30_out1);
        let reshape12_out1 = self.reshape12.val();
        let add31_out1 = conv2d13_out1.add(reshape12_out1);
        let constant157_out1 = self.constant157.val();
        let mul20_out1 = (constant157_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add31_out1);
        let constant158_out1 = self.constant158.val();
        let add32_out1 =
            mul20_out1.add((constant158_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish9_out1 = burn::tensor::activation::hard_swish(add32_out1);
        let constant159_out1 = self.constant159.val();
        let mul21_out1 = (constant159_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish9_out1);
        (mul21_out1, add17_out1, add28_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    constant160: burn::module::Param<Tensor<B, 1>>,
    conv2d14: Conv2d<B>,
    reshape13: burn::module::Param<Tensor<B, 4>>,
    constant161: burn::module::Param<Tensor<B, 1>>,
    constant162: burn::module::Param<Tensor<B, 1>>,
    constant163: burn::module::Param<Tensor<B, 1>>,
    constant164: burn::module::Param<Tensor<B, 1>>,
    conv2d15: Conv2d<B>,
    reshape14: burn::module::Param<Tensor<B, 4>>,
    constant165: burn::module::Param<Tensor<B, 1>>,
    constant166: burn::module::Param<Tensor<B, 1>>,
    constant167: burn::module::Param<Tensor<B, 1>>,
    constant168: burn::module::Param<Tensor<B, 1>>,
    conv2d16: Conv2d<B>,
    reshape15: burn::module::Param<Tensor<B, 4>>,
    constant169: burn::module::Param<Tensor<B, 1>>,
    constant170: burn::module::Param<Tensor<B, 1>>,
    constant171: burn::module::Param<Tensor<B, 1>>,
    constant172: burn::module::Param<Tensor<B, 1>>,
    conv2d17: Conv2d<B>,
    reshape16: burn::module::Param<Tensor<B, 4>>,
    constant175: burn::module::Param<Tensor<B, 1>>,
    constant176: burn::module::Param<Tensor<B, 1>>,
    constant177: burn::module::Param<Tensor<B, 1>>,
    constant178: burn::module::Param<Tensor<B, 1>>,
    conv2d18: Conv2d<B>,
    reshape17: burn::module::Param<Tensor<B, 4>>,
    constant179: burn::module::Param<Tensor<B, 1>>,
    constant180: burn::module::Param<Tensor<B, 1>>,
    constant181: burn::module::Param<Tensor<B, 1>>,
    constant182: burn::module::Param<Tensor<B, 1>>,
    conv2d19: Conv2d<B>,
    reshape18: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant160: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d14 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(false)
            .init(device);
        let reshape13: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let conv2d15 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape14: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let conv2d16 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(false)
            .init(device);
        let reshape15: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let conv2d17 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape16: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let conv2d18 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(false)
            .init(device);
        let reshape17: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let conv2d19 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape18: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
        );
        Self {
            constant160,
            conv2d14,
            reshape13,
            constant161,
            constant162,
            constant163,
            constant164,
            conv2d15,
            reshape14,
            constant165,
            constant166,
            constant167,
            constant168,
            conv2d16,
            reshape15,
            constant169,
            constant170,
            constant171,
            constant172,
            conv2d17,
            reshape16,
            constant175,
            constant176,
            constant177,
            constant178,
            conv2d18,
            reshape17,
            constant179,
            constant180,
            constant181,
            constant182,
            conv2d19,
            reshape18,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, mul21_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let constant160_out1 = self.constant160.val();
        let add33_out1 =
            mul21_out1.add((constant160_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d14_out1 = crate::model_arch::conv_fwd(&self.conv2d14, add33_out1);
        let reshape13_out1 = self.reshape13.val();
        let add34_out1 = conv2d14_out1.add(reshape13_out1);
        let constant161_out1 = self.constant161.val();
        let mul22_out1 = (constant161_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add34_out1);
        let constant162_out1 = self.constant162.val();
        let add35_out1 =
            mul22_out1.add((constant162_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish10_out1 = burn::tensor::activation::hard_swish(add35_out1);
        let constant163_out1 = self.constant163.val();
        let mul23_out1 = (constant163_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish10_out1);
        let constant164_out1 = self.constant164.val();
        let add36_out1 =
            mul23_out1.add((constant164_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d15_out1 = crate::model_arch::conv_fwd(&self.conv2d15, add36_out1);
        let reshape14_out1 = self.reshape14.val();
        let add37_out1 = conv2d15_out1.add(reshape14_out1);
        let constant165_out1 = self.constant165.val();
        let mul24_out1 = (constant165_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add37_out1);
        let constant166_out1 = self.constant166.val();
        let add38_out1 =
            mul24_out1.add((constant166_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish11_out1 = burn::tensor::activation::hard_swish(add38_out1);
        let constant167_out1 = self.constant167.val();
        let mul25_out1 = (constant167_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish11_out1);
        let constant168_out1 = self.constant168.val();
        let add39_out1 =
            mul25_out1.add((constant168_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d16_out1 = crate::model_arch::conv_fwd(&self.conv2d16, add39_out1);
        let reshape15_out1 = self.reshape15.val();
        let add40_out1 = conv2d16_out1.add(reshape15_out1);
        let constant169_out1 = self.constant169.val();
        let mul26_out1 = (constant169_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add40_out1);
        let constant170_out1 = self.constant170.val();
        let add41_out1 =
            mul26_out1.add((constant170_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish12_out1 = burn::tensor::activation::hard_swish(add41_out1);
        let constant171_out1 = self.constant171.val();
        let mul27_out1 = (constant171_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish12_out1);
        let constant172_out1 = self.constant172.val();
        let add42_out1 =
            mul27_out1.add((constant172_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d17_out1 = crate::model_arch::conv_fwd(&self.conv2d17, add42_out1);
        let reshape16_out1 = self.reshape16.val();
        let add43_out1 = conv2d17_out1.add(reshape16_out1);
        let constant175_out1 = self.constant175.val();
        let mul28_out1 = (constant175_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add43_out1);
        let constant176_out1 = self.constant176.val();
        let add44_out1 =
            mul28_out1.add((constant176_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish13_out1 = burn::tensor::activation::hard_swish(add44_out1);
        let constant177_out1 = self.constant177.val();
        let mul29_out1 = (constant177_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish13_out1);
        let constant178_out1 = self.constant178.val();
        let add45_out1 =
            mul29_out1.add((constant178_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d18_out1 = crate::model_arch::conv_fwd(&self.conv2d18, add45_out1);
        let reshape17_out1 = self.reshape17.val();
        let add46_out1 = conv2d18_out1.add(reshape17_out1);
        let constant179_out1 = self.constant179.val();
        let mul30_out1 = (constant179_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add46_out1);
        let constant180_out1 = self.constant180.val();
        let add47_out1 =
            mul30_out1.add((constant180_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish14_out1 = burn::tensor::activation::hard_swish(add47_out1);
        let constant181_out1 = self.constant181.val();
        let mul31_out1 = (constant181_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish14_out1);
        let constant182_out1 = self.constant182.val();
        let add48_out1 =
            mul31_out1.add((constant182_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d19_out1 = crate::model_arch::conv_fwd(&self.conv2d19, add48_out1);
        let reshape18_out1 = self.reshape18.val();
        let add49_out1 = conv2d19_out1.add(reshape18_out1);
        add49_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    constant183: burn::module::Param<Tensor<B, 1>>,
    constant184: burn::module::Param<Tensor<B, 1>>,
    constant185: burn::module::Param<Tensor<B, 1>>,
    constant186: burn::module::Param<Tensor<B, 1>>,
    conv2d20: Conv2d<B>,
    reshape19: burn::module::Param<Tensor<B, 4>>,
    constant187: burn::module::Param<Tensor<B, 1>>,
    constant188: burn::module::Param<Tensor<B, 1>>,
    constant189: burn::module::Param<Tensor<B, 1>>,
    constant190: burn::module::Param<Tensor<B, 1>>,
    conv2d21: Conv2d<B>,
    reshape20: burn::module::Param<Tensor<B, 4>>,
    constant191: burn::module::Param<Tensor<B, 1>>,
    constant192: burn::module::Param<Tensor<B, 1>>,
    constant193: burn::module::Param<Tensor<B, 1>>,
    constant194: burn::module::Param<Tensor<B, 1>>,
    conv2d22: Conv2d<B>,
    reshape21: burn::module::Param<Tensor<B, 4>>,
    constant197: burn::module::Param<Tensor<B, 1>>,
    constant198: burn::module::Param<Tensor<B, 1>>,
    globalaveragepool1: AdaptiveAvgPool2d,
    conv2d23: Conv2d<B>,
    reshape22: burn::module::Param<Tensor<B, 4>>,
    conv2d24: Conv2d<B>,
    reshape23: burn::module::Param<Tensor<B, 4>>,
    conv2d25: Conv2d<B>,
    reshape24: burn::module::Param<Tensor<B, 4>>,
    constant199: burn::module::Param<Tensor<B, 1>>,
    constant200: burn::module::Param<Tensor<B, 1>>,
    constant201: burn::module::Param<Tensor<B, 1>>,
    constant202: burn::module::Param<Tensor<B, 1>>,
    conv2d26: Conv2d<B>,
    reshape25: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
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
        let conv2d20 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(false)
            .init(device);
        let reshape19: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let conv2d21 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape20: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let conv2d22 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(false)
            .init(device);
        let reshape21: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
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
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d23 = Conv2dConfig::new([192, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape22: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 48, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 48, 1, 1].into(),
        );
        let conv2d24 = Conv2dConfig::new([48, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape23: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 192, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 192, 1, 1].into(),
        );
        let conv2d25 = Conv2dConfig::new([192, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape24: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
        );
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
        let conv2d26 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(false)
            .init(device);
        let reshape25: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
        );
        Self {
            constant183,
            constant184,
            constant185,
            constant186,
            conv2d20,
            reshape19,
            constant187,
            constant188,
            constant189,
            constant190,
            conv2d21,
            reshape20,
            constant191,
            constant192,
            constant193,
            constant194,
            conv2d22,
            reshape21,
            constant197,
            constant198,
            globalaveragepool1,
            conv2d23,
            reshape22,
            conv2d24,
            reshape23,
            conv2d25,
            reshape24,
            constant199,
            constant200,
            constant201,
            constant202,
            conv2d26,
            reshape25,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add49_out1: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let constant183_out1 = self.constant183.val();
        let mul32_out1 = (constant183_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add49_out1);
        let constant184_out1 = self.constant184.val();
        let add50_out1 =
            mul32_out1.add((constant184_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish15_out1 = burn::tensor::activation::hard_swish(add50_out1);
        let constant185_out1 = self.constant185.val();
        let mul33_out1 = (constant185_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish15_out1);
        let constant186_out1 = self.constant186.val();
        let add51_out1 =
            mul33_out1.add((constant186_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d20_out1 = crate::model_arch::conv_fwd(&self.conv2d20, add51_out1);
        let reshape19_out1 = self.reshape19.val();
        let add52_out1 = conv2d20_out1.add(reshape19_out1);
        let constant187_out1 = self.constant187.val();
        let mul34_out1 = (constant187_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add52_out1);
        let constant188_out1 = self.constant188.val();
        let add53_out1 =
            mul34_out1.add((constant188_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish16_out1 = burn::tensor::activation::hard_swish(add53_out1);
        let constant189_out1 = self.constant189.val();
        let mul35_out1 = (constant189_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish16_out1);
        let constant190_out1 = self.constant190.val();
        let add54_out1 =
            mul35_out1.add((constant190_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d21_out1 = crate::model_arch::conv_fwd(&self.conv2d21, add54_out1);
        let reshape20_out1 = self.reshape20.val();
        let add55_out1 = conv2d21_out1.add(reshape20_out1);
        let constant191_out1 = self.constant191.val();
        let mul36_out1 = (constant191_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add55_out1);
        let constant192_out1 = self.constant192.val();
        let add56_out1 =
            mul36_out1.add((constant192_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish17_out1 = burn::tensor::activation::hard_swish(add56_out1);
        let constant193_out1 = self.constant193.val();
        let mul37_out1 = (constant193_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish17_out1);
        let constant194_out1 = self.constant194.val();
        let add57_out1 =
            mul37_out1.add((constant194_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d22_out1 = crate::model_arch::conv_fwd(&self.conv2d22, add57_out1.clone());
        let reshape21_out1 = self.reshape21.val();
        let add58_out1 = conv2d22_out1.add(reshape21_out1);
        let constant197_out1 = self.constant197.val();
        let mul38_out1 = (constant197_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add58_out1);
        let constant198_out1 = self.constant198.val();
        let add59_out1 =
            mul38_out1.add((constant198_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(add59_out1.clone());
        let conv2d23_out1 = crate::model_arch::conv_fwd(&self.conv2d23, globalaveragepool1_out1);
        let reshape22_out1 = self.reshape22.val();
        let add60_out1 = conv2d23_out1.add(reshape22_out1);
        let relu1_out1 = burn::tensor::activation::relu(add60_out1);
        let conv2d24_out1 = crate::model_arch::conv_fwd(&self.conv2d24, relu1_out1);
        let reshape23_out1 = self.reshape23.val();
        let add61_out1 = conv2d24_out1.add(reshape23_out1);
        let hardsigmoid1_out1 =
            burn::tensor::activation::hard_sigmoid(add61_out1, 0.16666670143604279, 0.5);
        let mul39_out1 = add59_out1.mul(hardsigmoid1_out1);
        let conv2d25_out1 = crate::model_arch::conv_fwd(&self.conv2d25, mul39_out1);
        let reshape24_out1 = self.reshape24.val();
        let add62_out1 = conv2d25_out1.add(reshape24_out1);
        let constant199_out1 = self.constant199.val();
        let mul40_out1 = (constant199_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add62_out1);
        let constant200_out1 = self.constant200.val();
        let add63_out1 =
            mul40_out1.add((constant200_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish18_out1 = burn::tensor::activation::hard_swish(add63_out1);
        let constant201_out1 = self.constant201.val();
        let mul41_out1 = (constant201_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish18_out1);
        let constant202_out1 = self.constant202.val();
        let add64_out1 =
            mul41_out1.add((constant202_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d26_out1 = crate::model_arch::conv_fwd(&self.conv2d26, add64_out1);
        let reshape25_out1 = self.reshape25.val();
        let add65_out1 = conv2d26_out1.add(reshape25_out1);
        (add65_out1, add57_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule5<B: Backend> {
    constant203: burn::module::Param<Tensor<B, 1>>,
    constant204: burn::module::Param<Tensor<B, 1>>,
    constant205: burn::module::Param<Tensor<B, 1>>,
    constant206: burn::module::Param<Tensor<B, 1>>,
    globalaveragepool2: AdaptiveAvgPool2d,
    conv2d27: Conv2d<B>,
    reshape26: burn::module::Param<Tensor<B, 4>>,
    conv2d28: Conv2d<B>,
    reshape27: burn::module::Param<Tensor<B, 4>>,
    conv2d29: Conv2d<B>,
    reshape28: burn::module::Param<Tensor<B, 4>>,
    constant207: burn::module::Param<Tensor<B, 1>>,
    constant208: burn::module::Param<Tensor<B, 1>>,
    constant209: burn::module::Param<Tensor<B, 1>>,
    constant210: burn::module::Param<Tensor<B, 1>>,
    conv2d30: Conv2d<B>,
    reshape29: burn::module::Param<Tensor<B, 4>>,
    constant211: burn::module::Param<Tensor<B, 1>>,
    constant212: burn::module::Param<Tensor<B, 1>>,
    constant213: burn::module::Param<Tensor<B, 1>>,
    constant214: burn::module::Param<Tensor<B, 1>>,
    conv2d31: Conv2d<B>,
    reshape30: burn::module::Param<Tensor<B, 4>>,
    constant215: burn::module::Param<Tensor<B, 1>>,
    constant216: burn::module::Param<Tensor<B, 1>>,
    constant217: burn::module::Param<Tensor<B, 1>>,
    constant218: burn::module::Param<Tensor<B, 1>>,
    conv2d32: Conv2d<B>,
    reshape31: burn::module::Param<Tensor<B, 4>>,
    constant219: burn::module::Param<Tensor<B, 1>>,
    constant220: burn::module::Param<Tensor<B, 1>>,
    constant221: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule5<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
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
        let globalaveragepool2 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d27 = Conv2dConfig::new([384, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape26: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
        );
        let conv2d28 = Conv2dConfig::new([96, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape27: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
        );
        let conv2d29 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape28: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
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
        let conv2d30 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(false)
            .init(device);
        let reshape29: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
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
        let conv2d31 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape30: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
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
        let constant217: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant218: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d32 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(false)
            .init(device);
        let reshape31: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
        );
        let constant219: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant220: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant221: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            constant203,
            constant204,
            constant205,
            constant206,
            globalaveragepool2,
            conv2d27,
            reshape26,
            conv2d28,
            reshape27,
            conv2d29,
            reshape28,
            constant207,
            constant208,
            constant209,
            constant210,
            conv2d30,
            reshape29,
            constant211,
            constant212,
            constant213,
            constant214,
            conv2d31,
            reshape30,
            constant215,
            constant216,
            constant217,
            constant218,
            conv2d32,
            reshape31,
            constant219,
            constant220,
            constant221,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add65_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let constant203_out1 = self.constant203.val();
        let mul42_out1 = (constant203_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add65_out1);
        let constant204_out1 = self.constant204.val();
        let add66_out1 =
            mul42_out1.add((constant204_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish19_out1 = burn::tensor::activation::hard_swish(add66_out1);
        let constant205_out1 = self.constant205.val();
        let mul43_out1 = (constant205_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish19_out1);
        let constant206_out1 = self.constant206.val();
        let add67_out1 =
            mul43_out1.add((constant206_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let globalaveragepool2_out1 = self.globalaveragepool2.forward(add67_out1.clone());
        let conv2d27_out1 = crate::model_arch::conv_fwd(&self.conv2d27, globalaveragepool2_out1);
        let reshape26_out1 = self.reshape26.val();
        let add68_out1 = conv2d27_out1.add(reshape26_out1);
        let relu2_out1 = burn::tensor::activation::relu(add68_out1);
        let conv2d28_out1 = crate::model_arch::conv_fwd(&self.conv2d28, relu2_out1);
        let reshape27_out1 = self.reshape27.val();
        let add69_out1 = conv2d28_out1.add(reshape27_out1);
        let hardsigmoid2_out1 =
            burn::tensor::activation::hard_sigmoid(add69_out1, 0.16666670143604279, 0.5);
        let mul44_out1 = add67_out1.mul(hardsigmoid2_out1);
        let conv2d29_out1 = crate::model_arch::conv_fwd(&self.conv2d29, mul44_out1);
        let reshape28_out1 = self.reshape28.val();
        let add70_out1 = conv2d29_out1.add(reshape28_out1);
        let constant207_out1 = self.constant207.val();
        let mul45_out1 = (constant207_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add70_out1);
        let constant208_out1 = self.constant208.val();
        let add71_out1 =
            mul45_out1.add((constant208_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish20_out1 = burn::tensor::activation::hard_swish(add71_out1);
        let constant209_out1 = self.constant209.val();
        let mul46_out1 = (constant209_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish20_out1);
        let constant210_out1 = self.constant210.val();
        let add72_out1 =
            mul46_out1.add((constant210_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d30_out1 = crate::model_arch::conv_fwd(&self.conv2d30, add72_out1);
        let reshape29_out1 = self.reshape29.val();
        let add73_out1 = conv2d30_out1.add(reshape29_out1);
        let constant211_out1 = self.constant211.val();
        let mul47_out1 = (constant211_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add73_out1);
        let constant212_out1 = self.constant212.val();
        let add74_out1 =
            mul47_out1.add((constant212_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish21_out1 = burn::tensor::activation::hard_swish(add74_out1);
        let constant213_out1 = self.constant213.val();
        let mul48_out1 = (constant213_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish21_out1);
        let constant214_out1 = self.constant214.val();
        let add75_out1 =
            mul48_out1.add((constant214_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d31_out1 = crate::model_arch::conv_fwd(&self.conv2d31, add75_out1);
        let reshape30_out1 = self.reshape30.val();
        let add76_out1 = conv2d31_out1.add(reshape30_out1);
        let constant215_out1 = self.constant215.val();
        let mul49_out1 = (constant215_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add76_out1);
        let constant216_out1 = self.constant216.val();
        let add77_out1 =
            mul49_out1.add((constant216_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish22_out1 = burn::tensor::activation::hard_swish(add77_out1);
        let constant217_out1 = self.constant217.val();
        let mul50_out1 = (constant217_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish22_out1);
        let constant218_out1 = self.constant218.val();
        let add78_out1 =
            mul50_out1.add((constant218_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d32_out1 = crate::model_arch::conv_fwd(&self.conv2d32, add78_out1);
        let reshape31_out1 = self.reshape31.val();
        let add79_out1 = conv2d32_out1.add(reshape31_out1);
        let constant219_out1 = self.constant219.val();
        let mul51_out1 = (constant219_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add79_out1);
        let constant220_out1 = self.constant220.val();
        let add80_out1 =
            mul51_out1.add((constant220_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish23_out1 = burn::tensor::activation::hard_swish(add80_out1);
        let constant221_out1 = self.constant221.val();
        let mul52_out1 = (constant221_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish23_out1);
        mul52_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6<B: Backend> {
    constant222: burn::module::Param<Tensor<B, 1>>,
    conv2d33: Conv2d<B>,
    reshape32: burn::module::Param<Tensor<B, 4>>,
    constant223: burn::module::Param<Tensor<B, 1>>,
    constant224: burn::module::Param<Tensor<B, 1>>,
    constant225: burn::module::Param<Tensor<B, 1>>,
    constant226: burn::module::Param<Tensor<B, 1>>,
    conv2d34: Conv2d<B>,
    reshape33: burn::module::Param<Tensor<B, 4>>,
    conv2d35: Conv2d<B>,
    reshape34: burn::module::Param<Tensor<B, 4>>,
    conv2d36: Conv2d<B>,
    reshape35: burn::module::Param<Tensor<B, 4>>,
    conv2d37: Conv2d<B>,
    reshape36: burn::module::Param<Tensor<B, 4>>,
    conv2d38: Conv2d<B>,
    globalaveragepool3: AdaptiveAvgPool2d,
    conv2d39: Conv2d<B>,
    reshape37: burn::module::Param<Tensor<B, 4>>,
    conv2d40: Conv2d<B>,
    reshape38: burn::module::Param<Tensor<B, 4>>,
    conv2d41: Conv2d<B>,
    globalaveragepool4: AdaptiveAvgPool2d,
    conv2d42: Conv2d<B>,
    reshape39: burn::module::Param<Tensor<B, 4>>,
    conv2d43: Conv2d<B>,
    reshape40: burn::module::Param<Tensor<B, 4>>,
    conv2d44: Conv2d<B>,
    globalaveragepool5: AdaptiveAvgPool2d,
    conv2d45: Conv2d<B>,
    reshape41: burn::module::Param<Tensor<B, 4>>,
    conv2d46: Conv2d<B>,
    reshape42: burn::module::Param<Tensor<B, 4>>,
    conv2d47: Conv2d<B>,
    globalaveragepool6: AdaptiveAvgPool2d,
    conv2d48: Conv2d<B>,
    reshape43: burn::module::Param<Tensor<B, 4>>,
    conv2d49: Conv2d<B>,
    reshape44: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule6<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant222: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d33 = Conv2dConfig::new([384, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape32: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 384, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 384, 1, 1].into(),
        );
        let constant223: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant224: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant225: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant226: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d34 = Conv2dConfig::new([48, 12], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape33: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 12, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 12, 1, 1].into(),
        );
        let conv2d35 = Conv2dConfig::new([96, 18], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape34: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 18, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 18, 1, 1].into(),
        );
        let conv2d36 = Conv2dConfig::new([192, 42], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape35: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 42, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 42, 1, 1].into(),
        );
        let conv2d37 = Conv2dConfig::new([384, 360], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape36: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 360, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 360, 1, 1].into(),
        );
        let conv2d38 = Conv2dConfig::new([360, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool3 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d39 = Conv2dConfig::new([96, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape37: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let conv2d40 = Conv2dConfig::new([24, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape38: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
        );
        let conv2d41 = Conv2dConfig::new([42, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool4 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d42 = Conv2dConfig::new([96, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape39: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let conv2d43 = Conv2dConfig::new([24, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape40: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
        );
        let conv2d44 = Conv2dConfig::new([18, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool5 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d45 = Conv2dConfig::new([96, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape41: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let conv2d46 = Conv2dConfig::new([24, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape42: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
        );
        let conv2d47 = Conv2dConfig::new([12, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool6 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d48 = Conv2dConfig::new([96, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape43: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let conv2d49 = Conv2dConfig::new([24, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape44: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 96, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 96, 1, 1].into(),
        );
        Self {
            constant222,
            conv2d33,
            reshape32,
            constant223,
            constant224,
            constant225,
            constant226,
            conv2d34,
            reshape33,
            conv2d35,
            reshape34,
            conv2d36,
            reshape35,
            conv2d37,
            reshape36,
            conv2d38,
            globalaveragepool3,
            conv2d39,
            reshape37,
            conv2d40,
            reshape38,
            conv2d41,
            globalaveragepool4,
            conv2d42,
            reshape39,
            conv2d43,
            reshape40,
            conv2d44,
            globalaveragepool5,
            conv2d45,
            reshape41,
            conv2d46,
            reshape42,
            conv2d47,
            globalaveragepool6,
            conv2d48,
            reshape43,
            conv2d49,
            reshape44,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        mul52_out1: Tensor<B, 4>,
        add17_out1: Tensor<B, 4>,
        add28_out1: Tensor<B, 4>,
        add57_out1: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let constant222_out1 = self.constant222.val();
        let add81_out1 =
            mul52_out1.add((constant222_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d33_out1 = crate::model_arch::conv_fwd(&self.conv2d33, add81_out1);
        let reshape32_out1 = self.reshape32.val();
        let add82_out1 = conv2d33_out1.add(reshape32_out1);
        let constant223_out1 = self.constant223.val();
        let mul53_out1 = (constant223_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(add82_out1);
        let constant224_out1 = self.constant224.val();
        let add83_out1 =
            mul53_out1.add((constant224_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let hardswish24_out1 = burn::tensor::activation::hard_swish(add83_out1);
        let constant225_out1 = self.constant225.val();
        let mul54_out1 = (constant225_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(hardswish24_out1);
        let constant226_out1 = self.constant226.val();
        let add84_out1 =
            mul54_out1.add((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d34_out1 = crate::model_arch::conv_fwd(&self.conv2d34, add17_out1);
        let reshape33_out1 = self.reshape33.val();
        let add85_out1 = conv2d34_out1.add(reshape33_out1);
        let conv2d35_out1 = crate::model_arch::conv_fwd(&self.conv2d35, add28_out1);
        let reshape34_out1 = self.reshape34.val();
        let add86_out1 = conv2d35_out1.add(reshape34_out1);
        let conv2d36_out1 = crate::model_arch::conv_fwd(&self.conv2d36, add57_out1);
        let reshape35_out1 = self.reshape35.val();
        let add87_out1 = conv2d36_out1.add(reshape35_out1);
        let conv2d37_out1 = crate::model_arch::conv_fwd(&self.conv2d37, add84_out1);
        let reshape36_out1 = self.reshape36.val();
        let add88_out1 = conv2d37_out1.add(reshape36_out1);
        let conv2d38_out1 = crate::model_arch::conv_fwd(&self.conv2d38, add88_out1);
        let globalaveragepool3_out1 = self.globalaveragepool3.forward(conv2d38_out1.clone());
        let conv2d39_out1 = crate::model_arch::conv_fwd(&self.conv2d39, globalaveragepool3_out1);
        let reshape37_out1 = self.reshape37.val();
        let add89_out1 = conv2d39_out1.add(reshape37_out1);
        let relu3_out1 = burn::tensor::activation::relu(add89_out1);
        let conv2d40_out1 = crate::model_arch::conv_fwd(&self.conv2d40, relu3_out1);
        let reshape38_out1 = self.reshape38.val();
        let add90_out1 = conv2d40_out1.add(reshape38_out1);
        let hardsigmoid3_out1 =
            burn::tensor::activation::hard_sigmoid(add90_out1, 0.20000000298023224, 0.5);
        let mul55_out1 = conv2d38_out1.clone().mul(hardsigmoid3_out1);
        let add91_out1 = conv2d38_out1.add(mul55_out1);
        let conv2d41_out1 = crate::model_arch::conv_fwd(&self.conv2d41, add87_out1);
        let globalaveragepool4_out1 = self.globalaveragepool4.forward(conv2d41_out1.clone());
        let conv2d42_out1 = crate::model_arch::conv_fwd(&self.conv2d42, globalaveragepool4_out1);
        let reshape39_out1 = self.reshape39.val();
        let add92_out1 = conv2d42_out1.add(reshape39_out1);
        let relu4_out1 = burn::tensor::activation::relu(add92_out1);
        let conv2d43_out1 = crate::model_arch::conv_fwd(&self.conv2d43, relu4_out1);
        let reshape40_out1 = self.reshape40.val();
        let add93_out1 = conv2d43_out1.add(reshape40_out1);
        let hardsigmoid4_out1 =
            burn::tensor::activation::hard_sigmoid(add93_out1, 0.20000000298023224, 0.5);
        let mul56_out1 = conv2d41_out1.clone().mul(hardsigmoid4_out1);
        let add94_out1 = conv2d41_out1.add(mul56_out1);
        let conv2d44_out1 = crate::model_arch::conv_fwd(&self.conv2d44, add86_out1);
        let globalaveragepool5_out1 = self.globalaveragepool5.forward(conv2d44_out1.clone());
        let conv2d45_out1 = crate::model_arch::conv_fwd(&self.conv2d45, globalaveragepool5_out1);
        let reshape41_out1 = self.reshape41.val();
        let add95_out1 = conv2d45_out1.add(reshape41_out1);
        let relu5_out1 = burn::tensor::activation::relu(add95_out1);
        let conv2d46_out1 = crate::model_arch::conv_fwd(&self.conv2d46, relu5_out1);
        let reshape42_out1 = self.reshape42.val();
        let add96_out1 = conv2d46_out1.add(reshape42_out1);
        let hardsigmoid5_out1 =
            burn::tensor::activation::hard_sigmoid(add96_out1, 0.20000000298023224, 0.5);
        let mul57_out1 = conv2d44_out1.clone().mul(hardsigmoid5_out1);
        let add97_out1 = conv2d44_out1.add(mul57_out1);
        let conv2d47_out1 = crate::model_arch::conv_fwd(&self.conv2d47, add85_out1);
        let globalaveragepool6_out1 = self.globalaveragepool6.forward(conv2d47_out1.clone());
        let conv2d48_out1 = crate::model_arch::conv_fwd(&self.conv2d48, globalaveragepool6_out1);
        let reshape43_out1 = self.reshape43.val();
        let add98_out1 = conv2d48_out1.add(reshape43_out1);
        let relu6_out1 = burn::tensor::activation::relu(add98_out1);
        let conv2d49_out1 = crate::model_arch::conv_fwd(&self.conv2d49, relu6_out1);
        let reshape44_out1 = self.reshape44.val();
        let add99_out1 = conv2d49_out1.add(reshape44_out1);
        let hardsigmoid6_out1 =
            burn::tensor::activation::hard_sigmoid(add99_out1, 0.20000000298023224, 0.5);
        let mul58_out1 = conv2d47_out1.clone().mul(hardsigmoid6_out1);
        let add100_out1 = conv2d47_out1.add(mul58_out1);
        (add91_out1, add94_out1, add97_out1, add100_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule7<B: Backend> {
    resize1: burn::nn::interpolate::Interpolate2d,
    resize2: burn::nn::interpolate::Interpolate2d,
    resize3: burn::nn::interpolate::Interpolate2d,
    conv2d50: Conv2d<B>,
    globalaveragepool7: AdaptiveAvgPool2d,
    conv2d51: Conv2d<B>,
    reshape45: burn::module::Param<Tensor<B, 4>>,
    conv2d52: Conv2d<B>,
    reshape46: burn::module::Param<Tensor<B, 4>>,
    conv2d53: Conv2d<B>,
    globalaveragepool8: AdaptiveAvgPool2d,
    conv2d54: Conv2d<B>,
    reshape47: burn::module::Param<Tensor<B, 4>>,
    conv2d55: Conv2d<B>,
    reshape48: burn::module::Param<Tensor<B, 4>>,
    conv2d56: Conv2d<B>,
    globalaveragepool9: AdaptiveAvgPool2d,
    conv2d57: Conv2d<B>,
    reshape49: burn::module::Param<Tensor<B, 4>>,
    conv2d58: Conv2d<B>,
    reshape50: burn::module::Param<Tensor<B, 4>>,
    conv2d59: Conv2d<B>,
    globalaveragepool10: AdaptiveAvgPool2d,
    conv2d60: Conv2d<B>,
    reshape51: burn::module::Param<Tensor<B, 4>>,
    conv2d61: Conv2d<B>,
    reshape52: burn::module::Param<Tensor<B, 4>>,
    resize4: burn::nn::interpolate::Interpolate2d,
    resize5: burn::nn::interpolate::Interpolate2d,
    resize6: burn::nn::interpolate::Interpolate2d,
    conv2d62: Conv2d<B>,
    batchnormalization2: BatchNorm<B>,
    convtranspose2d1: ConvTranspose2d<B>,
    reshape53: burn::module::Param<Tensor<B, 4>>,
    batchnormalization3: BatchNorm<B>,
    convtranspose2d2: ConvTranspose2d<B>,
    reshape54: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule7<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let resize1 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize2 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize3 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d50 = Conv2dConfig::new([96, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool7 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d51 = Conv2dConfig::new([24, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape45: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 6, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 6, 1, 1].into(),
        );
        let conv2d52 = Conv2dConfig::new([6, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape46: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let conv2d53 = Conv2dConfig::new([96, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool8 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d54 = Conv2dConfig::new([24, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape47: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 6, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 6, 1, 1].into(),
        );
        let conv2d55 = Conv2dConfig::new([6, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape48: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let conv2d56 = Conv2dConfig::new([96, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool9 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d57 = Conv2dConfig::new([24, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape49: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 6, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 6, 1, 1].into(),
        );
        let conv2d58 = Conv2dConfig::new([6, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape50: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let conv2d59 = Conv2dConfig::new([96, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let globalaveragepool10 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d60 = Conv2dConfig::new([24, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape51: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 6, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 6, 1, 1].into(),
        );
        let conv2d61 = Conv2dConfig::new([6, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape52: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let resize4 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([8.0, 8.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize5 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([4.0, 4.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize6 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d62 = Conv2dConfig::new([96, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(24)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let convtranspose2d1 = ConvTranspose2dConfig::new([24, 24], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape53: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 24, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 24, 1, 1].into(),
        );
        let batchnormalization3 = BatchNormConfig::new(24)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let convtranspose2d2 = ConvTranspose2dConfig::new([24, 1], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let reshape54: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 1, 1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 1, 1].into(),
        );
        Self {
            resize1,
            resize2,
            resize3,
            conv2d50,
            globalaveragepool7,
            conv2d51,
            reshape45,
            conv2d52,
            reshape46,
            conv2d53,
            globalaveragepool8,
            conv2d54,
            reshape47,
            conv2d55,
            reshape48,
            conv2d56,
            globalaveragepool9,
            conv2d57,
            reshape49,
            conv2d58,
            reshape50,
            conv2d59,
            globalaveragepool10,
            conv2d60,
            reshape51,
            conv2d61,
            reshape52,
            resize4,
            resize5,
            resize6,
            conv2d62,
            batchnormalization2,
            convtranspose2d1,
            reshape53,
            batchnormalization3,
            convtranspose2d2,
            reshape54,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add91_out1: Tensor<B, 4>,
        add94_out1: Tensor<B, 4>,
        add97_out1: Tensor<B, 4>,
        add100_out1: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let resize1_out1 = self.resize1.forward(add91_out1.clone());
        let add101_out1 = add94_out1.add(resize1_out1);
        let resize2_out1 = self.resize2.forward(add101_out1.clone());
        let add102_out1 = add97_out1.add(resize2_out1);
        let resize3_out1 = self.resize3.forward(add102_out1.clone());
        let add103_out1 = add100_out1.add(resize3_out1);
        let conv2d50_out1 = crate::model_arch::conv_fwd(&self.conv2d50, add91_out1);
        let globalaveragepool7_out1 = self.globalaveragepool7.forward(conv2d50_out1.clone());
        let conv2d51_out1 = crate::model_arch::conv_fwd(&self.conv2d51, globalaveragepool7_out1);
        let reshape45_out1 = self.reshape45.val();
        let add104_out1 = conv2d51_out1.add(reshape45_out1);
        let relu7_out1 = burn::tensor::activation::relu(add104_out1);
        let conv2d52_out1 = crate::model_arch::conv_fwd(&self.conv2d52, relu7_out1);
        let reshape46_out1 = self.reshape46.val();
        let add105_out1 = conv2d52_out1.add(reshape46_out1);
        let hardsigmoid7_out1 =
            burn::tensor::activation::hard_sigmoid(add105_out1, 0.20000000298023224, 0.5);
        let mul59_out1 = conv2d50_out1.clone().mul(hardsigmoid7_out1);
        let add106_out1 = conv2d50_out1.add(mul59_out1);
        let conv2d53_out1 = crate::model_arch::conv_fwd(&self.conv2d53, add101_out1);
        let globalaveragepool8_out1 = self.globalaveragepool8.forward(conv2d53_out1.clone());
        let conv2d54_out1 = crate::model_arch::conv_fwd(&self.conv2d54, globalaveragepool8_out1);
        let reshape47_out1 = self.reshape47.val();
        let add107_out1 = conv2d54_out1.add(reshape47_out1);
        let relu8_out1 = burn::tensor::activation::relu(add107_out1);
        let conv2d55_out1 = crate::model_arch::conv_fwd(&self.conv2d55, relu8_out1);
        let reshape48_out1 = self.reshape48.val();
        let add108_out1 = conv2d55_out1.add(reshape48_out1);
        let hardsigmoid8_out1 =
            burn::tensor::activation::hard_sigmoid(add108_out1, 0.20000000298023224, 0.5);
        let mul60_out1 = conv2d53_out1.clone().mul(hardsigmoid8_out1);
        let add109_out1 = conv2d53_out1.add(mul60_out1);
        let conv2d56_out1 = crate::model_arch::conv_fwd(&self.conv2d56, add102_out1);
        let globalaveragepool9_out1 = self.globalaveragepool9.forward(conv2d56_out1.clone());
        let conv2d57_out1 = crate::model_arch::conv_fwd(&self.conv2d57, globalaveragepool9_out1);
        let reshape49_out1 = self.reshape49.val();
        let add110_out1 = conv2d57_out1.add(reshape49_out1);
        let relu9_out1 = burn::tensor::activation::relu(add110_out1);
        let conv2d58_out1 = crate::model_arch::conv_fwd(&self.conv2d58, relu9_out1);
        let reshape50_out1 = self.reshape50.val();
        let add111_out1 = conv2d58_out1.add(reshape50_out1);
        let hardsigmoid9_out1 =
            burn::tensor::activation::hard_sigmoid(add111_out1, 0.20000000298023224, 0.5);
        let mul61_out1 = conv2d56_out1.clone().mul(hardsigmoid9_out1);
        let add112_out1 = conv2d56_out1.add(mul61_out1);
        let conv2d59_out1 = crate::model_arch::conv_fwd(&self.conv2d59, add103_out1);
        let globalaveragepool10_out1 = self.globalaveragepool10.forward(conv2d59_out1.clone());
        let conv2d60_out1 = crate::model_arch::conv_fwd(&self.conv2d60, globalaveragepool10_out1);
        let reshape51_out1 = self.reshape51.val();
        let add113_out1 = conv2d60_out1.add(reshape51_out1);
        let relu10_out1 = burn::tensor::activation::relu(add113_out1);
        let conv2d61_out1 = crate::model_arch::conv_fwd(&self.conv2d61, relu10_out1);
        let reshape52_out1 = self.reshape52.val();
        let add114_out1 = conv2d61_out1.add(reshape52_out1);
        let hardsigmoid10_out1 =
            burn::tensor::activation::hard_sigmoid(add114_out1, 0.20000000298023224, 0.5);
        let mul62_out1 = conv2d59_out1.clone().mul(hardsigmoid10_out1);
        let add115_out1 = conv2d59_out1.add(mul62_out1);
        let resize4_out1 = self.resize4.forward(add106_out1);
        let resize5_out1 = self.resize5.forward(add109_out1);
        let resize6_out1 = self.resize6.forward(add112_out1);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [resize4_out1, resize5_out1, resize6_out1, add115_out1].into(),
            1,
        );
        let conv2d62_out1 = crate::model_arch::conv_fwd(&self.conv2d62, concat1_out1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(conv2d62_out1);
        let relu11_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(relu11_out1);
        let reshape53_out1 = self.reshape53.val();
        let add116_out1 = convtranspose2d1_out1.add(reshape53_out1);
        let batchnormalization3_out1 = self.batchnormalization3.forward(add116_out1);
        let relu12_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(relu12_out1);
        let reshape54_out1 = self.reshape54.val();
        let add117_out1 = convtranspose2d2_out1.add(reshape54_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(add117_out1);
        sigmoid1_out1
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
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let add16_out1 = self.submodule1.forward(x);
        let (mul21_out1, add17_out1, add28_out1) = self.submodule2.forward(add16_out1);
        let add49_out1 = self.submodule3.forward(mul21_out1);
        let (add65_out1, add57_out1) = self.submodule4.forward(add49_out1);
        let mul52_out1 = self.submodule5.forward(add65_out1);
        let (add91_out1, add94_out1, add97_out1, add100_out1) = self
            .submodule6
            .forward(mul52_out1, add17_out1, add28_out1, add57_out1);
        let sigmoid1_out1 =
            self.submodule7
                .forward(add91_out1, add94_out1, add97_out1, add100_out1);
        sigmoid1_out1
    }
}
