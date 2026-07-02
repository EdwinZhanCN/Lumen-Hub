// Generated from ONNX "src/siglip2-base/vision.fp32.onnx" by burn-onnx
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
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
    constant4: burn::module::Param<Tensor<B, 3>>,
    constant5: burn::module::Param<Tensor<B, 1>>,
    constant6: burn::module::Param<Tensor<B, 1>>,
    constant7: burn::module::Param<Tensor<B, 1>>,
    constant8: burn::module::Param<Tensor<B, 1>>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    constant12: burn::module::Param<Tensor<B, 1>>,
    constant13: burn::module::Param<Tensor<B, 1>>,
    constant14: burn::module::Param<Tensor<B, 1>>,
    linear4: Linear<B>,
    constant20: burn::module::Param<Tensor<B, 1>>,
    constant21: burn::module::Param<Tensor<B, 1>>,
    linear5: Linear<B>,
    constant24: burn::module::Param<Tensor<B, 1>>,
    constant25: burn::module::Param<Tensor<B, 1>>,
    constant26: burn::module::Param<Tensor<B, 1>>,
    constant27: burn::module::Param<Tensor<B, 1>>,
    linear6: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 768], [16, 16])
            .with_stride([16, 16])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant4: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 3>::zeros([1, 196, 768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 196, 768].into(),
        );
        let constant5: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant6: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant7: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant8: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear1 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear2 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear3 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant12: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant13: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant14: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear4 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant20: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant21: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear5 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let constant24: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([0.044714998453855515f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant25: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([0.7978845834732056f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant26: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([1f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant27: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([0.5f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let linear6 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        Self {
            conv2d1,
            constant4,
            constant5,
            constant6,
            constant7,
            constant8,
            linear1,
            linear2,
            linear3,
            constant12,
            constant13,
            constant14,
            linear4,
            constant20,
            constant21,
            linear5,
            constant24,
            constant25,
            constant26,
            constant27,
            linear6,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    ) {
        let conv2d1_out1 = crate::model_arch::conv_fwd(&self.conv2d1, pixel_values);
        let reshape1_out1 = conv2d1_out1.reshape([-1, 768, 196]);
        let transpose1_out1 = reshape1_out1.permute([0, 2, 1]);
        let constant4_out1 = self.constant4.val();
        let add1_out1 = transpose1_out1.add(constant4_out1);
        let reducemean1_out1 = { add1_out1.clone().mean_dim(2usize) };
        let sub1_out1 = add1_out1.clone().sub(reducemean1_out1);
        let constant5_out1 = self.constant5.val();
        let pow1_out1 = sub1_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean2_out1 = { pow1_out1.mean_dim(2usize) };
        let constant6_out1 = self.constant6.val();
        let add2_out1 =
            reducemean2_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt1_out1 = add2_out1.sqrt();
        let div1_out1 = sub1_out1.div(sqrt1_out1);
        let constant7_out1 = self.constant7.val();
        let mul1_out1 = div1_out1.mul((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant8_out1 = self.constant8.val();
        let add3_out1 = mul1_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear1_out1 = self.linear1.forward(add3_out1.clone());
        let linear2_out1 = self.linear2.forward(add3_out1.clone());
        let linear3_out1 = self.linear3.forward(add3_out1);
        let constant12_out1 = self.constant12.val();
        let add4_out1 = (constant12_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear1_out1);
        let constant13_out1 = self.constant13.val();
        let add5_out1 = (constant13_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear2_out1);
        let constant14_out1 = self.constant14.val();
        let add6_out1 = (constant14_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear3_out1);
        let reshape2_out1 = add4_out1.reshape([-1, 196, 12, 64]);
        let reshape3_out1 = add5_out1.reshape([-1, 196, 12, 64]);
        let reshape4_out1 = add6_out1.reshape([-1, 196, 12, 64]);
        let transpose2_out1 = reshape2_out1.permute([0, 2, 1, 3]);
        let transpose3_out1 = reshape4_out1.permute([0, 2, 1, 3]);
        let transpose4_out1 = reshape3_out1.permute([0, 2, 3, 1]);
        let matmul4_k_corrected = transpose4_out1.permute([0, 1, 3, 2]);
        let (matmul5_out1,) = {
            let q = transpose2_out1;
            let k = matmul4_k_corrected;
            let v = transpose3_out1;
            let matmul5_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul5_out1,)
        };
        let transpose5_out1 = matmul5_out1.permute([0, 2, 1, 3]);
        let reshape5_out1 = transpose5_out1.reshape([-1, 196, 768]);
        let linear4_out1 = self.linear4.forward(reshape5_out1);
        let add7_out1 = add1_out1.add(linear4_out1);
        let reducemean3_out1 = { add7_out1.clone().mean_dim(2usize) };
        let sub2_out1 = add7_out1.clone().sub(reducemean3_out1);
        let pow2_out1 = sub2_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean4_out1 = { pow2_out1.mean_dim(2usize) };
        let add8_out1 =
            reducemean4_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt2_out1 = add8_out1.sqrt();
        let div2_out1 = sub2_out1.div(sqrt2_out1);
        let constant20_out1 = self.constant20.val();
        let mul4_out1 = div2_out1.mul((constant20_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant21_out1 = self.constant21.val();
        let add9_out1 = mul4_out1.add((constant21_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear5_out1 = self.linear5.forward(add9_out1);
        let mul5_out1 = linear5_out1.clone().mul(linear5_out1.clone());
        let mul6_out1 = linear5_out1.clone().mul(mul5_out1);
        let constant24_out1 = self.constant24.val();
        let mul7_out1 = (constant24_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul6_out1);
        let add10_out1 = linear5_out1.clone().add(mul7_out1);
        let constant25_out1 = self.constant25.val();
        let mul8_out1 = (constant25_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add10_out1);
        let tanh1_out1 = mul8_out1.tanh();
        let constant26_out1 = self.constant26.val();
        let add11_out1 = (constant26_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh1_out1);
        let mul9_out1 = linear5_out1.mul(add11_out1);
        let constant27_out1 = self.constant27.val();
        let mul10_out1 = (constant27_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul9_out1);
        let linear6_out1 = self.linear6.forward(mul10_out1);
        let add12_out1 = add7_out1.add(linear6_out1);
        (
            add12_out1,
            constant5_out1,
            constant6_out1,
            constant24_out1,
            constant25_out1,
            constant26_out1,
            constant27_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    constant30: burn::module::Param<Tensor<B, 1>>,
    constant31: burn::module::Param<Tensor<B, 1>>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    linear9: Linear<B>,
    constant35: burn::module::Param<Tensor<B, 1>>,
    constant36: burn::module::Param<Tensor<B, 1>>,
    constant37: burn::module::Param<Tensor<B, 1>>,
    linear10: Linear<B>,
    constant40: burn::module::Param<Tensor<B, 1>>,
    constant41: burn::module::Param<Tensor<B, 1>>,
    linear11: Linear<B>,
    linear12: Linear<B>,
    constant46: burn::module::Param<Tensor<B, 1>>,
    constant47: burn::module::Param<Tensor<B, 1>>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    linear15: Linear<B>,
    constant51: burn::module::Param<Tensor<B, 1>>,
    constant52: burn::module::Param<Tensor<B, 1>>,
    constant53: burn::module::Param<Tensor<B, 1>>,
    linear16: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant30: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant31: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear7 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear8 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear9 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant35: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant36: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant37: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear10 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant40: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant41: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear11 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear12 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant46: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant47: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear13 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear14 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear15 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant51: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant52: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant53: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear16 = LinearConfig::new(768, 768).with_bias(true).init(device);
        Self {
            constant30,
            constant31,
            linear7,
            linear8,
            linear9,
            constant35,
            constant36,
            constant37,
            linear10,
            constant40,
            constant41,
            linear11,
            linear12,
            constant46,
            constant47,
            linear13,
            linear14,
            linear15,
            constant51,
            constant52,
            constant53,
            linear16,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add12_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean5_out1 = { add12_out1.clone().mean_dim(2usize) };
        let sub3_out1 = add12_out1.clone().sub(reducemean5_out1);
        let pow3_out1 = sub3_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean6_out1 = { pow3_out1.mean_dim(2usize) };
        let add13_out1 =
            reducemean6_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt3_out1 = add13_out1.sqrt();
        let div3_out1 = sub3_out1.div(sqrt3_out1);
        let constant30_out1 = self.constant30.val();
        let mul11_out1 = div3_out1.mul((constant30_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant31_out1 = self.constant31.val();
        let add14_out1 = mul11_out1.add((constant31_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear7_out1 = self.linear7.forward(add14_out1.clone());
        let linear8_out1 = self.linear8.forward(add14_out1.clone());
        let linear9_out1 = self.linear9.forward(add14_out1);
        let constant35_out1 = self.constant35.val();
        let add15_out1 = (constant35_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear7_out1);
        let constant36_out1 = self.constant36.val();
        let add16_out1 = (constant36_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear8_out1);
        let constant37_out1 = self.constant37.val();
        let add17_out1 = (constant37_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear9_out1);
        let reshape6_out1 = add15_out1.reshape([-1, 196, 12, 64]);
        let reshape7_out1 = add16_out1.reshape([-1, 196, 12, 64]);
        let reshape8_out1 = add17_out1.reshape([-1, 196, 12, 64]);
        let transpose6_out1 = reshape6_out1.permute([0, 2, 1, 3]);
        let transpose7_out1 = reshape8_out1.permute([0, 2, 1, 3]);
        let transpose8_out1 = reshape7_out1.permute([0, 2, 3, 1]);
        let matmul12_k_corrected = transpose8_out1.permute([0, 1, 3, 2]);
        let (matmul13_out1,) = {
            let q = transpose6_out1;
            let k = matmul12_k_corrected;
            let v = transpose7_out1;
            let matmul13_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul13_out1,)
        };
        let transpose9_out1 = matmul13_out1.permute([0, 2, 1, 3]);
        let reshape9_out1 = transpose9_out1.reshape([-1, 196, 768]);
        let linear10_out1 = self.linear10.forward(reshape9_out1);
        let add18_out1 = add12_out1.add(linear10_out1);
        let reducemean7_out1 = { add18_out1.clone().mean_dim(2usize) };
        let sub4_out1 = add18_out1.clone().sub(reducemean7_out1);
        let pow4_out1 = sub4_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean8_out1 = { pow4_out1.mean_dim(2usize) };
        let add19_out1 =
            reducemean8_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt4_out1 = add19_out1.sqrt();
        let div4_out1 = sub4_out1.div(sqrt4_out1);
        let constant40_out1 = self.constant40.val();
        let mul14_out1 = div4_out1.mul((constant40_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant41_out1 = self.constant41.val();
        let add20_out1 = mul14_out1.add((constant41_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear11_out1 = self.linear11.forward(add20_out1);
        let mul15_out1 = linear11_out1.clone().mul(linear11_out1.clone());
        let mul16_out1 = linear11_out1.clone().mul(mul15_out1);
        let mul17_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul16_out1);
        let add21_out1 = linear11_out1.clone().add(mul17_out1);
        let mul18_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add21_out1);
        let tanh2_out1 = mul18_out1.tanh();
        let add22_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh2_out1);
        let mul19_out1 = linear11_out1.mul(add22_out1);
        let mul20_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul19_out1);
        let linear12_out1 = self.linear12.forward(mul20_out1);
        let add23_out1 = add18_out1.add(linear12_out1);
        let reducemean9_out1 = { add23_out1.clone().mean_dim(2usize) };
        let sub5_out1 = add23_out1.clone().sub(reducemean9_out1);
        let pow5_out1 = sub5_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean10_out1 = { pow5_out1.mean_dim(2usize) };
        let add24_out1 = reducemean10_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt5_out1 = add24_out1.sqrt();
        let div5_out1 = sub5_out1.div(sqrt5_out1);
        let constant46_out1 = self.constant46.val();
        let mul21_out1 = div5_out1.mul((constant46_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant47_out1 = self.constant47.val();
        let add25_out1 = mul21_out1.add((constant47_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear13_out1 = self.linear13.forward(add25_out1.clone());
        let linear14_out1 = self.linear14.forward(add25_out1.clone());
        let linear15_out1 = self.linear15.forward(add25_out1);
        let constant51_out1 = self.constant51.val();
        let add26_out1 = (constant51_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear13_out1);
        let constant52_out1 = self.constant52.val();
        let add27_out1 = (constant52_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear14_out1);
        let constant53_out1 = self.constant53.val();
        let add28_out1 = (constant53_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear15_out1);
        let reshape10_out1 = add26_out1.reshape([-1, 196, 12, 64]);
        let reshape11_out1 = add27_out1.reshape([-1, 196, 12, 64]);
        let reshape12_out1 = add28_out1.reshape([-1, 196, 12, 64]);
        let transpose10_out1 = reshape10_out1.permute([0, 2, 1, 3]);
        let transpose11_out1 = reshape12_out1.permute([0, 2, 1, 3]);
        let transpose12_out1 = reshape11_out1.permute([0, 2, 3, 1]);
        let matmul20_k_corrected = transpose12_out1.permute([0, 1, 3, 2]);
        let (matmul21_out1,) = {
            let q = transpose10_out1;
            let k = matmul20_k_corrected;
            let v = transpose11_out1;
            let matmul21_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul21_out1,)
        };
        let transpose13_out1 = matmul21_out1.permute([0, 2, 1, 3]);
        let reshape13_out1 = transpose13_out1.reshape([-1, 196, 768]);
        let linear16_out1 = self.linear16.forward(reshape13_out1);
        let add29_out1 = add23_out1.add(linear16_out1);
        add29_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    constant56: burn::module::Param<Tensor<B, 1>>,
    constant57: burn::module::Param<Tensor<B, 1>>,
    linear17: Linear<B>,
    linear18: Linear<B>,
    constant62: burn::module::Param<Tensor<B, 1>>,
    constant63: burn::module::Param<Tensor<B, 1>>,
    linear19: Linear<B>,
    linear20: Linear<B>,
    linear21: Linear<B>,
    constant67: burn::module::Param<Tensor<B, 1>>,
    constant68: burn::module::Param<Tensor<B, 1>>,
    constant69: burn::module::Param<Tensor<B, 1>>,
    linear22: Linear<B>,
    constant72: burn::module::Param<Tensor<B, 1>>,
    constant73: burn::module::Param<Tensor<B, 1>>,
    linear23: Linear<B>,
    linear24: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant56: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant57: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear17 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear18 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant62: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant63: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear19 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear20 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear21 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant67: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant68: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant69: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear22 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant72: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant73: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear23 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear24 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        Self {
            constant56,
            constant57,
            linear17,
            linear18,
            constant62,
            constant63,
            linear19,
            linear20,
            linear21,
            constant67,
            constant68,
            constant69,
            linear22,
            constant72,
            constant73,
            linear23,
            linear24,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add29_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean11_out1 = { add29_out1.clone().mean_dim(2usize) };
        let sub6_out1 = add29_out1.clone().sub(reducemean11_out1);
        let pow6_out1 = sub6_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean12_out1 = { pow6_out1.mean_dim(2usize) };
        let add30_out1 =
            reducemean12_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt6_out1 = add30_out1.sqrt();
        let div6_out1 = sub6_out1.div(sqrt6_out1);
        let constant56_out1 = self.constant56.val();
        let mul24_out1 = div6_out1.mul((constant56_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant57_out1 = self.constant57.val();
        let add31_out1 = mul24_out1.add((constant57_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear17_out1 = self.linear17.forward(add31_out1);
        let mul25_out1 = linear17_out1.clone().mul(linear17_out1.clone());
        let mul26_out1 = linear17_out1.clone().mul(mul25_out1);
        let mul27_out1 = (constant24_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul26_out1);
        let add32_out1 = linear17_out1.clone().add(mul27_out1);
        let mul28_out1 = (constant25_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add32_out1);
        let tanh3_out1 = mul28_out1.tanh();
        let add33_out1 = (constant26_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh3_out1);
        let mul29_out1 = linear17_out1.mul(add33_out1);
        let mul30_out1 = (constant27_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul29_out1);
        let linear18_out1 = self.linear18.forward(mul30_out1);
        let add34_out1 = add29_out1.add(linear18_out1);
        let reducemean13_out1 = { add34_out1.clone().mean_dim(2usize) };
        let sub7_out1 = add34_out1.clone().sub(reducemean13_out1);
        let pow7_out1 = sub7_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean14_out1 = { pow7_out1.mean_dim(2usize) };
        let add35_out1 =
            reducemean14_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt7_out1 = add35_out1.sqrt();
        let div7_out1 = sub7_out1.div(sqrt7_out1);
        let constant62_out1 = self.constant62.val();
        let mul31_out1 = div7_out1.mul((constant62_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant63_out1 = self.constant63.val();
        let add36_out1 = mul31_out1.add((constant63_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear19_out1 = self.linear19.forward(add36_out1.clone());
        let linear20_out1 = self.linear20.forward(add36_out1.clone());
        let linear21_out1 = self.linear21.forward(add36_out1);
        let constant67_out1 = self.constant67.val();
        let add37_out1 = (constant67_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear19_out1);
        let constant68_out1 = self.constant68.val();
        let add38_out1 = (constant68_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear20_out1);
        let constant69_out1 = self.constant69.val();
        let add39_out1 = (constant69_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear21_out1);
        let reshape14_out1 = add37_out1.reshape([-1, 196, 12, 64]);
        let reshape15_out1 = add38_out1.reshape([-1, 196, 12, 64]);
        let reshape16_out1 = add39_out1.reshape([-1, 196, 12, 64]);
        let transpose14_out1 = reshape14_out1.permute([0, 2, 1, 3]);
        let transpose15_out1 = reshape16_out1.permute([0, 2, 1, 3]);
        let transpose16_out1 = reshape15_out1.permute([0, 2, 3, 1]);
        let matmul28_k_corrected = transpose16_out1.permute([0, 1, 3, 2]);
        let (matmul29_out1,) = {
            let q = transpose14_out1;
            let k = matmul28_k_corrected;
            let v = transpose15_out1;
            let matmul29_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul29_out1,)
        };
        let transpose17_out1 = matmul29_out1.permute([0, 2, 1, 3]);
        let reshape17_out1 = transpose17_out1.reshape([-1, 196, 768]);
        let linear22_out1 = self.linear22.forward(reshape17_out1);
        let add40_out1 = add34_out1.add(linear22_out1);
        let reducemean15_out1 = { add40_out1.clone().mean_dim(2usize) };
        let sub8_out1 = add40_out1.clone().sub(reducemean15_out1);
        let pow8_out1 = sub8_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean16_out1 = { pow8_out1.mean_dim(2usize) };
        let add41_out1 = reducemean16_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt8_out1 = add41_out1.sqrt();
        let div8_out1 = sub8_out1.div(sqrt8_out1);
        let constant72_out1 = self.constant72.val();
        let mul34_out1 = div8_out1.mul((constant72_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant73_out1 = self.constant73.val();
        let add42_out1 = mul34_out1.add((constant73_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear23_out1 = self.linear23.forward(add42_out1);
        let mul35_out1 = linear23_out1.clone().mul(linear23_out1.clone());
        let mul36_out1 = linear23_out1.clone().mul(mul35_out1);
        let mul37_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul36_out1);
        let add43_out1 = linear23_out1.clone().add(mul37_out1);
        let mul38_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add43_out1);
        let tanh4_out1 = mul38_out1.tanh();
        let add44_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh4_out1);
        let mul39_out1 = linear23_out1.mul(add44_out1);
        let mul40_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul39_out1);
        let linear24_out1 = self.linear24.forward(mul40_out1);
        let add45_out1 = add40_out1.add(linear24_out1);
        add45_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    constant78: burn::module::Param<Tensor<B, 1>>,
    constant79: burn::module::Param<Tensor<B, 1>>,
    linear25: Linear<B>,
    linear26: Linear<B>,
    linear27: Linear<B>,
    constant83: burn::module::Param<Tensor<B, 1>>,
    constant84: burn::module::Param<Tensor<B, 1>>,
    constant85: burn::module::Param<Tensor<B, 1>>,
    linear28: Linear<B>,
    constant88: burn::module::Param<Tensor<B, 1>>,
    constant89: burn::module::Param<Tensor<B, 1>>,
    linear29: Linear<B>,
    linear30: Linear<B>,
    constant94: burn::module::Param<Tensor<B, 1>>,
    constant95: burn::module::Param<Tensor<B, 1>>,
    linear31: Linear<B>,
    linear32: Linear<B>,
    linear33: Linear<B>,
    constant99: burn::module::Param<Tensor<B, 1>>,
    constant100: burn::module::Param<Tensor<B, 1>>,
    constant101: burn::module::Param<Tensor<B, 1>>,
    linear34: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant78: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant79: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear25 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear26 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear27 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant83: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant84: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant85: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear28 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant88: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant89: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear29 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear30 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant94: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant95: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear31 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear32 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear33 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant99: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant100: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant101: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear34 = LinearConfig::new(768, 768).with_bias(true).init(device);
        Self {
            constant78,
            constant79,
            linear25,
            linear26,
            linear27,
            constant83,
            constant84,
            constant85,
            linear28,
            constant88,
            constant89,
            linear29,
            linear30,
            constant94,
            constant95,
            linear31,
            linear32,
            linear33,
            constant99,
            constant100,
            constant101,
            linear34,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add45_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean17_out1 = { add45_out1.clone().mean_dim(2usize) };
        let sub9_out1 = add45_out1.clone().sub(reducemean17_out1);
        let pow9_out1 = sub9_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean18_out1 = { pow9_out1.mean_dim(2usize) };
        let add46_out1 =
            reducemean18_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt9_out1 = add46_out1.sqrt();
        let div9_out1 = sub9_out1.div(sqrt9_out1);
        let constant78_out1 = self.constant78.val();
        let mul41_out1 = div9_out1.mul((constant78_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant79_out1 = self.constant79.val();
        let add47_out1 = mul41_out1.add((constant79_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear25_out1 = self.linear25.forward(add47_out1.clone());
        let linear26_out1 = self.linear26.forward(add47_out1.clone());
        let linear27_out1 = self.linear27.forward(add47_out1);
        let constant83_out1 = self.constant83.val();
        let add48_out1 = (constant83_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear25_out1);
        let constant84_out1 = self.constant84.val();
        let add49_out1 = (constant84_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear26_out1);
        let constant85_out1 = self.constant85.val();
        let add50_out1 = (constant85_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear27_out1);
        let reshape18_out1 = add48_out1.reshape([-1, 196, 12, 64]);
        let reshape19_out1 = add49_out1.reshape([-1, 196, 12, 64]);
        let reshape20_out1 = add50_out1.reshape([-1, 196, 12, 64]);
        let transpose18_out1 = reshape18_out1.permute([0, 2, 1, 3]);
        let transpose19_out1 = reshape20_out1.permute([0, 2, 1, 3]);
        let transpose20_out1 = reshape19_out1.permute([0, 2, 3, 1]);
        let matmul36_k_corrected = transpose20_out1.permute([0, 1, 3, 2]);
        let (matmul37_out1,) = {
            let q = transpose18_out1;
            let k = matmul36_k_corrected;
            let v = transpose19_out1;
            let matmul37_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul37_out1,)
        };
        let transpose21_out1 = matmul37_out1.permute([0, 2, 1, 3]);
        let reshape21_out1 = transpose21_out1.reshape([-1, 196, 768]);
        let linear28_out1 = self.linear28.forward(reshape21_out1);
        let add51_out1 = add45_out1.add(linear28_out1);
        let reducemean19_out1 = { add51_out1.clone().mean_dim(2usize) };
        let sub10_out1 = add51_out1.clone().sub(reducemean19_out1);
        let pow10_out1 = sub10_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean20_out1 = { pow10_out1.mean_dim(2usize) };
        let add52_out1 =
            reducemean20_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt10_out1 = add52_out1.sqrt();
        let div10_out1 = sub10_out1.div(sqrt10_out1);
        let constant88_out1 = self.constant88.val();
        let mul44_out1 = div10_out1.mul((constant88_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant89_out1 = self.constant89.val();
        let add53_out1 = mul44_out1.add((constant89_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear29_out1 = self.linear29.forward(add53_out1);
        let mul45_out1 = linear29_out1.clone().mul(linear29_out1.clone());
        let mul46_out1 = linear29_out1.clone().mul(mul45_out1);
        let mul47_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul46_out1);
        let add54_out1 = linear29_out1.clone().add(mul47_out1);
        let mul48_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add54_out1);
        let tanh5_out1 = mul48_out1.tanh();
        let add55_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh5_out1);
        let mul49_out1 = linear29_out1.mul(add55_out1);
        let mul50_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul49_out1);
        let linear30_out1 = self.linear30.forward(mul50_out1);
        let add56_out1 = add51_out1.add(linear30_out1);
        let reducemean21_out1 = { add56_out1.clone().mean_dim(2usize) };
        let sub11_out1 = add56_out1.clone().sub(reducemean21_out1);
        let pow11_out1 = sub11_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean22_out1 = { pow11_out1.mean_dim(2usize) };
        let add57_out1 = reducemean22_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt11_out1 = add57_out1.sqrt();
        let div11_out1 = sub11_out1.div(sqrt11_out1);
        let constant94_out1 = self.constant94.val();
        let mul51_out1 = div11_out1.mul((constant94_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant95_out1 = self.constant95.val();
        let add58_out1 = mul51_out1.add((constant95_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear31_out1 = self.linear31.forward(add58_out1.clone());
        let linear32_out1 = self.linear32.forward(add58_out1.clone());
        let linear33_out1 = self.linear33.forward(add58_out1);
        let constant99_out1 = self.constant99.val();
        let add59_out1 = (constant99_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear31_out1);
        let constant100_out1 = self.constant100.val();
        let add60_out1 = (constant100_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear32_out1);
        let constant101_out1 = self.constant101.val();
        let add61_out1 = (constant101_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear33_out1);
        let reshape22_out1 = add59_out1.reshape([-1, 196, 12, 64]);
        let reshape23_out1 = add60_out1.reshape([-1, 196, 12, 64]);
        let reshape24_out1 = add61_out1.reshape([-1, 196, 12, 64]);
        let transpose22_out1 = reshape22_out1.permute([0, 2, 1, 3]);
        let transpose23_out1 = reshape24_out1.permute([0, 2, 1, 3]);
        let transpose24_out1 = reshape23_out1.permute([0, 2, 3, 1]);
        let matmul44_k_corrected = transpose24_out1.permute([0, 1, 3, 2]);
        let (matmul45_out1,) = {
            let q = transpose22_out1;
            let k = matmul44_k_corrected;
            let v = transpose23_out1;
            let matmul45_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul45_out1,)
        };
        let transpose25_out1 = matmul45_out1.permute([0, 2, 1, 3]);
        let reshape25_out1 = transpose25_out1.reshape([-1, 196, 768]);
        let linear34_out1 = self.linear34.forward(reshape25_out1);
        let add62_out1 = add56_out1.add(linear34_out1);
        add62_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5<B: Backend> {
    constant104: burn::module::Param<Tensor<B, 1>>,
    constant105: burn::module::Param<Tensor<B, 1>>,
    linear35: Linear<B>,
    linear36: Linear<B>,
    constant110: burn::module::Param<Tensor<B, 1>>,
    constant111: burn::module::Param<Tensor<B, 1>>,
    linear37: Linear<B>,
    linear38: Linear<B>,
    linear39: Linear<B>,
    constant115: burn::module::Param<Tensor<B, 1>>,
    constant116: burn::module::Param<Tensor<B, 1>>,
    constant117: burn::module::Param<Tensor<B, 1>>,
    linear40: Linear<B>,
    constant120: burn::module::Param<Tensor<B, 1>>,
    constant121: burn::module::Param<Tensor<B, 1>>,
    linear41: Linear<B>,
    linear42: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule5<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant104: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant105: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear35 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear36 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant110: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant111: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear37 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear38 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear39 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant115: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant116: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant117: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear40 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant120: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant121: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear41 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear42 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        Self {
            constant104,
            constant105,
            linear35,
            linear36,
            constant110,
            constant111,
            linear37,
            linear38,
            linear39,
            constant115,
            constant116,
            constant117,
            linear40,
            constant120,
            constant121,
            linear41,
            linear42,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add62_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean23_out1 = { add62_out1.clone().mean_dim(2usize) };
        let sub12_out1 = add62_out1.clone().sub(reducemean23_out1);
        let pow12_out1 = sub12_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean24_out1 = { pow12_out1.mean_dim(2usize) };
        let add63_out1 =
            reducemean24_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt12_out1 = add63_out1.sqrt();
        let div12_out1 = sub12_out1.div(sqrt12_out1);
        let constant104_out1 = self.constant104.val();
        let mul54_out1 = div12_out1.mul((constant104_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant105_out1 = self.constant105.val();
        let add64_out1 = mul54_out1.add((constant105_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear35_out1 = self.linear35.forward(add64_out1);
        let mul55_out1 = linear35_out1.clone().mul(linear35_out1.clone());
        let mul56_out1 = linear35_out1.clone().mul(mul55_out1);
        let mul57_out1 = (constant24_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul56_out1);
        let add65_out1 = linear35_out1.clone().add(mul57_out1);
        let mul58_out1 = (constant25_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add65_out1);
        let tanh6_out1 = mul58_out1.tanh();
        let add66_out1 = (constant26_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh6_out1);
        let mul59_out1 = linear35_out1.mul(add66_out1);
        let mul60_out1 = (constant27_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul59_out1);
        let linear36_out1 = self.linear36.forward(mul60_out1);
        let add67_out1 = add62_out1.add(linear36_out1);
        let reducemean25_out1 = { add67_out1.clone().mean_dim(2usize) };
        let sub13_out1 = add67_out1.clone().sub(reducemean25_out1);
        let pow13_out1 = sub13_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean26_out1 = { pow13_out1.mean_dim(2usize) };
        let add68_out1 =
            reducemean26_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt13_out1 = add68_out1.sqrt();
        let div13_out1 = sub13_out1.div(sqrt13_out1);
        let constant110_out1 = self.constant110.val();
        let mul61_out1 = div13_out1.mul((constant110_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant111_out1 = self.constant111.val();
        let add69_out1 = mul61_out1.add((constant111_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear37_out1 = self.linear37.forward(add69_out1.clone());
        let linear38_out1 = self.linear38.forward(add69_out1.clone());
        let linear39_out1 = self.linear39.forward(add69_out1);
        let constant115_out1 = self.constant115.val();
        let add70_out1 = (constant115_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear37_out1);
        let constant116_out1 = self.constant116.val();
        let add71_out1 = (constant116_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear38_out1);
        let constant117_out1 = self.constant117.val();
        let add72_out1 = (constant117_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear39_out1);
        let reshape26_out1 = add70_out1.reshape([-1, 196, 12, 64]);
        let reshape27_out1 = add71_out1.reshape([-1, 196, 12, 64]);
        let reshape28_out1 = add72_out1.reshape([-1, 196, 12, 64]);
        let transpose26_out1 = reshape26_out1.permute([0, 2, 1, 3]);
        let transpose27_out1 = reshape28_out1.permute([0, 2, 1, 3]);
        let transpose28_out1 = reshape27_out1.permute([0, 2, 3, 1]);
        let matmul52_k_corrected = transpose28_out1.permute([0, 1, 3, 2]);
        let (matmul53_out1,) = {
            let q = transpose26_out1;
            let k = matmul52_k_corrected;
            let v = transpose27_out1;
            let matmul53_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul53_out1,)
        };
        let transpose29_out1 = matmul53_out1.permute([0, 2, 1, 3]);
        let reshape29_out1 = transpose29_out1.reshape([-1, 196, 768]);
        let linear40_out1 = self.linear40.forward(reshape29_out1);
        let add73_out1 = add67_out1.add(linear40_out1);
        let reducemean27_out1 = { add73_out1.clone().mean_dim(2usize) };
        let sub14_out1 = add73_out1.clone().sub(reducemean27_out1);
        let pow14_out1 = sub14_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean28_out1 = { pow14_out1.mean_dim(2usize) };
        let add74_out1 = reducemean28_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt14_out1 = add74_out1.sqrt();
        let div14_out1 = sub14_out1.div(sqrt14_out1);
        let constant120_out1 = self.constant120.val();
        let mul64_out1 = div14_out1.mul((constant120_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant121_out1 = self.constant121.val();
        let add75_out1 = mul64_out1.add((constant121_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear41_out1 = self.linear41.forward(add75_out1);
        let mul65_out1 = linear41_out1.clone().mul(linear41_out1.clone());
        let mul66_out1 = linear41_out1.clone().mul(mul65_out1);
        let mul67_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul66_out1);
        let add76_out1 = linear41_out1.clone().add(mul67_out1);
        let mul68_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add76_out1);
        let tanh7_out1 = mul68_out1.tanh();
        let add77_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh7_out1);
        let mul69_out1 = linear41_out1.mul(add77_out1);
        let mul70_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul69_out1);
        let linear42_out1 = self.linear42.forward(mul70_out1);
        let add78_out1 = add73_out1.add(linear42_out1);
        add78_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6<B: Backend> {
    constant126: burn::module::Param<Tensor<B, 1>>,
    constant127: burn::module::Param<Tensor<B, 1>>,
    linear43: Linear<B>,
    linear44: Linear<B>,
    linear45: Linear<B>,
    constant131: burn::module::Param<Tensor<B, 1>>,
    constant132: burn::module::Param<Tensor<B, 1>>,
    constant133: burn::module::Param<Tensor<B, 1>>,
    linear46: Linear<B>,
    constant136: burn::module::Param<Tensor<B, 1>>,
    constant137: burn::module::Param<Tensor<B, 1>>,
    linear47: Linear<B>,
    linear48: Linear<B>,
    constant142: burn::module::Param<Tensor<B, 1>>,
    constant143: burn::module::Param<Tensor<B, 1>>,
    linear49: Linear<B>,
    linear50: Linear<B>,
    linear51: Linear<B>,
    constant147: burn::module::Param<Tensor<B, 1>>,
    constant148: burn::module::Param<Tensor<B, 1>>,
    constant149: burn::module::Param<Tensor<B, 1>>,
    linear52: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule6<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant126: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant127: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear43 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear44 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear45 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant131: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant132: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant133: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear46 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant136: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant137: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear47 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear48 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant142: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant143: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear49 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear50 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear51 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant147: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant148: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant149: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear52 = LinearConfig::new(768, 768).with_bias(true).init(device);
        Self {
            constant126,
            constant127,
            linear43,
            linear44,
            linear45,
            constant131,
            constant132,
            constant133,
            linear46,
            constant136,
            constant137,
            linear47,
            linear48,
            constant142,
            constant143,
            linear49,
            linear50,
            linear51,
            constant147,
            constant148,
            constant149,
            linear52,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add78_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean29_out1 = { add78_out1.clone().mean_dim(2usize) };
        let sub15_out1 = add78_out1.clone().sub(reducemean29_out1);
        let pow15_out1 = sub15_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean30_out1 = { pow15_out1.mean_dim(2usize) };
        let add79_out1 =
            reducemean30_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt15_out1 = add79_out1.sqrt();
        let div15_out1 = sub15_out1.div(sqrt15_out1);
        let constant126_out1 = self.constant126.val();
        let mul71_out1 = div15_out1.mul((constant126_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant127_out1 = self.constant127.val();
        let add80_out1 = mul71_out1.add((constant127_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear43_out1 = self.linear43.forward(add80_out1.clone());
        let linear44_out1 = self.linear44.forward(add80_out1.clone());
        let linear45_out1 = self.linear45.forward(add80_out1);
        let constant131_out1 = self.constant131.val();
        let add81_out1 = (constant131_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear43_out1);
        let constant132_out1 = self.constant132.val();
        let add82_out1 = (constant132_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear44_out1);
        let constant133_out1 = self.constant133.val();
        let add83_out1 = (constant133_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear45_out1);
        let reshape30_out1 = add81_out1.reshape([-1, 196, 12, 64]);
        let reshape31_out1 = add82_out1.reshape([-1, 196, 12, 64]);
        let reshape32_out1 = add83_out1.reshape([-1, 196, 12, 64]);
        let transpose30_out1 = reshape30_out1.permute([0, 2, 1, 3]);
        let transpose31_out1 = reshape32_out1.permute([0, 2, 1, 3]);
        let transpose32_out1 = reshape31_out1.permute([0, 2, 3, 1]);
        let matmul60_k_corrected = transpose32_out1.permute([0, 1, 3, 2]);
        let (matmul61_out1,) = {
            let q = transpose30_out1;
            let k = matmul60_k_corrected;
            let v = transpose31_out1;
            let matmul61_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul61_out1,)
        };
        let transpose33_out1 = matmul61_out1.permute([0, 2, 1, 3]);
        let reshape33_out1 = transpose33_out1.reshape([-1, 196, 768]);
        let linear46_out1 = self.linear46.forward(reshape33_out1);
        let add84_out1 = add78_out1.add(linear46_out1);
        let reducemean31_out1 = { add84_out1.clone().mean_dim(2usize) };
        let sub16_out1 = add84_out1.clone().sub(reducemean31_out1);
        let pow16_out1 = sub16_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean32_out1 = { pow16_out1.mean_dim(2usize) };
        let add85_out1 =
            reducemean32_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt16_out1 = add85_out1.sqrt();
        let div16_out1 = sub16_out1.div(sqrt16_out1);
        let constant136_out1 = self.constant136.val();
        let mul74_out1 = div16_out1.mul((constant136_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant137_out1 = self.constant137.val();
        let add86_out1 = mul74_out1.add((constant137_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear47_out1 = self.linear47.forward(add86_out1);
        let mul75_out1 = linear47_out1.clone().mul(linear47_out1.clone());
        let mul76_out1 = linear47_out1.clone().mul(mul75_out1);
        let mul77_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul76_out1);
        let add87_out1 = linear47_out1.clone().add(mul77_out1);
        let mul78_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add87_out1);
        let tanh8_out1 = mul78_out1.tanh();
        let add88_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh8_out1);
        let mul79_out1 = linear47_out1.mul(add88_out1);
        let mul80_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul79_out1);
        let linear48_out1 = self.linear48.forward(mul80_out1);
        let add89_out1 = add84_out1.add(linear48_out1);
        let reducemean33_out1 = { add89_out1.clone().mean_dim(2usize) };
        let sub17_out1 = add89_out1.clone().sub(reducemean33_out1);
        let pow17_out1 = sub17_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean34_out1 = { pow17_out1.mean_dim(2usize) };
        let add90_out1 = reducemean34_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt17_out1 = add90_out1.sqrt();
        let div17_out1 = sub17_out1.div(sqrt17_out1);
        let constant142_out1 = self.constant142.val();
        let mul81_out1 = div17_out1.mul((constant142_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant143_out1 = self.constant143.val();
        let add91_out1 = mul81_out1.add((constant143_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear49_out1 = self.linear49.forward(add91_out1.clone());
        let linear50_out1 = self.linear50.forward(add91_out1.clone());
        let linear51_out1 = self.linear51.forward(add91_out1);
        let constant147_out1 = self.constant147.val();
        let add92_out1 = (constant147_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear49_out1);
        let constant148_out1 = self.constant148.val();
        let add93_out1 = (constant148_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear50_out1);
        let constant149_out1 = self.constant149.val();
        let add94_out1 = (constant149_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear51_out1);
        let reshape34_out1 = add92_out1.reshape([-1, 196, 12, 64]);
        let reshape35_out1 = add93_out1.reshape([-1, 196, 12, 64]);
        let reshape36_out1 = add94_out1.reshape([-1, 196, 12, 64]);
        let transpose34_out1 = reshape34_out1.permute([0, 2, 1, 3]);
        let transpose35_out1 = reshape36_out1.permute([0, 2, 1, 3]);
        let transpose36_out1 = reshape35_out1.permute([0, 2, 3, 1]);
        let matmul68_k_corrected = transpose36_out1.permute([0, 1, 3, 2]);
        let (matmul69_out1,) = {
            let q = transpose34_out1;
            let k = matmul68_k_corrected;
            let v = transpose35_out1;
            let matmul69_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul69_out1,)
        };
        let transpose37_out1 = matmul69_out1.permute([0, 2, 1, 3]);
        let reshape37_out1 = transpose37_out1.reshape([-1, 196, 768]);
        let linear52_out1 = self.linear52.forward(reshape37_out1);
        let add95_out1 = add89_out1.add(linear52_out1);
        add95_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule7<B: Backend> {
    constant152: burn::module::Param<Tensor<B, 1>>,
    constant153: burn::module::Param<Tensor<B, 1>>,
    linear53: Linear<B>,
    linear54: Linear<B>,
    constant158: burn::module::Param<Tensor<B, 1>>,
    constant159: burn::module::Param<Tensor<B, 1>>,
    linear55: Linear<B>,
    linear56: Linear<B>,
    linear57: Linear<B>,
    constant163: burn::module::Param<Tensor<B, 1>>,
    constant164: burn::module::Param<Tensor<B, 1>>,
    constant165: burn::module::Param<Tensor<B, 1>>,
    linear58: Linear<B>,
    constant168: burn::module::Param<Tensor<B, 1>>,
    constant169: burn::module::Param<Tensor<B, 1>>,
    linear59: Linear<B>,
    linear60: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule7<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant152: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant153: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear53 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear54 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant158: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant159: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear55 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear56 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear57 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant163: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant164: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant165: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear58 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant168: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant169: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear59 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear60 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        Self {
            constant152,
            constant153,
            linear53,
            linear54,
            constant158,
            constant159,
            linear55,
            linear56,
            linear57,
            constant163,
            constant164,
            constant165,
            linear58,
            constant168,
            constant169,
            linear59,
            linear60,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add95_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean35_out1 = { add95_out1.clone().mean_dim(2usize) };
        let sub18_out1 = add95_out1.clone().sub(reducemean35_out1);
        let pow18_out1 = sub18_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean36_out1 = { pow18_out1.mean_dim(2usize) };
        let add96_out1 =
            reducemean36_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt18_out1 = add96_out1.sqrt();
        let div18_out1 = sub18_out1.div(sqrt18_out1);
        let constant152_out1 = self.constant152.val();
        let mul84_out1 = div18_out1.mul((constant152_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant153_out1 = self.constant153.val();
        let add97_out1 = mul84_out1.add((constant153_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear53_out1 = self.linear53.forward(add97_out1);
        let mul85_out1 = linear53_out1.clone().mul(linear53_out1.clone());
        let mul86_out1 = linear53_out1.clone().mul(mul85_out1);
        let mul87_out1 = (constant24_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul86_out1);
        let add98_out1 = linear53_out1.clone().add(mul87_out1);
        let mul88_out1 = (constant25_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add98_out1);
        let tanh9_out1 = mul88_out1.tanh();
        let add99_out1 = (constant26_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh9_out1);
        let mul89_out1 = linear53_out1.mul(add99_out1);
        let mul90_out1 = (constant27_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul89_out1);
        let linear54_out1 = self.linear54.forward(mul90_out1);
        let add100_out1 = add95_out1.add(linear54_out1);
        let reducemean37_out1 = { add100_out1.clone().mean_dim(2usize) };
        let sub19_out1 = add100_out1.clone().sub(reducemean37_out1);
        let pow19_out1 = sub19_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean38_out1 = { pow19_out1.mean_dim(2usize) };
        let add101_out1 =
            reducemean38_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt19_out1 = add101_out1.sqrt();
        let div19_out1 = sub19_out1.div(sqrt19_out1);
        let constant158_out1 = self.constant158.val();
        let mul91_out1 = div19_out1.mul((constant158_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant159_out1 = self.constant159.val();
        let add102_out1 = mul91_out1.add((constant159_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear55_out1 = self.linear55.forward(add102_out1.clone());
        let linear56_out1 = self.linear56.forward(add102_out1.clone());
        let linear57_out1 = self.linear57.forward(add102_out1);
        let constant163_out1 = self.constant163.val();
        let add103_out1 = (constant163_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear55_out1);
        let constant164_out1 = self.constant164.val();
        let add104_out1 = (constant164_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear56_out1);
        let constant165_out1 = self.constant165.val();
        let add105_out1 = (constant165_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear57_out1);
        let reshape38_out1 = add103_out1.reshape([-1, 196, 12, 64]);
        let reshape39_out1 = add104_out1.reshape([-1, 196, 12, 64]);
        let reshape40_out1 = add105_out1.reshape([-1, 196, 12, 64]);
        let transpose38_out1 = reshape38_out1.permute([0, 2, 1, 3]);
        let transpose39_out1 = reshape40_out1.permute([0, 2, 1, 3]);
        let transpose40_out1 = reshape39_out1.permute([0, 2, 3, 1]);
        let matmul76_k_corrected = transpose40_out1.permute([0, 1, 3, 2]);
        let (matmul77_out1,) = {
            let q = transpose38_out1;
            let k = matmul76_k_corrected;
            let v = transpose39_out1;
            let matmul77_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul77_out1,)
        };
        let transpose41_out1 = matmul77_out1.permute([0, 2, 1, 3]);
        let reshape41_out1 = transpose41_out1.reshape([-1, 196, 768]);
        let linear58_out1 = self.linear58.forward(reshape41_out1);
        let add106_out1 = add100_out1.add(linear58_out1);
        let reducemean39_out1 = { add106_out1.clone().mean_dim(2usize) };
        let sub20_out1 = add106_out1.clone().sub(reducemean39_out1);
        let pow20_out1 = sub20_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean40_out1 = { pow20_out1.mean_dim(2usize) };
        let add107_out1 = reducemean40_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt20_out1 = add107_out1.sqrt();
        let div20_out1 = sub20_out1.div(sqrt20_out1);
        let constant168_out1 = self.constant168.val();
        let mul94_out1 = div20_out1.mul((constant168_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant169_out1 = self.constant169.val();
        let add108_out1 = mul94_out1.add((constant169_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear59_out1 = self.linear59.forward(add108_out1);
        let mul95_out1 = linear59_out1.clone().mul(linear59_out1.clone());
        let mul96_out1 = linear59_out1.clone().mul(mul95_out1);
        let mul97_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul96_out1);
        let add109_out1 = linear59_out1.clone().add(mul97_out1);
        let mul98_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add109_out1);
        let tanh10_out1 = mul98_out1.tanh();
        let add110_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh10_out1);
        let mul99_out1 = linear59_out1.mul(add110_out1);
        let mul100_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul99_out1);
        let linear60_out1 = self.linear60.forward(mul100_out1);
        let add111_out1 = add106_out1.add(linear60_out1);
        add111_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule8<B: Backend> {
    constant174: burn::module::Param<Tensor<B, 1>>,
    constant175: burn::module::Param<Tensor<B, 1>>,
    linear61: Linear<B>,
    linear62: Linear<B>,
    linear63: Linear<B>,
    constant179: burn::module::Param<Tensor<B, 1>>,
    constant180: burn::module::Param<Tensor<B, 1>>,
    constant181: burn::module::Param<Tensor<B, 1>>,
    linear64: Linear<B>,
    constant184: burn::module::Param<Tensor<B, 1>>,
    constant185: burn::module::Param<Tensor<B, 1>>,
    linear65: Linear<B>,
    linear66: Linear<B>,
    constant190: burn::module::Param<Tensor<B, 1>>,
    constant191: burn::module::Param<Tensor<B, 1>>,
    linear67: Linear<B>,
    linear68: Linear<B>,
    linear69: Linear<B>,
    constant195: burn::module::Param<Tensor<B, 1>>,
    constant196: burn::module::Param<Tensor<B, 1>>,
    constant197: burn::module::Param<Tensor<B, 1>>,
    linear70: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule8<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant174: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant175: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear61 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear62 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear63 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant179: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant180: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant181: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear64 = LinearConfig::new(768, 768).with_bias(true).init(device);
        let constant184: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant185: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear65 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear66 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant190: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant191: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear67 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear68 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let linear69 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant195: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant196: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant197: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear70 = LinearConfig::new(768, 768).with_bias(true).init(device);
        Self {
            constant174,
            constant175,
            linear61,
            linear62,
            linear63,
            constant179,
            constant180,
            constant181,
            linear64,
            constant184,
            constant185,
            linear65,
            linear66,
            constant190,
            constant191,
            linear67,
            linear68,
            linear69,
            constant195,
            constant196,
            constant197,
            linear70,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add111_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean41_out1 = { add111_out1.clone().mean_dim(2usize) };
        let sub21_out1 = add111_out1.clone().sub(reducemean41_out1);
        let pow21_out1 = sub21_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean42_out1 = { pow21_out1.mean_dim(2usize) };
        let add112_out1 =
            reducemean42_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt21_out1 = add112_out1.sqrt();
        let div21_out1 = sub21_out1.div(sqrt21_out1);
        let constant174_out1 = self.constant174.val();
        let mul101_out1 = div21_out1.mul((constant174_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant175_out1 = self.constant175.val();
        let add113_out1 = mul101_out1.add((constant175_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear61_out1 = self.linear61.forward(add113_out1.clone());
        let linear62_out1 = self.linear62.forward(add113_out1.clone());
        let linear63_out1 = self.linear63.forward(add113_out1);
        let constant179_out1 = self.constant179.val();
        let add114_out1 = (constant179_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear61_out1);
        let constant180_out1 = self.constant180.val();
        let add115_out1 = (constant180_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear62_out1);
        let constant181_out1 = self.constant181.val();
        let add116_out1 = (constant181_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear63_out1);
        let reshape42_out1 = add114_out1.reshape([-1, 196, 12, 64]);
        let reshape43_out1 = add115_out1.reshape([-1, 196, 12, 64]);
        let reshape44_out1 = add116_out1.reshape([-1, 196, 12, 64]);
        let transpose42_out1 = reshape42_out1.permute([0, 2, 1, 3]);
        let transpose43_out1 = reshape44_out1.permute([0, 2, 1, 3]);
        let transpose44_out1 = reshape43_out1.permute([0, 2, 3, 1]);
        let matmul84_k_corrected = transpose44_out1.permute([0, 1, 3, 2]);
        let (matmul85_out1,) = {
            let q = transpose42_out1;
            let k = matmul84_k_corrected;
            let v = transpose43_out1;
            let matmul85_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul85_out1,)
        };
        let transpose45_out1 = matmul85_out1.permute([0, 2, 1, 3]);
        let reshape45_out1 = transpose45_out1.reshape([-1, 196, 768]);
        let linear64_out1 = self.linear64.forward(reshape45_out1);
        let add117_out1 = add111_out1.add(linear64_out1);
        let reducemean43_out1 = { add117_out1.clone().mean_dim(2usize) };
        let sub22_out1 = add117_out1.clone().sub(reducemean43_out1);
        let pow22_out1 = sub22_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean44_out1 = { pow22_out1.mean_dim(2usize) };
        let add118_out1 =
            reducemean44_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt22_out1 = add118_out1.sqrt();
        let div22_out1 = sub22_out1.div(sqrt22_out1);
        let constant184_out1 = self.constant184.val();
        let mul104_out1 = div22_out1.mul((constant184_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant185_out1 = self.constant185.val();
        let add119_out1 = mul104_out1.add((constant185_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear65_out1 = self.linear65.forward(add119_out1);
        let mul105_out1 = linear65_out1.clone().mul(linear65_out1.clone());
        let mul106_out1 = linear65_out1.clone().mul(mul105_out1);
        let mul107_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul106_out1);
        let add120_out1 = linear65_out1.clone().add(mul107_out1);
        let mul108_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add120_out1);
        let tanh11_out1 = mul108_out1.tanh();
        let add121_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh11_out1);
        let mul109_out1 = linear65_out1.mul(add121_out1);
        let mul110_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul109_out1);
        let linear66_out1 = self.linear66.forward(mul110_out1);
        let add122_out1 = add117_out1.add(linear66_out1);
        let reducemean45_out1 = { add122_out1.clone().mean_dim(2usize) };
        let sub23_out1 = add122_out1.clone().sub(reducemean45_out1);
        let pow23_out1 = sub23_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean46_out1 = { pow23_out1.mean_dim(2usize) };
        let add123_out1 = reducemean46_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt23_out1 = add123_out1.sqrt();
        let div23_out1 = sub23_out1.div(sqrt23_out1);
        let constant190_out1 = self.constant190.val();
        let mul111_out1 = div23_out1.mul((constant190_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant191_out1 = self.constant191.val();
        let add124_out1 = mul111_out1.add((constant191_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear67_out1 = self.linear67.forward(add124_out1.clone());
        let linear68_out1 = self.linear68.forward(add124_out1.clone());
        let linear69_out1 = self.linear69.forward(add124_out1);
        let constant195_out1 = self.constant195.val();
        let add125_out1 = (constant195_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear67_out1);
        let constant196_out1 = self.constant196.val();
        let add126_out1 = (constant196_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear68_out1);
        let constant197_out1 = self.constant197.val();
        let add127_out1 = (constant197_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear69_out1);
        let reshape46_out1 = add125_out1.reshape([-1, 196, 12, 64]);
        let reshape47_out1 = add126_out1.reshape([-1, 196, 12, 64]);
        let reshape48_out1 = add127_out1.reshape([-1, 196, 12, 64]);
        let transpose46_out1 = reshape46_out1.permute([0, 2, 1, 3]);
        let transpose47_out1 = reshape48_out1.permute([0, 2, 1, 3]);
        let transpose48_out1 = reshape47_out1.permute([0, 2, 3, 1]);
        let matmul92_k_corrected = transpose48_out1.permute([0, 1, 3, 2]);
        let (matmul93_out1,) = {
            let q = transpose46_out1;
            let k = matmul92_k_corrected;
            let v = transpose47_out1;
            let matmul93_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1249999925494194f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul93_out1,)
        };
        let transpose49_out1 = matmul93_out1.permute([0, 2, 1, 3]);
        let reshape49_out1 = transpose49_out1.reshape([-1, 196, 768]);
        let linear70_out1 = self.linear70.forward(reshape49_out1);
        let add128_out1 = add122_out1.add(linear70_out1);
        add128_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule9<B: Backend> {
    constant200: burn::module::Param<Tensor<B, 1>>,
    constant201: burn::module::Param<Tensor<B, 1>>,
    linear71: Linear<B>,
    linear72: Linear<B>,
    constant206: burn::module::Param<Tensor<B, 1>>,
    constant207: burn::module::Param<Tensor<B, 1>>,
    linear73: Linear<B>,
    constant211: burn::module::Param<Tensor<B, 1>>,
    constant214: burn::module::Param<Tensor<B, 3>>,
    linear74: Linear<B>,
    constant217: burn::module::Param<Tensor<B, 1>>,
    constant221: burn::module::Param<Tensor<B, 1>>,
    linear75: Linear<B>,
    constant226: burn::module::Param<Tensor<B, 1>>,
    constant227: burn::module::Param<Tensor<B, 1>>,
    linear76: Linear<B>,
    linear77: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule9<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant200: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant201: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear71 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear72 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let constant206: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant207: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear73 = LinearConfig::new(768, 1536).with_bias(false).init(device);
        let constant211: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1536], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1536].into(),
        );
        let constant214: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 3>::zeros([1, 1, 768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 768].into(),
        );
        let linear74 = LinearConfig::new(768, 768).with_bias(false).init(device);
        let constant217: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant221: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::from([0.125f64]),
                    (device, burn::tensor::DType::F32),
                )
            },
            device.clone(),
            false,
            [1].into(),
        );
        let linear75 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let constant226: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let constant227: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [768].into(),
        );
        let linear76 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear77 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        Self {
            constant200,
            constant201,
            linear71,
            linear72,
            constant206,
            constant207,
            linear73,
            constant211,
            constant214,
            linear74,
            constant217,
            constant221,
            linear75,
            constant226,
            constant227,
            linear76,
            linear77,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add128_out1: Tensor<B, 3>,
        constant5_out1: Tensor<B, 1>,
        constant6_out1: Tensor<B, 1>,
        constant24_out1: Tensor<B, 1>,
        constant25_out1: Tensor<B, 1>,
        constant26_out1: Tensor<B, 1>,
        constant27_out1: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let reducemean47_out1 = { add128_out1.clone().mean_dim(2usize) };
        let sub24_out1 = add128_out1.clone().sub(reducemean47_out1);
        let pow24_out1 = sub24_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean48_out1 = { pow24_out1.mean_dim(2usize) };
        let add129_out1 =
            reducemean48_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt24_out1 = add129_out1.sqrt();
        let div24_out1 = sub24_out1.div(sqrt24_out1);
        let constant200_out1 = self.constant200.val();
        let mul114_out1 = div24_out1.mul((constant200_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant201_out1 = self.constant201.val();
        let add130_out1 = mul114_out1.add((constant201_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear71_out1 = self.linear71.forward(add130_out1);
        let mul115_out1 = linear71_out1.clone().mul(linear71_out1.clone());
        let mul116_out1 = linear71_out1.clone().mul(mul115_out1);
        let mul117_out1 = (constant24_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul116_out1);
        let add131_out1 = linear71_out1.clone().add(mul117_out1);
        let mul118_out1 = (constant25_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add131_out1);
        let tanh12_out1 = mul118_out1.tanh();
        let add132_out1 = (constant26_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh12_out1);
        let mul119_out1 = linear71_out1.mul(add132_out1);
        let mul120_out1 = (constant27_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul119_out1);
        let linear72_out1 = self.linear72.forward(mul120_out1);
        let add133_out1 = add128_out1.add(linear72_out1);
        let reducemean49_out1 = { add133_out1.clone().mean_dim(2usize) };
        let sub25_out1 = add133_out1.sub(reducemean49_out1);
        let pow25_out1 = sub25_out1
            .clone()
            .powf((constant5_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean50_out1 = { pow25_out1.mean_dim(2usize) };
        let add134_out1 =
            reducemean50_out1.add((constant6_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt25_out1 = add134_out1.sqrt();
        let div25_out1 = sub25_out1.div(sqrt25_out1);
        let constant206_out1 = self.constant206.val();
        let mul121_out1 = div25_out1.mul((constant206_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant207_out1 = self.constant207.val();
        let add135_out1 = mul121_out1.add((constant207_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape1_out1: [i64; 3] = {
            let axes = &add135_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let transpose50_out1 = add135_out1.permute([1, 0, 2]);
        let gather1_out1 = shape1_out1[0] as i64;
        let linear73_out1 = self.linear73.forward(transpose50_out1);
        let unsqueeze1_out1 = [gather1_out1 as i64];
        let constant211_out1 = self.constant211.val();
        let add136_out1 = (constant211_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear73_out1);
        let constant212_out1: [i64; 1] = [1i64];
        let concat1_out1: [i64; 3usize] = [
            &unsqueeze1_out1[..],
            &constant212_out1[..],
            &constant212_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape50_out1 = add136_out1.reshape([196, -1, 2, 768]);
        let constant214_out1 = self.constant214.val();
        let tile1_out1 = {
            let __repeats: alloc::vec::Vec<usize> =
                concat1_out1.iter().map(|&v| v as usize).collect();
            constant214_out1.repeat(&__repeats)
        };
        let unsqueeze2_out1: Tensor<B, 5> = reshape50_out1.unsqueeze_dims::<5>(&[0]);
        let transpose51_out1 = tile1_out1.permute([1, 0, 2]);
        let transpose52_out1 = unsqueeze2_out1.permute([3, 1, 2, 0, 4]);
        let linear74_out1 = self.linear74.forward(transpose51_out1);
        let squeeze1_out1 = transpose52_out1.squeeze_dims::<4>(&[3]);
        let constant217_out1 = self.constant217.val();
        let add137_out1 = (constant217_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear74_out1);
        let gather2_out1 = {
            let sliced = squeeze1_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather3_out1 = {
            let sliced = squeeze1_out1.slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape51_out1 = add137_out1.reshape([1, -1, 64]);
        let reshape52_out1 = gather2_out1.reshape([196, -1, 64]);
        let reshape53_out1 = gather3_out1.reshape([196, -1, 64]);
        let transpose53_out1 = reshape51_out1.permute([1, 0, 2]);
        let transpose54_out1 = reshape53_out1.permute([1, 0, 2]);
        let transpose55_out1 = reshape52_out1.permute([1, 2, 0]);
        let constant221_out1 = self.constant221.val();
        let mul122_out1 =
            transpose53_out1.mul((constant221_out1).unsqueeze_dims(&[0isize, 1isize]));
        let matmul99_out1 = mul122_out1.matmul(transpose55_out1);
        let softmax13_out1 = burn::tensor::activation::softmax(matmul99_out1, 2);
        let matmul100_out1 = softmax13_out1.matmul(transpose54_out1);
        let transpose56_out1 = matmul100_out1.permute([1, 0, 2]);
        let reshape54_out1 = transpose56_out1.reshape([-1, 768]);
        let linear75_out1 = self.linear75.forward(reshape54_out1);
        let reshape55_out1 = linear75_out1.reshape([1, -1, 768]);
        let transpose57_out1 = reshape55_out1.permute([1, 0, 2]);
        let reducemean51_out1 = { transpose57_out1.clone().mean_dim(2usize) };
        let sub26_out1 = transpose57_out1.clone().sub(reducemean51_out1);
        let pow26_out1 = sub26_out1
            .clone()
            .powf((constant5_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean52_out1 = { pow26_out1.mean_dim(2usize) };
        let add138_out1 = reducemean52_out1.add((constant6_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt26_out1 = add138_out1.sqrt();
        let div26_out1 = sub26_out1.div(sqrt26_out1);
        let constant226_out1 = self.constant226.val();
        let mul123_out1 = div26_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant227_out1 = self.constant227.val();
        let add139_out1 = mul123_out1.add((constant227_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear76_out1 = self.linear76.forward(add139_out1);
        let mul124_out1 = linear76_out1.clone().mul(linear76_out1.clone());
        let mul125_out1 = linear76_out1.clone().mul(mul124_out1);
        let mul126_out1 = (constant24_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul125_out1);
        let add140_out1 = linear76_out1.clone().add(mul126_out1);
        let mul127_out1 = (constant25_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add140_out1);
        let tanh13_out1 = mul127_out1.tanh();
        let add141_out1 = (constant26_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh13_out1);
        let mul128_out1 = linear76_out1.mul(add141_out1);
        let mul129_out1 = (constant27_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul128_out1);
        let linear77_out1 = self.linear77.forward(mul129_out1);
        let add142_out1 = transpose57_out1.add(linear77_out1);
        let gather4_out1 = {
            let sliced = add142_out1.slice(s![.., 0, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        gather4_out1
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
    submodule8: Submodule8<B>,
    submodule9: Submodule9<B>,
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
        let submodule8 = Submodule8::new(device);
        let submodule9 = Submodule9::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            submodule6,
            submodule7,
            submodule8,
            submodule9,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 2> {
        let (
            add12_out1,
            constant5_out1,
            constant6_out1,
            constant24_out1,
            constant25_out1,
            constant26_out1,
            constant27_out1,
        ) = self.submodule1.forward(pixel_values);
        let add29_out1 = self.submodule2.forward(
            add12_out1,
            constant5_out1.clone(),
            constant6_out1.clone(),
            constant24_out1.clone(),
            constant25_out1.clone(),
            constant26_out1.clone(),
            constant27_out1.clone(),
        );
        let add45_out1 = self.submodule3.forward(
            add29_out1,
            constant5_out1.clone(),
            constant6_out1.clone(),
            constant24_out1.clone(),
            constant25_out1.clone(),
            constant26_out1.clone(),
            constant27_out1.clone(),
        );
        let add62_out1 = self.submodule4.forward(
            add45_out1,
            constant5_out1.clone(),
            constant6_out1.clone(),
            constant24_out1.clone(),
            constant25_out1.clone(),
            constant26_out1.clone(),
            constant27_out1.clone(),
        );
        let add78_out1 = self.submodule5.forward(
            add62_out1,
            constant5_out1.clone(),
            constant6_out1.clone(),
            constant24_out1.clone(),
            constant25_out1.clone(),
            constant26_out1.clone(),
            constant27_out1.clone(),
        );
        let add95_out1 = self.submodule6.forward(
            add78_out1,
            constant5_out1.clone(),
            constant6_out1.clone(),
            constant24_out1.clone(),
            constant25_out1.clone(),
            constant26_out1.clone(),
            constant27_out1.clone(),
        );
        let add111_out1 = self.submodule7.forward(
            add95_out1,
            constant5_out1.clone(),
            constant6_out1.clone(),
            constant24_out1.clone(),
            constant25_out1.clone(),
            constant26_out1.clone(),
            constant27_out1.clone(),
        );
        let add128_out1 = self.submodule8.forward(
            add111_out1,
            constant5_out1.clone(),
            constant6_out1.clone(),
            constant24_out1.clone(),
            constant25_out1.clone(),
            constant26_out1.clone(),
            constant27_out1.clone(),
        );
        let gather4_out1 = self.submodule9.forward(
            add128_out1,
            constant5_out1,
            constant6_out1,
            constant24_out1,
            constant25_out1,
            constant26_out1,
            constant27_out1,
        );
        gather4_out1
    }
}
