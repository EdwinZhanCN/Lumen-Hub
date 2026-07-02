// Generated from ONNX "src/siglip2-so400m/text.fp32.onnx" by burn-onnx
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    constant1: burn::module::Param<Tensor<B, 2>>,
    constant4: burn::module::Param<Tensor<B, 2, Int>>,
    constant6: burn::module::Param<Tensor<B, 2>>,
    constant7: burn::module::Param<Tensor<B, 1>>,
    constant8: burn::module::Param<Tensor<B, 1>>,
    constant9: burn::module::Param<Tensor<B, 1>>,
    constant10: burn::module::Param<Tensor<B, 1>>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    constant15: burn::module::Param<Tensor<B, 1>>,
    constant16: burn::module::Param<Tensor<B, 1>>,
    constant17: burn::module::Param<Tensor<B, 1>>,
    linear4: Linear<B>,
    constant24: burn::module::Param<Tensor<B, 1>>,
    constant25: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant1: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2>::zeros([256000, 1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [256000, 1152].into(),
        );
        let constant4: burn::module::Param<Tensor<B, 2, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2, Int>::zeros([1, 64], (device, burn::tensor::DType::I64))
            },
            device.clone(),
            false,
            [1, 64].into(),
        );
        let constant6: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2>::zeros([64, 1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [64, 1152].into(),
        );
        let constant7: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant8: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant9: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant10: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear1 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear2 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear3 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant15: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant16: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant17: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear4 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant24: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant25: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        Self {
            constant1,
            constant4,
            constant6,
            constant7,
            constant8,
            constant9,
            constant10,
            linear1,
            linear2,
            linear3,
            constant15,
            constant16,
            constant17,
            linear4,
            constant24,
            constant25,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        [i64; 1],
        [i64; 1],
        [i64; 1],
    ) {
        let shape1_out1: [i64; 2] = {
            let axes = &input_ids.clone().dims()[0..2];
            let mut output = [0i64; 2];
            for i in 0..2 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant1_out1 = self.constant1.val();
        let gather1_out1 = constant1_out1.take::<2, 3>(0, input_ids);
        let gather2_out1 = shape1_out1[1] as i64;
        let unsqueeze1_out1 = [gather2_out1 as i64];
        let constant4_out1 = self.constant4.val();
        let slice1_out1 = constant4_out1.slice(s![.., 0..unsqueeze1_out1[0]]);
        let constant6_out1 = self.constant6.val();
        let gather3_out1 = constant6_out1.take::<2, 3>(0, slice1_out1);
        let add1_out1 = gather1_out1.add(gather3_out1);
        let reducemean1_out1 = { add1_out1.clone().mean_dim(2usize) };
        let sub1_out1 = add1_out1.clone().sub(reducemean1_out1);
        let constant7_out1 = self.constant7.val();
        let pow1_out1 = sub1_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean2_out1 = { pow1_out1.mean_dim(2usize) };
        let constant8_out1 = self.constant8.val();
        let add2_out1 =
            reducemean2_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt1_out1 = add2_out1.sqrt();
        let div1_out1 = sub1_out1.div(sqrt1_out1);
        let constant9_out1 = self.constant9.val();
        let mul1_out1 = div1_out1.mul((constant9_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant10_out1 = self.constant10.val();
        let add3_out1 = mul1_out1.add((constant10_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape2_out1: [i64; 3] = {
            let axes = &add3_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear1_out1 = self.linear1.forward(add3_out1.clone());
        let linear2_out1 = self.linear2.forward(add3_out1.clone());
        let linear3_out1 = self.linear3.forward(add3_out1);
        let gather4_out1 = shape2_out1[0] as i64;
        let gather5_out1 = shape2_out1[1] as i64;
        let constant15_out1 = self.constant15.val();
        let add4_out1 = (constant15_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear1_out1);
        let constant16_out1 = self.constant16.val();
        let add5_out1 = (constant16_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear2_out1);
        let constant17_out1 = self.constant17.val();
        let add6_out1 = (constant17_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear3_out1);
        let unsqueeze2_out1 = [gather4_out1 as i64];
        let unsqueeze3_out1 = [gather5_out1 as i64];
        let constant18_out1: [i64; 1] = [16i64];
        let constant19_out1: [i64; 1] = [72i64];
        let concat1_out1: [i64; 4usize] = [
            &unsqueeze2_out1[..],
            &unsqueeze3_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let constant20_out1: [i64; 1] = [1152i64];
        let concat2_out1: [i64; 3usize] = [
            &unsqueeze2_out1[..],
            &unsqueeze3_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape1_out1 = add4_out1.reshape(concat1_out1);
        let reshape2_out1 = add5_out1.reshape(concat1_out1);
        let reshape3_out1 = add6_out1.reshape(concat1_out1);
        let transpose1_out1 = reshape1_out1.permute([0, 2, 1, 3]);
        let transpose2_out1 = reshape3_out1.permute([0, 2, 1, 3]);
        let transpose3_out1 = reshape2_out1.permute([0, 2, 3, 1]);
        let matmul4_k_corrected = transpose3_out1.permute([0, 1, 3, 2]);
        let (matmul5_out1,) = {
            let q = transpose1_out1;
            let k = matmul4_k_corrected;
            let v = transpose2_out1;
            let matmul5_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul5_out1,)
        };
        let transpose4_out1 = matmul5_out1.permute([0, 2, 1, 3]);
        let reshape4_out1 = transpose4_out1.reshape(concat2_out1);
        let linear4_out1 = self.linear4.forward(reshape4_out1);
        let add7_out1 = add1_out1.add(linear4_out1);
        let reducemean3_out1 = { add7_out1.clone().mean_dim(2usize) };
        let sub2_out1 = add7_out1.clone().sub(reducemean3_out1);
        let pow2_out1 = sub2_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean4_out1 = { pow2_out1.mean_dim(2usize) };
        let add8_out1 =
            reducemean4_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt2_out1 = add8_out1.sqrt();
        let div2_out1 = sub2_out1.div(sqrt2_out1);
        let constant24_out1 = self.constant24.val();
        let mul4_out1 = div2_out1.mul((constant24_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant25_out1 = self.constant25.val();
        let add9_out1 = mul4_out1.add((constant25_out1).unsqueeze_dims(&[0isize, 1isize]));
        (
            add9_out1,
            add7_out1,
            constant7_out1,
            constant8_out1,
            constant18_out1,
            constant19_out1,
            constant20_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    linear5: Linear<B>,
    constant28: burn::module::Param<Tensor<B, 1>>,
    constant29: burn::module::Param<Tensor<B, 1>>,
    constant30: burn::module::Param<Tensor<B, 1>>,
    constant31: burn::module::Param<Tensor<B, 1>>,
    linear6: Linear<B>,
    constant34: burn::module::Param<Tensor<B, 1>>,
    constant35: burn::module::Param<Tensor<B, 1>>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    linear9: Linear<B>,
    constant39: burn::module::Param<Tensor<B, 1>>,
    constant40: burn::module::Param<Tensor<B, 1>>,
    constant41: burn::module::Param<Tensor<B, 1>>,
    linear10: Linear<B>,
    constant44: burn::module::Param<Tensor<B, 1>>,
    constant45: burn::module::Param<Tensor<B, 1>>,
    linear11: Linear<B>,
    linear12: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let linear5 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let constant28: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant29: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant30: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant31: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let linear6 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant34: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant35: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear7 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear8 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear9 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant39: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant40: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant41: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear10 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant44: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant45: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear11 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear12 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            linear5,
            constant28,
            constant29,
            constant30,
            constant31,
            linear6,
            constant34,
            constant35,
            linear7,
            linear8,
            linear9,
            constant39,
            constant40,
            constant41,
            linear10,
            constant44,
            constant45,
            linear11,
            linear12,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add9_out1: Tensor<B, 3>,
        add7_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    ) {
        let linear5_out1 = self.linear5.forward(add9_out1);
        let mul5_out1 = linear5_out1.clone().mul(linear5_out1.clone());
        let mul6_out1 = linear5_out1.clone().mul(mul5_out1);
        let constant28_out1 = self.constant28.val();
        let mul7_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul6_out1);
        let add10_out1 = linear5_out1.clone().add(mul7_out1);
        let constant29_out1 = self.constant29.val();
        let mul8_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add10_out1);
        let tanh1_out1 = mul8_out1.tanh();
        let constant30_out1 = self.constant30.val();
        let add11_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh1_out1);
        let mul9_out1 = linear5_out1.mul(add11_out1);
        let constant31_out1 = self.constant31.val();
        let mul10_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul9_out1);
        let linear6_out1 = self.linear6.forward(mul10_out1);
        let add12_out1 = add7_out1.add(linear6_out1);
        let reducemean5_out1 = { add12_out1.clone().mean_dim(2usize) };
        let sub3_out1 = add12_out1.clone().sub(reducemean5_out1);
        let pow3_out1 = sub3_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean6_out1 = { pow3_out1.mean_dim(2usize) };
        let add13_out1 =
            reducemean6_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt3_out1 = add13_out1.sqrt();
        let div3_out1 = sub3_out1.div(sqrt3_out1);
        let constant34_out1 = self.constant34.val();
        let mul11_out1 = div3_out1.mul((constant34_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant35_out1 = self.constant35.val();
        let add14_out1 = mul11_out1.add((constant35_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape3_out1: [i64; 3] = {
            let axes = &add14_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear7_out1 = self.linear7.forward(add14_out1.clone());
        let linear8_out1 = self.linear8.forward(add14_out1.clone());
        let linear9_out1 = self.linear9.forward(add14_out1);
        let gather6_out1 = shape3_out1[0] as i64;
        let gather7_out1 = shape3_out1[1] as i64;
        let constant39_out1 = self.constant39.val();
        let add15_out1 = (constant39_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear7_out1);
        let constant40_out1 = self.constant40.val();
        let add16_out1 = (constant40_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear8_out1);
        let constant41_out1 = self.constant41.val();
        let add17_out1 = (constant41_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear9_out1);
        let unsqueeze4_out1 = [gather6_out1 as i64];
        let unsqueeze5_out1 = [gather7_out1 as i64];
        let concat3_out1: [i64; 4usize] = [
            &unsqueeze4_out1[..],
            &unsqueeze5_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat4_out1: [i64; 3usize] = [
            &unsqueeze4_out1[..],
            &unsqueeze5_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape5_out1 = add15_out1.reshape(concat3_out1);
        let reshape6_out1 = add16_out1.reshape(concat3_out1);
        let reshape7_out1 = add17_out1.reshape(concat3_out1);
        let transpose5_out1 = reshape5_out1.permute([0, 2, 1, 3]);
        let transpose6_out1 = reshape7_out1.permute([0, 2, 1, 3]);
        let transpose7_out1 = reshape6_out1.permute([0, 2, 3, 1]);
        let matmul12_k_corrected = transpose7_out1.permute([0, 1, 3, 2]);
        let (matmul13_out1,) = {
            let q = transpose5_out1;
            let k = matmul12_k_corrected;
            let v = transpose6_out1;
            let matmul13_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul13_out1,)
        };
        let transpose8_out1 = matmul13_out1.permute([0, 2, 1, 3]);
        let reshape8_out1 = transpose8_out1.reshape(concat4_out1);
        let linear10_out1 = self.linear10.forward(reshape8_out1);
        let add18_out1 = add12_out1.add(linear10_out1);
        let reducemean7_out1 = { add18_out1.clone().mean_dim(2usize) };
        let sub4_out1 = add18_out1.clone().sub(reducemean7_out1);
        let pow4_out1 = sub4_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean8_out1 = { pow4_out1.mean_dim(2usize) };
        let add19_out1 = reducemean8_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt4_out1 = add19_out1.sqrt();
        let div4_out1 = sub4_out1.div(sqrt4_out1);
        let constant44_out1 = self.constant44.val();
        let mul14_out1 = div4_out1.mul((constant44_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant45_out1 = self.constant45.val();
        let add20_out1 = mul14_out1.add((constant45_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear11_out1 = self.linear11.forward(add20_out1);
        let mul15_out1 = linear11_out1.clone().mul(linear11_out1.clone());
        let mul16_out1 = linear11_out1.clone().mul(mul15_out1);
        let mul17_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul16_out1);
        let add21_out1 = linear11_out1.clone().add(mul17_out1);
        let mul18_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add21_out1);
        let tanh2_out1 = mul18_out1.tanh();
        let add22_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh2_out1);
        let mul19_out1 = linear11_out1.mul(add22_out1);
        let mul20_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul19_out1);
        let linear12_out1 = self.linear12.forward(mul20_out1);
        let add23_out1 = add18_out1.add(linear12_out1);
        (
            add23_out1,
            constant28_out1,
            constant29_out1,
            constant30_out1,
            constant31_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    constant50: burn::module::Param<Tensor<B, 1>>,
    constant51: burn::module::Param<Tensor<B, 1>>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    linear15: Linear<B>,
    constant55: burn::module::Param<Tensor<B, 1>>,
    constant56: burn::module::Param<Tensor<B, 1>>,
    constant57: burn::module::Param<Tensor<B, 1>>,
    linear16: Linear<B>,
    constant60: burn::module::Param<Tensor<B, 1>>,
    constant61: burn::module::Param<Tensor<B, 1>>,
    linear17: Linear<B>,
    linear18: Linear<B>,
    constant66: burn::module::Param<Tensor<B, 1>>,
    constant67: burn::module::Param<Tensor<B, 1>>,
    linear19: Linear<B>,
    linear20: Linear<B>,
    linear21: Linear<B>,
    constant71: burn::module::Param<Tensor<B, 1>>,
    constant72: burn::module::Param<Tensor<B, 1>>,
    constant73: burn::module::Param<Tensor<B, 1>>,
    linear22: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant50: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant51: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear13 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear14 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear15 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant55: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant56: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant57: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear16 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant60: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant61: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear17 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear18 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant66: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant67: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear19 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear20 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear21 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant71: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant72: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant73: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear22 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant50,
            constant51,
            linear13,
            linear14,
            linear15,
            constant55,
            constant56,
            constant57,
            linear16,
            constant60,
            constant61,
            linear17,
            linear18,
            constant66,
            constant67,
            linear19,
            linear20,
            linear21,
            constant71,
            constant72,
            constant73,
            linear22,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add23_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean9_out1 = { add23_out1.clone().mean_dim(2usize) };
        let sub5_out1 = add23_out1.clone().sub(reducemean9_out1);
        let pow5_out1 = sub5_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean10_out1 = { pow5_out1.mean_dim(2usize) };
        let add24_out1 =
            reducemean10_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt5_out1 = add24_out1.sqrt();
        let div5_out1 = sub5_out1.div(sqrt5_out1);
        let constant50_out1 = self.constant50.val();
        let mul21_out1 = div5_out1.mul((constant50_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant51_out1 = self.constant51.val();
        let add25_out1 = mul21_out1.add((constant51_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape4_out1: [i64; 3] = {
            let axes = &add25_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear13_out1 = self.linear13.forward(add25_out1.clone());
        let linear14_out1 = self.linear14.forward(add25_out1.clone());
        let linear15_out1 = self.linear15.forward(add25_out1);
        let gather8_out1 = shape4_out1[0] as i64;
        let gather9_out1 = shape4_out1[1] as i64;
        let constant55_out1 = self.constant55.val();
        let add26_out1 = (constant55_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear13_out1);
        let constant56_out1 = self.constant56.val();
        let add27_out1 = (constant56_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear14_out1);
        let constant57_out1 = self.constant57.val();
        let add28_out1 = (constant57_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear15_out1);
        let unsqueeze6_out1 = [gather8_out1 as i64];
        let unsqueeze7_out1 = [gather9_out1 as i64];
        let concat5_out1: [i64; 4usize] = [
            &unsqueeze6_out1[..],
            &unsqueeze7_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat6_out1: [i64; 3usize] = [
            &unsqueeze6_out1[..],
            &unsqueeze7_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape9_out1 = add26_out1.reshape(concat5_out1);
        let reshape10_out1 = add27_out1.reshape(concat5_out1);
        let reshape11_out1 = add28_out1.reshape(concat5_out1);
        let transpose9_out1 = reshape9_out1.permute([0, 2, 1, 3]);
        let transpose10_out1 = reshape11_out1.permute([0, 2, 1, 3]);
        let transpose11_out1 = reshape10_out1.permute([0, 2, 3, 1]);
        let matmul20_k_corrected = transpose11_out1.permute([0, 1, 3, 2]);
        let (matmul21_out1,) = {
            let q = transpose9_out1;
            let k = matmul20_k_corrected;
            let v = transpose10_out1;
            let matmul21_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul21_out1,)
        };
        let transpose12_out1 = matmul21_out1.permute([0, 2, 1, 3]);
        let reshape12_out1 = transpose12_out1.reshape(concat6_out1);
        let linear16_out1 = self.linear16.forward(reshape12_out1);
        let add29_out1 = add23_out1.add(linear16_out1);
        let reducemean11_out1 = { add29_out1.clone().mean_dim(2usize) };
        let sub6_out1 = add29_out1.clone().sub(reducemean11_out1);
        let pow6_out1 = sub6_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean12_out1 = { pow6_out1.mean_dim(2usize) };
        let add30_out1 =
            reducemean12_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt6_out1 = add30_out1.sqrt();
        let div6_out1 = sub6_out1.div(sqrt6_out1);
        let constant60_out1 = self.constant60.val();
        let mul24_out1 = div6_out1.mul((constant60_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant61_out1 = self.constant61.val();
        let add31_out1 = mul24_out1.add((constant61_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear17_out1 = self.linear17.forward(add31_out1);
        let mul25_out1 = linear17_out1.clone().mul(linear17_out1.clone());
        let mul26_out1 = linear17_out1.clone().mul(mul25_out1);
        let mul27_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul26_out1);
        let add32_out1 = linear17_out1.clone().add(mul27_out1);
        let mul28_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add32_out1);
        let tanh3_out1 = mul28_out1.tanh();
        let add33_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh3_out1);
        let mul29_out1 = linear17_out1.mul(add33_out1);
        let mul30_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul29_out1);
        let linear18_out1 = self.linear18.forward(mul30_out1);
        let add34_out1 = add29_out1.add(linear18_out1);
        let reducemean13_out1 = { add34_out1.clone().mean_dim(2usize) };
        let sub7_out1 = add34_out1.clone().sub(reducemean13_out1);
        let pow7_out1 = sub7_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean14_out1 = { pow7_out1.mean_dim(2usize) };
        let add35_out1 = reducemean14_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt7_out1 = add35_out1.sqrt();
        let div7_out1 = sub7_out1.div(sqrt7_out1);
        let constant66_out1 = self.constant66.val();
        let mul31_out1 = div7_out1.mul((constant66_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant67_out1 = self.constant67.val();
        let add36_out1 = mul31_out1.add((constant67_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape5_out1: [i64; 3] = {
            let axes = &add36_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear19_out1 = self.linear19.forward(add36_out1.clone());
        let linear20_out1 = self.linear20.forward(add36_out1.clone());
        let linear21_out1 = self.linear21.forward(add36_out1);
        let gather10_out1 = shape5_out1[0] as i64;
        let gather11_out1 = shape5_out1[1] as i64;
        let constant71_out1 = self.constant71.val();
        let add37_out1 = (constant71_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear19_out1);
        let constant72_out1 = self.constant72.val();
        let add38_out1 = (constant72_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear20_out1);
        let constant73_out1 = self.constant73.val();
        let add39_out1 = (constant73_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear21_out1);
        let unsqueeze8_out1 = [gather10_out1 as i64];
        let unsqueeze9_out1 = [gather11_out1 as i64];
        let concat7_out1: [i64; 4usize] = [
            &unsqueeze8_out1[..],
            &unsqueeze9_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat8_out1: [i64; 3usize] = [
            &unsqueeze8_out1[..],
            &unsqueeze9_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape13_out1 = add37_out1.reshape(concat7_out1);
        let reshape14_out1 = add38_out1.reshape(concat7_out1);
        let reshape15_out1 = add39_out1.reshape(concat7_out1);
        let transpose13_out1 = reshape13_out1.permute([0, 2, 1, 3]);
        let transpose14_out1 = reshape15_out1.permute([0, 2, 1, 3]);
        let transpose15_out1 = reshape14_out1.permute([0, 2, 3, 1]);
        let matmul28_k_corrected = transpose15_out1.permute([0, 1, 3, 2]);
        let (matmul29_out1,) = {
            let q = transpose13_out1;
            let k = matmul28_k_corrected;
            let v = transpose14_out1;
            let matmul29_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul29_out1,)
        };
        let transpose16_out1 = matmul29_out1.permute([0, 2, 1, 3]);
        let reshape16_out1 = transpose16_out1.reshape(concat8_out1);
        let linear22_out1 = self.linear22.forward(reshape16_out1);
        let add40_out1 = add34_out1.add(linear22_out1);
        add40_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    constant76: burn::module::Param<Tensor<B, 1>>,
    constant77: burn::module::Param<Tensor<B, 1>>,
    linear23: Linear<B>,
    linear24: Linear<B>,
    constant82: burn::module::Param<Tensor<B, 1>>,
    constant83: burn::module::Param<Tensor<B, 1>>,
    linear25: Linear<B>,
    linear26: Linear<B>,
    linear27: Linear<B>,
    constant87: burn::module::Param<Tensor<B, 1>>,
    constant88: burn::module::Param<Tensor<B, 1>>,
    constant89: burn::module::Param<Tensor<B, 1>>,
    linear28: Linear<B>,
    constant92: burn::module::Param<Tensor<B, 1>>,
    constant93: burn::module::Param<Tensor<B, 1>>,
    linear29: Linear<B>,
    linear30: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant76: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant77: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear23 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear24 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant82: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant83: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear25 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear26 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear27 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant87: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant88: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant89: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear28 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant92: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant93: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear29 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear30 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant76,
            constant77,
            linear23,
            linear24,
            constant82,
            constant83,
            linear25,
            linear26,
            linear27,
            constant87,
            constant88,
            constant89,
            linear28,
            constant92,
            constant93,
            linear29,
            linear30,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add40_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean15_out1 = { add40_out1.clone().mean_dim(2usize) };
        let sub8_out1 = add40_out1.clone().sub(reducemean15_out1);
        let pow8_out1 = sub8_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean16_out1 = { pow8_out1.mean_dim(2usize) };
        let add41_out1 =
            reducemean16_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt8_out1 = add41_out1.sqrt();
        let div8_out1 = sub8_out1.div(sqrt8_out1);
        let constant76_out1 = self.constant76.val();
        let mul34_out1 = div8_out1.mul((constant76_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant77_out1 = self.constant77.val();
        let add42_out1 = mul34_out1.add((constant77_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear23_out1 = self.linear23.forward(add42_out1);
        let mul35_out1 = linear23_out1.clone().mul(linear23_out1.clone());
        let mul36_out1 = linear23_out1.clone().mul(mul35_out1);
        let mul37_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul36_out1);
        let add43_out1 = linear23_out1.clone().add(mul37_out1);
        let mul38_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add43_out1);
        let tanh4_out1 = mul38_out1.tanh();
        let add44_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh4_out1);
        let mul39_out1 = linear23_out1.mul(add44_out1);
        let mul40_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul39_out1);
        let linear24_out1 = self.linear24.forward(mul40_out1);
        let add45_out1 = add40_out1.add(linear24_out1);
        let reducemean17_out1 = { add45_out1.clone().mean_dim(2usize) };
        let sub9_out1 = add45_out1.clone().sub(reducemean17_out1);
        let pow9_out1 = sub9_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean18_out1 = { pow9_out1.mean_dim(2usize) };
        let add46_out1 =
            reducemean18_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt9_out1 = add46_out1.sqrt();
        let div9_out1 = sub9_out1.div(sqrt9_out1);
        let constant82_out1 = self.constant82.val();
        let mul41_out1 = div9_out1.mul((constant82_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant83_out1 = self.constant83.val();
        let add47_out1 = mul41_out1.add((constant83_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape6_out1: [i64; 3] = {
            let axes = &add47_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear25_out1 = self.linear25.forward(add47_out1.clone());
        let linear26_out1 = self.linear26.forward(add47_out1.clone());
        let linear27_out1 = self.linear27.forward(add47_out1);
        let gather12_out1 = shape6_out1[0] as i64;
        let gather13_out1 = shape6_out1[1] as i64;
        let constant87_out1 = self.constant87.val();
        let add48_out1 = (constant87_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear25_out1);
        let constant88_out1 = self.constant88.val();
        let add49_out1 = (constant88_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear26_out1);
        let constant89_out1 = self.constant89.val();
        let add50_out1 = (constant89_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear27_out1);
        let unsqueeze10_out1 = [gather12_out1 as i64];
        let unsqueeze11_out1 = [gather13_out1 as i64];
        let concat9_out1: [i64; 4usize] = [
            &unsqueeze10_out1[..],
            &unsqueeze11_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat10_out1: [i64; 3usize] = [
            &unsqueeze10_out1[..],
            &unsqueeze11_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape17_out1 = add48_out1.reshape(concat9_out1);
        let reshape18_out1 = add49_out1.reshape(concat9_out1);
        let reshape19_out1 = add50_out1.reshape(concat9_out1);
        let transpose17_out1 = reshape17_out1.permute([0, 2, 1, 3]);
        let transpose18_out1 = reshape19_out1.permute([0, 2, 1, 3]);
        let transpose19_out1 = reshape18_out1.permute([0, 2, 3, 1]);
        let matmul36_k_corrected = transpose19_out1.permute([0, 1, 3, 2]);
        let (matmul37_out1,) = {
            let q = transpose17_out1;
            let k = matmul36_k_corrected;
            let v = transpose18_out1;
            let matmul37_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul37_out1,)
        };
        let transpose20_out1 = matmul37_out1.permute([0, 2, 1, 3]);
        let reshape20_out1 = transpose20_out1.reshape(concat10_out1);
        let linear28_out1 = self.linear28.forward(reshape20_out1);
        let add51_out1 = add45_out1.add(linear28_out1);
        let reducemean19_out1 = { add51_out1.clone().mean_dim(2usize) };
        let sub10_out1 = add51_out1.clone().sub(reducemean19_out1);
        let pow10_out1 = sub10_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean20_out1 = { pow10_out1.mean_dim(2usize) };
        let add52_out1 = reducemean20_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt10_out1 = add52_out1.sqrt();
        let div10_out1 = sub10_out1.div(sqrt10_out1);
        let constant92_out1 = self.constant92.val();
        let mul44_out1 = div10_out1.mul((constant92_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant93_out1 = self.constant93.val();
        let add53_out1 = mul44_out1.add((constant93_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear29_out1 = self.linear29.forward(add53_out1);
        let mul45_out1 = linear29_out1.clone().mul(linear29_out1.clone());
        let mul46_out1 = linear29_out1.clone().mul(mul45_out1);
        let mul47_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul46_out1);
        let add54_out1 = linear29_out1.clone().add(mul47_out1);
        let mul48_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add54_out1);
        let tanh5_out1 = mul48_out1.tanh();
        let add55_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh5_out1);
        let mul49_out1 = linear29_out1.mul(add55_out1);
        let mul50_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul49_out1);
        let linear30_out1 = self.linear30.forward(mul50_out1);
        let add56_out1 = add51_out1.add(linear30_out1);
        add56_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5<B: Backend> {
    constant98: burn::module::Param<Tensor<B, 1>>,
    constant99: burn::module::Param<Tensor<B, 1>>,
    linear31: Linear<B>,
    linear32: Linear<B>,
    linear33: Linear<B>,
    constant103: burn::module::Param<Tensor<B, 1>>,
    constant104: burn::module::Param<Tensor<B, 1>>,
    constant105: burn::module::Param<Tensor<B, 1>>,
    linear34: Linear<B>,
    constant108: burn::module::Param<Tensor<B, 1>>,
    constant109: burn::module::Param<Tensor<B, 1>>,
    linear35: Linear<B>,
    linear36: Linear<B>,
    constant114: burn::module::Param<Tensor<B, 1>>,
    constant115: burn::module::Param<Tensor<B, 1>>,
    linear37: Linear<B>,
    linear38: Linear<B>,
    linear39: Linear<B>,
    constant119: burn::module::Param<Tensor<B, 1>>,
    constant120: burn::module::Param<Tensor<B, 1>>,
    constant121: burn::module::Param<Tensor<B, 1>>,
    linear40: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule5<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant98: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant99: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear31 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear32 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear33 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant103: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant104: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant105: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear34 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant108: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant109: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear35 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear36 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant114: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant115: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear37 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear38 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear39 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant119: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant120: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant121: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear40 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant98,
            constant99,
            linear31,
            linear32,
            linear33,
            constant103,
            constant104,
            constant105,
            linear34,
            constant108,
            constant109,
            linear35,
            linear36,
            constant114,
            constant115,
            linear37,
            linear38,
            linear39,
            constant119,
            constant120,
            constant121,
            linear40,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add56_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean21_out1 = { add56_out1.clone().mean_dim(2usize) };
        let sub11_out1 = add56_out1.clone().sub(reducemean21_out1);
        let pow11_out1 = sub11_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean22_out1 = { pow11_out1.mean_dim(2usize) };
        let add57_out1 =
            reducemean22_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt11_out1 = add57_out1.sqrt();
        let div11_out1 = sub11_out1.div(sqrt11_out1);
        let constant98_out1 = self.constant98.val();
        let mul51_out1 = div11_out1.mul((constant98_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant99_out1 = self.constant99.val();
        let add58_out1 = mul51_out1.add((constant99_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape7_out1: [i64; 3] = {
            let axes = &add58_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear31_out1 = self.linear31.forward(add58_out1.clone());
        let linear32_out1 = self.linear32.forward(add58_out1.clone());
        let linear33_out1 = self.linear33.forward(add58_out1);
        let gather14_out1 = shape7_out1[0] as i64;
        let gather15_out1 = shape7_out1[1] as i64;
        let constant103_out1 = self.constant103.val();
        let add59_out1 = (constant103_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear31_out1);
        let constant104_out1 = self.constant104.val();
        let add60_out1 = (constant104_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear32_out1);
        let constant105_out1 = self.constant105.val();
        let add61_out1 = (constant105_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear33_out1);
        let unsqueeze12_out1 = [gather14_out1 as i64];
        let unsqueeze13_out1 = [gather15_out1 as i64];
        let concat11_out1: [i64; 4usize] = [
            &unsqueeze12_out1[..],
            &unsqueeze13_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat12_out1: [i64; 3usize] = [
            &unsqueeze12_out1[..],
            &unsqueeze13_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape21_out1 = add59_out1.reshape(concat11_out1);
        let reshape22_out1 = add60_out1.reshape(concat11_out1);
        let reshape23_out1 = add61_out1.reshape(concat11_out1);
        let transpose21_out1 = reshape21_out1.permute([0, 2, 1, 3]);
        let transpose22_out1 = reshape23_out1.permute([0, 2, 1, 3]);
        let transpose23_out1 = reshape22_out1.permute([0, 2, 3, 1]);
        let matmul44_k_corrected = transpose23_out1.permute([0, 1, 3, 2]);
        let (matmul45_out1,) = {
            let q = transpose21_out1;
            let k = matmul44_k_corrected;
            let v = transpose22_out1;
            let matmul45_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul45_out1,)
        };
        let transpose24_out1 = matmul45_out1.permute([0, 2, 1, 3]);
        let reshape24_out1 = transpose24_out1.reshape(concat12_out1);
        let linear34_out1 = self.linear34.forward(reshape24_out1);
        let add62_out1 = add56_out1.add(linear34_out1);
        let reducemean23_out1 = { add62_out1.clone().mean_dim(2usize) };
        let sub12_out1 = add62_out1.clone().sub(reducemean23_out1);
        let pow12_out1 = sub12_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean24_out1 = { pow12_out1.mean_dim(2usize) };
        let add63_out1 =
            reducemean24_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt12_out1 = add63_out1.sqrt();
        let div12_out1 = sub12_out1.div(sqrt12_out1);
        let constant108_out1 = self.constant108.val();
        let mul54_out1 = div12_out1.mul((constant108_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant109_out1 = self.constant109.val();
        let add64_out1 = mul54_out1.add((constant109_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear35_out1 = self.linear35.forward(add64_out1);
        let mul55_out1 = linear35_out1.clone().mul(linear35_out1.clone());
        let mul56_out1 = linear35_out1.clone().mul(mul55_out1);
        let mul57_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul56_out1);
        let add65_out1 = linear35_out1.clone().add(mul57_out1);
        let mul58_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add65_out1);
        let tanh6_out1 = mul58_out1.tanh();
        let add66_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh6_out1);
        let mul59_out1 = linear35_out1.mul(add66_out1);
        let mul60_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul59_out1);
        let linear36_out1 = self.linear36.forward(mul60_out1);
        let add67_out1 = add62_out1.add(linear36_out1);
        let reducemean25_out1 = { add67_out1.clone().mean_dim(2usize) };
        let sub13_out1 = add67_out1.clone().sub(reducemean25_out1);
        let pow13_out1 = sub13_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean26_out1 = { pow13_out1.mean_dim(2usize) };
        let add68_out1 = reducemean26_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt13_out1 = add68_out1.sqrt();
        let div13_out1 = sub13_out1.div(sqrt13_out1);
        let constant114_out1 = self.constant114.val();
        let mul61_out1 = div13_out1.mul((constant114_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant115_out1 = self.constant115.val();
        let add69_out1 = mul61_out1.add((constant115_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape8_out1: [i64; 3] = {
            let axes = &add69_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear37_out1 = self.linear37.forward(add69_out1.clone());
        let linear38_out1 = self.linear38.forward(add69_out1.clone());
        let linear39_out1 = self.linear39.forward(add69_out1);
        let gather16_out1 = shape8_out1[0] as i64;
        let gather17_out1 = shape8_out1[1] as i64;
        let constant119_out1 = self.constant119.val();
        let add70_out1 = (constant119_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear37_out1);
        let constant120_out1 = self.constant120.val();
        let add71_out1 = (constant120_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear38_out1);
        let constant121_out1 = self.constant121.val();
        let add72_out1 = (constant121_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear39_out1);
        let unsqueeze14_out1 = [gather16_out1 as i64];
        let unsqueeze15_out1 = [gather17_out1 as i64];
        let concat13_out1: [i64; 4usize] = [
            &unsqueeze14_out1[..],
            &unsqueeze15_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat14_out1: [i64; 3usize] = [
            &unsqueeze14_out1[..],
            &unsqueeze15_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape25_out1 = add70_out1.reshape(concat13_out1);
        let reshape26_out1 = add71_out1.reshape(concat13_out1);
        let reshape27_out1 = add72_out1.reshape(concat13_out1);
        let transpose25_out1 = reshape25_out1.permute([0, 2, 1, 3]);
        let transpose26_out1 = reshape27_out1.permute([0, 2, 1, 3]);
        let transpose27_out1 = reshape26_out1.permute([0, 2, 3, 1]);
        let matmul52_k_corrected = transpose27_out1.permute([0, 1, 3, 2]);
        let (matmul53_out1,) = {
            let q = transpose25_out1;
            let k = matmul52_k_corrected;
            let v = transpose26_out1;
            let matmul53_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul53_out1,)
        };
        let transpose28_out1 = matmul53_out1.permute([0, 2, 1, 3]);
        let reshape28_out1 = transpose28_out1.reshape(concat14_out1);
        let linear40_out1 = self.linear40.forward(reshape28_out1);
        let add73_out1 = add67_out1.add(linear40_out1);
        add73_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6<B: Backend> {
    constant124: burn::module::Param<Tensor<B, 1>>,
    constant125: burn::module::Param<Tensor<B, 1>>,
    linear41: Linear<B>,
    linear42: Linear<B>,
    constant130: burn::module::Param<Tensor<B, 1>>,
    constant131: burn::module::Param<Tensor<B, 1>>,
    linear43: Linear<B>,
    linear44: Linear<B>,
    linear45: Linear<B>,
    constant135: burn::module::Param<Tensor<B, 1>>,
    constant136: burn::module::Param<Tensor<B, 1>>,
    constant137: burn::module::Param<Tensor<B, 1>>,
    linear46: Linear<B>,
    constant140: burn::module::Param<Tensor<B, 1>>,
    constant141: burn::module::Param<Tensor<B, 1>>,
    linear47: Linear<B>,
    linear48: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule6<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant124: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant125: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear41 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear42 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant130: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant131: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear43 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear44 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear45 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant135: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant136: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant137: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear46 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant140: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant141: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear47 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear48 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant124,
            constant125,
            linear41,
            linear42,
            constant130,
            constant131,
            linear43,
            linear44,
            linear45,
            constant135,
            constant136,
            constant137,
            linear46,
            constant140,
            constant141,
            linear47,
            linear48,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add73_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean27_out1 = { add73_out1.clone().mean_dim(2usize) };
        let sub14_out1 = add73_out1.clone().sub(reducemean27_out1);
        let pow14_out1 = sub14_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean28_out1 = { pow14_out1.mean_dim(2usize) };
        let add74_out1 =
            reducemean28_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt14_out1 = add74_out1.sqrt();
        let div14_out1 = sub14_out1.div(sqrt14_out1);
        let constant124_out1 = self.constant124.val();
        let mul64_out1 = div14_out1.mul((constant124_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant125_out1 = self.constant125.val();
        let add75_out1 = mul64_out1.add((constant125_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear41_out1 = self.linear41.forward(add75_out1);
        let mul65_out1 = linear41_out1.clone().mul(linear41_out1.clone());
        let mul66_out1 = linear41_out1.clone().mul(mul65_out1);
        let mul67_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul66_out1);
        let add76_out1 = linear41_out1.clone().add(mul67_out1);
        let mul68_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add76_out1);
        let tanh7_out1 = mul68_out1.tanh();
        let add77_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh7_out1);
        let mul69_out1 = linear41_out1.mul(add77_out1);
        let mul70_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul69_out1);
        let linear42_out1 = self.linear42.forward(mul70_out1);
        let add78_out1 = add73_out1.add(linear42_out1);
        let reducemean29_out1 = { add78_out1.clone().mean_dim(2usize) };
        let sub15_out1 = add78_out1.clone().sub(reducemean29_out1);
        let pow15_out1 = sub15_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean30_out1 = { pow15_out1.mean_dim(2usize) };
        let add79_out1 =
            reducemean30_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt15_out1 = add79_out1.sqrt();
        let div15_out1 = sub15_out1.div(sqrt15_out1);
        let constant130_out1 = self.constant130.val();
        let mul71_out1 = div15_out1.mul((constant130_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant131_out1 = self.constant131.val();
        let add80_out1 = mul71_out1.add((constant131_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape9_out1: [i64; 3] = {
            let axes = &add80_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear43_out1 = self.linear43.forward(add80_out1.clone());
        let linear44_out1 = self.linear44.forward(add80_out1.clone());
        let linear45_out1 = self.linear45.forward(add80_out1);
        let gather18_out1 = shape9_out1[0] as i64;
        let gather19_out1 = shape9_out1[1] as i64;
        let constant135_out1 = self.constant135.val();
        let add81_out1 = (constant135_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear43_out1);
        let constant136_out1 = self.constant136.val();
        let add82_out1 = (constant136_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear44_out1);
        let constant137_out1 = self.constant137.val();
        let add83_out1 = (constant137_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear45_out1);
        let unsqueeze16_out1 = [gather18_out1 as i64];
        let unsqueeze17_out1 = [gather19_out1 as i64];
        let concat15_out1: [i64; 4usize] = [
            &unsqueeze16_out1[..],
            &unsqueeze17_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat16_out1: [i64; 3usize] = [
            &unsqueeze16_out1[..],
            &unsqueeze17_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape29_out1 = add81_out1.reshape(concat15_out1);
        let reshape30_out1 = add82_out1.reshape(concat15_out1);
        let reshape31_out1 = add83_out1.reshape(concat15_out1);
        let transpose29_out1 = reshape29_out1.permute([0, 2, 1, 3]);
        let transpose30_out1 = reshape31_out1.permute([0, 2, 1, 3]);
        let transpose31_out1 = reshape30_out1.permute([0, 2, 3, 1]);
        let matmul60_k_corrected = transpose31_out1.permute([0, 1, 3, 2]);
        let (matmul61_out1,) = {
            let q = transpose29_out1;
            let k = matmul60_k_corrected;
            let v = transpose30_out1;
            let matmul61_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul61_out1,)
        };
        let transpose32_out1 = matmul61_out1.permute([0, 2, 1, 3]);
        let reshape32_out1 = transpose32_out1.reshape(concat16_out1);
        let linear46_out1 = self.linear46.forward(reshape32_out1);
        let add84_out1 = add78_out1.add(linear46_out1);
        let reducemean31_out1 = { add84_out1.clone().mean_dim(2usize) };
        let sub16_out1 = add84_out1.clone().sub(reducemean31_out1);
        let pow16_out1 = sub16_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean32_out1 = { pow16_out1.mean_dim(2usize) };
        let add85_out1 = reducemean32_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt16_out1 = add85_out1.sqrt();
        let div16_out1 = sub16_out1.div(sqrt16_out1);
        let constant140_out1 = self.constant140.val();
        let mul74_out1 = div16_out1.mul((constant140_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant141_out1 = self.constant141.val();
        let add86_out1 = mul74_out1.add((constant141_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear47_out1 = self.linear47.forward(add86_out1);
        let mul75_out1 = linear47_out1.clone().mul(linear47_out1.clone());
        let mul76_out1 = linear47_out1.clone().mul(mul75_out1);
        let mul77_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul76_out1);
        let add87_out1 = linear47_out1.clone().add(mul77_out1);
        let mul78_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add87_out1);
        let tanh8_out1 = mul78_out1.tanh();
        let add88_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh8_out1);
        let mul79_out1 = linear47_out1.mul(add88_out1);
        let mul80_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul79_out1);
        let linear48_out1 = self.linear48.forward(mul80_out1);
        let add89_out1 = add84_out1.add(linear48_out1);
        add89_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule7<B: Backend> {
    constant146: burn::module::Param<Tensor<B, 1>>,
    constant147: burn::module::Param<Tensor<B, 1>>,
    linear49: Linear<B>,
    linear50: Linear<B>,
    linear51: Linear<B>,
    constant151: burn::module::Param<Tensor<B, 1>>,
    constant152: burn::module::Param<Tensor<B, 1>>,
    constant153: burn::module::Param<Tensor<B, 1>>,
    linear52: Linear<B>,
    constant156: burn::module::Param<Tensor<B, 1>>,
    constant157: burn::module::Param<Tensor<B, 1>>,
    linear53: Linear<B>,
    linear54: Linear<B>,
    constant162: burn::module::Param<Tensor<B, 1>>,
    constant163: burn::module::Param<Tensor<B, 1>>,
    linear55: Linear<B>,
    linear56: Linear<B>,
    linear57: Linear<B>,
    constant167: burn::module::Param<Tensor<B, 1>>,
    constant168: burn::module::Param<Tensor<B, 1>>,
    constant169: burn::module::Param<Tensor<B, 1>>,
    linear58: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule7<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant146: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant147: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear49 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear50 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear51 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant151: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant152: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant153: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear52 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant156: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant157: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear53 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear54 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant162: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant163: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear55 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear56 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear57 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant167: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant168: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant169: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear58 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant146,
            constant147,
            linear49,
            linear50,
            linear51,
            constant151,
            constant152,
            constant153,
            linear52,
            constant156,
            constant157,
            linear53,
            linear54,
            constant162,
            constant163,
            linear55,
            linear56,
            linear57,
            constant167,
            constant168,
            constant169,
            linear58,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add89_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean33_out1 = { add89_out1.clone().mean_dim(2usize) };
        let sub17_out1 = add89_out1.clone().sub(reducemean33_out1);
        let pow17_out1 = sub17_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean34_out1 = { pow17_out1.mean_dim(2usize) };
        let add90_out1 =
            reducemean34_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt17_out1 = add90_out1.sqrt();
        let div17_out1 = sub17_out1.div(sqrt17_out1);
        let constant146_out1 = self.constant146.val();
        let mul81_out1 = div17_out1.mul((constant146_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant147_out1 = self.constant147.val();
        let add91_out1 = mul81_out1.add((constant147_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape10_out1: [i64; 3] = {
            let axes = &add91_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear49_out1 = self.linear49.forward(add91_out1.clone());
        let linear50_out1 = self.linear50.forward(add91_out1.clone());
        let linear51_out1 = self.linear51.forward(add91_out1);
        let gather20_out1 = shape10_out1[0] as i64;
        let gather21_out1 = shape10_out1[1] as i64;
        let constant151_out1 = self.constant151.val();
        let add92_out1 = (constant151_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear49_out1);
        let constant152_out1 = self.constant152.val();
        let add93_out1 = (constant152_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear50_out1);
        let constant153_out1 = self.constant153.val();
        let add94_out1 = (constant153_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear51_out1);
        let unsqueeze18_out1 = [gather20_out1 as i64];
        let unsqueeze19_out1 = [gather21_out1 as i64];
        let concat17_out1: [i64; 4usize] = [
            &unsqueeze18_out1[..],
            &unsqueeze19_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat18_out1: [i64; 3usize] = [
            &unsqueeze18_out1[..],
            &unsqueeze19_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape33_out1 = add92_out1.reshape(concat17_out1);
        let reshape34_out1 = add93_out1.reshape(concat17_out1);
        let reshape35_out1 = add94_out1.reshape(concat17_out1);
        let transpose33_out1 = reshape33_out1.permute([0, 2, 1, 3]);
        let transpose34_out1 = reshape35_out1.permute([0, 2, 1, 3]);
        let transpose35_out1 = reshape34_out1.permute([0, 2, 3, 1]);
        let matmul68_k_corrected = transpose35_out1.permute([0, 1, 3, 2]);
        let (matmul69_out1,) = {
            let q = transpose33_out1;
            let k = matmul68_k_corrected;
            let v = transpose34_out1;
            let matmul69_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul69_out1,)
        };
        let transpose36_out1 = matmul69_out1.permute([0, 2, 1, 3]);
        let reshape36_out1 = transpose36_out1.reshape(concat18_out1);
        let linear52_out1 = self.linear52.forward(reshape36_out1);
        let add95_out1 = add89_out1.add(linear52_out1);
        let reducemean35_out1 = { add95_out1.clone().mean_dim(2usize) };
        let sub18_out1 = add95_out1.clone().sub(reducemean35_out1);
        let pow18_out1 = sub18_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean36_out1 = { pow18_out1.mean_dim(2usize) };
        let add96_out1 =
            reducemean36_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt18_out1 = add96_out1.sqrt();
        let div18_out1 = sub18_out1.div(sqrt18_out1);
        let constant156_out1 = self.constant156.val();
        let mul84_out1 = div18_out1.mul((constant156_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant157_out1 = self.constant157.val();
        let add97_out1 = mul84_out1.add((constant157_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear53_out1 = self.linear53.forward(add97_out1);
        let mul85_out1 = linear53_out1.clone().mul(linear53_out1.clone());
        let mul86_out1 = linear53_out1.clone().mul(mul85_out1);
        let mul87_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul86_out1);
        let add98_out1 = linear53_out1.clone().add(mul87_out1);
        let mul88_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add98_out1);
        let tanh9_out1 = mul88_out1.tanh();
        let add99_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh9_out1);
        let mul89_out1 = linear53_out1.mul(add99_out1);
        let mul90_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul89_out1);
        let linear54_out1 = self.linear54.forward(mul90_out1);
        let add100_out1 = add95_out1.add(linear54_out1);
        let reducemean37_out1 = { add100_out1.clone().mean_dim(2usize) };
        let sub19_out1 = add100_out1.clone().sub(reducemean37_out1);
        let pow19_out1 = sub19_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean38_out1 = { pow19_out1.mean_dim(2usize) };
        let add101_out1 = reducemean38_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt19_out1 = add101_out1.sqrt();
        let div19_out1 = sub19_out1.div(sqrt19_out1);
        let constant162_out1 = self.constant162.val();
        let mul91_out1 = div19_out1.mul((constant162_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant163_out1 = self.constant163.val();
        let add102_out1 = mul91_out1.add((constant163_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape11_out1: [i64; 3] = {
            let axes = &add102_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear55_out1 = self.linear55.forward(add102_out1.clone());
        let linear56_out1 = self.linear56.forward(add102_out1.clone());
        let linear57_out1 = self.linear57.forward(add102_out1);
        let gather22_out1 = shape11_out1[0] as i64;
        let gather23_out1 = shape11_out1[1] as i64;
        let constant167_out1 = self.constant167.val();
        let add103_out1 = (constant167_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear55_out1);
        let constant168_out1 = self.constant168.val();
        let add104_out1 = (constant168_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear56_out1);
        let constant169_out1 = self.constant169.val();
        let add105_out1 = (constant169_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear57_out1);
        let unsqueeze20_out1 = [gather22_out1 as i64];
        let unsqueeze21_out1 = [gather23_out1 as i64];
        let concat19_out1: [i64; 4usize] = [
            &unsqueeze20_out1[..],
            &unsqueeze21_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat20_out1: [i64; 3usize] = [
            &unsqueeze20_out1[..],
            &unsqueeze21_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape37_out1 = add103_out1.reshape(concat19_out1);
        let reshape38_out1 = add104_out1.reshape(concat19_out1);
        let reshape39_out1 = add105_out1.reshape(concat19_out1);
        let transpose37_out1 = reshape37_out1.permute([0, 2, 1, 3]);
        let transpose38_out1 = reshape39_out1.permute([0, 2, 1, 3]);
        let transpose39_out1 = reshape38_out1.permute([0, 2, 3, 1]);
        let matmul76_k_corrected = transpose39_out1.permute([0, 1, 3, 2]);
        let (matmul77_out1,) = {
            let q = transpose37_out1;
            let k = matmul76_k_corrected;
            let v = transpose38_out1;
            let matmul77_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul77_out1,)
        };
        let transpose40_out1 = matmul77_out1.permute([0, 2, 1, 3]);
        let reshape40_out1 = transpose40_out1.reshape(concat20_out1);
        let linear58_out1 = self.linear58.forward(reshape40_out1);
        let add106_out1 = add100_out1.add(linear58_out1);
        add106_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule8<B: Backend> {
    constant172: burn::module::Param<Tensor<B, 1>>,
    constant173: burn::module::Param<Tensor<B, 1>>,
    linear59: Linear<B>,
    linear60: Linear<B>,
    constant178: burn::module::Param<Tensor<B, 1>>,
    constant179: burn::module::Param<Tensor<B, 1>>,
    linear61: Linear<B>,
    linear62: Linear<B>,
    linear63: Linear<B>,
    constant183: burn::module::Param<Tensor<B, 1>>,
    constant184: burn::module::Param<Tensor<B, 1>>,
    constant185: burn::module::Param<Tensor<B, 1>>,
    linear64: Linear<B>,
    constant188: burn::module::Param<Tensor<B, 1>>,
    constant189: burn::module::Param<Tensor<B, 1>>,
    linear65: Linear<B>,
    linear66: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule8<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant172: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant173: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear59 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear60 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant178: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant179: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear61 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear62 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear63 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant183: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant184: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant185: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear64 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant188: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant189: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear65 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear66 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant172,
            constant173,
            linear59,
            linear60,
            constant178,
            constant179,
            linear61,
            linear62,
            linear63,
            constant183,
            constant184,
            constant185,
            linear64,
            constant188,
            constant189,
            linear65,
            linear66,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add106_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean39_out1 = { add106_out1.clone().mean_dim(2usize) };
        let sub20_out1 = add106_out1.clone().sub(reducemean39_out1);
        let pow20_out1 = sub20_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean40_out1 = { pow20_out1.mean_dim(2usize) };
        let add107_out1 =
            reducemean40_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt20_out1 = add107_out1.sqrt();
        let div20_out1 = sub20_out1.div(sqrt20_out1);
        let constant172_out1 = self.constant172.val();
        let mul94_out1 = div20_out1.mul((constant172_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant173_out1 = self.constant173.val();
        let add108_out1 = mul94_out1.add((constant173_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear59_out1 = self.linear59.forward(add108_out1);
        let mul95_out1 = linear59_out1.clone().mul(linear59_out1.clone());
        let mul96_out1 = linear59_out1.clone().mul(mul95_out1);
        let mul97_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul96_out1);
        let add109_out1 = linear59_out1.clone().add(mul97_out1);
        let mul98_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add109_out1);
        let tanh10_out1 = mul98_out1.tanh();
        let add110_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh10_out1);
        let mul99_out1 = linear59_out1.mul(add110_out1);
        let mul100_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul99_out1);
        let linear60_out1 = self.linear60.forward(mul100_out1);
        let add111_out1 = add106_out1.add(linear60_out1);
        let reducemean41_out1 = { add111_out1.clone().mean_dim(2usize) };
        let sub21_out1 = add111_out1.clone().sub(reducemean41_out1);
        let pow21_out1 = sub21_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean42_out1 = { pow21_out1.mean_dim(2usize) };
        let add112_out1 =
            reducemean42_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt21_out1 = add112_out1.sqrt();
        let div21_out1 = sub21_out1.div(sqrt21_out1);
        let constant178_out1 = self.constant178.val();
        let mul101_out1 = div21_out1.mul((constant178_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant179_out1 = self.constant179.val();
        let add113_out1 = mul101_out1.add((constant179_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape12_out1: [i64; 3] = {
            let axes = &add113_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear61_out1 = self.linear61.forward(add113_out1.clone());
        let linear62_out1 = self.linear62.forward(add113_out1.clone());
        let linear63_out1 = self.linear63.forward(add113_out1);
        let gather24_out1 = shape12_out1[0] as i64;
        let gather25_out1 = shape12_out1[1] as i64;
        let constant183_out1 = self.constant183.val();
        let add114_out1 = (constant183_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear61_out1);
        let constant184_out1 = self.constant184.val();
        let add115_out1 = (constant184_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear62_out1);
        let constant185_out1 = self.constant185.val();
        let add116_out1 = (constant185_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear63_out1);
        let unsqueeze22_out1 = [gather24_out1 as i64];
        let unsqueeze23_out1 = [gather25_out1 as i64];
        let concat21_out1: [i64; 4usize] = [
            &unsqueeze22_out1[..],
            &unsqueeze23_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat22_out1: [i64; 3usize] = [
            &unsqueeze22_out1[..],
            &unsqueeze23_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape41_out1 = add114_out1.reshape(concat21_out1);
        let reshape42_out1 = add115_out1.reshape(concat21_out1);
        let reshape43_out1 = add116_out1.reshape(concat21_out1);
        let transpose41_out1 = reshape41_out1.permute([0, 2, 1, 3]);
        let transpose42_out1 = reshape43_out1.permute([0, 2, 1, 3]);
        let transpose43_out1 = reshape42_out1.permute([0, 2, 3, 1]);
        let matmul84_k_corrected = transpose43_out1.permute([0, 1, 3, 2]);
        let (matmul85_out1,) = {
            let q = transpose41_out1;
            let k = matmul84_k_corrected;
            let v = transpose42_out1;
            let matmul85_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul85_out1,)
        };
        let transpose44_out1 = matmul85_out1.permute([0, 2, 1, 3]);
        let reshape44_out1 = transpose44_out1.reshape(concat22_out1);
        let linear64_out1 = self.linear64.forward(reshape44_out1);
        let add117_out1 = add111_out1.add(linear64_out1);
        let reducemean43_out1 = { add117_out1.clone().mean_dim(2usize) };
        let sub22_out1 = add117_out1.clone().sub(reducemean43_out1);
        let pow22_out1 = sub22_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean44_out1 = { pow22_out1.mean_dim(2usize) };
        let add118_out1 = reducemean44_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt22_out1 = add118_out1.sqrt();
        let div22_out1 = sub22_out1.div(sqrt22_out1);
        let constant188_out1 = self.constant188.val();
        let mul104_out1 = div22_out1.mul((constant188_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant189_out1 = self.constant189.val();
        let add119_out1 = mul104_out1.add((constant189_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear65_out1 = self.linear65.forward(add119_out1);
        let mul105_out1 = linear65_out1.clone().mul(linear65_out1.clone());
        let mul106_out1 = linear65_out1.clone().mul(mul105_out1);
        let mul107_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul106_out1);
        let add120_out1 = linear65_out1.clone().add(mul107_out1);
        let mul108_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add120_out1);
        let tanh11_out1 = mul108_out1.tanh();
        let add121_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh11_out1);
        let mul109_out1 = linear65_out1.mul(add121_out1);
        let mul110_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul109_out1);
        let linear66_out1 = self.linear66.forward(mul110_out1);
        let add122_out1 = add117_out1.add(linear66_out1);
        add122_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule9<B: Backend> {
    constant194: burn::module::Param<Tensor<B, 1>>,
    constant195: burn::module::Param<Tensor<B, 1>>,
    linear67: Linear<B>,
    linear68: Linear<B>,
    linear69: Linear<B>,
    constant199: burn::module::Param<Tensor<B, 1>>,
    constant200: burn::module::Param<Tensor<B, 1>>,
    constant201: burn::module::Param<Tensor<B, 1>>,
    linear70: Linear<B>,
    constant204: burn::module::Param<Tensor<B, 1>>,
    constant205: burn::module::Param<Tensor<B, 1>>,
    linear71: Linear<B>,
    linear72: Linear<B>,
    constant210: burn::module::Param<Tensor<B, 1>>,
    constant211: burn::module::Param<Tensor<B, 1>>,
    linear73: Linear<B>,
    linear74: Linear<B>,
    linear75: Linear<B>,
    constant215: burn::module::Param<Tensor<B, 1>>,
    constant216: burn::module::Param<Tensor<B, 1>>,
    constant217: burn::module::Param<Tensor<B, 1>>,
    linear76: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule9<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant194: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant195: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear67 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear68 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear69 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant199: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant200: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant201: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear70 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant204: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant205: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear71 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear72 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant210: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant211: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear73 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear74 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear75 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant215: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant216: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant217: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear76 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant194,
            constant195,
            linear67,
            linear68,
            linear69,
            constant199,
            constant200,
            constant201,
            linear70,
            constant204,
            constant205,
            linear71,
            linear72,
            constant210,
            constant211,
            linear73,
            linear74,
            linear75,
            constant215,
            constant216,
            constant217,
            linear76,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add122_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean45_out1 = { add122_out1.clone().mean_dim(2usize) };
        let sub23_out1 = add122_out1.clone().sub(reducemean45_out1);
        let pow23_out1 = sub23_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean46_out1 = { pow23_out1.mean_dim(2usize) };
        let add123_out1 =
            reducemean46_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt23_out1 = add123_out1.sqrt();
        let div23_out1 = sub23_out1.div(sqrt23_out1);
        let constant194_out1 = self.constant194.val();
        let mul111_out1 = div23_out1.mul((constant194_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant195_out1 = self.constant195.val();
        let add124_out1 = mul111_out1.add((constant195_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape13_out1: [i64; 3] = {
            let axes = &add124_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear67_out1 = self.linear67.forward(add124_out1.clone());
        let linear68_out1 = self.linear68.forward(add124_out1.clone());
        let linear69_out1 = self.linear69.forward(add124_out1);
        let gather26_out1 = shape13_out1[0] as i64;
        let gather27_out1 = shape13_out1[1] as i64;
        let constant199_out1 = self.constant199.val();
        let add125_out1 = (constant199_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear67_out1);
        let constant200_out1 = self.constant200.val();
        let add126_out1 = (constant200_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear68_out1);
        let constant201_out1 = self.constant201.val();
        let add127_out1 = (constant201_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear69_out1);
        let unsqueeze24_out1 = [gather26_out1 as i64];
        let unsqueeze25_out1 = [gather27_out1 as i64];
        let concat23_out1: [i64; 4usize] = [
            &unsqueeze24_out1[..],
            &unsqueeze25_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat24_out1: [i64; 3usize] = [
            &unsqueeze24_out1[..],
            &unsqueeze25_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape45_out1 = add125_out1.reshape(concat23_out1);
        let reshape46_out1 = add126_out1.reshape(concat23_out1);
        let reshape47_out1 = add127_out1.reshape(concat23_out1);
        let transpose45_out1 = reshape45_out1.permute([0, 2, 1, 3]);
        let transpose46_out1 = reshape47_out1.permute([0, 2, 1, 3]);
        let transpose47_out1 = reshape46_out1.permute([0, 2, 3, 1]);
        let matmul92_k_corrected = transpose47_out1.permute([0, 1, 3, 2]);
        let (matmul93_out1,) = {
            let q = transpose45_out1;
            let k = matmul92_k_corrected;
            let v = transpose46_out1;
            let matmul93_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul93_out1,)
        };
        let transpose48_out1 = matmul93_out1.permute([0, 2, 1, 3]);
        let reshape48_out1 = transpose48_out1.reshape(concat24_out1);
        let linear70_out1 = self.linear70.forward(reshape48_out1);
        let add128_out1 = add122_out1.add(linear70_out1);
        let reducemean47_out1 = { add128_out1.clone().mean_dim(2usize) };
        let sub24_out1 = add128_out1.clone().sub(reducemean47_out1);
        let pow24_out1 = sub24_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean48_out1 = { pow24_out1.mean_dim(2usize) };
        let add129_out1 =
            reducemean48_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt24_out1 = add129_out1.sqrt();
        let div24_out1 = sub24_out1.div(sqrt24_out1);
        let constant204_out1 = self.constant204.val();
        let mul114_out1 = div24_out1.mul((constant204_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant205_out1 = self.constant205.val();
        let add130_out1 = mul114_out1.add((constant205_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear71_out1 = self.linear71.forward(add130_out1);
        let mul115_out1 = linear71_out1.clone().mul(linear71_out1.clone());
        let mul116_out1 = linear71_out1.clone().mul(mul115_out1);
        let mul117_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul116_out1);
        let add131_out1 = linear71_out1.clone().add(mul117_out1);
        let mul118_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add131_out1);
        let tanh12_out1 = mul118_out1.tanh();
        let add132_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh12_out1);
        let mul119_out1 = linear71_out1.mul(add132_out1);
        let mul120_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul119_out1);
        let linear72_out1 = self.linear72.forward(mul120_out1);
        let add133_out1 = add128_out1.add(linear72_out1);
        let reducemean49_out1 = { add133_out1.clone().mean_dim(2usize) };
        let sub25_out1 = add133_out1.clone().sub(reducemean49_out1);
        let pow25_out1 = sub25_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean50_out1 = { pow25_out1.mean_dim(2usize) };
        let add134_out1 = reducemean50_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt25_out1 = add134_out1.sqrt();
        let div25_out1 = sub25_out1.div(sqrt25_out1);
        let constant210_out1 = self.constant210.val();
        let mul121_out1 = div25_out1.mul((constant210_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant211_out1 = self.constant211.val();
        let add135_out1 = mul121_out1.add((constant211_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape14_out1: [i64; 3] = {
            let axes = &add135_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear73_out1 = self.linear73.forward(add135_out1.clone());
        let linear74_out1 = self.linear74.forward(add135_out1.clone());
        let linear75_out1 = self.linear75.forward(add135_out1);
        let gather28_out1 = shape14_out1[0] as i64;
        let gather29_out1 = shape14_out1[1] as i64;
        let constant215_out1 = self.constant215.val();
        let add136_out1 = (constant215_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear73_out1);
        let constant216_out1 = self.constant216.val();
        let add137_out1 = (constant216_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear74_out1);
        let constant217_out1 = self.constant217.val();
        let add138_out1 = (constant217_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear75_out1);
        let unsqueeze26_out1 = [gather28_out1 as i64];
        let unsqueeze27_out1 = [gather29_out1 as i64];
        let concat25_out1: [i64; 4usize] = [
            &unsqueeze26_out1[..],
            &unsqueeze27_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat26_out1: [i64; 3usize] = [
            &unsqueeze26_out1[..],
            &unsqueeze27_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape49_out1 = add136_out1.reshape(concat25_out1);
        let reshape50_out1 = add137_out1.reshape(concat25_out1);
        let reshape51_out1 = add138_out1.reshape(concat25_out1);
        let transpose49_out1 = reshape49_out1.permute([0, 2, 1, 3]);
        let transpose50_out1 = reshape51_out1.permute([0, 2, 1, 3]);
        let transpose51_out1 = reshape50_out1.permute([0, 2, 3, 1]);
        let matmul100_k_corrected = transpose51_out1.permute([0, 1, 3, 2]);
        let (matmul101_out1,) = {
            let q = transpose49_out1;
            let k = matmul100_k_corrected;
            let v = transpose50_out1;
            let matmul101_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul101_out1,)
        };
        let transpose52_out1 = matmul101_out1.permute([0, 2, 1, 3]);
        let reshape52_out1 = transpose52_out1.reshape(concat26_out1);
        let linear76_out1 = self.linear76.forward(reshape52_out1);
        let add139_out1 = add133_out1.add(linear76_out1);
        add139_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule10<B: Backend> {
    constant220: burn::module::Param<Tensor<B, 1>>,
    constant221: burn::module::Param<Tensor<B, 1>>,
    linear77: Linear<B>,
    linear78: Linear<B>,
    constant226: burn::module::Param<Tensor<B, 1>>,
    constant227: burn::module::Param<Tensor<B, 1>>,
    linear79: Linear<B>,
    linear80: Linear<B>,
    linear81: Linear<B>,
    constant231: burn::module::Param<Tensor<B, 1>>,
    constant232: burn::module::Param<Tensor<B, 1>>,
    constant233: burn::module::Param<Tensor<B, 1>>,
    linear82: Linear<B>,
    constant236: burn::module::Param<Tensor<B, 1>>,
    constant237: burn::module::Param<Tensor<B, 1>>,
    linear83: Linear<B>,
    linear84: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule10<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant220: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant221: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear77 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear78 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant226: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant227: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear79 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear80 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear81 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant231: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant232: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant233: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear82 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant236: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant237: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear83 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear84 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant220,
            constant221,
            linear77,
            linear78,
            constant226,
            constant227,
            linear79,
            linear80,
            linear81,
            constant231,
            constant232,
            constant233,
            linear82,
            constant236,
            constant237,
            linear83,
            linear84,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add139_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean51_out1 = { add139_out1.clone().mean_dim(2usize) };
        let sub26_out1 = add139_out1.clone().sub(reducemean51_out1);
        let pow26_out1 = sub26_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean52_out1 = { pow26_out1.mean_dim(2usize) };
        let add140_out1 =
            reducemean52_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt26_out1 = add140_out1.sqrt();
        let div26_out1 = sub26_out1.div(sqrt26_out1);
        let constant220_out1 = self.constant220.val();
        let mul124_out1 = div26_out1.mul((constant220_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant221_out1 = self.constant221.val();
        let add141_out1 = mul124_out1.add((constant221_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear77_out1 = self.linear77.forward(add141_out1);
        let mul125_out1 = linear77_out1.clone().mul(linear77_out1.clone());
        let mul126_out1 = linear77_out1.clone().mul(mul125_out1);
        let mul127_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul126_out1);
        let add142_out1 = linear77_out1.clone().add(mul127_out1);
        let mul128_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add142_out1);
        let tanh13_out1 = mul128_out1.tanh();
        let add143_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh13_out1);
        let mul129_out1 = linear77_out1.mul(add143_out1);
        let mul130_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul129_out1);
        let linear78_out1 = self.linear78.forward(mul130_out1);
        let add144_out1 = add139_out1.add(linear78_out1);
        let reducemean53_out1 = { add144_out1.clone().mean_dim(2usize) };
        let sub27_out1 = add144_out1.clone().sub(reducemean53_out1);
        let pow27_out1 = sub27_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean54_out1 = { pow27_out1.mean_dim(2usize) };
        let add145_out1 =
            reducemean54_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt27_out1 = add145_out1.sqrt();
        let div27_out1 = sub27_out1.div(sqrt27_out1);
        let constant226_out1 = self.constant226.val();
        let mul131_out1 = div27_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant227_out1 = self.constant227.val();
        let add146_out1 = mul131_out1.add((constant227_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape15_out1: [i64; 3] = {
            let axes = &add146_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear79_out1 = self.linear79.forward(add146_out1.clone());
        let linear80_out1 = self.linear80.forward(add146_out1.clone());
        let linear81_out1 = self.linear81.forward(add146_out1);
        let gather30_out1 = shape15_out1[0] as i64;
        let gather31_out1 = shape15_out1[1] as i64;
        let constant231_out1 = self.constant231.val();
        let add147_out1 = (constant231_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear79_out1);
        let constant232_out1 = self.constant232.val();
        let add148_out1 = (constant232_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear80_out1);
        let constant233_out1 = self.constant233.val();
        let add149_out1 = (constant233_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear81_out1);
        let unsqueeze28_out1 = [gather30_out1 as i64];
        let unsqueeze29_out1 = [gather31_out1 as i64];
        let concat27_out1: [i64; 4usize] = [
            &unsqueeze28_out1[..],
            &unsqueeze29_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat28_out1: [i64; 3usize] = [
            &unsqueeze28_out1[..],
            &unsqueeze29_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape53_out1 = add147_out1.reshape(concat27_out1);
        let reshape54_out1 = add148_out1.reshape(concat27_out1);
        let reshape55_out1 = add149_out1.reshape(concat27_out1);
        let transpose53_out1 = reshape53_out1.permute([0, 2, 1, 3]);
        let transpose54_out1 = reshape55_out1.permute([0, 2, 1, 3]);
        let transpose55_out1 = reshape54_out1.permute([0, 2, 3, 1]);
        let matmul108_k_corrected = transpose55_out1.permute([0, 1, 3, 2]);
        let (matmul109_out1,) = {
            let q = transpose53_out1;
            let k = matmul108_k_corrected;
            let v = transpose54_out1;
            let matmul109_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul109_out1,)
        };
        let transpose56_out1 = matmul109_out1.permute([0, 2, 1, 3]);
        let reshape56_out1 = transpose56_out1.reshape(concat28_out1);
        let linear82_out1 = self.linear82.forward(reshape56_out1);
        let add150_out1 = add144_out1.add(linear82_out1);
        let reducemean55_out1 = { add150_out1.clone().mean_dim(2usize) };
        let sub28_out1 = add150_out1.clone().sub(reducemean55_out1);
        let pow28_out1 = sub28_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean56_out1 = { pow28_out1.mean_dim(2usize) };
        let add151_out1 = reducemean56_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt28_out1 = add151_out1.sqrt();
        let div28_out1 = sub28_out1.div(sqrt28_out1);
        let constant236_out1 = self.constant236.val();
        let mul134_out1 = div28_out1.mul((constant236_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant237_out1 = self.constant237.val();
        let add152_out1 = mul134_out1.add((constant237_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear83_out1 = self.linear83.forward(add152_out1);
        let mul135_out1 = linear83_out1.clone().mul(linear83_out1.clone());
        let mul136_out1 = linear83_out1.clone().mul(mul135_out1);
        let mul137_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul136_out1);
        let add153_out1 = linear83_out1.clone().add(mul137_out1);
        let mul138_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add153_out1);
        let tanh14_out1 = mul138_out1.tanh();
        let add154_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh14_out1);
        let mul139_out1 = linear83_out1.mul(add154_out1);
        let mul140_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul139_out1);
        let linear84_out1 = self.linear84.forward(mul140_out1);
        let add155_out1 = add150_out1.add(linear84_out1);
        add155_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule11<B: Backend> {
    constant242: burn::module::Param<Tensor<B, 1>>,
    constant243: burn::module::Param<Tensor<B, 1>>,
    linear85: Linear<B>,
    linear86: Linear<B>,
    linear87: Linear<B>,
    constant247: burn::module::Param<Tensor<B, 1>>,
    constant248: burn::module::Param<Tensor<B, 1>>,
    constant249: burn::module::Param<Tensor<B, 1>>,
    linear88: Linear<B>,
    constant252: burn::module::Param<Tensor<B, 1>>,
    constant253: burn::module::Param<Tensor<B, 1>>,
    linear89: Linear<B>,
    linear90: Linear<B>,
    constant258: burn::module::Param<Tensor<B, 1>>,
    constant259: burn::module::Param<Tensor<B, 1>>,
    linear91: Linear<B>,
    linear92: Linear<B>,
    linear93: Linear<B>,
    constant263: burn::module::Param<Tensor<B, 1>>,
    constant264: burn::module::Param<Tensor<B, 1>>,
    constant265: burn::module::Param<Tensor<B, 1>>,
    linear94: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule11<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant242: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant243: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear85 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear86 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear87 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant247: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant248: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant249: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear88 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant252: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant253: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear89 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear90 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant258: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant259: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear91 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear92 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear93 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant263: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant264: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant265: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear94 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant242,
            constant243,
            linear85,
            linear86,
            linear87,
            constant247,
            constant248,
            constant249,
            linear88,
            constant252,
            constant253,
            linear89,
            linear90,
            constant258,
            constant259,
            linear91,
            linear92,
            linear93,
            constant263,
            constant264,
            constant265,
            linear94,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add155_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean57_out1 = { add155_out1.clone().mean_dim(2usize) };
        let sub29_out1 = add155_out1.clone().sub(reducemean57_out1);
        let pow29_out1 = sub29_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean58_out1 = { pow29_out1.mean_dim(2usize) };
        let add156_out1 =
            reducemean58_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt29_out1 = add156_out1.sqrt();
        let div29_out1 = sub29_out1.div(sqrt29_out1);
        let constant242_out1 = self.constant242.val();
        let mul141_out1 = div29_out1.mul((constant242_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant243_out1 = self.constant243.val();
        let add157_out1 = mul141_out1.add((constant243_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape16_out1: [i64; 3] = {
            let axes = &add157_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear85_out1 = self.linear85.forward(add157_out1.clone());
        let linear86_out1 = self.linear86.forward(add157_out1.clone());
        let linear87_out1 = self.linear87.forward(add157_out1);
        let gather32_out1 = shape16_out1[0] as i64;
        let gather33_out1 = shape16_out1[1] as i64;
        let constant247_out1 = self.constant247.val();
        let add158_out1 = (constant247_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear85_out1);
        let constant248_out1 = self.constant248.val();
        let add159_out1 = (constant248_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear86_out1);
        let constant249_out1 = self.constant249.val();
        let add160_out1 = (constant249_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear87_out1);
        let unsqueeze30_out1 = [gather32_out1 as i64];
        let unsqueeze31_out1 = [gather33_out1 as i64];
        let concat29_out1: [i64; 4usize] = [
            &unsqueeze30_out1[..],
            &unsqueeze31_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat30_out1: [i64; 3usize] = [
            &unsqueeze30_out1[..],
            &unsqueeze31_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape57_out1 = add158_out1.reshape(concat29_out1);
        let reshape58_out1 = add159_out1.reshape(concat29_out1);
        let reshape59_out1 = add160_out1.reshape(concat29_out1);
        let transpose57_out1 = reshape57_out1.permute([0, 2, 1, 3]);
        let transpose58_out1 = reshape59_out1.permute([0, 2, 1, 3]);
        let transpose59_out1 = reshape58_out1.permute([0, 2, 3, 1]);
        let matmul116_k_corrected = transpose59_out1.permute([0, 1, 3, 2]);
        let (matmul117_out1,) = {
            let q = transpose57_out1;
            let k = matmul116_k_corrected;
            let v = transpose58_out1;
            let matmul117_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul117_out1,)
        };
        let transpose60_out1 = matmul117_out1.permute([0, 2, 1, 3]);
        let reshape60_out1 = transpose60_out1.reshape(concat30_out1);
        let linear88_out1 = self.linear88.forward(reshape60_out1);
        let add161_out1 = add155_out1.add(linear88_out1);
        let reducemean59_out1 = { add161_out1.clone().mean_dim(2usize) };
        let sub30_out1 = add161_out1.clone().sub(reducemean59_out1);
        let pow30_out1 = sub30_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean60_out1 = { pow30_out1.mean_dim(2usize) };
        let add162_out1 =
            reducemean60_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt30_out1 = add162_out1.sqrt();
        let div30_out1 = sub30_out1.div(sqrt30_out1);
        let constant252_out1 = self.constant252.val();
        let mul144_out1 = div30_out1.mul((constant252_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant253_out1 = self.constant253.val();
        let add163_out1 = mul144_out1.add((constant253_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear89_out1 = self.linear89.forward(add163_out1);
        let mul145_out1 = linear89_out1.clone().mul(linear89_out1.clone());
        let mul146_out1 = linear89_out1.clone().mul(mul145_out1);
        let mul147_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul146_out1);
        let add164_out1 = linear89_out1.clone().add(mul147_out1);
        let mul148_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add164_out1);
        let tanh15_out1 = mul148_out1.tanh();
        let add165_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh15_out1);
        let mul149_out1 = linear89_out1.mul(add165_out1);
        let mul150_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul149_out1);
        let linear90_out1 = self.linear90.forward(mul150_out1);
        let add166_out1 = add161_out1.add(linear90_out1);
        let reducemean61_out1 = { add166_out1.clone().mean_dim(2usize) };
        let sub31_out1 = add166_out1.clone().sub(reducemean61_out1);
        let pow31_out1 = sub31_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean62_out1 = { pow31_out1.mean_dim(2usize) };
        let add167_out1 = reducemean62_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt31_out1 = add167_out1.sqrt();
        let div31_out1 = sub31_out1.div(sqrt31_out1);
        let constant258_out1 = self.constant258.val();
        let mul151_out1 = div31_out1.mul((constant258_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant259_out1 = self.constant259.val();
        let add168_out1 = mul151_out1.add((constant259_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape17_out1: [i64; 3] = {
            let axes = &add168_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear91_out1 = self.linear91.forward(add168_out1.clone());
        let linear92_out1 = self.linear92.forward(add168_out1.clone());
        let linear93_out1 = self.linear93.forward(add168_out1);
        let gather34_out1 = shape17_out1[0] as i64;
        let gather35_out1 = shape17_out1[1] as i64;
        let constant263_out1 = self.constant263.val();
        let add169_out1 = (constant263_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear91_out1);
        let constant264_out1 = self.constant264.val();
        let add170_out1 = (constant264_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear92_out1);
        let constant265_out1 = self.constant265.val();
        let add171_out1 = (constant265_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear93_out1);
        let unsqueeze32_out1 = [gather34_out1 as i64];
        let unsqueeze33_out1 = [gather35_out1 as i64];
        let concat31_out1: [i64; 4usize] = [
            &unsqueeze32_out1[..],
            &unsqueeze33_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat32_out1: [i64; 3usize] = [
            &unsqueeze32_out1[..],
            &unsqueeze33_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape61_out1 = add169_out1.reshape(concat31_out1);
        let reshape62_out1 = add170_out1.reshape(concat31_out1);
        let reshape63_out1 = add171_out1.reshape(concat31_out1);
        let transpose61_out1 = reshape61_out1.permute([0, 2, 1, 3]);
        let transpose62_out1 = reshape63_out1.permute([0, 2, 1, 3]);
        let transpose63_out1 = reshape62_out1.permute([0, 2, 3, 1]);
        let matmul124_k_corrected = transpose63_out1.permute([0, 1, 3, 2]);
        let (matmul125_out1,) = {
            let q = transpose61_out1;
            let k = matmul124_k_corrected;
            let v = transpose62_out1;
            let matmul125_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul125_out1,)
        };
        let transpose64_out1 = matmul125_out1.permute([0, 2, 1, 3]);
        let reshape64_out1 = transpose64_out1.reshape(concat32_out1);
        let linear94_out1 = self.linear94.forward(reshape64_out1);
        let add172_out1 = add166_out1.add(linear94_out1);
        add172_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule12<B: Backend> {
    constant268: burn::module::Param<Tensor<B, 1>>,
    constant269: burn::module::Param<Tensor<B, 1>>,
    linear95: Linear<B>,
    linear96: Linear<B>,
    constant274: burn::module::Param<Tensor<B, 1>>,
    constant275: burn::module::Param<Tensor<B, 1>>,
    linear97: Linear<B>,
    linear98: Linear<B>,
    linear99: Linear<B>,
    constant279: burn::module::Param<Tensor<B, 1>>,
    constant280: burn::module::Param<Tensor<B, 1>>,
    constant281: burn::module::Param<Tensor<B, 1>>,
    linear100: Linear<B>,
    constant284: burn::module::Param<Tensor<B, 1>>,
    constant285: burn::module::Param<Tensor<B, 1>>,
    linear101: Linear<B>,
    linear102: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule12<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant268: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant269: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear95 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear96 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant274: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant275: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear97 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear98 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear99 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant279: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant280: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant281: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear100 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant284: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant285: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear101 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear102 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant268,
            constant269,
            linear95,
            linear96,
            constant274,
            constant275,
            linear97,
            linear98,
            linear99,
            constant279,
            constant280,
            constant281,
            linear100,
            constant284,
            constant285,
            linear101,
            linear102,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add172_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean63_out1 = { add172_out1.clone().mean_dim(2usize) };
        let sub32_out1 = add172_out1.clone().sub(reducemean63_out1);
        let pow32_out1 = sub32_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean64_out1 = { pow32_out1.mean_dim(2usize) };
        let add173_out1 =
            reducemean64_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt32_out1 = add173_out1.sqrt();
        let div32_out1 = sub32_out1.div(sqrt32_out1);
        let constant268_out1 = self.constant268.val();
        let mul154_out1 = div32_out1.mul((constant268_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant269_out1 = self.constant269.val();
        let add174_out1 = mul154_out1.add((constant269_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear95_out1 = self.linear95.forward(add174_out1);
        let mul155_out1 = linear95_out1.clone().mul(linear95_out1.clone());
        let mul156_out1 = linear95_out1.clone().mul(mul155_out1);
        let mul157_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul156_out1);
        let add175_out1 = linear95_out1.clone().add(mul157_out1);
        let mul158_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add175_out1);
        let tanh16_out1 = mul158_out1.tanh();
        let add176_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh16_out1);
        let mul159_out1 = linear95_out1.mul(add176_out1);
        let mul160_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul159_out1);
        let linear96_out1 = self.linear96.forward(mul160_out1);
        let add177_out1 = add172_out1.add(linear96_out1);
        let reducemean65_out1 = { add177_out1.clone().mean_dim(2usize) };
        let sub33_out1 = add177_out1.clone().sub(reducemean65_out1);
        let pow33_out1 = sub33_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean66_out1 = { pow33_out1.mean_dim(2usize) };
        let add178_out1 =
            reducemean66_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt33_out1 = add178_out1.sqrt();
        let div33_out1 = sub33_out1.div(sqrt33_out1);
        let constant274_out1 = self.constant274.val();
        let mul161_out1 = div33_out1.mul((constant274_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant275_out1 = self.constant275.val();
        let add179_out1 = mul161_out1.add((constant275_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape18_out1: [i64; 3] = {
            let axes = &add179_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear97_out1 = self.linear97.forward(add179_out1.clone());
        let linear98_out1 = self.linear98.forward(add179_out1.clone());
        let linear99_out1 = self.linear99.forward(add179_out1);
        let gather36_out1 = shape18_out1[0] as i64;
        let gather37_out1 = shape18_out1[1] as i64;
        let constant279_out1 = self.constant279.val();
        let add180_out1 = (constant279_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear97_out1);
        let constant280_out1 = self.constant280.val();
        let add181_out1 = (constant280_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear98_out1);
        let constant281_out1 = self.constant281.val();
        let add182_out1 = (constant281_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear99_out1);
        let unsqueeze34_out1 = [gather36_out1 as i64];
        let unsqueeze35_out1 = [gather37_out1 as i64];
        let concat33_out1: [i64; 4usize] = [
            &unsqueeze34_out1[..],
            &unsqueeze35_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat34_out1: [i64; 3usize] = [
            &unsqueeze34_out1[..],
            &unsqueeze35_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape65_out1 = add180_out1.reshape(concat33_out1);
        let reshape66_out1 = add181_out1.reshape(concat33_out1);
        let reshape67_out1 = add182_out1.reshape(concat33_out1);
        let transpose65_out1 = reshape65_out1.permute([0, 2, 1, 3]);
        let transpose66_out1 = reshape67_out1.permute([0, 2, 1, 3]);
        let transpose67_out1 = reshape66_out1.permute([0, 2, 3, 1]);
        let matmul132_k_corrected = transpose67_out1.permute([0, 1, 3, 2]);
        let (matmul133_out1,) = {
            let q = transpose65_out1;
            let k = matmul132_k_corrected;
            let v = transpose66_out1;
            let matmul133_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul133_out1,)
        };
        let transpose68_out1 = matmul133_out1.permute([0, 2, 1, 3]);
        let reshape68_out1 = transpose68_out1.reshape(concat34_out1);
        let linear100_out1 = self.linear100.forward(reshape68_out1);
        let add183_out1 = add177_out1.add(linear100_out1);
        let reducemean67_out1 = { add183_out1.clone().mean_dim(2usize) };
        let sub34_out1 = add183_out1.clone().sub(reducemean67_out1);
        let pow34_out1 = sub34_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean68_out1 = { pow34_out1.mean_dim(2usize) };
        let add184_out1 = reducemean68_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt34_out1 = add184_out1.sqrt();
        let div34_out1 = sub34_out1.div(sqrt34_out1);
        let constant284_out1 = self.constant284.val();
        let mul164_out1 = div34_out1.mul((constant284_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant285_out1 = self.constant285.val();
        let add185_out1 = mul164_out1.add((constant285_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear101_out1 = self.linear101.forward(add185_out1);
        let mul165_out1 = linear101_out1.clone().mul(linear101_out1.clone());
        let mul166_out1 = linear101_out1.clone().mul(mul165_out1);
        let mul167_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul166_out1);
        let add186_out1 = linear101_out1.clone().add(mul167_out1);
        let mul168_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add186_out1);
        let tanh17_out1 = mul168_out1.tanh();
        let add187_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh17_out1);
        let mul169_out1 = linear101_out1.mul(add187_out1);
        let mul170_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul169_out1);
        let linear102_out1 = self.linear102.forward(mul170_out1);
        let add188_out1 = add183_out1.add(linear102_out1);
        add188_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule13<B: Backend> {
    constant290: burn::module::Param<Tensor<B, 1>>,
    constant291: burn::module::Param<Tensor<B, 1>>,
    linear103: Linear<B>,
    linear104: Linear<B>,
    linear105: Linear<B>,
    constant295: burn::module::Param<Tensor<B, 1>>,
    constant296: burn::module::Param<Tensor<B, 1>>,
    constant297: burn::module::Param<Tensor<B, 1>>,
    linear106: Linear<B>,
    constant300: burn::module::Param<Tensor<B, 1>>,
    constant301: burn::module::Param<Tensor<B, 1>>,
    linear107: Linear<B>,
    linear108: Linear<B>,
    constant306: burn::module::Param<Tensor<B, 1>>,
    constant307: burn::module::Param<Tensor<B, 1>>,
    linear109: Linear<B>,
    linear110: Linear<B>,
    linear111: Linear<B>,
    constant311: burn::module::Param<Tensor<B, 1>>,
    constant312: burn::module::Param<Tensor<B, 1>>,
    constant313: burn::module::Param<Tensor<B, 1>>,
    linear112: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule13<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant290: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant291: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear103 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear104 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear105 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant295: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant296: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant297: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear106 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant300: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant301: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear107 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear108 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant306: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant307: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear109 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear110 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear111 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant311: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant312: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant313: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear112 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant290,
            constant291,
            linear103,
            linear104,
            linear105,
            constant295,
            constant296,
            constant297,
            linear106,
            constant300,
            constant301,
            linear107,
            linear108,
            constant306,
            constant307,
            linear109,
            linear110,
            linear111,
            constant311,
            constant312,
            constant313,
            linear112,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add188_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean69_out1 = { add188_out1.clone().mean_dim(2usize) };
        let sub35_out1 = add188_out1.clone().sub(reducemean69_out1);
        let pow35_out1 = sub35_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean70_out1 = { pow35_out1.mean_dim(2usize) };
        let add189_out1 =
            reducemean70_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt35_out1 = add189_out1.sqrt();
        let div35_out1 = sub35_out1.div(sqrt35_out1);
        let constant290_out1 = self.constant290.val();
        let mul171_out1 = div35_out1.mul((constant290_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant291_out1 = self.constant291.val();
        let add190_out1 = mul171_out1.add((constant291_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape19_out1: [i64; 3] = {
            let axes = &add190_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear103_out1 = self.linear103.forward(add190_out1.clone());
        let linear104_out1 = self.linear104.forward(add190_out1.clone());
        let linear105_out1 = self.linear105.forward(add190_out1);
        let gather38_out1 = shape19_out1[0] as i64;
        let gather39_out1 = shape19_out1[1] as i64;
        let constant295_out1 = self.constant295.val();
        let add191_out1 = (constant295_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear103_out1);
        let constant296_out1 = self.constant296.val();
        let add192_out1 = (constant296_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear104_out1);
        let constant297_out1 = self.constant297.val();
        let add193_out1 = (constant297_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear105_out1);
        let unsqueeze36_out1 = [gather38_out1 as i64];
        let unsqueeze37_out1 = [gather39_out1 as i64];
        let concat35_out1: [i64; 4usize] = [
            &unsqueeze36_out1[..],
            &unsqueeze37_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat36_out1: [i64; 3usize] = [
            &unsqueeze36_out1[..],
            &unsqueeze37_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape69_out1 = add191_out1.reshape(concat35_out1);
        let reshape70_out1 = add192_out1.reshape(concat35_out1);
        let reshape71_out1 = add193_out1.reshape(concat35_out1);
        let transpose69_out1 = reshape69_out1.permute([0, 2, 1, 3]);
        let transpose70_out1 = reshape71_out1.permute([0, 2, 1, 3]);
        let transpose71_out1 = reshape70_out1.permute([0, 2, 3, 1]);
        let matmul140_k_corrected = transpose71_out1.permute([0, 1, 3, 2]);
        let (matmul141_out1,) = {
            let q = transpose69_out1;
            let k = matmul140_k_corrected;
            let v = transpose70_out1;
            let matmul141_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul141_out1,)
        };
        let transpose72_out1 = matmul141_out1.permute([0, 2, 1, 3]);
        let reshape72_out1 = transpose72_out1.reshape(concat36_out1);
        let linear106_out1 = self.linear106.forward(reshape72_out1);
        let add194_out1 = add188_out1.add(linear106_out1);
        let reducemean71_out1 = { add194_out1.clone().mean_dim(2usize) };
        let sub36_out1 = add194_out1.clone().sub(reducemean71_out1);
        let pow36_out1 = sub36_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean72_out1 = { pow36_out1.mean_dim(2usize) };
        let add195_out1 =
            reducemean72_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt36_out1 = add195_out1.sqrt();
        let div36_out1 = sub36_out1.div(sqrt36_out1);
        let constant300_out1 = self.constant300.val();
        let mul174_out1 = div36_out1.mul((constant300_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant301_out1 = self.constant301.val();
        let add196_out1 = mul174_out1.add((constant301_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear107_out1 = self.linear107.forward(add196_out1);
        let mul175_out1 = linear107_out1.clone().mul(linear107_out1.clone());
        let mul176_out1 = linear107_out1.clone().mul(mul175_out1);
        let mul177_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul176_out1);
        let add197_out1 = linear107_out1.clone().add(mul177_out1);
        let mul178_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add197_out1);
        let tanh18_out1 = mul178_out1.tanh();
        let add198_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh18_out1);
        let mul179_out1 = linear107_out1.mul(add198_out1);
        let mul180_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul179_out1);
        let linear108_out1 = self.linear108.forward(mul180_out1);
        let add199_out1 = add194_out1.add(linear108_out1);
        let reducemean73_out1 = { add199_out1.clone().mean_dim(2usize) };
        let sub37_out1 = add199_out1.clone().sub(reducemean73_out1);
        let pow37_out1 = sub37_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean74_out1 = { pow37_out1.mean_dim(2usize) };
        let add200_out1 = reducemean74_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt37_out1 = add200_out1.sqrt();
        let div37_out1 = sub37_out1.div(sqrt37_out1);
        let constant306_out1 = self.constant306.val();
        let mul181_out1 = div37_out1.mul((constant306_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant307_out1 = self.constant307.val();
        let add201_out1 = mul181_out1.add((constant307_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape20_out1: [i64; 3] = {
            let axes = &add201_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear109_out1 = self.linear109.forward(add201_out1.clone());
        let linear110_out1 = self.linear110.forward(add201_out1.clone());
        let linear111_out1 = self.linear111.forward(add201_out1);
        let gather40_out1 = shape20_out1[0] as i64;
        let gather41_out1 = shape20_out1[1] as i64;
        let constant311_out1 = self.constant311.val();
        let add202_out1 = (constant311_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear109_out1);
        let constant312_out1 = self.constant312.val();
        let add203_out1 = (constant312_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear110_out1);
        let constant313_out1 = self.constant313.val();
        let add204_out1 = (constant313_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear111_out1);
        let unsqueeze38_out1 = [gather40_out1 as i64];
        let unsqueeze39_out1 = [gather41_out1 as i64];
        let concat37_out1: [i64; 4usize] = [
            &unsqueeze38_out1[..],
            &unsqueeze39_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat38_out1: [i64; 3usize] = [
            &unsqueeze38_out1[..],
            &unsqueeze39_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape73_out1 = add202_out1.reshape(concat37_out1);
        let reshape74_out1 = add203_out1.reshape(concat37_out1);
        let reshape75_out1 = add204_out1.reshape(concat37_out1);
        let transpose73_out1 = reshape73_out1.permute([0, 2, 1, 3]);
        let transpose74_out1 = reshape75_out1.permute([0, 2, 1, 3]);
        let transpose75_out1 = reshape74_out1.permute([0, 2, 3, 1]);
        let matmul148_k_corrected = transpose75_out1.permute([0, 1, 3, 2]);
        let (matmul149_out1,) = {
            let q = transpose73_out1;
            let k = matmul148_k_corrected;
            let v = transpose74_out1;
            let matmul149_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul149_out1,)
        };
        let transpose76_out1 = matmul149_out1.permute([0, 2, 1, 3]);
        let reshape76_out1 = transpose76_out1.reshape(concat38_out1);
        let linear112_out1 = self.linear112.forward(reshape76_out1);
        let add205_out1 = add199_out1.add(linear112_out1);
        add205_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule14<B: Backend> {
    constant316: burn::module::Param<Tensor<B, 1>>,
    constant317: burn::module::Param<Tensor<B, 1>>,
    linear113: Linear<B>,
    linear114: Linear<B>,
    constant322: burn::module::Param<Tensor<B, 1>>,
    constant323: burn::module::Param<Tensor<B, 1>>,
    linear115: Linear<B>,
    linear116: Linear<B>,
    linear117: Linear<B>,
    constant327: burn::module::Param<Tensor<B, 1>>,
    constant328: burn::module::Param<Tensor<B, 1>>,
    constant329: burn::module::Param<Tensor<B, 1>>,
    linear118: Linear<B>,
    constant332: burn::module::Param<Tensor<B, 1>>,
    constant333: burn::module::Param<Tensor<B, 1>>,
    linear119: Linear<B>,
    linear120: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule14<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant316: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant317: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear113 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear114 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant322: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant323: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear115 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear116 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear117 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant327: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant328: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant329: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear118 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant332: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant333: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear119 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear120 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant316,
            constant317,
            linear113,
            linear114,
            constant322,
            constant323,
            linear115,
            linear116,
            linear117,
            constant327,
            constant328,
            constant329,
            linear118,
            constant332,
            constant333,
            linear119,
            linear120,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add205_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean75_out1 = { add205_out1.clone().mean_dim(2usize) };
        let sub38_out1 = add205_out1.clone().sub(reducemean75_out1);
        let pow38_out1 = sub38_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean76_out1 = { pow38_out1.mean_dim(2usize) };
        let add206_out1 =
            reducemean76_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt38_out1 = add206_out1.sqrt();
        let div38_out1 = sub38_out1.div(sqrt38_out1);
        let constant316_out1 = self.constant316.val();
        let mul184_out1 = div38_out1.mul((constant316_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant317_out1 = self.constant317.val();
        let add207_out1 = mul184_out1.add((constant317_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear113_out1 = self.linear113.forward(add207_out1);
        let mul185_out1 = linear113_out1.clone().mul(linear113_out1.clone());
        let mul186_out1 = linear113_out1.clone().mul(mul185_out1);
        let mul187_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul186_out1);
        let add208_out1 = linear113_out1.clone().add(mul187_out1);
        let mul188_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add208_out1);
        let tanh19_out1 = mul188_out1.tanh();
        let add209_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh19_out1);
        let mul189_out1 = linear113_out1.mul(add209_out1);
        let mul190_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul189_out1);
        let linear114_out1 = self.linear114.forward(mul190_out1);
        let add210_out1 = add205_out1.add(linear114_out1);
        let reducemean77_out1 = { add210_out1.clone().mean_dim(2usize) };
        let sub39_out1 = add210_out1.clone().sub(reducemean77_out1);
        let pow39_out1 = sub39_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean78_out1 = { pow39_out1.mean_dim(2usize) };
        let add211_out1 =
            reducemean78_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt39_out1 = add211_out1.sqrt();
        let div39_out1 = sub39_out1.div(sqrt39_out1);
        let constant322_out1 = self.constant322.val();
        let mul191_out1 = div39_out1.mul((constant322_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant323_out1 = self.constant323.val();
        let add212_out1 = mul191_out1.add((constant323_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape21_out1: [i64; 3] = {
            let axes = &add212_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear115_out1 = self.linear115.forward(add212_out1.clone());
        let linear116_out1 = self.linear116.forward(add212_out1.clone());
        let linear117_out1 = self.linear117.forward(add212_out1);
        let gather42_out1 = shape21_out1[0] as i64;
        let gather43_out1 = shape21_out1[1] as i64;
        let constant327_out1 = self.constant327.val();
        let add213_out1 = (constant327_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear115_out1);
        let constant328_out1 = self.constant328.val();
        let add214_out1 = (constant328_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear116_out1);
        let constant329_out1 = self.constant329.val();
        let add215_out1 = (constant329_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear117_out1);
        let unsqueeze40_out1 = [gather42_out1 as i64];
        let unsqueeze41_out1 = [gather43_out1 as i64];
        let concat39_out1: [i64; 4usize] = [
            &unsqueeze40_out1[..],
            &unsqueeze41_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat40_out1: [i64; 3usize] = [
            &unsqueeze40_out1[..],
            &unsqueeze41_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape77_out1 = add213_out1.reshape(concat39_out1);
        let reshape78_out1 = add214_out1.reshape(concat39_out1);
        let reshape79_out1 = add215_out1.reshape(concat39_out1);
        let transpose77_out1 = reshape77_out1.permute([0, 2, 1, 3]);
        let transpose78_out1 = reshape79_out1.permute([0, 2, 1, 3]);
        let transpose79_out1 = reshape78_out1.permute([0, 2, 3, 1]);
        let matmul156_k_corrected = transpose79_out1.permute([0, 1, 3, 2]);
        let (matmul157_out1,) = {
            let q = transpose77_out1;
            let k = matmul156_k_corrected;
            let v = transpose78_out1;
            let matmul157_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul157_out1,)
        };
        let transpose80_out1 = matmul157_out1.permute([0, 2, 1, 3]);
        let reshape80_out1 = transpose80_out1.reshape(concat40_out1);
        let linear118_out1 = self.linear118.forward(reshape80_out1);
        let add216_out1 = add210_out1.add(linear118_out1);
        let reducemean79_out1 = { add216_out1.clone().mean_dim(2usize) };
        let sub40_out1 = add216_out1.clone().sub(reducemean79_out1);
        let pow40_out1 = sub40_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean80_out1 = { pow40_out1.mean_dim(2usize) };
        let add217_out1 = reducemean80_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt40_out1 = add217_out1.sqrt();
        let div40_out1 = sub40_out1.div(sqrt40_out1);
        let constant332_out1 = self.constant332.val();
        let mul194_out1 = div40_out1.mul((constant332_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant333_out1 = self.constant333.val();
        let add218_out1 = mul194_out1.add((constant333_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear119_out1 = self.linear119.forward(add218_out1);
        let mul195_out1 = linear119_out1.clone().mul(linear119_out1.clone());
        let mul196_out1 = linear119_out1.clone().mul(mul195_out1);
        let mul197_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul196_out1);
        let add219_out1 = linear119_out1.clone().add(mul197_out1);
        let mul198_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add219_out1);
        let tanh20_out1 = mul198_out1.tanh();
        let add220_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh20_out1);
        let mul199_out1 = linear119_out1.mul(add220_out1);
        let mul200_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul199_out1);
        let linear120_out1 = self.linear120.forward(mul200_out1);
        let add221_out1 = add216_out1.add(linear120_out1);
        add221_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule15<B: Backend> {
    constant338: burn::module::Param<Tensor<B, 1>>,
    constant339: burn::module::Param<Tensor<B, 1>>,
    linear121: Linear<B>,
    linear122: Linear<B>,
    linear123: Linear<B>,
    constant343: burn::module::Param<Tensor<B, 1>>,
    constant344: burn::module::Param<Tensor<B, 1>>,
    constant345: burn::module::Param<Tensor<B, 1>>,
    linear124: Linear<B>,
    constant348: burn::module::Param<Tensor<B, 1>>,
    constant349: burn::module::Param<Tensor<B, 1>>,
    linear125: Linear<B>,
    linear126: Linear<B>,
    constant354: burn::module::Param<Tensor<B, 1>>,
    constant355: burn::module::Param<Tensor<B, 1>>,
    linear127: Linear<B>,
    linear128: Linear<B>,
    linear129: Linear<B>,
    constant359: burn::module::Param<Tensor<B, 1>>,
    constant360: burn::module::Param<Tensor<B, 1>>,
    constant361: burn::module::Param<Tensor<B, 1>>,
    linear130: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule15<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant338: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant339: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear121 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear122 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear123 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant343: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant344: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant345: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear124 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant348: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant349: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear125 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear126 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant354: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant355: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear127 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear128 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear129 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant359: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant360: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant361: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear130 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant338,
            constant339,
            linear121,
            linear122,
            linear123,
            constant343,
            constant344,
            constant345,
            linear124,
            constant348,
            constant349,
            linear125,
            linear126,
            constant354,
            constant355,
            linear127,
            linear128,
            linear129,
            constant359,
            constant360,
            constant361,
            linear130,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add221_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean81_out1 = { add221_out1.clone().mean_dim(2usize) };
        let sub41_out1 = add221_out1.clone().sub(reducemean81_out1);
        let pow41_out1 = sub41_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean82_out1 = { pow41_out1.mean_dim(2usize) };
        let add222_out1 =
            reducemean82_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt41_out1 = add222_out1.sqrt();
        let div41_out1 = sub41_out1.div(sqrt41_out1);
        let constant338_out1 = self.constant338.val();
        let mul201_out1 = div41_out1.mul((constant338_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant339_out1 = self.constant339.val();
        let add223_out1 = mul201_out1.add((constant339_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape22_out1: [i64; 3] = {
            let axes = &add223_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear121_out1 = self.linear121.forward(add223_out1.clone());
        let linear122_out1 = self.linear122.forward(add223_out1.clone());
        let linear123_out1 = self.linear123.forward(add223_out1);
        let gather44_out1 = shape22_out1[0] as i64;
        let gather45_out1 = shape22_out1[1] as i64;
        let constant343_out1 = self.constant343.val();
        let add224_out1 = (constant343_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear121_out1);
        let constant344_out1 = self.constant344.val();
        let add225_out1 = (constant344_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear122_out1);
        let constant345_out1 = self.constant345.val();
        let add226_out1 = (constant345_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear123_out1);
        let unsqueeze42_out1 = [gather44_out1 as i64];
        let unsqueeze43_out1 = [gather45_out1 as i64];
        let concat41_out1: [i64; 4usize] = [
            &unsqueeze42_out1[..],
            &unsqueeze43_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat42_out1: [i64; 3usize] = [
            &unsqueeze42_out1[..],
            &unsqueeze43_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape81_out1 = add224_out1.reshape(concat41_out1);
        let reshape82_out1 = add225_out1.reshape(concat41_out1);
        let reshape83_out1 = add226_out1.reshape(concat41_out1);
        let transpose81_out1 = reshape81_out1.permute([0, 2, 1, 3]);
        let transpose82_out1 = reshape83_out1.permute([0, 2, 1, 3]);
        let transpose83_out1 = reshape82_out1.permute([0, 2, 3, 1]);
        let matmul164_k_corrected = transpose83_out1.permute([0, 1, 3, 2]);
        let (matmul165_out1,) = {
            let q = transpose81_out1;
            let k = matmul164_k_corrected;
            let v = transpose82_out1;
            let matmul165_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul165_out1,)
        };
        let transpose84_out1 = matmul165_out1.permute([0, 2, 1, 3]);
        let reshape84_out1 = transpose84_out1.reshape(concat42_out1);
        let linear124_out1 = self.linear124.forward(reshape84_out1);
        let add227_out1 = add221_out1.add(linear124_out1);
        let reducemean83_out1 = { add227_out1.clone().mean_dim(2usize) };
        let sub42_out1 = add227_out1.clone().sub(reducemean83_out1);
        let pow42_out1 = sub42_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean84_out1 = { pow42_out1.mean_dim(2usize) };
        let add228_out1 =
            reducemean84_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt42_out1 = add228_out1.sqrt();
        let div42_out1 = sub42_out1.div(sqrt42_out1);
        let constant348_out1 = self.constant348.val();
        let mul204_out1 = div42_out1.mul((constant348_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant349_out1 = self.constant349.val();
        let add229_out1 = mul204_out1.add((constant349_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear125_out1 = self.linear125.forward(add229_out1);
        let mul205_out1 = linear125_out1.clone().mul(linear125_out1.clone());
        let mul206_out1 = linear125_out1.clone().mul(mul205_out1);
        let mul207_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul206_out1);
        let add230_out1 = linear125_out1.clone().add(mul207_out1);
        let mul208_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add230_out1);
        let tanh21_out1 = mul208_out1.tanh();
        let add231_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh21_out1);
        let mul209_out1 = linear125_out1.mul(add231_out1);
        let mul210_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul209_out1);
        let linear126_out1 = self.linear126.forward(mul210_out1);
        let add232_out1 = add227_out1.add(linear126_out1);
        let reducemean85_out1 = { add232_out1.clone().mean_dim(2usize) };
        let sub43_out1 = add232_out1.clone().sub(reducemean85_out1);
        let pow43_out1 = sub43_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean86_out1 = { pow43_out1.mean_dim(2usize) };
        let add233_out1 = reducemean86_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt43_out1 = add233_out1.sqrt();
        let div43_out1 = sub43_out1.div(sqrt43_out1);
        let constant354_out1 = self.constant354.val();
        let mul211_out1 = div43_out1.mul((constant354_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant355_out1 = self.constant355.val();
        let add234_out1 = mul211_out1.add((constant355_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape23_out1: [i64; 3] = {
            let axes = &add234_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear127_out1 = self.linear127.forward(add234_out1.clone());
        let linear128_out1 = self.linear128.forward(add234_out1.clone());
        let linear129_out1 = self.linear129.forward(add234_out1);
        let gather46_out1 = shape23_out1[0] as i64;
        let gather47_out1 = shape23_out1[1] as i64;
        let constant359_out1 = self.constant359.val();
        let add235_out1 = (constant359_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear127_out1);
        let constant360_out1 = self.constant360.val();
        let add236_out1 = (constant360_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear128_out1);
        let constant361_out1 = self.constant361.val();
        let add237_out1 = (constant361_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear129_out1);
        let unsqueeze44_out1 = [gather46_out1 as i64];
        let unsqueeze45_out1 = [gather47_out1 as i64];
        let concat43_out1: [i64; 4usize] = [
            &unsqueeze44_out1[..],
            &unsqueeze45_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat44_out1: [i64; 3usize] = [
            &unsqueeze44_out1[..],
            &unsqueeze45_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape85_out1 = add235_out1.reshape(concat43_out1);
        let reshape86_out1 = add236_out1.reshape(concat43_out1);
        let reshape87_out1 = add237_out1.reshape(concat43_out1);
        let transpose85_out1 = reshape85_out1.permute([0, 2, 1, 3]);
        let transpose86_out1 = reshape87_out1.permute([0, 2, 1, 3]);
        let transpose87_out1 = reshape86_out1.permute([0, 2, 3, 1]);
        let matmul172_k_corrected = transpose87_out1.permute([0, 1, 3, 2]);
        let (matmul173_out1,) = {
            let q = transpose85_out1;
            let k = matmul172_k_corrected;
            let v = transpose86_out1;
            let matmul173_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul173_out1,)
        };
        let transpose88_out1 = matmul173_out1.permute([0, 2, 1, 3]);
        let reshape88_out1 = transpose88_out1.reshape(concat44_out1);
        let linear130_out1 = self.linear130.forward(reshape88_out1);
        let add238_out1 = add232_out1.add(linear130_out1);
        add238_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule16<B: Backend> {
    constant364: burn::module::Param<Tensor<B, 1>>,
    constant365: burn::module::Param<Tensor<B, 1>>,
    linear131: Linear<B>,
    linear132: Linear<B>,
    constant370: burn::module::Param<Tensor<B, 1>>,
    constant371: burn::module::Param<Tensor<B, 1>>,
    linear133: Linear<B>,
    linear134: Linear<B>,
    linear135: Linear<B>,
    constant375: burn::module::Param<Tensor<B, 1>>,
    constant376: burn::module::Param<Tensor<B, 1>>,
    constant377: burn::module::Param<Tensor<B, 1>>,
    linear136: Linear<B>,
    constant380: burn::module::Param<Tensor<B, 1>>,
    constant381: burn::module::Param<Tensor<B, 1>>,
    linear137: Linear<B>,
    linear138: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule16<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant364: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant365: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear131 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear132 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant370: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant371: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear133 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear134 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear135 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant375: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant376: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant377: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear136 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant380: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant381: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear137 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear138 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant364,
            constant365,
            linear131,
            linear132,
            constant370,
            constant371,
            linear133,
            linear134,
            linear135,
            constant375,
            constant376,
            constant377,
            linear136,
            constant380,
            constant381,
            linear137,
            linear138,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add238_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean87_out1 = { add238_out1.clone().mean_dim(2usize) };
        let sub44_out1 = add238_out1.clone().sub(reducemean87_out1);
        let pow44_out1 = sub44_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean88_out1 = { pow44_out1.mean_dim(2usize) };
        let add239_out1 =
            reducemean88_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt44_out1 = add239_out1.sqrt();
        let div44_out1 = sub44_out1.div(sqrt44_out1);
        let constant364_out1 = self.constant364.val();
        let mul214_out1 = div44_out1.mul((constant364_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant365_out1 = self.constant365.val();
        let add240_out1 = mul214_out1.add((constant365_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear131_out1 = self.linear131.forward(add240_out1);
        let mul215_out1 = linear131_out1.clone().mul(linear131_out1.clone());
        let mul216_out1 = linear131_out1.clone().mul(mul215_out1);
        let mul217_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul216_out1);
        let add241_out1 = linear131_out1.clone().add(mul217_out1);
        let mul218_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add241_out1);
        let tanh22_out1 = mul218_out1.tanh();
        let add242_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh22_out1);
        let mul219_out1 = linear131_out1.mul(add242_out1);
        let mul220_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul219_out1);
        let linear132_out1 = self.linear132.forward(mul220_out1);
        let add243_out1 = add238_out1.add(linear132_out1);
        let reducemean89_out1 = { add243_out1.clone().mean_dim(2usize) };
        let sub45_out1 = add243_out1.clone().sub(reducemean89_out1);
        let pow45_out1 = sub45_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean90_out1 = { pow45_out1.mean_dim(2usize) };
        let add244_out1 =
            reducemean90_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt45_out1 = add244_out1.sqrt();
        let div45_out1 = sub45_out1.div(sqrt45_out1);
        let constant370_out1 = self.constant370.val();
        let mul221_out1 = div45_out1.mul((constant370_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant371_out1 = self.constant371.val();
        let add245_out1 = mul221_out1.add((constant371_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape24_out1: [i64; 3] = {
            let axes = &add245_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear133_out1 = self.linear133.forward(add245_out1.clone());
        let linear134_out1 = self.linear134.forward(add245_out1.clone());
        let linear135_out1 = self.linear135.forward(add245_out1);
        let gather48_out1 = shape24_out1[0] as i64;
        let gather49_out1 = shape24_out1[1] as i64;
        let constant375_out1 = self.constant375.val();
        let add246_out1 = (constant375_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear133_out1);
        let constant376_out1 = self.constant376.val();
        let add247_out1 = (constant376_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear134_out1);
        let constant377_out1 = self.constant377.val();
        let add248_out1 = (constant377_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear135_out1);
        let unsqueeze46_out1 = [gather48_out1 as i64];
        let unsqueeze47_out1 = [gather49_out1 as i64];
        let concat45_out1: [i64; 4usize] = [
            &unsqueeze46_out1[..],
            &unsqueeze47_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat46_out1: [i64; 3usize] = [
            &unsqueeze46_out1[..],
            &unsqueeze47_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape89_out1 = add246_out1.reshape(concat45_out1);
        let reshape90_out1 = add247_out1.reshape(concat45_out1);
        let reshape91_out1 = add248_out1.reshape(concat45_out1);
        let transpose89_out1 = reshape89_out1.permute([0, 2, 1, 3]);
        let transpose90_out1 = reshape91_out1.permute([0, 2, 1, 3]);
        let transpose91_out1 = reshape90_out1.permute([0, 2, 3, 1]);
        let matmul180_k_corrected = transpose91_out1.permute([0, 1, 3, 2]);
        let (matmul181_out1,) = {
            let q = transpose89_out1;
            let k = matmul180_k_corrected;
            let v = transpose90_out1;
            let matmul181_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul181_out1,)
        };
        let transpose92_out1 = matmul181_out1.permute([0, 2, 1, 3]);
        let reshape92_out1 = transpose92_out1.reshape(concat46_out1);
        let linear136_out1 = self.linear136.forward(reshape92_out1);
        let add249_out1 = add243_out1.add(linear136_out1);
        let reducemean91_out1 = { add249_out1.clone().mean_dim(2usize) };
        let sub46_out1 = add249_out1.clone().sub(reducemean91_out1);
        let pow46_out1 = sub46_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean92_out1 = { pow46_out1.mean_dim(2usize) };
        let add250_out1 = reducemean92_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt46_out1 = add250_out1.sqrt();
        let div46_out1 = sub46_out1.div(sqrt46_out1);
        let constant380_out1 = self.constant380.val();
        let mul224_out1 = div46_out1.mul((constant380_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant381_out1 = self.constant381.val();
        let add251_out1 = mul224_out1.add((constant381_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear137_out1 = self.linear137.forward(add251_out1);
        let mul225_out1 = linear137_out1.clone().mul(linear137_out1.clone());
        let mul226_out1 = linear137_out1.clone().mul(mul225_out1);
        let mul227_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul226_out1);
        let add252_out1 = linear137_out1.clone().add(mul227_out1);
        let mul228_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add252_out1);
        let tanh23_out1 = mul228_out1.tanh();
        let add253_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh23_out1);
        let mul229_out1 = linear137_out1.mul(add253_out1);
        let mul230_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul229_out1);
        let linear138_out1 = self.linear138.forward(mul230_out1);
        let add254_out1 = add249_out1.add(linear138_out1);
        add254_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule17<B: Backend> {
    constant386: burn::module::Param<Tensor<B, 1>>,
    constant387: burn::module::Param<Tensor<B, 1>>,
    linear139: Linear<B>,
    linear140: Linear<B>,
    linear141: Linear<B>,
    constant391: burn::module::Param<Tensor<B, 1>>,
    constant392: burn::module::Param<Tensor<B, 1>>,
    constant393: burn::module::Param<Tensor<B, 1>>,
    linear142: Linear<B>,
    constant396: burn::module::Param<Tensor<B, 1>>,
    constant397: burn::module::Param<Tensor<B, 1>>,
    linear143: Linear<B>,
    linear144: Linear<B>,
    constant402: burn::module::Param<Tensor<B, 1>>,
    constant403: burn::module::Param<Tensor<B, 1>>,
    linear145: Linear<B>,
    linear146: Linear<B>,
    linear147: Linear<B>,
    constant407: burn::module::Param<Tensor<B, 1>>,
    constant408: burn::module::Param<Tensor<B, 1>>,
    constant409: burn::module::Param<Tensor<B, 1>>,
    linear148: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule17<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant386: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant387: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear139 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear140 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear141 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant391: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant392: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant393: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear142 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant396: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant397: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear143 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear144 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant402: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant403: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear145 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear146 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear147 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant407: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant408: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant409: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear148 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        Self {
            constant386,
            constant387,
            linear139,
            linear140,
            linear141,
            constant391,
            constant392,
            constant393,
            linear142,
            constant396,
            constant397,
            linear143,
            linear144,
            constant402,
            constant403,
            linear145,
            linear146,
            linear147,
            constant407,
            constant408,
            constant409,
            linear148,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add254_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reducemean93_out1 = { add254_out1.clone().mean_dim(2usize) };
        let sub47_out1 = add254_out1.clone().sub(reducemean93_out1);
        let pow47_out1 = sub47_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean94_out1 = { pow47_out1.mean_dim(2usize) };
        let add255_out1 =
            reducemean94_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt47_out1 = add255_out1.sqrt();
        let div47_out1 = sub47_out1.div(sqrt47_out1);
        let constant386_out1 = self.constant386.val();
        let mul231_out1 = div47_out1.mul((constant386_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant387_out1 = self.constant387.val();
        let add256_out1 = mul231_out1.add((constant387_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape25_out1: [i64; 3] = {
            let axes = &add256_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear139_out1 = self.linear139.forward(add256_out1.clone());
        let linear140_out1 = self.linear140.forward(add256_out1.clone());
        let linear141_out1 = self.linear141.forward(add256_out1);
        let gather50_out1 = shape25_out1[0] as i64;
        let gather51_out1 = shape25_out1[1] as i64;
        let constant391_out1 = self.constant391.val();
        let add257_out1 = (constant391_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear139_out1);
        let constant392_out1 = self.constant392.val();
        let add258_out1 = (constant392_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear140_out1);
        let constant393_out1 = self.constant393.val();
        let add259_out1 = (constant393_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear141_out1);
        let unsqueeze48_out1 = [gather50_out1 as i64];
        let unsqueeze49_out1 = [gather51_out1 as i64];
        let concat47_out1: [i64; 4usize] = [
            &unsqueeze48_out1[..],
            &unsqueeze49_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat48_out1: [i64; 3usize] = [
            &unsqueeze48_out1[..],
            &unsqueeze49_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape93_out1 = add257_out1.reshape(concat47_out1);
        let reshape94_out1 = add258_out1.reshape(concat47_out1);
        let reshape95_out1 = add259_out1.reshape(concat47_out1);
        let transpose93_out1 = reshape93_out1.permute([0, 2, 1, 3]);
        let transpose94_out1 = reshape95_out1.permute([0, 2, 1, 3]);
        let transpose95_out1 = reshape94_out1.permute([0, 2, 3, 1]);
        let matmul188_k_corrected = transpose95_out1.permute([0, 1, 3, 2]);
        let (matmul189_out1,) = {
            let q = transpose93_out1;
            let k = matmul188_k_corrected;
            let v = transpose94_out1;
            let matmul189_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul189_out1,)
        };
        let transpose96_out1 = matmul189_out1.permute([0, 2, 1, 3]);
        let reshape96_out1 = transpose96_out1.reshape(concat48_out1);
        let linear142_out1 = self.linear142.forward(reshape96_out1);
        let add260_out1 = add254_out1.add(linear142_out1);
        let reducemean95_out1 = { add260_out1.clone().mean_dim(2usize) };
        let sub48_out1 = add260_out1.clone().sub(reducemean95_out1);
        let pow48_out1 = sub48_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean96_out1 = { pow48_out1.mean_dim(2usize) };
        let add261_out1 =
            reducemean96_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt48_out1 = add261_out1.sqrt();
        let div48_out1 = sub48_out1.div(sqrt48_out1);
        let constant396_out1 = self.constant396.val();
        let mul234_out1 = div48_out1.mul((constant396_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant397_out1 = self.constant397.val();
        let add262_out1 = mul234_out1.add((constant397_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear143_out1 = self.linear143.forward(add262_out1);
        let mul235_out1 = linear143_out1.clone().mul(linear143_out1.clone());
        let mul236_out1 = linear143_out1.clone().mul(mul235_out1);
        let mul237_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul236_out1);
        let add263_out1 = linear143_out1.clone().add(mul237_out1);
        let mul238_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add263_out1);
        let tanh24_out1 = mul238_out1.tanh();
        let add264_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh24_out1);
        let mul239_out1 = linear143_out1.mul(add264_out1);
        let mul240_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul239_out1);
        let linear144_out1 = self.linear144.forward(mul240_out1);
        let add265_out1 = add260_out1.add(linear144_out1);
        let reducemean97_out1 = { add265_out1.clone().mean_dim(2usize) };
        let sub49_out1 = add265_out1.clone().sub(reducemean97_out1);
        let pow49_out1 = sub49_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean98_out1 = { pow49_out1.mean_dim(2usize) };
        let add266_out1 = reducemean98_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt49_out1 = add266_out1.sqrt();
        let div49_out1 = sub49_out1.div(sqrt49_out1);
        let constant402_out1 = self.constant402.val();
        let mul241_out1 = div49_out1.mul((constant402_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant403_out1 = self.constant403.val();
        let add267_out1 = mul241_out1.add((constant403_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape26_out1: [i64; 3] = {
            let axes = &add267_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear145_out1 = self.linear145.forward(add267_out1.clone());
        let linear146_out1 = self.linear146.forward(add267_out1.clone());
        let linear147_out1 = self.linear147.forward(add267_out1);
        let gather52_out1 = shape26_out1[0] as i64;
        let gather53_out1 = shape26_out1[1] as i64;
        let constant407_out1 = self.constant407.val();
        let add268_out1 = (constant407_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear145_out1);
        let constant408_out1 = self.constant408.val();
        let add269_out1 = (constant408_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear146_out1);
        let constant409_out1 = self.constant409.val();
        let add270_out1 = (constant409_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear147_out1);
        let unsqueeze50_out1 = [gather52_out1 as i64];
        let unsqueeze51_out1 = [gather53_out1 as i64];
        let concat49_out1: [i64; 4usize] = [
            &unsqueeze50_out1[..],
            &unsqueeze51_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat50_out1: [i64; 3usize] = [
            &unsqueeze50_out1[..],
            &unsqueeze51_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape97_out1 = add268_out1.reshape(concat49_out1);
        let reshape98_out1 = add269_out1.reshape(concat49_out1);
        let reshape99_out1 = add270_out1.reshape(concat49_out1);
        let transpose97_out1 = reshape97_out1.permute([0, 2, 1, 3]);
        let transpose98_out1 = reshape99_out1.permute([0, 2, 1, 3]);
        let transpose99_out1 = reshape98_out1.permute([0, 2, 3, 1]);
        let matmul196_k_corrected = transpose99_out1.permute([0, 1, 3, 2]);
        let (matmul197_out1,) = {
            let q = transpose97_out1;
            let k = matmul196_k_corrected;
            let v = transpose98_out1;
            let matmul197_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul197_out1,)
        };
        let transpose100_out1 = matmul197_out1.permute([0, 2, 1, 3]);
        let reshape100_out1 = transpose100_out1.reshape(concat50_out1);
        let linear148_out1 = self.linear148.forward(reshape100_out1);
        let add271_out1 = add265_out1.add(linear148_out1);
        add271_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule18<B: Backend> {
    constant412: burn::module::Param<Tensor<B, 1>>,
    constant413: burn::module::Param<Tensor<B, 1>>,
    linear149: Linear<B>,
    linear150: Linear<B>,
    constant418: burn::module::Param<Tensor<B, 1>>,
    constant419: burn::module::Param<Tensor<B, 1>>,
    linear151: Linear<B>,
    linear152: Linear<B>,
    linear153: Linear<B>,
    constant423: burn::module::Param<Tensor<B, 1>>,
    constant424: burn::module::Param<Tensor<B, 1>>,
    constant425: burn::module::Param<Tensor<B, 1>>,
    linear154: Linear<B>,
    constant428: burn::module::Param<Tensor<B, 1>>,
    constant429: burn::module::Param<Tensor<B, 1>>,
    linear155: Linear<B>,
    linear156: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule18<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant412: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant413: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear149 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear150 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant418: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant419: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear151 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear152 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear153 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant423: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant424: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant425: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear154 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant428: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant429: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear155 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear156 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        Self {
            constant412,
            constant413,
            linear149,
            linear150,
            constant418,
            constant419,
            linear151,
            linear152,
            linear153,
            constant423,
            constant424,
            constant425,
            linear154,
            constant428,
            constant429,
            linear155,
            linear156,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add271_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
    ) -> Tensor<B, 3> {
        let reducemean99_out1 = { add271_out1.clone().mean_dim(2usize) };
        let sub50_out1 = add271_out1.clone().sub(reducemean99_out1);
        let pow50_out1 = sub50_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean100_out1 = { pow50_out1.mean_dim(2usize) };
        let add272_out1 =
            reducemean100_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt50_out1 = add272_out1.sqrt();
        let div50_out1 = sub50_out1.div(sqrt50_out1);
        let constant412_out1 = self.constant412.val();
        let mul244_out1 = div50_out1.mul((constant412_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant413_out1 = self.constant413.val();
        let add273_out1 = mul244_out1.add((constant413_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear149_out1 = self.linear149.forward(add273_out1);
        let mul245_out1 = linear149_out1.clone().mul(linear149_out1.clone());
        let mul246_out1 = linear149_out1.clone().mul(mul245_out1);
        let mul247_out1 = (constant28_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul246_out1);
        let add274_out1 = linear149_out1.clone().add(mul247_out1);
        let mul248_out1 = (constant29_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add274_out1);
        let tanh25_out1 = mul248_out1.tanh();
        let add275_out1 = (constant30_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh25_out1);
        let mul249_out1 = linear149_out1.mul(add275_out1);
        let mul250_out1 = (constant31_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul249_out1);
        let linear150_out1 = self.linear150.forward(mul250_out1);
        let add276_out1 = add271_out1.add(linear150_out1);
        let reducemean101_out1 = { add276_out1.clone().mean_dim(2usize) };
        let sub51_out1 = add276_out1.clone().sub(reducemean101_out1);
        let pow51_out1 = sub51_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean102_out1 = { pow51_out1.mean_dim(2usize) };
        let add277_out1 =
            reducemean102_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt51_out1 = add277_out1.sqrt();
        let div51_out1 = sub51_out1.div(sqrt51_out1);
        let constant418_out1 = self.constant418.val();
        let mul251_out1 = div51_out1.mul((constant418_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant419_out1 = self.constant419.val();
        let add278_out1 = mul251_out1.add((constant419_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape27_out1: [i64; 3] = {
            let axes = &add278_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear151_out1 = self.linear151.forward(add278_out1.clone());
        let linear152_out1 = self.linear152.forward(add278_out1.clone());
        let linear153_out1 = self.linear153.forward(add278_out1);
        let gather54_out1 = shape27_out1[0] as i64;
        let gather55_out1 = shape27_out1[1] as i64;
        let constant423_out1 = self.constant423.val();
        let add279_out1 = (constant423_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear151_out1);
        let constant424_out1 = self.constant424.val();
        let add280_out1 = (constant424_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear152_out1);
        let constant425_out1 = self.constant425.val();
        let add281_out1 = (constant425_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear153_out1);
        let unsqueeze52_out1 = [gather54_out1 as i64];
        let unsqueeze53_out1 = [gather55_out1 as i64];
        let concat51_out1: [i64; 4usize] = [
            &unsqueeze52_out1[..],
            &unsqueeze53_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat52_out1: [i64; 3usize] = [
            &unsqueeze52_out1[..],
            &unsqueeze53_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape101_out1 = add279_out1.reshape(concat51_out1);
        let reshape102_out1 = add280_out1.reshape(concat51_out1);
        let reshape103_out1 = add281_out1.reshape(concat51_out1);
        let transpose101_out1 = reshape101_out1.permute([0, 2, 1, 3]);
        let transpose102_out1 = reshape103_out1.permute([0, 2, 1, 3]);
        let transpose103_out1 = reshape102_out1.permute([0, 2, 3, 1]);
        let matmul204_k_corrected = transpose103_out1.permute([0, 1, 3, 2]);
        let (matmul205_out1,) = {
            let q = transpose101_out1;
            let k = matmul204_k_corrected;
            let v = transpose102_out1;
            let matmul205_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul205_out1,)
        };
        let transpose104_out1 = matmul205_out1.permute([0, 2, 1, 3]);
        let reshape104_out1 = transpose104_out1.reshape(concat52_out1);
        let linear154_out1 = self.linear154.forward(reshape104_out1);
        let add282_out1 = add276_out1.add(linear154_out1);
        let reducemean103_out1 = { add282_out1.clone().mean_dim(2usize) };
        let sub52_out1 = add282_out1.clone().sub(reducemean103_out1);
        let pow52_out1 = sub52_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean104_out1 = { pow52_out1.mean_dim(2usize) };
        let add283_out1 =
            reducemean104_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt52_out1 = add283_out1.sqrt();
        let div52_out1 = sub52_out1.div(sqrt52_out1);
        let constant428_out1 = self.constant428.val();
        let mul254_out1 = div52_out1.mul((constant428_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant429_out1 = self.constant429.val();
        let add284_out1 = mul254_out1.add((constant429_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear155_out1 = self.linear155.forward(add284_out1);
        let mul255_out1 = linear155_out1.clone().mul(linear155_out1.clone());
        let mul256_out1 = linear155_out1.clone().mul(mul255_out1);
        let mul257_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul256_out1);
        let add285_out1 = linear155_out1.clone().add(mul257_out1);
        let mul258_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add285_out1);
        let tanh26_out1 = mul258_out1.tanh();
        let add286_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh26_out1);
        let mul259_out1 = linear155_out1.mul(add286_out1);
        let mul260_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul259_out1);
        let linear156_out1 = self.linear156.forward(mul260_out1);
        let add287_out1 = add282_out1.add(linear156_out1);
        add287_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule19<B: Backend> {
    constant434: burn::module::Param<Tensor<B, 1>>,
    constant435: burn::module::Param<Tensor<B, 1>>,
    linear157: Linear<B>,
    linear158: Linear<B>,
    linear159: Linear<B>,
    constant439: burn::module::Param<Tensor<B, 1>>,
    constant440: burn::module::Param<Tensor<B, 1>>,
    constant441: burn::module::Param<Tensor<B, 1>>,
    linear160: Linear<B>,
    constant444: burn::module::Param<Tensor<B, 1>>,
    constant445: burn::module::Param<Tensor<B, 1>>,
    linear161: Linear<B>,
    linear162: Linear<B>,
    constant450: burn::module::Param<Tensor<B, 1>>,
    constant451: burn::module::Param<Tensor<B, 1>>,
    linear163: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule19<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant434: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant435: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear157 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear158 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let linear159 = LinearConfig::new(1152, 1152).with_bias(false).init(device);
        let constant439: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant440: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant441: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear160 = LinearConfig::new(1152, 1152).with_bias(true).init(device);
        let constant444: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant445: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear161 = LinearConfig::new(1152, 4304).with_bias(true).init(device);
        let linear162 = LinearConfig::new(4304, 1152).with_bias(true).init(device);
        let constant450: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let constant451: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1152], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1152].into(),
        );
        let linear163 = LinearConfig::new(1152, 1152)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            constant434,
            constant435,
            linear157,
            linear158,
            linear159,
            constant439,
            constant440,
            constant441,
            linear160,
            constant444,
            constant445,
            linear161,
            linear162,
            constant450,
            constant451,
            linear163,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add287_out1: Tensor<B, 3>,
        constant7_out1: Tensor<B, 1>,
        constant8_out1: Tensor<B, 1>,
        constant18_out1: [i64; 1],
        constant19_out1: [i64; 1],
        constant20_out1: [i64; 1],
        constant28_out1: Tensor<B, 1>,
        constant29_out1: Tensor<B, 1>,
        constant30_out1: Tensor<B, 1>,
        constant31_out1: Tensor<B, 1>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let reducemean105_out1 = { add287_out1.clone().mean_dim(2usize) };
        let sub53_out1 = add287_out1.clone().sub(reducemean105_out1);
        let pow53_out1 = sub53_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean106_out1 = { pow53_out1.mean_dim(2usize) };
        let add288_out1 =
            reducemean106_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt53_out1 = add288_out1.sqrt();
        let div53_out1 = sub53_out1.div(sqrt53_out1);
        let constant434_out1 = self.constant434.val();
        let mul261_out1 = div53_out1.mul((constant434_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant435_out1 = self.constant435.val();
        let add289_out1 = mul261_out1.add((constant435_out1).unsqueeze_dims(&[0isize, 1isize]));
        let shape28_out1: [i64; 3] = {
            let axes = &add289_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let linear157_out1 = self.linear157.forward(add289_out1.clone());
        let linear158_out1 = self.linear158.forward(add289_out1.clone());
        let linear159_out1 = self.linear159.forward(add289_out1);
        let gather56_out1 = shape28_out1[0] as i64;
        let gather57_out1 = shape28_out1[1] as i64;
        let constant439_out1 = self.constant439.val();
        let add290_out1 = (constant439_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear157_out1);
        let constant440_out1 = self.constant440.val();
        let add291_out1 = (constant440_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear158_out1);
        let constant441_out1 = self.constant441.val();
        let add292_out1 = (constant441_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(linear159_out1);
        let unsqueeze54_out1 = [gather56_out1 as i64];
        let unsqueeze55_out1 = [gather57_out1 as i64];
        let concat53_out1: [i64; 4usize] = [
            &unsqueeze54_out1[..],
            &unsqueeze55_out1[..],
            &constant18_out1[..],
            &constant19_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let concat54_out1: [i64; 3usize] = [
            &unsqueeze54_out1[..],
            &unsqueeze55_out1[..],
            &constant20_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape105_out1 = add290_out1.reshape(concat53_out1);
        let reshape106_out1 = add291_out1.reshape(concat53_out1);
        let reshape107_out1 = add292_out1.reshape(concat53_out1);
        let transpose105_out1 = reshape105_out1.permute([0, 2, 1, 3]);
        let transpose106_out1 = reshape107_out1.permute([0, 2, 1, 3]);
        let transpose107_out1 = reshape106_out1.permute([0, 2, 3, 1]);
        let matmul212_k_corrected = transpose107_out1.permute([0, 1, 3, 2]);
        let (matmul213_out1,) = {
            let q = transpose105_out1;
            let k = matmul212_k_corrected;
            let v = transpose106_out1;
            let matmul213_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.1178511381149292f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul213_out1,)
        };
        let transpose108_out1 = matmul213_out1.permute([0, 2, 1, 3]);
        let reshape108_out1 = transpose108_out1.reshape(concat54_out1);
        let linear160_out1 = self.linear160.forward(reshape108_out1);
        let add293_out1 = add287_out1.add(linear160_out1);
        let reducemean107_out1 = { add293_out1.clone().mean_dim(2usize) };
        let sub54_out1 = add293_out1.clone().sub(reducemean107_out1);
        let pow54_out1 = sub54_out1
            .clone()
            .powf((constant7_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean108_out1 = { pow54_out1.mean_dim(2usize) };
        let add294_out1 =
            reducemean108_out1.add((constant8_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt54_out1 = add294_out1.sqrt();
        let div54_out1 = sub54_out1.div(sqrt54_out1);
        let constant444_out1 = self.constant444.val();
        let mul264_out1 = div54_out1.mul((constant444_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant445_out1 = self.constant445.val();
        let add295_out1 = mul264_out1.add((constant445_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear161_out1 = self.linear161.forward(add295_out1);
        let mul265_out1 = linear161_out1.clone().mul(linear161_out1.clone());
        let mul266_out1 = linear161_out1.clone().mul(mul265_out1);
        let mul267_out1 = (constant28_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul266_out1);
        let add296_out1 = linear161_out1.clone().add(mul267_out1);
        let mul268_out1 = (constant29_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(add296_out1);
        let tanh27_out1 = mul268_out1.tanh();
        let add297_out1 = (constant30_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .add(tanh27_out1);
        let mul269_out1 = linear161_out1.mul(add297_out1);
        let mul270_out1 = (constant31_out1)
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul269_out1);
        let linear162_out1 = self.linear162.forward(mul270_out1);
        let add298_out1 = add293_out1.add(linear162_out1);
        let reducemean109_out1 = { add298_out1.clone().mean_dim(2usize) };
        let sub55_out1 = add298_out1.sub(reducemean109_out1);
        let pow55_out1 = sub55_out1
            .clone()
            .powf((constant7_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean110_out1 = { pow55_out1.mean_dim(2usize) };
        let add299_out1 =
            reducemean110_out1.add((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt55_out1 = add299_out1.sqrt();
        let div55_out1 = sub55_out1.div(sqrt55_out1);
        let constant450_out1 = self.constant450.val();
        let mul271_out1 = div55_out1.mul((constant450_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant451_out1 = self.constant451.val();
        let add300_out1 = mul271_out1.add((constant451_out1).unsqueeze_dims(&[0isize, 1isize]));
        let gather58_out1 = {
            let sliced = add300_out1.clone().slice(s![.., -1, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let linear163_out1 = self.linear163.forward(gather58_out1);
        (add300_out1, linear163_out1)
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
    submodule10: Submodule10<B>,
    submodule11: Submodule11<B>,
    submodule12: Submodule12<B>,
    submodule13: Submodule13<B>,
    submodule14: Submodule14<B>,
    submodule15: Submodule15<B>,
    submodule16: Submodule16<B>,
    submodule17: Submodule17<B>,
    submodule18: Submodule18<B>,
    submodule19: Submodule19<B>,
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
        let submodule10 = Submodule10::new(device);
        let submodule11 = Submodule11::new(device);
        let submodule12 = Submodule12::new(device);
        let submodule13 = Submodule13::new(device);
        let submodule14 = Submodule14::new(device);
        let submodule15 = Submodule15::new(device);
        let submodule16 = Submodule16::new(device);
        let submodule17 = Submodule17::new(device);
        let submodule18 = Submodule18::new(device);
        let submodule19 = Submodule19::new(device);
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
            submodule10,
            submodule11,
            submodule12,
            submodule13,
            submodule14,
            submodule15,
            submodule16,
            submodule17,
            submodule18,
            submodule19,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let (
            add9_out1,
            add7_out1,
            constant7_out1,
            constant8_out1,
            constant18_out1,
            constant19_out1,
            constant20_out1,
        ) = self.submodule1.forward(input_ids);
        let (add23_out1, constant28_out1, constant29_out1, constant30_out1, constant31_out1) =
            self.submodule2.forward(
                add9_out1,
                add7_out1,
                constant7_out1.clone(),
                constant8_out1.clone(),
                constant18_out1.clone(),
                constant19_out1.clone(),
                constant20_out1.clone(),
            );
        let add40_out1 = self.submodule3.forward(
            add23_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add56_out1 = self.submodule4.forward(
            add40_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let add73_out1 = self.submodule5.forward(
            add56_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add89_out1 = self.submodule6.forward(
            add73_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let add106_out1 = self.submodule7.forward(
            add89_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add122_out1 = self.submodule8.forward(
            add106_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let add139_out1 = self.submodule9.forward(
            add122_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add155_out1 = self.submodule10.forward(
            add139_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let add172_out1 = self.submodule11.forward(
            add155_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add188_out1 = self.submodule12.forward(
            add172_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let add205_out1 = self.submodule13.forward(
            add188_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add221_out1 = self.submodule14.forward(
            add205_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let add238_out1 = self.submodule15.forward(
            add221_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add254_out1 = self.submodule16.forward(
            add238_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let add271_out1 = self.submodule17.forward(
            add254_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
        );
        let add287_out1 = self.submodule18.forward(
            add271_out1,
            constant7_out1.clone(),
            constant8_out1.clone(),
            constant28_out1.clone(),
            constant29_out1.clone(),
            constant30_out1.clone(),
            constant31_out1.clone(),
            constant18_out1.clone(),
            constant19_out1.clone(),
            constant20_out1.clone(),
        );
        let (add300_out1, linear163_out1) = self.submodule19.forward(
            add287_out1,
            constant7_out1,
            constant8_out1,
            constant18_out1,
            constant19_out1,
            constant20_out1,
            constant28_out1,
            constant29_out1,
            constant30_out1,
            constant31_out1,
        );
        (add300_out1, linear163_out1)
    }
}
