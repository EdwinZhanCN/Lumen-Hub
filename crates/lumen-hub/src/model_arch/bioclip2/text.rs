// Generated from ONNX "src/bioclip2/text.onnx" by burn-onnx
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    constant111: burn::module::Param<Tensor<B, 2>>,
    constant1: burn::module::Param<Tensor<B, 2>>,
    layernormalization1: LayerNorm<B>,
    linear1: Linear<B>,
    constant116: burn::module::Param<Tensor<B, 1>>,
    constant115: burn::module::Param<Tensor<B, 4>>,
    linear2: Linear<B>,
    layernormalization2: LayerNorm<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant111: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2>::zeros([49408, 768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [49408, 768].into(),
        );
        let constant1: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2>::zeros([77, 768], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [77, 768].into(),
        );
        let layernormalization1 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear1 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant116: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let constant115: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear2 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization2 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear3 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear4 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        Self {
            constant111,
            constant1,
            layernormalization1,
            linear1,
            constant116,
            constant115,
            linear2,
            layernormalization2,
            linear3,
            linear4,
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
        [i64; 4],
        [i64; 3],
        [i64; 4],
        [i64; 1],
        Tensor<B, 1>,
        [i64; 2],
        [i64; 3],
        i64,
        [i64; 1],
        [i64; 1],
    ) {
        let shape1_out1: [i64; 1] = {
            let axes = &input_ids.clone().dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let squeeze1_out1 = shape1_out1[0] as i64;
        let constant111_out1 = self.constant111.val();
        let gather1_out1 = constant111_out1.take::<2, 3>(0, input_ids);
        let constant1_out1 = self.constant1.val();
        let add1_out1 = gather1_out1.add((constant1_out1).unsqueeze_dims(&[0isize]));
        let layernormalization1_out1 = {
            let dtype = add1_out1.clone().dtype();
            self.layernormalization1
                .forward(add1_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose1_out1 = layernormalization1_out1.permute([1, 0, 2]);
        let linear1_out1 = self.linear1.forward(transpose1_out1);
        let constant167_out1: [i64; 1] = [77i64];
        let constant168_out1: [i64; 1] = [3i64];
        let constant169_out1: [i64; 1] = [768i64];
        let concat1_out1: [i64; 4usize] = [
            &constant167_out1[..],
            &shape1_out1[..],
            &constant168_out1[..],
            &constant169_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape1_out1 = linear1_out1.reshape(concat1_out1);
        let unsqueeze1_out1: Tensor<B, 5> = reshape1_out1.unsqueeze_dims::<5>(&[0]);
        let transpose2_out1 = unsqueeze1_out1.permute([3, 1, 2, 0, 4]);
        let squeeze2_out1 = transpose2_out1.squeeze_dims::<4>(&[-2]);
        let gather2_out1 = {
            let sliced = squeeze2_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather3_out1 = {
            let sliced = squeeze2_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather4_out1 = {
            let sliced = squeeze2_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let constant173_out1 = 12i64;
        let mul1_out1 = squeeze1_out1 * constant173_out1;
        let constant174_out1: [i64; 1] = [64i64];
        let concat2_out1: [i64; 3usize] = [
            &constant167_out1[..],
            &[mul1_out1][..],
            &constant174_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape3_out1 = gather2_out1.reshape(concat2_out1);
        let transpose3_out1 = reshape3_out1.permute([1, 0, 2]);
        let reshape4_out1 = gather3_out1.reshape(concat2_out1);
        let transpose4_out1 = reshape4_out1.permute([1, 0, 2]);
        let reshape5_out1 = gather4_out1.reshape(concat2_out1);
        let transpose5_out1 = reshape5_out1.permute([1, 0, 2]);
        let constant175_out1: [i64; 1] = [12i64];
        let concat3_out1: [i64; 4usize] = [
            &shape1_out1[..],
            &constant175_out1[..],
            &constant167_out1[..],
            &constant174_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape6_out1 = transpose3_out1.reshape(concat3_out1);
        let reshape7_out1 = transpose4_out1.reshape(concat3_out1);
        let reshape8_out1 = transpose5_out1.reshape(concat3_out1);
        let shape2_out1: [i64; 4] = {
            let axes = &reshape7_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 1] = shape2_out1[3..4].try_into().unwrap();
        let slice2_out1: [i64; 1] = shape2_out1[2..3].try_into().unwrap();
        let slice3_out1: [i64; 2] = shape2_out1[0..2].try_into().unwrap();
        let constant165_out1: [i64; 1] = [-1i64];
        let concat4_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice2_out1[..], &slice1_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape9_out1 = reshape7_out1.reshape(concat4_out1);
        let transpose6_out1 = reshape9_out1.permute([0, 2, 1]);
        let concat5_out1: [i64; 4usize] = [&slice3_out1[..], &slice1_out1[..], &slice2_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape10_out1 = transpose6_out1.reshape(concat5_out1);
        let constant116_out1 = self.constant116.val();
        let mul2_out1 =
            reshape6_out1.mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul3_out1 = reshape10_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul2_out1 = mul2_out1.matmul(mul3_out1);
        let constant115_out1 = self.constant115.val();
        let add2_out1 = matmul2_out1.add(constant115_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(add2_out1, 3);
        let matmul3_out1 = softmax1_out1.matmul(reshape8_out1);
        let transpose7_out1 = matmul3_out1.permute([2, 0, 1, 3]);
        let constant178_out1 = 77i64;
        let mul4_out1 = squeeze1_out1 * constant178_out1;
        let concat6_out1: [i64; 2usize] = [&[mul4_out1][..], &constant169_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape12_out1 = transpose7_out1.reshape(concat6_out1);
        let linear2_out1 = self.linear2.forward(reshape12_out1);
        let concat7_out1: [i64; 3usize] = [
            &constant167_out1[..],
            &shape1_out1[..],
            &constant169_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape13_out1 = linear2_out1.reshape(concat7_out1);
        let transpose8_out1 = reshape13_out1.permute([1, 0, 2]);
        let add3_out1 = add1_out1.add(transpose8_out1);
        let layernormalization2_out1 = {
            let dtype = add3_out1.clone().dtype();
            self.layernormalization2
                .forward(add3_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear3_out1 = self.linear3.forward(layernormalization2_out1);
        let gelu1_out1 = burn::tensor::activation::gelu(linear3_out1);
        let linear4_out1 = self.linear4.forward(gelu1_out1);
        let add4_out1 = add3_out1.add(linear4_out1);
        (
            add4_out1,
            concat1_out1,
            concat2_out1,
            concat3_out1,
            constant165_out1,
            constant116_out1,
            concat6_out1,
            concat7_out1,
            squeeze1_out1,
            shape1_out1,
            constant169_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    layernormalization3: LayerNorm<B>,
    linear5: Linear<B>,
    constant120: burn::module::Param<Tensor<B, 4>>,
    linear6: Linear<B>,
    layernormalization4: LayerNorm<B>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    layernormalization5: LayerNorm<B>,
    linear9: Linear<B>,
    constant124: burn::module::Param<Tensor<B, 4>>,
    linear10: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization3 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear5 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant120: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear6 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization4 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear7 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear8 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization5 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear9 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant124: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear10 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization3,
            linear5,
            constant120,
            linear6,
            layernormalization4,
            linear7,
            linear8,
            layernormalization5,
            linear9,
            constant124,
            linear10,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add4_out1: Tensor<B, 3>,
        concat1_out1: [i64; 4],
        concat2_out1: [i64; 3],
        concat3_out1: [i64; 4],
        constant165_out1: [i64; 1],
        constant116_out1: Tensor<B, 1>,
        concat6_out1: [i64; 2],
        concat7_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization3_out1 = {
            let dtype = add4_out1.clone().dtype();
            self.layernormalization3
                .forward(add4_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose9_out1 = layernormalization3_out1.permute([1, 0, 2]);
        let linear5_out1 = self.linear5.forward(transpose9_out1);
        let reshape14_out1 = linear5_out1.reshape(concat1_out1);
        let unsqueeze2_out1: Tensor<B, 5> = reshape14_out1.unsqueeze_dims::<5>(&[0]);
        let transpose10_out1 = unsqueeze2_out1.permute([3, 1, 2, 0, 4]);
        let squeeze3_out1 = transpose10_out1.squeeze_dims::<4>(&[-2]);
        let gather5_out1 = {
            let sliced = squeeze3_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather6_out1 = {
            let sliced = squeeze3_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather7_out1 = {
            let sliced = squeeze3_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape15_out1 = gather5_out1.reshape(concat2_out1);
        let transpose11_out1 = reshape15_out1.permute([1, 0, 2]);
        let reshape16_out1 = gather6_out1.reshape(concat2_out1);
        let transpose12_out1 = reshape16_out1.permute([1, 0, 2]);
        let reshape17_out1 = gather7_out1.reshape(concat2_out1);
        let transpose13_out1 = reshape17_out1.permute([1, 0, 2]);
        let reshape18_out1 = transpose11_out1.reshape(concat3_out1);
        let reshape19_out1 = transpose12_out1.reshape(concat3_out1);
        let reshape20_out1 = transpose13_out1.reshape(concat3_out1);
        let shape3_out1: [i64; 4] = {
            let axes = &reshape19_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice4_out1: [i64; 1] = shape3_out1[3..4].try_into().unwrap();
        let slice5_out1: [i64; 1] = shape3_out1[2..3].try_into().unwrap();
        let slice6_out1: [i64; 2] = shape3_out1[0..2].try_into().unwrap();
        let concat8_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice5_out1[..], &slice4_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape21_out1 = reshape19_out1.reshape(concat8_out1);
        let transpose14_out1 = reshape21_out1.permute([0, 2, 1]);
        let concat9_out1: [i64; 4usize] = [&slice6_out1[..], &slice4_out1[..], &slice5_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape22_out1 = transpose14_out1.reshape(concat9_out1);
        let mul5_out1 = reshape18_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul6_out1 = reshape22_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul7_out1 = mul5_out1.matmul(mul6_out1);
        let constant120_out1 = self.constant120.val();
        let add5_out1 = matmul7_out1.add(constant120_out1);
        let softmax2_out1 = burn::tensor::activation::softmax(add5_out1, 3);
        let matmul8_out1 = softmax2_out1.matmul(reshape20_out1);
        let transpose15_out1 = matmul8_out1.permute([2, 0, 1, 3]);
        let reshape23_out1 = transpose15_out1.reshape(concat6_out1);
        let linear6_out1 = self.linear6.forward(reshape23_out1);
        let reshape24_out1 = linear6_out1.reshape(concat7_out1);
        let transpose16_out1 = reshape24_out1.permute([1, 0, 2]);
        let add6_out1 = add4_out1.add(transpose16_out1);
        let layernormalization4_out1 = {
            let dtype = add6_out1.clone().dtype();
            self.layernormalization4
                .forward(add6_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear7_out1 = self.linear7.forward(layernormalization4_out1);
        let gelu2_out1 = burn::tensor::activation::gelu(linear7_out1);
        let linear8_out1 = self.linear8.forward(gelu2_out1);
        let add7_out1 = add6_out1.add(linear8_out1);
        let layernormalization5_out1 = {
            let dtype = add7_out1.clone().dtype();
            self.layernormalization5
                .forward(add7_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose17_out1 = layernormalization5_out1.permute([1, 0, 2]);
        let linear9_out1 = self.linear9.forward(transpose17_out1);
        let reshape25_out1 = linear9_out1.reshape(concat1_out1);
        let unsqueeze3_out1: Tensor<B, 5> = reshape25_out1.unsqueeze_dims::<5>(&[0]);
        let transpose18_out1 = unsqueeze3_out1.permute([3, 1, 2, 0, 4]);
        let squeeze4_out1 = transpose18_out1.squeeze_dims::<4>(&[-2]);
        let gather8_out1 = {
            let sliced = squeeze4_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather9_out1 = {
            let sliced = squeeze4_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather10_out1 = {
            let sliced = squeeze4_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape26_out1 = gather8_out1.reshape(concat2_out1);
        let transpose19_out1 = reshape26_out1.permute([1, 0, 2]);
        let reshape27_out1 = gather9_out1.reshape(concat2_out1);
        let transpose20_out1 = reshape27_out1.permute([1, 0, 2]);
        let reshape28_out1 = gather10_out1.reshape(concat2_out1);
        let transpose21_out1 = reshape28_out1.permute([1, 0, 2]);
        let reshape29_out1 = transpose19_out1.reshape(concat3_out1);
        let reshape30_out1 = transpose20_out1.reshape(concat3_out1);
        let reshape31_out1 = transpose21_out1.reshape(concat3_out1);
        let shape4_out1: [i64; 4] = {
            let axes = &reshape30_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice7_out1: [i64; 1] = shape4_out1[3..4].try_into().unwrap();
        let slice8_out1: [i64; 1] = shape4_out1[2..3].try_into().unwrap();
        let slice9_out1: [i64; 2] = shape4_out1[0..2].try_into().unwrap();
        let concat10_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice8_out1[..], &slice7_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape32_out1 = reshape30_out1.reshape(concat10_out1);
        let transpose22_out1 = reshape32_out1.permute([0, 2, 1]);
        let concat11_out1: [i64; 4usize] = [&slice9_out1[..], &slice7_out1[..], &slice8_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape33_out1 = transpose22_out1.reshape(concat11_out1);
        let mul7_out1 = reshape29_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul8_out1 =
            reshape33_out1.mul((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul12_out1 = mul7_out1.matmul(mul8_out1);
        let constant124_out1 = self.constant124.val();
        let add8_out1 = matmul12_out1.add(constant124_out1);
        let softmax3_out1 = burn::tensor::activation::softmax(add8_out1, 3);
        let matmul13_out1 = softmax3_out1.matmul(reshape31_out1);
        let transpose23_out1 = matmul13_out1.permute([2, 0, 1, 3]);
        let reshape34_out1 = transpose23_out1.reshape(concat6_out1);
        let linear10_out1 = self.linear10.forward(reshape34_out1);
        let reshape35_out1 = linear10_out1.reshape(concat7_out1);
        let transpose24_out1 = reshape35_out1.permute([1, 0, 2]);
        let add9_out1 = add7_out1.add(transpose24_out1);
        add9_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    layernormalization6: LayerNorm<B>,
    linear11: Linear<B>,
    linear12: Linear<B>,
    layernormalization7: LayerNorm<B>,
    linear13: Linear<B>,
    constant128: burn::module::Param<Tensor<B, 4>>,
    linear14: Linear<B>,
    layernormalization8: LayerNorm<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    layernormalization9: LayerNorm<B>,
    linear17: Linear<B>,
    constant132: burn::module::Param<Tensor<B, 4>>,
    linear18: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization6 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear11 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear12 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization7 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant128: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear14 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization8 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear16 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization9 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear17 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant132: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear18 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization6,
            linear11,
            linear12,
            layernormalization7,
            linear13,
            constant128,
            linear14,
            layernormalization8,
            linear15,
            linear16,
            layernormalization9,
            linear17,
            constant132,
            linear18,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add9_out1: Tensor<B, 3>,
        concat1_out1: [i64; 4],
        concat2_out1: [i64; 3],
        concat3_out1: [i64; 4],
        constant165_out1: [i64; 1],
        constant116_out1: Tensor<B, 1>,
        concat6_out1: [i64; 2],
        concat7_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization6_out1 = {
            let dtype = add9_out1.clone().dtype();
            self.layernormalization6
                .forward(add9_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear11_out1 = self.linear11.forward(layernormalization6_out1);
        let gelu3_out1 = burn::tensor::activation::gelu(linear11_out1);
        let linear12_out1 = self.linear12.forward(gelu3_out1);
        let add10_out1 = add9_out1.add(linear12_out1);
        let layernormalization7_out1 = {
            let dtype = add10_out1.clone().dtype();
            self.layernormalization7
                .forward(add10_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose25_out1 = layernormalization7_out1.permute([1, 0, 2]);
        let linear13_out1 = self.linear13.forward(transpose25_out1);
        let reshape36_out1 = linear13_out1.reshape(concat1_out1);
        let unsqueeze4_out1: Tensor<B, 5> = reshape36_out1.unsqueeze_dims::<5>(&[0]);
        let transpose26_out1 = unsqueeze4_out1.permute([3, 1, 2, 0, 4]);
        let squeeze5_out1 = transpose26_out1.squeeze_dims::<4>(&[-2]);
        let gather11_out1 = {
            let sliced = squeeze5_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather12_out1 = {
            let sliced = squeeze5_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather13_out1 = {
            let sliced = squeeze5_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape37_out1 = gather11_out1.reshape(concat2_out1);
        let transpose27_out1 = reshape37_out1.permute([1, 0, 2]);
        let reshape38_out1 = gather12_out1.reshape(concat2_out1);
        let transpose28_out1 = reshape38_out1.permute([1, 0, 2]);
        let reshape39_out1 = gather13_out1.reshape(concat2_out1);
        let transpose29_out1 = reshape39_out1.permute([1, 0, 2]);
        let reshape40_out1 = transpose27_out1.reshape(concat3_out1);
        let reshape41_out1 = transpose28_out1.reshape(concat3_out1);
        let reshape42_out1 = transpose29_out1.reshape(concat3_out1);
        let shape5_out1: [i64; 4] = {
            let axes = &reshape41_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice10_out1: [i64; 1] = shape5_out1[3..4].try_into().unwrap();
        let slice11_out1: [i64; 1] = shape5_out1[2..3].try_into().unwrap();
        let slice12_out1: [i64; 2] = shape5_out1[0..2].try_into().unwrap();
        let concat12_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice11_out1[..], &slice10_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape43_out1 = reshape41_out1.reshape(concat12_out1);
        let transpose30_out1 = reshape43_out1.permute([0, 2, 1]);
        let concat13_out1: [i64; 4usize] =
            [&slice12_out1[..], &slice10_out1[..], &slice11_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape44_out1 = transpose30_out1.reshape(concat13_out1);
        let mul9_out1 = reshape40_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul10_out1 = reshape44_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul17_out1 = mul9_out1.matmul(mul10_out1);
        let constant128_out1 = self.constant128.val();
        let add11_out1 = matmul17_out1.add(constant128_out1);
        let softmax4_out1 = burn::tensor::activation::softmax(add11_out1, 3);
        let matmul18_out1 = softmax4_out1.matmul(reshape42_out1);
        let transpose31_out1 = matmul18_out1.permute([2, 0, 1, 3]);
        let reshape45_out1 = transpose31_out1.reshape(concat6_out1);
        let linear14_out1 = self.linear14.forward(reshape45_out1);
        let reshape46_out1 = linear14_out1.reshape(concat7_out1);
        let transpose32_out1 = reshape46_out1.permute([1, 0, 2]);
        let add12_out1 = add10_out1.add(transpose32_out1);
        let layernormalization8_out1 = {
            let dtype = add12_out1.clone().dtype();
            self.layernormalization8
                .forward(add12_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear15_out1 = self.linear15.forward(layernormalization8_out1);
        let gelu4_out1 = burn::tensor::activation::gelu(linear15_out1);
        let linear16_out1 = self.linear16.forward(gelu4_out1);
        let add13_out1 = add12_out1.add(linear16_out1);
        let layernormalization9_out1 = {
            let dtype = add13_out1.clone().dtype();
            self.layernormalization9
                .forward(add13_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose33_out1 = layernormalization9_out1.permute([1, 0, 2]);
        let linear17_out1 = self.linear17.forward(transpose33_out1);
        let reshape47_out1 = linear17_out1.reshape(concat1_out1);
        let unsqueeze5_out1: Tensor<B, 5> = reshape47_out1.unsqueeze_dims::<5>(&[0]);
        let transpose34_out1 = unsqueeze5_out1.permute([3, 1, 2, 0, 4]);
        let squeeze6_out1 = transpose34_out1.squeeze_dims::<4>(&[-2]);
        let gather14_out1 = {
            let sliced = squeeze6_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather15_out1 = {
            let sliced = squeeze6_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather16_out1 = {
            let sliced = squeeze6_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape48_out1 = gather14_out1.reshape(concat2_out1);
        let transpose35_out1 = reshape48_out1.permute([1, 0, 2]);
        let reshape49_out1 = gather15_out1.reshape(concat2_out1);
        let transpose36_out1 = reshape49_out1.permute([1, 0, 2]);
        let reshape50_out1 = gather16_out1.reshape(concat2_out1);
        let transpose37_out1 = reshape50_out1.permute([1, 0, 2]);
        let reshape51_out1 = transpose35_out1.reshape(concat3_out1);
        let reshape52_out1 = transpose36_out1.reshape(concat3_out1);
        let reshape53_out1 = transpose37_out1.reshape(concat3_out1);
        let shape6_out1: [i64; 4] = {
            let axes = &reshape52_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice13_out1: [i64; 1] = shape6_out1[3..4].try_into().unwrap();
        let slice14_out1: [i64; 1] = shape6_out1[2..3].try_into().unwrap();
        let slice15_out1: [i64; 2] = shape6_out1[0..2].try_into().unwrap();
        let concat14_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice14_out1[..], &slice13_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape54_out1 = reshape52_out1.reshape(concat14_out1);
        let transpose38_out1 = reshape54_out1.permute([0, 2, 1]);
        let concat15_out1: [i64; 4usize] =
            [&slice15_out1[..], &slice13_out1[..], &slice14_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape55_out1 = transpose38_out1.reshape(concat15_out1);
        let mul11_out1 = reshape51_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul12_out1 =
            reshape55_out1.mul((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul22_out1 = mul11_out1.matmul(mul12_out1);
        let constant132_out1 = self.constant132.val();
        let add14_out1 = matmul22_out1.add(constant132_out1);
        let softmax5_out1 = burn::tensor::activation::softmax(add14_out1, 3);
        let matmul23_out1 = softmax5_out1.matmul(reshape53_out1);
        let transpose39_out1 = matmul23_out1.permute([2, 0, 1, 3]);
        let reshape56_out1 = transpose39_out1.reshape(concat6_out1);
        let linear18_out1 = self.linear18.forward(reshape56_out1);
        let reshape57_out1 = linear18_out1.reshape(concat7_out1);
        let transpose40_out1 = reshape57_out1.permute([1, 0, 2]);
        let add15_out1 = add13_out1.add(transpose40_out1);
        add15_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    layernormalization10: LayerNorm<B>,
    linear19: Linear<B>,
    linear20: Linear<B>,
    layernormalization11: LayerNorm<B>,
    linear21: Linear<B>,
    constant136: burn::module::Param<Tensor<B, 4>>,
    linear22: Linear<B>,
    layernormalization12: LayerNorm<B>,
    linear23: Linear<B>,
    linear24: Linear<B>,
    layernormalization13: LayerNorm<B>,
    linear25: Linear<B>,
    constant140: burn::module::Param<Tensor<B, 4>>,
    linear26: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization10 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear19 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear20 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization11 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear21 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant136: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear22 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization12 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear23 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear24 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization13 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear25 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant140: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear26 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization10,
            linear19,
            linear20,
            layernormalization11,
            linear21,
            constant136,
            linear22,
            layernormalization12,
            linear23,
            linear24,
            layernormalization13,
            linear25,
            constant140,
            linear26,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add15_out1: Tensor<B, 3>,
        concat1_out1: [i64; 4],
        concat2_out1: [i64; 3],
        concat3_out1: [i64; 4],
        constant165_out1: [i64; 1],
        constant116_out1: Tensor<B, 1>,
        concat6_out1: [i64; 2],
        concat7_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization10_out1 = {
            let dtype = add15_out1.clone().dtype();
            self.layernormalization10
                .forward(add15_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear19_out1 = self.linear19.forward(layernormalization10_out1);
        let gelu5_out1 = burn::tensor::activation::gelu(linear19_out1);
        let linear20_out1 = self.linear20.forward(gelu5_out1);
        let add16_out1 = add15_out1.add(linear20_out1);
        let layernormalization11_out1 = {
            let dtype = add16_out1.clone().dtype();
            self.layernormalization11
                .forward(add16_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose41_out1 = layernormalization11_out1.permute([1, 0, 2]);
        let linear21_out1 = self.linear21.forward(transpose41_out1);
        let reshape58_out1 = linear21_out1.reshape(concat1_out1);
        let unsqueeze6_out1: Tensor<B, 5> = reshape58_out1.unsqueeze_dims::<5>(&[0]);
        let transpose42_out1 = unsqueeze6_out1.permute([3, 1, 2, 0, 4]);
        let squeeze7_out1 = transpose42_out1.squeeze_dims::<4>(&[-2]);
        let gather17_out1 = {
            let sliced = squeeze7_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather18_out1 = {
            let sliced = squeeze7_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather19_out1 = {
            let sliced = squeeze7_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape59_out1 = gather17_out1.reshape(concat2_out1);
        let transpose43_out1 = reshape59_out1.permute([1, 0, 2]);
        let reshape60_out1 = gather18_out1.reshape(concat2_out1);
        let transpose44_out1 = reshape60_out1.permute([1, 0, 2]);
        let reshape61_out1 = gather19_out1.reshape(concat2_out1);
        let transpose45_out1 = reshape61_out1.permute([1, 0, 2]);
        let reshape62_out1 = transpose43_out1.reshape(concat3_out1);
        let reshape63_out1 = transpose44_out1.reshape(concat3_out1);
        let reshape64_out1 = transpose45_out1.reshape(concat3_out1);
        let shape7_out1: [i64; 4] = {
            let axes = &reshape63_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice16_out1: [i64; 1] = shape7_out1[3..4].try_into().unwrap();
        let slice17_out1: [i64; 1] = shape7_out1[2..3].try_into().unwrap();
        let slice18_out1: [i64; 2] = shape7_out1[0..2].try_into().unwrap();
        let concat16_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice17_out1[..], &slice16_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape65_out1 = reshape63_out1.reshape(concat16_out1);
        let transpose46_out1 = reshape65_out1.permute([0, 2, 1]);
        let concat17_out1: [i64; 4usize] =
            [&slice18_out1[..], &slice16_out1[..], &slice17_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape66_out1 = transpose46_out1.reshape(concat17_out1);
        let mul13_out1 = reshape62_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul14_out1 = reshape66_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul27_out1 = mul13_out1.matmul(mul14_out1);
        let constant136_out1 = self.constant136.val();
        let add17_out1 = matmul27_out1.add(constant136_out1);
        let softmax6_out1 = burn::tensor::activation::softmax(add17_out1, 3);
        let matmul28_out1 = softmax6_out1.matmul(reshape64_out1);
        let transpose47_out1 = matmul28_out1.permute([2, 0, 1, 3]);
        let reshape67_out1 = transpose47_out1.reshape(concat6_out1);
        let linear22_out1 = self.linear22.forward(reshape67_out1);
        let reshape68_out1 = linear22_out1.reshape(concat7_out1);
        let transpose48_out1 = reshape68_out1.permute([1, 0, 2]);
        let add18_out1 = add16_out1.add(transpose48_out1);
        let layernormalization12_out1 = {
            let dtype = add18_out1.clone().dtype();
            self.layernormalization12
                .forward(add18_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear23_out1 = self.linear23.forward(layernormalization12_out1);
        let gelu6_out1 = burn::tensor::activation::gelu(linear23_out1);
        let linear24_out1 = self.linear24.forward(gelu6_out1);
        let add19_out1 = add18_out1.add(linear24_out1);
        let layernormalization13_out1 = {
            let dtype = add19_out1.clone().dtype();
            self.layernormalization13
                .forward(add19_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose49_out1 = layernormalization13_out1.permute([1, 0, 2]);
        let linear25_out1 = self.linear25.forward(transpose49_out1);
        let reshape69_out1 = linear25_out1.reshape(concat1_out1);
        let unsqueeze7_out1: Tensor<B, 5> = reshape69_out1.unsqueeze_dims::<5>(&[0]);
        let transpose50_out1 = unsqueeze7_out1.permute([3, 1, 2, 0, 4]);
        let squeeze8_out1 = transpose50_out1.squeeze_dims::<4>(&[-2]);
        let gather20_out1 = {
            let sliced = squeeze8_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather21_out1 = {
            let sliced = squeeze8_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather22_out1 = {
            let sliced = squeeze8_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape70_out1 = gather20_out1.reshape(concat2_out1);
        let transpose51_out1 = reshape70_out1.permute([1, 0, 2]);
        let reshape71_out1 = gather21_out1.reshape(concat2_out1);
        let transpose52_out1 = reshape71_out1.permute([1, 0, 2]);
        let reshape72_out1 = gather22_out1.reshape(concat2_out1);
        let transpose53_out1 = reshape72_out1.permute([1, 0, 2]);
        let reshape73_out1 = transpose51_out1.reshape(concat3_out1);
        let reshape74_out1 = transpose52_out1.reshape(concat3_out1);
        let reshape75_out1 = transpose53_out1.reshape(concat3_out1);
        let shape8_out1: [i64; 4] = {
            let axes = &reshape74_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice19_out1: [i64; 1] = shape8_out1[3..4].try_into().unwrap();
        let slice20_out1: [i64; 1] = shape8_out1[2..3].try_into().unwrap();
        let slice21_out1: [i64; 2] = shape8_out1[0..2].try_into().unwrap();
        let concat18_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice20_out1[..], &slice19_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape76_out1 = reshape74_out1.reshape(concat18_out1);
        let transpose54_out1 = reshape76_out1.permute([0, 2, 1]);
        let concat19_out1: [i64; 4usize] =
            [&slice21_out1[..], &slice19_out1[..], &slice20_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape77_out1 = transpose54_out1.reshape(concat19_out1);
        let mul15_out1 = reshape73_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul16_out1 =
            reshape77_out1.mul((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul32_out1 = mul15_out1.matmul(mul16_out1);
        let constant140_out1 = self.constant140.val();
        let add20_out1 = matmul32_out1.add(constant140_out1);
        let softmax7_out1 = burn::tensor::activation::softmax(add20_out1, 3);
        let matmul33_out1 = softmax7_out1.matmul(reshape75_out1);
        let transpose55_out1 = matmul33_out1.permute([2, 0, 1, 3]);
        let reshape78_out1 = transpose55_out1.reshape(concat6_out1);
        let linear26_out1 = self.linear26.forward(reshape78_out1);
        let reshape79_out1 = linear26_out1.reshape(concat7_out1);
        let transpose56_out1 = reshape79_out1.permute([1, 0, 2]);
        let add21_out1 = add19_out1.add(transpose56_out1);
        add21_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5<B: Backend> {
    layernormalization14: LayerNorm<B>,
    linear27: Linear<B>,
    linear28: Linear<B>,
    layernormalization15: LayerNorm<B>,
    linear29: Linear<B>,
    constant144: burn::module::Param<Tensor<B, 4>>,
    linear30: Linear<B>,
    layernormalization16: LayerNorm<B>,
    linear31: Linear<B>,
    linear32: Linear<B>,
    layernormalization17: LayerNorm<B>,
    linear33: Linear<B>,
    constant148: burn::module::Param<Tensor<B, 4>>,
    linear34: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule5<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization14 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear27 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear28 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization15 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear29 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant144: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear30 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization16 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear31 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear32 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization17 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear33 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant148: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear34 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization14,
            linear27,
            linear28,
            layernormalization15,
            linear29,
            constant144,
            linear30,
            layernormalization16,
            linear31,
            linear32,
            layernormalization17,
            linear33,
            constant148,
            linear34,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add21_out1: Tensor<B, 3>,
        concat1_out1: [i64; 4],
        concat2_out1: [i64; 3],
        concat3_out1: [i64; 4],
        constant165_out1: [i64; 1],
        constant116_out1: Tensor<B, 1>,
        concat6_out1: [i64; 2],
        concat7_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization14_out1 = {
            let dtype = add21_out1.clone().dtype();
            self.layernormalization14
                .forward(add21_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear27_out1 = self.linear27.forward(layernormalization14_out1);
        let gelu7_out1 = burn::tensor::activation::gelu(linear27_out1);
        let linear28_out1 = self.linear28.forward(gelu7_out1);
        let add22_out1 = add21_out1.add(linear28_out1);
        let layernormalization15_out1 = {
            let dtype = add22_out1.clone().dtype();
            self.layernormalization15
                .forward(add22_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose57_out1 = layernormalization15_out1.permute([1, 0, 2]);
        let linear29_out1 = self.linear29.forward(transpose57_out1);
        let reshape80_out1 = linear29_out1.reshape(concat1_out1);
        let unsqueeze8_out1: Tensor<B, 5> = reshape80_out1.unsqueeze_dims::<5>(&[0]);
        let transpose58_out1 = unsqueeze8_out1.permute([3, 1, 2, 0, 4]);
        let squeeze9_out1 = transpose58_out1.squeeze_dims::<4>(&[-2]);
        let gather23_out1 = {
            let sliced = squeeze9_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather24_out1 = {
            let sliced = squeeze9_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather25_out1 = {
            let sliced = squeeze9_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape81_out1 = gather23_out1.reshape(concat2_out1);
        let transpose59_out1 = reshape81_out1.permute([1, 0, 2]);
        let reshape82_out1 = gather24_out1.reshape(concat2_out1);
        let transpose60_out1 = reshape82_out1.permute([1, 0, 2]);
        let reshape83_out1 = gather25_out1.reshape(concat2_out1);
        let transpose61_out1 = reshape83_out1.permute([1, 0, 2]);
        let reshape84_out1 = transpose59_out1.reshape(concat3_out1);
        let reshape85_out1 = transpose60_out1.reshape(concat3_out1);
        let reshape86_out1 = transpose61_out1.reshape(concat3_out1);
        let shape9_out1: [i64; 4] = {
            let axes = &reshape85_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice22_out1: [i64; 1] = shape9_out1[3..4].try_into().unwrap();
        let slice23_out1: [i64; 1] = shape9_out1[2..3].try_into().unwrap();
        let slice24_out1: [i64; 2] = shape9_out1[0..2].try_into().unwrap();
        let concat20_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice23_out1[..], &slice22_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape87_out1 = reshape85_out1.reshape(concat20_out1);
        let transpose62_out1 = reshape87_out1.permute([0, 2, 1]);
        let concat21_out1: [i64; 4usize] =
            [&slice24_out1[..], &slice22_out1[..], &slice23_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape88_out1 = transpose62_out1.reshape(concat21_out1);
        let mul17_out1 = reshape84_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul18_out1 = reshape88_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul37_out1 = mul17_out1.matmul(mul18_out1);
        let constant144_out1 = self.constant144.val();
        let add23_out1 = matmul37_out1.add(constant144_out1);
        let softmax8_out1 = burn::tensor::activation::softmax(add23_out1, 3);
        let matmul38_out1 = softmax8_out1.matmul(reshape86_out1);
        let transpose63_out1 = matmul38_out1.permute([2, 0, 1, 3]);
        let reshape89_out1 = transpose63_out1.reshape(concat6_out1);
        let linear30_out1 = self.linear30.forward(reshape89_out1);
        let reshape90_out1 = linear30_out1.reshape(concat7_out1);
        let transpose64_out1 = reshape90_out1.permute([1, 0, 2]);
        let add24_out1 = add22_out1.add(transpose64_out1);
        let layernormalization16_out1 = {
            let dtype = add24_out1.clone().dtype();
            self.layernormalization16
                .forward(add24_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear31_out1 = self.linear31.forward(layernormalization16_out1);
        let gelu8_out1 = burn::tensor::activation::gelu(linear31_out1);
        let linear32_out1 = self.linear32.forward(gelu8_out1);
        let add25_out1 = add24_out1.add(linear32_out1);
        let layernormalization17_out1 = {
            let dtype = add25_out1.clone().dtype();
            self.layernormalization17
                .forward(add25_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose65_out1 = layernormalization17_out1.permute([1, 0, 2]);
        let linear33_out1 = self.linear33.forward(transpose65_out1);
        let reshape91_out1 = linear33_out1.reshape(concat1_out1);
        let unsqueeze9_out1: Tensor<B, 5> = reshape91_out1.unsqueeze_dims::<5>(&[0]);
        let transpose66_out1 = unsqueeze9_out1.permute([3, 1, 2, 0, 4]);
        let squeeze10_out1 = transpose66_out1.squeeze_dims::<4>(&[-2]);
        let gather26_out1 = {
            let sliced = squeeze10_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather27_out1 = {
            let sliced = squeeze10_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather28_out1 = {
            let sliced = squeeze10_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape92_out1 = gather26_out1.reshape(concat2_out1);
        let transpose67_out1 = reshape92_out1.permute([1, 0, 2]);
        let reshape93_out1 = gather27_out1.reshape(concat2_out1);
        let transpose68_out1 = reshape93_out1.permute([1, 0, 2]);
        let reshape94_out1 = gather28_out1.reshape(concat2_out1);
        let transpose69_out1 = reshape94_out1.permute([1, 0, 2]);
        let reshape95_out1 = transpose67_out1.reshape(concat3_out1);
        let reshape96_out1 = transpose68_out1.reshape(concat3_out1);
        let reshape97_out1 = transpose69_out1.reshape(concat3_out1);
        let shape10_out1: [i64; 4] = {
            let axes = &reshape96_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice25_out1: [i64; 1] = shape10_out1[3..4].try_into().unwrap();
        let slice26_out1: [i64; 1] = shape10_out1[2..3].try_into().unwrap();
        let slice27_out1: [i64; 2] = shape10_out1[0..2].try_into().unwrap();
        let concat22_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice26_out1[..], &slice25_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape98_out1 = reshape96_out1.reshape(concat22_out1);
        let transpose70_out1 = reshape98_out1.permute([0, 2, 1]);
        let concat23_out1: [i64; 4usize] =
            [&slice27_out1[..], &slice25_out1[..], &slice26_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape99_out1 = transpose70_out1.reshape(concat23_out1);
        let mul19_out1 = reshape95_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul20_out1 =
            reshape99_out1.mul((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul42_out1 = mul19_out1.matmul(mul20_out1);
        let constant148_out1 = self.constant148.val();
        let add26_out1 = matmul42_out1.add(constant148_out1);
        let softmax9_out1 = burn::tensor::activation::softmax(add26_out1, 3);
        let matmul43_out1 = softmax9_out1.matmul(reshape97_out1);
        let transpose71_out1 = matmul43_out1.permute([2, 0, 1, 3]);
        let reshape100_out1 = transpose71_out1.reshape(concat6_out1);
        let linear34_out1 = self.linear34.forward(reshape100_out1);
        let reshape101_out1 = linear34_out1.reshape(concat7_out1);
        let transpose72_out1 = reshape101_out1.permute([1, 0, 2]);
        let add27_out1 = add25_out1.add(transpose72_out1);
        add27_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6<B: Backend> {
    layernormalization18: LayerNorm<B>,
    linear35: Linear<B>,
    linear36: Linear<B>,
    layernormalization19: LayerNorm<B>,
    linear37: Linear<B>,
    constant152: burn::module::Param<Tensor<B, 4>>,
    linear38: Linear<B>,
    layernormalization20: LayerNorm<B>,
    linear39: Linear<B>,
    linear40: Linear<B>,
    layernormalization21: LayerNorm<B>,
    linear41: Linear<B>,
    constant156: burn::module::Param<Tensor<B, 4>>,
    linear42: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule6<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization18 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear35 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear36 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization19 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear37 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant152: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear38 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization20 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear39 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear40 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization21 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear41 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant156: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear42 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization18,
            linear35,
            linear36,
            layernormalization19,
            linear37,
            constant152,
            linear38,
            layernormalization20,
            linear39,
            linear40,
            layernormalization21,
            linear41,
            constant156,
            linear42,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add27_out1: Tensor<B, 3>,
        concat1_out1: [i64; 4],
        concat2_out1: [i64; 3],
        concat3_out1: [i64; 4],
        constant165_out1: [i64; 1],
        constant116_out1: Tensor<B, 1>,
        concat6_out1: [i64; 2],
        concat7_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization18_out1 = {
            let dtype = add27_out1.clone().dtype();
            self.layernormalization18
                .forward(add27_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear35_out1 = self.linear35.forward(layernormalization18_out1);
        let gelu9_out1 = burn::tensor::activation::gelu(linear35_out1);
        let linear36_out1 = self.linear36.forward(gelu9_out1);
        let add28_out1 = add27_out1.add(linear36_out1);
        let layernormalization19_out1 = {
            let dtype = add28_out1.clone().dtype();
            self.layernormalization19
                .forward(add28_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose73_out1 = layernormalization19_out1.permute([1, 0, 2]);
        let linear37_out1 = self.linear37.forward(transpose73_out1);
        let reshape102_out1 = linear37_out1.reshape(concat1_out1);
        let unsqueeze10_out1: Tensor<B, 5> = reshape102_out1.unsqueeze_dims::<5>(&[0]);
        let transpose74_out1 = unsqueeze10_out1.permute([3, 1, 2, 0, 4]);
        let squeeze11_out1 = transpose74_out1.squeeze_dims::<4>(&[-2]);
        let gather29_out1 = {
            let sliced = squeeze11_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather30_out1 = {
            let sliced = squeeze11_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather31_out1 = {
            let sliced = squeeze11_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape103_out1 = gather29_out1.reshape(concat2_out1);
        let transpose75_out1 = reshape103_out1.permute([1, 0, 2]);
        let reshape104_out1 = gather30_out1.reshape(concat2_out1);
        let transpose76_out1 = reshape104_out1.permute([1, 0, 2]);
        let reshape105_out1 = gather31_out1.reshape(concat2_out1);
        let transpose77_out1 = reshape105_out1.permute([1, 0, 2]);
        let reshape106_out1 = transpose75_out1.reshape(concat3_out1);
        let reshape107_out1 = transpose76_out1.reshape(concat3_out1);
        let reshape108_out1 = transpose77_out1.reshape(concat3_out1);
        let shape11_out1: [i64; 4] = {
            let axes = &reshape107_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice28_out1: [i64; 1] = shape11_out1[3..4].try_into().unwrap();
        let slice29_out1: [i64; 1] = shape11_out1[2..3].try_into().unwrap();
        let slice30_out1: [i64; 2] = shape11_out1[0..2].try_into().unwrap();
        let concat24_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice29_out1[..], &slice28_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape109_out1 = reshape107_out1.reshape(concat24_out1);
        let transpose78_out1 = reshape109_out1.permute([0, 2, 1]);
        let concat25_out1: [i64; 4usize] =
            [&slice30_out1[..], &slice28_out1[..], &slice29_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape110_out1 = transpose78_out1.reshape(concat25_out1);
        let mul21_out1 = reshape106_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul22_out1 = reshape110_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul47_out1 = mul21_out1.matmul(mul22_out1);
        let constant152_out1 = self.constant152.val();
        let add29_out1 = matmul47_out1.add(constant152_out1);
        let softmax10_out1 = burn::tensor::activation::softmax(add29_out1, 3);
        let matmul48_out1 = softmax10_out1.matmul(reshape108_out1);
        let transpose79_out1 = matmul48_out1.permute([2, 0, 1, 3]);
        let reshape111_out1 = transpose79_out1.reshape(concat6_out1);
        let linear38_out1 = self.linear38.forward(reshape111_out1);
        let reshape112_out1 = linear38_out1.reshape(concat7_out1);
        let transpose80_out1 = reshape112_out1.permute([1, 0, 2]);
        let add30_out1 = add28_out1.add(transpose80_out1);
        let layernormalization20_out1 = {
            let dtype = add30_out1.clone().dtype();
            self.layernormalization20
                .forward(add30_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear39_out1 = self.linear39.forward(layernormalization20_out1);
        let gelu10_out1 = burn::tensor::activation::gelu(linear39_out1);
        let linear40_out1 = self.linear40.forward(gelu10_out1);
        let add31_out1 = add30_out1.add(linear40_out1);
        let layernormalization21_out1 = {
            let dtype = add31_out1.clone().dtype();
            self.layernormalization21
                .forward(add31_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose81_out1 = layernormalization21_out1.permute([1, 0, 2]);
        let linear41_out1 = self.linear41.forward(transpose81_out1);
        let reshape113_out1 = linear41_out1.reshape(concat1_out1);
        let unsqueeze11_out1: Tensor<B, 5> = reshape113_out1.unsqueeze_dims::<5>(&[0]);
        let transpose82_out1 = unsqueeze11_out1.permute([3, 1, 2, 0, 4]);
        let squeeze12_out1 = transpose82_out1.squeeze_dims::<4>(&[-2]);
        let gather32_out1 = {
            let sliced = squeeze12_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather33_out1 = {
            let sliced = squeeze12_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather34_out1 = {
            let sliced = squeeze12_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape114_out1 = gather32_out1.reshape(concat2_out1);
        let transpose83_out1 = reshape114_out1.permute([1, 0, 2]);
        let reshape115_out1 = gather33_out1.reshape(concat2_out1);
        let transpose84_out1 = reshape115_out1.permute([1, 0, 2]);
        let reshape116_out1 = gather34_out1.reshape(concat2_out1);
        let transpose85_out1 = reshape116_out1.permute([1, 0, 2]);
        let reshape117_out1 = transpose83_out1.reshape(concat3_out1);
        let reshape118_out1 = transpose84_out1.reshape(concat3_out1);
        let reshape119_out1 = transpose85_out1.reshape(concat3_out1);
        let shape12_out1: [i64; 4] = {
            let axes = &reshape118_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice31_out1: [i64; 1] = shape12_out1[3..4].try_into().unwrap();
        let slice32_out1: [i64; 1] = shape12_out1[2..3].try_into().unwrap();
        let slice33_out1: [i64; 2] = shape12_out1[0..2].try_into().unwrap();
        let concat26_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice32_out1[..], &slice31_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape120_out1 = reshape118_out1.reshape(concat26_out1);
        let transpose86_out1 = reshape120_out1.permute([0, 2, 1]);
        let concat27_out1: [i64; 4usize] =
            [&slice33_out1[..], &slice31_out1[..], &slice32_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape121_out1 = transpose86_out1.reshape(concat27_out1);
        let mul23_out1 = reshape117_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul24_out1 =
            reshape121_out1.mul((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul52_out1 = mul23_out1.matmul(mul24_out1);
        let constant156_out1 = self.constant156.val();
        let add32_out1 = matmul52_out1.add(constant156_out1);
        let softmax11_out1 = burn::tensor::activation::softmax(add32_out1, 3);
        let matmul53_out1 = softmax11_out1.matmul(reshape119_out1);
        let transpose87_out1 = matmul53_out1.permute([2, 0, 1, 3]);
        let reshape122_out1 = transpose87_out1.reshape(concat6_out1);
        let linear42_out1 = self.linear42.forward(reshape122_out1);
        let reshape123_out1 = linear42_out1.reshape(concat7_out1);
        let transpose88_out1 = reshape123_out1.permute([1, 0, 2]);
        let add33_out1 = add31_out1.add(transpose88_out1);
        add33_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule7<B: Backend> {
    layernormalization22: LayerNorm<B>,
    linear43: Linear<B>,
    linear44: Linear<B>,
    layernormalization23: LayerNorm<B>,
    linear45: Linear<B>,
    constant160: burn::module::Param<Tensor<B, 4>>,
    linear46: Linear<B>,
    layernormalization24: LayerNorm<B>,
    linear47: Linear<B>,
    linear48: Linear<B>,
    layernormalization25: LayerNorm<B>,
    linear49: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule7<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization22 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear43 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear44 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization23 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear45 = LinearConfig::new(768, 2304).with_bias(true).init(device);
        let constant160: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 4>::zeros([1, 1, 77, 77], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 77, 77].into(),
        );
        let linear46 = LinearConfig::new(768, 768)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization24 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear47 = LinearConfig::new(768, 3072).with_bias(true).init(device);
        let linear48 = LinearConfig::new(3072, 768).with_bias(true).init(device);
        let layernormalization25 = LayerNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear49 = LinearConfig::new(768, 768).with_bias(false).init(device);
        Self {
            layernormalization22,
            linear43,
            linear44,
            layernormalization23,
            linear45,
            constant160,
            linear46,
            layernormalization24,
            linear47,
            linear48,
            layernormalization25,
            linear49,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add33_out1: Tensor<B, 3>,
        concat1_out1: [i64; 4],
        concat2_out1: [i64; 3],
        concat3_out1: [i64; 4],
        constant165_out1: [i64; 1],
        constant116_out1: Tensor<B, 1>,
        concat6_out1: [i64; 2],
        concat7_out1: [i64; 3],
        squeeze1_out1: i64,
        input_ids: Tensor<B, 2, Int>,
        shape1_out1: [i64; 1],
        constant169_out1: [i64; 1],
    ) -> Tensor<B, 2> {
        let layernormalization22_out1 = {
            let dtype = add33_out1.clone().dtype();
            self.layernormalization22
                .forward(add33_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear43_out1 = self.linear43.forward(layernormalization22_out1);
        let gelu11_out1 = burn::tensor::activation::gelu(linear43_out1);
        let linear44_out1 = self.linear44.forward(gelu11_out1);
        let add34_out1 = add33_out1.add(linear44_out1);
        let layernormalization23_out1 = {
            let dtype = add34_out1.clone().dtype();
            self.layernormalization23
                .forward(add34_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose89_out1 = layernormalization23_out1.permute([1, 0, 2]);
        let linear45_out1 = self.linear45.forward(transpose89_out1);
        let reshape124_out1 = linear45_out1.reshape(concat1_out1);
        let unsqueeze12_out1: Tensor<B, 5> = reshape124_out1.unsqueeze_dims::<5>(&[0]);
        let transpose90_out1 = unsqueeze12_out1.permute([3, 1, 2, 0, 4]);
        let squeeze13_out1 = transpose90_out1.squeeze_dims::<4>(&[-2]);
        let gather35_out1 = {
            let sliced = squeeze13_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather36_out1 = {
            let sliced = squeeze13_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather37_out1 = {
            let sliced = squeeze13_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape125_out1 = gather35_out1.reshape(concat2_out1);
        let transpose91_out1 = reshape125_out1.permute([1, 0, 2]);
        let reshape126_out1 = gather36_out1.reshape(concat2_out1);
        let transpose92_out1 = reshape126_out1.permute([1, 0, 2]);
        let reshape127_out1 = gather37_out1.reshape(concat2_out1);
        let transpose93_out1 = reshape127_out1.permute([1, 0, 2]);
        let reshape128_out1 = transpose91_out1.reshape(concat3_out1);
        let reshape129_out1 = transpose92_out1.reshape(concat3_out1);
        let reshape130_out1 = transpose93_out1.reshape(concat3_out1);
        let shape13_out1: [i64; 4] = {
            let axes = &reshape129_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice34_out1: [i64; 1] = shape13_out1[3..4].try_into().unwrap();
        let slice35_out1: [i64; 1] = shape13_out1[2..3].try_into().unwrap();
        let slice36_out1: [i64; 2] = shape13_out1[0..2].try_into().unwrap();
        let concat28_out1: [i64; 3usize] =
            [&constant165_out1[..], &slice35_out1[..], &slice34_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape131_out1 = reshape129_out1.reshape(concat28_out1);
        let transpose94_out1 = reshape131_out1.permute([0, 2, 1]);
        let concat29_out1: [i64; 4usize] =
            [&slice36_out1[..], &slice34_out1[..], &slice35_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape132_out1 = transpose94_out1.reshape(concat29_out1);
        let mul25_out1 = reshape128_out1
            .mul((constant116_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul26_out1 =
            reshape132_out1.mul((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul57_out1 = mul25_out1.matmul(mul26_out1);
        let constant160_out1 = self.constant160.val();
        let add35_out1 = matmul57_out1.add(constant160_out1);
        let softmax12_out1 = burn::tensor::activation::softmax(add35_out1, 3);
        let matmul58_out1 = softmax12_out1.matmul(reshape130_out1);
        let transpose95_out1 = matmul58_out1.permute([2, 0, 1, 3]);
        let reshape133_out1 = transpose95_out1.reshape(concat6_out1);
        let linear46_out1 = self.linear46.forward(reshape133_out1);
        let reshape134_out1 = linear46_out1.reshape(concat7_out1);
        let transpose96_out1 = reshape134_out1.permute([1, 0, 2]);
        let add36_out1 = add34_out1.add(transpose96_out1);
        let layernormalization24_out1 = {
            let dtype = add36_out1.clone().dtype();
            self.layernormalization24
                .forward(add36_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear47_out1 = self.linear47.forward(layernormalization24_out1);
        let gelu12_out1 = burn::tensor::activation::gelu(linear47_out1);
        let linear48_out1 = self.linear48.forward(gelu12_out1);
        let add37_out1 = add36_out1.add(linear48_out1);
        let layernormalization25_out1 = {
            let dtype = add37_out1.dtype();
            self.layernormalization25
                .forward(add37_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let range1_out1 = {
            let __start = 0i64;
            let __limit = squeeze1_out1;
            let __delta = 1i64;
            assert!(__delta != 0);
            let __n = ((__limit - __start) as f64 / __delta as f64)
                .ceil()
                .max(0.0) as i64;
            Tensor::arange(0..__n, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(__delta)
                .add_scalar(__start)
        };
        let argmax_result = input_ids.argmax(1);
        let argmax1_out1 = argmax_result
            .squeeze_dim::<1usize>(1)
            .cast(burn::tensor::DType::I64);
        let unsqueeze13_out1: Tensor<B, 2, Int> = range1_out1.unsqueeze_dims::<2>(&[-1]);
        let unsqueeze14_out1: Tensor<B, 2, Int> = argmax1_out1.unsqueeze_dims::<2>(&[-1]);
        let concat30_out1 =
            burn::tensor::Tensor::cat([unsqueeze13_out1, unsqueeze14_out1].into(), 1);
        let gathernd1_out1 = {
            let __nd_data_dims = layernormalization25_out1.dims();
            let __nd_indices = concat30_out1.cast(burn::tensor::DType::I64);
            let __nd_idx_dims = __nd_indices.dims();
            let __nd_k = __nd_idx_dims[2 - 1];
            let mut __nd_dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(__nd_k);
            for __nd_i in 0..__nd_k {
                __nd_dim_sizes.push(__nd_data_dims[0 + __nd_i] as i64);
            }
            let mut __nd_bcast_shape = [1usize; 2];
            __nd_bcast_shape[2 - 1] = __nd_k;
            let __nd_dims_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::from(__nd_dim_sizes.as_slice()),
                (&self.device, burn::tensor::DType::I64),
            )
            .reshape(__nd_bcast_shape);
            let __nd_mask = __nd_indices.clone().lower_elem(0i64);
            let __nd_corrected = __nd_indices.clone() + __nd_dims_tensor;
            let __nd_indices_norm = __nd_indices.mask_where(__nd_mask, __nd_corrected);
            let __gather_nd_aug = __nd_indices_norm;
            layernormalization25_out1.gather_nd(__gather_nd_aug)
        };
        let linear49_out1 = self.linear49.forward(gathernd1_out1);
        let reducel21_out1 = {
            let input_dtype = linear49_out1.clone().dtype();
            linear49_out1
                .clone()
                .square()
                .sum_dim(1usize)
                .cast(burn::tensor::DType::F32)
                .sqrt()
                .cast(input_dtype)
        };
        let clip1_out1 = {
            let __clip_min = 0.0000000000009999999960041972f64;
            reducel21_out1.clamp_min(__clip_min)
        };
        let concat31_out1: [i64; 2usize] = [&shape1_out1[..], &constant169_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let expand1_out1 = {
            let onnx_shape: [i64; 2usize] = concat31_out1;
            let input_dims = clip1_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..2usize {
                let dim_offset = 2usize - 2usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            clip1_out1.expand(shape)
        };
        let div1_out1 = linear49_out1.div(expand1_out1);
        div1_out1
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
        Self::from_file(
            "/Volumes/CodeBase/Projects/Burn-Convert/burn_onnx_demo/target/debug/build/burn_onnx_demo-5924b412d15e0ca8/out/model/bioclip2-text/text.bpk",
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
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let (
            add4_out1,
            concat1_out1,
            concat2_out1,
            concat3_out1,
            constant165_out1,
            constant116_out1,
            concat6_out1,
            concat7_out1,
            squeeze1_out1,
            shape1_out1,
            constant169_out1,
        ) = self.submodule1.forward(input_ids.clone());
        let add9_out1 = self.submodule2.forward(
            add4_out1,
            concat1_out1.clone(),
            concat2_out1.clone(),
            concat3_out1.clone(),
            constant165_out1.clone(),
            constant116_out1.clone(),
            concat6_out1.clone(),
            concat7_out1.clone(),
        );
        let add15_out1 = self.submodule3.forward(
            add9_out1,
            concat1_out1.clone(),
            concat2_out1.clone(),
            concat3_out1.clone(),
            constant165_out1.clone(),
            constant116_out1.clone(),
            concat6_out1.clone(),
            concat7_out1.clone(),
        );
        let add21_out1 = self.submodule4.forward(
            add15_out1,
            concat1_out1.clone(),
            concat2_out1.clone(),
            concat3_out1.clone(),
            constant165_out1.clone(),
            constant116_out1.clone(),
            concat6_out1.clone(),
            concat7_out1.clone(),
        );
        let add27_out1 = self.submodule5.forward(
            add21_out1,
            concat1_out1.clone(),
            concat2_out1.clone(),
            concat3_out1.clone(),
            constant165_out1.clone(),
            constant116_out1.clone(),
            concat6_out1.clone(),
            concat7_out1.clone(),
        );
        let add33_out1 = self.submodule6.forward(
            add27_out1,
            concat1_out1.clone(),
            concat2_out1.clone(),
            concat3_out1.clone(),
            constant165_out1.clone(),
            constant116_out1.clone(),
            concat6_out1.clone(),
            concat7_out1.clone(),
        );
        let div1_out1 = self.submodule7.forward(
            add33_out1,
            concat1_out1,
            concat2_out1,
            concat3_out1,
            constant165_out1,
            constant116_out1,
            concat6_out1,
            concat7_out1,
            squeeze1_out1,
            input_ids,
            shape1_out1,
            constant169_out1,
        );
        div1_out1
    }
}
