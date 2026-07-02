// Generated from ONNX "src/bioclip2/vision.onnx" by burn-onnx
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
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
    constant224: burn::module::Param<Tensor<B, 3>>,
    constant1: burn::module::Param<Tensor<B, 2>>,
    layernormalization1: LayerNorm<B>,
    layernormalization2: LayerNorm<B>,
    linear1: Linear<B>,
    constant226: burn::module::Param<Tensor<B, 1>>,
    linear2: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 1024], [14, 14])
            .with_stride([14, 14])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant224: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 3>::zeros([1, 1, 1024], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1, 1, 1024].into(),
        );
        let constant1: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 2>::zeros([257, 1024], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [257, 1024].into(),
        );
        let layernormalization1 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization2 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear1 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let constant226: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| {
                Tensor::<B, 1>::zeros([1], (device, burn::tensor::DType::F32))
            },
            device.clone(),
            false,
            [1].into(),
        );
        let linear2 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            conv2d1,
            constant224,
            constant1,
            layernormalization1,
            layernormalization2,
            linear1,
            constant226,
            linear2,
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
        [i64; 4],
        [i64; 3],
        [i64; 4],
        [i64; 1],
        Tensor<B, 1>,
        [i64; 2],
        [i64; 3],
        [i64; 1],
    ) {
        let shape1_out1: [i64; 1] = {
            let axes = &pixel_values.clone().dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let squeeze1_out1 = shape1_out1[0] as i64;
        let conv2d1_out1 = crate::model_arch::conv_fwd(&self.conv2d1, pixel_values);
        let constant298_out1: [i64; 1] = [-1i64];
        let constant301_out1: [i64; 1] = [1024i64];
        let concat1_out1: [i64; 3usize] = [
            &shape1_out1[..],
            &constant301_out1[..],
            &constant298_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape1_out1 = conv2d1_out1.reshape(concat1_out1);
        let transpose1_out1 = reshape1_out1.permute([0, 2, 1]);
        let constant302_out1: [i64; 1] = [1i64];
        let concat2_out1: [i64; 3usize] = [
            &shape1_out1[..],
            &constant302_out1[..],
            &constant302_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let constant224_out1 = self.constant224.val();
        let expand1_out1 = {
            let onnx_shape: [i64; 3usize] = concat2_out1;
            let input_dims = constant224_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..3usize {
                let dim_offset = 3usize - 3usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            constant224_out1.expand(shape)
        };
        let concat3_out1 = burn::tensor::Tensor::cat([expand1_out1, transpose1_out1].into(), 1);
        let constant1_out1 = self.constant1.val();
        let add1_out1 = concat3_out1.add((constant1_out1).unsqueeze_dims(&[0isize]));
        let layernormalization1_out1 = {
            let dtype = add1_out1.dtype();
            self.layernormalization1
                .forward(add1_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let layernormalization2_out1 = {
            let dtype = layernormalization1_out1.clone().dtype();
            self.layernormalization2
                .forward(
                    layernormalization1_out1
                        .clone()
                        .cast(burn::tensor::DType::F32),
                )
                .cast(dtype)
        };
        let transpose2_out1 = layernormalization2_out1.permute([1, 0, 2]);
        let linear1_out1 = self.linear1.forward(transpose2_out1);
        let constant303_out1: [i64; 1] = [257i64];
        let constant304_out1: [i64; 1] = [3i64];
        let concat4_out1: [i64; 4usize] = [
            &constant303_out1[..],
            &shape1_out1[..],
            &constant304_out1[..],
            &constant301_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape2_out1 = linear1_out1.reshape(concat4_out1);
        let unsqueeze1_out1: Tensor<B, 5> = reshape2_out1.unsqueeze_dims::<5>(&[0]);
        let transpose3_out1 = unsqueeze1_out1.permute([3, 1, 2, 0, 4]);
        let squeeze2_out1 = transpose3_out1.squeeze_dims::<4>(&[-2]);
        let gather1_out1 = {
            let sliced = squeeze2_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather2_out1 = {
            let sliced = squeeze2_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather3_out1 = {
            let sliced = squeeze2_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let constant309_out1 = 16i64;
        let mul1_out1 = squeeze1_out1 * constant309_out1;
        let constant310_out1: [i64; 1] = [64i64];
        let concat5_out1: [i64; 3usize] = [
            &constant303_out1[..],
            &[mul1_out1][..],
            &constant310_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape4_out1 = gather1_out1.reshape(concat5_out1);
        let transpose4_out1 = reshape4_out1.permute([1, 0, 2]);
        let reshape5_out1 = gather2_out1.reshape(concat5_out1);
        let transpose5_out1 = reshape5_out1.permute([1, 0, 2]);
        let reshape6_out1 = gather3_out1.reshape(concat5_out1);
        let transpose6_out1 = reshape6_out1.permute([1, 0, 2]);
        let constant311_out1: [i64; 1] = [16i64];
        let concat6_out1: [i64; 4usize] = [
            &shape1_out1[..],
            &constant311_out1[..],
            &constant303_out1[..],
            &constant310_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape7_out1 = transpose4_out1.reshape(concat6_out1);
        let reshape8_out1 = transpose5_out1.reshape(concat6_out1);
        let reshape9_out1 = transpose6_out1.reshape(concat6_out1);
        let shape2_out1: [i64; 4] = {
            let axes = &reshape8_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 1] = shape2_out1[3..4].try_into().unwrap();
        let slice2_out1: [i64; 1] = shape2_out1[2..3].try_into().unwrap();
        let slice3_out1: [i64; 2] = shape2_out1[0..2].try_into().unwrap();
        let concat7_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice2_out1[..], &slice1_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape10_out1 = reshape8_out1.reshape(concat7_out1);
        let transpose7_out1 = reshape10_out1.permute([0, 2, 1]);
        let concat8_out1: [i64; 4usize] = [&slice3_out1[..], &slice1_out1[..], &slice2_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape11_out1 = transpose7_out1.reshape(concat8_out1);
        let constant226_out1 = self.constant226.val();
        let mul2_out1 =
            reshape7_out1.mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul3_out1 = reshape11_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul2_out1 = mul2_out1.matmul(mul3_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(matmul2_out1, 3);
        let matmul3_out1 = softmax1_out1.matmul(reshape9_out1);
        let transpose8_out1 = matmul3_out1.permute([2, 0, 1, 3]);
        let constant314_out1 = 257i64;
        let mul4_out1 = squeeze1_out1 * constant314_out1;
        let concat9_out1: [i64; 2usize] = [&[mul4_out1][..], &constant301_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape13_out1 = transpose8_out1.reshape(concat9_out1);
        let linear2_out1 = self.linear2.forward(reshape13_out1);
        let concat10_out1: [i64; 3usize] = [
            &constant303_out1[..],
            &shape1_out1[..],
            &constant301_out1[..],
        ]
        .concat()
        .try_into()
        .unwrap();
        let reshape14_out1 = linear2_out1.reshape(concat10_out1);
        let transpose9_out1 = reshape14_out1.permute([1, 0, 2]);
        let add2_out1 = layernormalization1_out1.add(transpose9_out1);
        (
            add2_out1,
            concat4_out1,
            concat5_out1,
            concat6_out1,
            constant298_out1,
            constant226_out1,
            concat9_out1,
            concat10_out1,
            shape1_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    layernormalization3: LayerNorm<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    layernormalization4: LayerNorm<B>,
    linear5: Linear<B>,
    linear6: Linear<B>,
    layernormalization5: LayerNorm<B>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    layernormalization6: LayerNorm<B>,
    linear9: Linear<B>,
    linear10: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization3 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear3 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear4 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization4 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear5 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear6 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization5 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear7 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear8 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization6 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear9 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear10 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization3,
            linear3,
            linear4,
            layernormalization4,
            linear5,
            linear6,
            layernormalization5,
            linear7,
            linear8,
            layernormalization6,
            linear9,
            linear10,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add2_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization3_out1 = {
            let dtype = add2_out1.clone().dtype();
            self.layernormalization3
                .forward(add2_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear3_out1 = self.linear3.forward(layernormalization3_out1);
        let gelu1_out1 = burn::tensor::activation::gelu(linear3_out1);
        let linear4_out1 = self.linear4.forward(gelu1_out1);
        let add3_out1 = add2_out1.add(linear4_out1);
        let layernormalization4_out1 = {
            let dtype = add3_out1.clone().dtype();
            self.layernormalization4
                .forward(add3_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose10_out1 = layernormalization4_out1.permute([1, 0, 2]);
        let linear5_out1 = self.linear5.forward(transpose10_out1);
        let reshape15_out1 = linear5_out1.reshape(concat4_out1);
        let unsqueeze2_out1: Tensor<B, 5> = reshape15_out1.unsqueeze_dims::<5>(&[0]);
        let transpose11_out1 = unsqueeze2_out1.permute([3, 1, 2, 0, 4]);
        let squeeze3_out1 = transpose11_out1.squeeze_dims::<4>(&[-2]);
        let gather4_out1 = {
            let sliced = squeeze3_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather5_out1 = {
            let sliced = squeeze3_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather6_out1 = {
            let sliced = squeeze3_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape16_out1 = gather4_out1.reshape(concat5_out1);
        let transpose12_out1 = reshape16_out1.permute([1, 0, 2]);
        let reshape17_out1 = gather5_out1.reshape(concat5_out1);
        let transpose13_out1 = reshape17_out1.permute([1, 0, 2]);
        let reshape18_out1 = gather6_out1.reshape(concat5_out1);
        let transpose14_out1 = reshape18_out1.permute([1, 0, 2]);
        let reshape19_out1 = transpose12_out1.reshape(concat6_out1);
        let reshape20_out1 = transpose13_out1.reshape(concat6_out1);
        let reshape21_out1 = transpose14_out1.reshape(concat6_out1);
        let shape3_out1: [i64; 4] = {
            let axes = &reshape20_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice4_out1: [i64; 1] = shape3_out1[3..4].try_into().unwrap();
        let slice5_out1: [i64; 1] = shape3_out1[2..3].try_into().unwrap();
        let slice6_out1: [i64; 2] = shape3_out1[0..2].try_into().unwrap();
        let concat11_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice5_out1[..], &slice4_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape22_out1 = reshape20_out1.reshape(concat11_out1);
        let transpose15_out1 = reshape22_out1.permute([0, 2, 1]);
        let concat12_out1: [i64; 4usize] = [&slice6_out1[..], &slice4_out1[..], &slice5_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape23_out1 = transpose15_out1.reshape(concat12_out1);
        let mul5_out1 = reshape19_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul6_out1 = reshape23_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul7_out1 = mul5_out1.matmul(mul6_out1);
        let softmax2_out1 = burn::tensor::activation::softmax(matmul7_out1, 3);
        let matmul8_out1 = softmax2_out1.matmul(reshape21_out1);
        let transpose16_out1 = matmul8_out1.permute([2, 0, 1, 3]);
        let reshape24_out1 = transpose16_out1.reshape(concat9_out1);
        let linear6_out1 = self.linear6.forward(reshape24_out1);
        let reshape25_out1 = linear6_out1.reshape(concat10_out1);
        let transpose17_out1 = reshape25_out1.permute([1, 0, 2]);
        let add4_out1 = add3_out1.add(transpose17_out1);
        let layernormalization5_out1 = {
            let dtype = add4_out1.clone().dtype();
            self.layernormalization5
                .forward(add4_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear7_out1 = self.linear7.forward(layernormalization5_out1);
        let gelu2_out1 = burn::tensor::activation::gelu(linear7_out1);
        let linear8_out1 = self.linear8.forward(gelu2_out1);
        let add5_out1 = add4_out1.add(linear8_out1);
        let layernormalization6_out1 = {
            let dtype = add5_out1.clone().dtype();
            self.layernormalization6
                .forward(add5_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose18_out1 = layernormalization6_out1.permute([1, 0, 2]);
        let linear9_out1 = self.linear9.forward(transpose18_out1);
        let reshape26_out1 = linear9_out1.reshape(concat4_out1);
        let unsqueeze3_out1: Tensor<B, 5> = reshape26_out1.unsqueeze_dims::<5>(&[0]);
        let transpose19_out1 = unsqueeze3_out1.permute([3, 1, 2, 0, 4]);
        let squeeze4_out1 = transpose19_out1.squeeze_dims::<4>(&[-2]);
        let gather7_out1 = {
            let sliced = squeeze4_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather8_out1 = {
            let sliced = squeeze4_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather9_out1 = {
            let sliced = squeeze4_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape27_out1 = gather7_out1.reshape(concat5_out1);
        let transpose20_out1 = reshape27_out1.permute([1, 0, 2]);
        let reshape28_out1 = gather8_out1.reshape(concat5_out1);
        let transpose21_out1 = reshape28_out1.permute([1, 0, 2]);
        let reshape29_out1 = gather9_out1.reshape(concat5_out1);
        let transpose22_out1 = reshape29_out1.permute([1, 0, 2]);
        let reshape30_out1 = transpose20_out1.reshape(concat6_out1);
        let reshape31_out1 = transpose21_out1.reshape(concat6_out1);
        let reshape32_out1 = transpose22_out1.reshape(concat6_out1);
        let shape4_out1: [i64; 4] = {
            let axes = &reshape31_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice7_out1: [i64; 1] = shape4_out1[3..4].try_into().unwrap();
        let slice8_out1: [i64; 1] = shape4_out1[2..3].try_into().unwrap();
        let slice9_out1: [i64; 2] = shape4_out1[0..2].try_into().unwrap();
        let concat13_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice8_out1[..], &slice7_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape33_out1 = reshape31_out1.reshape(concat13_out1);
        let transpose23_out1 = reshape33_out1.permute([0, 2, 1]);
        let concat14_out1: [i64; 4usize] = [&slice9_out1[..], &slice7_out1[..], &slice8_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape34_out1 = transpose23_out1.reshape(concat14_out1);
        let mul7_out1 = reshape30_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul8_out1 =
            reshape34_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul12_out1 = mul7_out1.matmul(mul8_out1);
        let softmax3_out1 = burn::tensor::activation::softmax(matmul12_out1, 3);
        let matmul13_out1 = softmax3_out1.matmul(reshape32_out1);
        let transpose24_out1 = matmul13_out1.permute([2, 0, 1, 3]);
        let reshape35_out1 = transpose24_out1.reshape(concat9_out1);
        let linear10_out1 = self.linear10.forward(reshape35_out1);
        let reshape36_out1 = linear10_out1.reshape(concat10_out1);
        let transpose25_out1 = reshape36_out1.permute([1, 0, 2]);
        let add6_out1 = add5_out1.add(transpose25_out1);
        add6_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    layernormalization7: LayerNorm<B>,
    linear11: Linear<B>,
    linear12: Linear<B>,
    layernormalization8: LayerNorm<B>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    layernormalization9: LayerNorm<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    layernormalization10: LayerNorm<B>,
    linear17: Linear<B>,
    linear18: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization7 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear11 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear12 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization8 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear14 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization9 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear16 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization10 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear17 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear18 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization7,
            linear11,
            linear12,
            layernormalization8,
            linear13,
            linear14,
            layernormalization9,
            linear15,
            linear16,
            layernormalization10,
            linear17,
            linear18,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add6_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization7_out1 = {
            let dtype = add6_out1.clone().dtype();
            self.layernormalization7
                .forward(add6_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear11_out1 = self.linear11.forward(layernormalization7_out1);
        let gelu3_out1 = burn::tensor::activation::gelu(linear11_out1);
        let linear12_out1 = self.linear12.forward(gelu3_out1);
        let add7_out1 = add6_out1.add(linear12_out1);
        let layernormalization8_out1 = {
            let dtype = add7_out1.clone().dtype();
            self.layernormalization8
                .forward(add7_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose26_out1 = layernormalization8_out1.permute([1, 0, 2]);
        let linear13_out1 = self.linear13.forward(transpose26_out1);
        let reshape37_out1 = linear13_out1.reshape(concat4_out1);
        let unsqueeze4_out1: Tensor<B, 5> = reshape37_out1.unsqueeze_dims::<5>(&[0]);
        let transpose27_out1 = unsqueeze4_out1.permute([3, 1, 2, 0, 4]);
        let squeeze5_out1 = transpose27_out1.squeeze_dims::<4>(&[-2]);
        let gather10_out1 = {
            let sliced = squeeze5_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather11_out1 = {
            let sliced = squeeze5_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather12_out1 = {
            let sliced = squeeze5_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape38_out1 = gather10_out1.reshape(concat5_out1);
        let transpose28_out1 = reshape38_out1.permute([1, 0, 2]);
        let reshape39_out1 = gather11_out1.reshape(concat5_out1);
        let transpose29_out1 = reshape39_out1.permute([1, 0, 2]);
        let reshape40_out1 = gather12_out1.reshape(concat5_out1);
        let transpose30_out1 = reshape40_out1.permute([1, 0, 2]);
        let reshape41_out1 = transpose28_out1.reshape(concat6_out1);
        let reshape42_out1 = transpose29_out1.reshape(concat6_out1);
        let reshape43_out1 = transpose30_out1.reshape(concat6_out1);
        let shape5_out1: [i64; 4] = {
            let axes = &reshape42_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice10_out1: [i64; 1] = shape5_out1[3..4].try_into().unwrap();
        let slice11_out1: [i64; 1] = shape5_out1[2..3].try_into().unwrap();
        let slice12_out1: [i64; 2] = shape5_out1[0..2].try_into().unwrap();
        let concat15_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice11_out1[..], &slice10_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape44_out1 = reshape42_out1.reshape(concat15_out1);
        let transpose31_out1 = reshape44_out1.permute([0, 2, 1]);
        let concat16_out1: [i64; 4usize] =
            [&slice12_out1[..], &slice10_out1[..], &slice11_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape45_out1 = transpose31_out1.reshape(concat16_out1);
        let mul9_out1 = reshape41_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul10_out1 = reshape45_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul17_out1 = mul9_out1.matmul(mul10_out1);
        let softmax4_out1 = burn::tensor::activation::softmax(matmul17_out1, 3);
        let matmul18_out1 = softmax4_out1.matmul(reshape43_out1);
        let transpose32_out1 = matmul18_out1.permute([2, 0, 1, 3]);
        let reshape46_out1 = transpose32_out1.reshape(concat9_out1);
        let linear14_out1 = self.linear14.forward(reshape46_out1);
        let reshape47_out1 = linear14_out1.reshape(concat10_out1);
        let transpose33_out1 = reshape47_out1.permute([1, 0, 2]);
        let add8_out1 = add7_out1.add(transpose33_out1);
        let layernormalization9_out1 = {
            let dtype = add8_out1.clone().dtype();
            self.layernormalization9
                .forward(add8_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear15_out1 = self.linear15.forward(layernormalization9_out1);
        let gelu4_out1 = burn::tensor::activation::gelu(linear15_out1);
        let linear16_out1 = self.linear16.forward(gelu4_out1);
        let add9_out1 = add8_out1.add(linear16_out1);
        let layernormalization10_out1 = {
            let dtype = add9_out1.clone().dtype();
            self.layernormalization10
                .forward(add9_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose34_out1 = layernormalization10_out1.permute([1, 0, 2]);
        let linear17_out1 = self.linear17.forward(transpose34_out1);
        let reshape48_out1 = linear17_out1.reshape(concat4_out1);
        let unsqueeze5_out1: Tensor<B, 5> = reshape48_out1.unsqueeze_dims::<5>(&[0]);
        let transpose35_out1 = unsqueeze5_out1.permute([3, 1, 2, 0, 4]);
        let squeeze6_out1 = transpose35_out1.squeeze_dims::<4>(&[-2]);
        let gather13_out1 = {
            let sliced = squeeze6_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather14_out1 = {
            let sliced = squeeze6_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather15_out1 = {
            let sliced = squeeze6_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape49_out1 = gather13_out1.reshape(concat5_out1);
        let transpose36_out1 = reshape49_out1.permute([1, 0, 2]);
        let reshape50_out1 = gather14_out1.reshape(concat5_out1);
        let transpose37_out1 = reshape50_out1.permute([1, 0, 2]);
        let reshape51_out1 = gather15_out1.reshape(concat5_out1);
        let transpose38_out1 = reshape51_out1.permute([1, 0, 2]);
        let reshape52_out1 = transpose36_out1.reshape(concat6_out1);
        let reshape53_out1 = transpose37_out1.reshape(concat6_out1);
        let reshape54_out1 = transpose38_out1.reshape(concat6_out1);
        let shape6_out1: [i64; 4] = {
            let axes = &reshape53_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice13_out1: [i64; 1] = shape6_out1[3..4].try_into().unwrap();
        let slice14_out1: [i64; 1] = shape6_out1[2..3].try_into().unwrap();
        let slice15_out1: [i64; 2] = shape6_out1[0..2].try_into().unwrap();
        let concat17_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice14_out1[..], &slice13_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape55_out1 = reshape53_out1.reshape(concat17_out1);
        let transpose39_out1 = reshape55_out1.permute([0, 2, 1]);
        let concat18_out1: [i64; 4usize] =
            [&slice15_out1[..], &slice13_out1[..], &slice14_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape56_out1 = transpose39_out1.reshape(concat18_out1);
        let mul11_out1 = reshape52_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul12_out1 =
            reshape56_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul22_out1 = mul11_out1.matmul(mul12_out1);
        let softmax5_out1 = burn::tensor::activation::softmax(matmul22_out1, 3);
        let matmul23_out1 = softmax5_out1.matmul(reshape54_out1);
        let transpose40_out1 = matmul23_out1.permute([2, 0, 1, 3]);
        let reshape57_out1 = transpose40_out1.reshape(concat9_out1);
        let linear18_out1 = self.linear18.forward(reshape57_out1);
        let reshape58_out1 = linear18_out1.reshape(concat10_out1);
        let transpose41_out1 = reshape58_out1.permute([1, 0, 2]);
        let add10_out1 = add9_out1.add(transpose41_out1);
        add10_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    layernormalization11: LayerNorm<B>,
    linear19: Linear<B>,
    linear20: Linear<B>,
    layernormalization12: LayerNorm<B>,
    linear21: Linear<B>,
    linear22: Linear<B>,
    layernormalization13: LayerNorm<B>,
    linear23: Linear<B>,
    linear24: Linear<B>,
    layernormalization14: LayerNorm<B>,
    linear25: Linear<B>,
    linear26: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization11 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear19 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear20 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization12 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear21 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear22 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization13 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear23 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear24 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization14 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear25 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear26 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization11,
            linear19,
            linear20,
            layernormalization12,
            linear21,
            linear22,
            layernormalization13,
            linear23,
            linear24,
            layernormalization14,
            linear25,
            linear26,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add10_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization11_out1 = {
            let dtype = add10_out1.clone().dtype();
            self.layernormalization11
                .forward(add10_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear19_out1 = self.linear19.forward(layernormalization11_out1);
        let gelu5_out1 = burn::tensor::activation::gelu(linear19_out1);
        let linear20_out1 = self.linear20.forward(gelu5_out1);
        let add11_out1 = add10_out1.add(linear20_out1);
        let layernormalization12_out1 = {
            let dtype = add11_out1.clone().dtype();
            self.layernormalization12
                .forward(add11_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose42_out1 = layernormalization12_out1.permute([1, 0, 2]);
        let linear21_out1 = self.linear21.forward(transpose42_out1);
        let reshape59_out1 = linear21_out1.reshape(concat4_out1);
        let unsqueeze6_out1: Tensor<B, 5> = reshape59_out1.unsqueeze_dims::<5>(&[0]);
        let transpose43_out1 = unsqueeze6_out1.permute([3, 1, 2, 0, 4]);
        let squeeze7_out1 = transpose43_out1.squeeze_dims::<4>(&[-2]);
        let gather16_out1 = {
            let sliced = squeeze7_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather17_out1 = {
            let sliced = squeeze7_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather18_out1 = {
            let sliced = squeeze7_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape60_out1 = gather16_out1.reshape(concat5_out1);
        let transpose44_out1 = reshape60_out1.permute([1, 0, 2]);
        let reshape61_out1 = gather17_out1.reshape(concat5_out1);
        let transpose45_out1 = reshape61_out1.permute([1, 0, 2]);
        let reshape62_out1 = gather18_out1.reshape(concat5_out1);
        let transpose46_out1 = reshape62_out1.permute([1, 0, 2]);
        let reshape63_out1 = transpose44_out1.reshape(concat6_out1);
        let reshape64_out1 = transpose45_out1.reshape(concat6_out1);
        let reshape65_out1 = transpose46_out1.reshape(concat6_out1);
        let shape7_out1: [i64; 4] = {
            let axes = &reshape64_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice16_out1: [i64; 1] = shape7_out1[3..4].try_into().unwrap();
        let slice17_out1: [i64; 1] = shape7_out1[2..3].try_into().unwrap();
        let slice18_out1: [i64; 2] = shape7_out1[0..2].try_into().unwrap();
        let concat19_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice17_out1[..], &slice16_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape66_out1 = reshape64_out1.reshape(concat19_out1);
        let transpose47_out1 = reshape66_out1.permute([0, 2, 1]);
        let concat20_out1: [i64; 4usize] =
            [&slice18_out1[..], &slice16_out1[..], &slice17_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape67_out1 = transpose47_out1.reshape(concat20_out1);
        let mul13_out1 = reshape63_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul14_out1 = reshape67_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul27_out1 = mul13_out1.matmul(mul14_out1);
        let softmax6_out1 = burn::tensor::activation::softmax(matmul27_out1, 3);
        let matmul28_out1 = softmax6_out1.matmul(reshape65_out1);
        let transpose48_out1 = matmul28_out1.permute([2, 0, 1, 3]);
        let reshape68_out1 = transpose48_out1.reshape(concat9_out1);
        let linear22_out1 = self.linear22.forward(reshape68_out1);
        let reshape69_out1 = linear22_out1.reshape(concat10_out1);
        let transpose49_out1 = reshape69_out1.permute([1, 0, 2]);
        let add12_out1 = add11_out1.add(transpose49_out1);
        let layernormalization13_out1 = {
            let dtype = add12_out1.clone().dtype();
            self.layernormalization13
                .forward(add12_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear23_out1 = self.linear23.forward(layernormalization13_out1);
        let gelu6_out1 = burn::tensor::activation::gelu(linear23_out1);
        let linear24_out1 = self.linear24.forward(gelu6_out1);
        let add13_out1 = add12_out1.add(linear24_out1);
        let layernormalization14_out1 = {
            let dtype = add13_out1.clone().dtype();
            self.layernormalization14
                .forward(add13_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose50_out1 = layernormalization14_out1.permute([1, 0, 2]);
        let linear25_out1 = self.linear25.forward(transpose50_out1);
        let reshape70_out1 = linear25_out1.reshape(concat4_out1);
        let unsqueeze7_out1: Tensor<B, 5> = reshape70_out1.unsqueeze_dims::<5>(&[0]);
        let transpose51_out1 = unsqueeze7_out1.permute([3, 1, 2, 0, 4]);
        let squeeze8_out1 = transpose51_out1.squeeze_dims::<4>(&[-2]);
        let gather19_out1 = {
            let sliced = squeeze8_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather20_out1 = {
            let sliced = squeeze8_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather21_out1 = {
            let sliced = squeeze8_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape71_out1 = gather19_out1.reshape(concat5_out1);
        let transpose52_out1 = reshape71_out1.permute([1, 0, 2]);
        let reshape72_out1 = gather20_out1.reshape(concat5_out1);
        let transpose53_out1 = reshape72_out1.permute([1, 0, 2]);
        let reshape73_out1 = gather21_out1.reshape(concat5_out1);
        let transpose54_out1 = reshape73_out1.permute([1, 0, 2]);
        let reshape74_out1 = transpose52_out1.reshape(concat6_out1);
        let reshape75_out1 = transpose53_out1.reshape(concat6_out1);
        let reshape76_out1 = transpose54_out1.reshape(concat6_out1);
        let shape8_out1: [i64; 4] = {
            let axes = &reshape75_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice19_out1: [i64; 1] = shape8_out1[3..4].try_into().unwrap();
        let slice20_out1: [i64; 1] = shape8_out1[2..3].try_into().unwrap();
        let slice21_out1: [i64; 2] = shape8_out1[0..2].try_into().unwrap();
        let concat21_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice20_out1[..], &slice19_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape77_out1 = reshape75_out1.reshape(concat21_out1);
        let transpose55_out1 = reshape77_out1.permute([0, 2, 1]);
        let concat22_out1: [i64; 4usize] =
            [&slice21_out1[..], &slice19_out1[..], &slice20_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape78_out1 = transpose55_out1.reshape(concat22_out1);
        let mul15_out1 = reshape74_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul16_out1 =
            reshape78_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul32_out1 = mul15_out1.matmul(mul16_out1);
        let softmax7_out1 = burn::tensor::activation::softmax(matmul32_out1, 3);
        let matmul33_out1 = softmax7_out1.matmul(reshape76_out1);
        let transpose56_out1 = matmul33_out1.permute([2, 0, 1, 3]);
        let reshape79_out1 = transpose56_out1.reshape(concat9_out1);
        let linear26_out1 = self.linear26.forward(reshape79_out1);
        let reshape80_out1 = linear26_out1.reshape(concat10_out1);
        let transpose57_out1 = reshape80_out1.permute([1, 0, 2]);
        let add14_out1 = add13_out1.add(transpose57_out1);
        add14_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5<B: Backend> {
    layernormalization15: LayerNorm<B>,
    linear27: Linear<B>,
    linear28: Linear<B>,
    layernormalization16: LayerNorm<B>,
    linear29: Linear<B>,
    linear30: Linear<B>,
    layernormalization17: LayerNorm<B>,
    linear31: Linear<B>,
    linear32: Linear<B>,
    layernormalization18: LayerNorm<B>,
    linear33: Linear<B>,
    linear34: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule5<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization15 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear27 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear28 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization16 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear29 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear30 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization17 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear31 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear32 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization18 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear33 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear34 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization15,
            linear27,
            linear28,
            layernormalization16,
            linear29,
            linear30,
            layernormalization17,
            linear31,
            linear32,
            layernormalization18,
            linear33,
            linear34,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add14_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization15_out1 = {
            let dtype = add14_out1.clone().dtype();
            self.layernormalization15
                .forward(add14_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear27_out1 = self.linear27.forward(layernormalization15_out1);
        let gelu7_out1 = burn::tensor::activation::gelu(linear27_out1);
        let linear28_out1 = self.linear28.forward(gelu7_out1);
        let add15_out1 = add14_out1.add(linear28_out1);
        let layernormalization16_out1 = {
            let dtype = add15_out1.clone().dtype();
            self.layernormalization16
                .forward(add15_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose58_out1 = layernormalization16_out1.permute([1, 0, 2]);
        let linear29_out1 = self.linear29.forward(transpose58_out1);
        let reshape81_out1 = linear29_out1.reshape(concat4_out1);
        let unsqueeze8_out1: Tensor<B, 5> = reshape81_out1.unsqueeze_dims::<5>(&[0]);
        let transpose59_out1 = unsqueeze8_out1.permute([3, 1, 2, 0, 4]);
        let squeeze9_out1 = transpose59_out1.squeeze_dims::<4>(&[-2]);
        let gather22_out1 = {
            let sliced = squeeze9_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather23_out1 = {
            let sliced = squeeze9_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather24_out1 = {
            let sliced = squeeze9_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape82_out1 = gather22_out1.reshape(concat5_out1);
        let transpose60_out1 = reshape82_out1.permute([1, 0, 2]);
        let reshape83_out1 = gather23_out1.reshape(concat5_out1);
        let transpose61_out1 = reshape83_out1.permute([1, 0, 2]);
        let reshape84_out1 = gather24_out1.reshape(concat5_out1);
        let transpose62_out1 = reshape84_out1.permute([1, 0, 2]);
        let reshape85_out1 = transpose60_out1.reshape(concat6_out1);
        let reshape86_out1 = transpose61_out1.reshape(concat6_out1);
        let reshape87_out1 = transpose62_out1.reshape(concat6_out1);
        let shape9_out1: [i64; 4] = {
            let axes = &reshape86_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice22_out1: [i64; 1] = shape9_out1[3..4].try_into().unwrap();
        let slice23_out1: [i64; 1] = shape9_out1[2..3].try_into().unwrap();
        let slice24_out1: [i64; 2] = shape9_out1[0..2].try_into().unwrap();
        let concat23_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice23_out1[..], &slice22_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape88_out1 = reshape86_out1.reshape(concat23_out1);
        let transpose63_out1 = reshape88_out1.permute([0, 2, 1]);
        let concat24_out1: [i64; 4usize] =
            [&slice24_out1[..], &slice22_out1[..], &slice23_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape89_out1 = transpose63_out1.reshape(concat24_out1);
        let mul17_out1 = reshape85_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul18_out1 = reshape89_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul37_out1 = mul17_out1.matmul(mul18_out1);
        let softmax8_out1 = burn::tensor::activation::softmax(matmul37_out1, 3);
        let matmul38_out1 = softmax8_out1.matmul(reshape87_out1);
        let transpose64_out1 = matmul38_out1.permute([2, 0, 1, 3]);
        let reshape90_out1 = transpose64_out1.reshape(concat9_out1);
        let linear30_out1 = self.linear30.forward(reshape90_out1);
        let reshape91_out1 = linear30_out1.reshape(concat10_out1);
        let transpose65_out1 = reshape91_out1.permute([1, 0, 2]);
        let add16_out1 = add15_out1.add(transpose65_out1);
        let layernormalization17_out1 = {
            let dtype = add16_out1.clone().dtype();
            self.layernormalization17
                .forward(add16_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear31_out1 = self.linear31.forward(layernormalization17_out1);
        let gelu8_out1 = burn::tensor::activation::gelu(linear31_out1);
        let linear32_out1 = self.linear32.forward(gelu8_out1);
        let add17_out1 = add16_out1.add(linear32_out1);
        let layernormalization18_out1 = {
            let dtype = add17_out1.clone().dtype();
            self.layernormalization18
                .forward(add17_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose66_out1 = layernormalization18_out1.permute([1, 0, 2]);
        let linear33_out1 = self.linear33.forward(transpose66_out1);
        let reshape92_out1 = linear33_out1.reshape(concat4_out1);
        let unsqueeze9_out1: Tensor<B, 5> = reshape92_out1.unsqueeze_dims::<5>(&[0]);
        let transpose67_out1 = unsqueeze9_out1.permute([3, 1, 2, 0, 4]);
        let squeeze10_out1 = transpose67_out1.squeeze_dims::<4>(&[-2]);
        let gather25_out1 = {
            let sliced = squeeze10_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather26_out1 = {
            let sliced = squeeze10_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather27_out1 = {
            let sliced = squeeze10_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape93_out1 = gather25_out1.reshape(concat5_out1);
        let transpose68_out1 = reshape93_out1.permute([1, 0, 2]);
        let reshape94_out1 = gather26_out1.reshape(concat5_out1);
        let transpose69_out1 = reshape94_out1.permute([1, 0, 2]);
        let reshape95_out1 = gather27_out1.reshape(concat5_out1);
        let transpose70_out1 = reshape95_out1.permute([1, 0, 2]);
        let reshape96_out1 = transpose68_out1.reshape(concat6_out1);
        let reshape97_out1 = transpose69_out1.reshape(concat6_out1);
        let reshape98_out1 = transpose70_out1.reshape(concat6_out1);
        let shape10_out1: [i64; 4] = {
            let axes = &reshape97_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice25_out1: [i64; 1] = shape10_out1[3..4].try_into().unwrap();
        let slice26_out1: [i64; 1] = shape10_out1[2..3].try_into().unwrap();
        let slice27_out1: [i64; 2] = shape10_out1[0..2].try_into().unwrap();
        let concat25_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice26_out1[..], &slice25_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape99_out1 = reshape97_out1.reshape(concat25_out1);
        let transpose71_out1 = reshape99_out1.permute([0, 2, 1]);
        let concat26_out1: [i64; 4usize] =
            [&slice27_out1[..], &slice25_out1[..], &slice26_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape100_out1 = transpose71_out1.reshape(concat26_out1);
        let mul19_out1 = reshape96_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul20_out1 =
            reshape100_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul42_out1 = mul19_out1.matmul(mul20_out1);
        let softmax9_out1 = burn::tensor::activation::softmax(matmul42_out1, 3);
        let matmul43_out1 = softmax9_out1.matmul(reshape98_out1);
        let transpose72_out1 = matmul43_out1.permute([2, 0, 1, 3]);
        let reshape101_out1 = transpose72_out1.reshape(concat9_out1);
        let linear34_out1 = self.linear34.forward(reshape101_out1);
        let reshape102_out1 = linear34_out1.reshape(concat10_out1);
        let transpose73_out1 = reshape102_out1.permute([1, 0, 2]);
        let add18_out1 = add17_out1.add(transpose73_out1);
        add18_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6<B: Backend> {
    layernormalization19: LayerNorm<B>,
    linear35: Linear<B>,
    linear36: Linear<B>,
    layernormalization20: LayerNorm<B>,
    linear37: Linear<B>,
    linear38: Linear<B>,
    layernormalization21: LayerNorm<B>,
    linear39: Linear<B>,
    linear40: Linear<B>,
    layernormalization22: LayerNorm<B>,
    linear41: Linear<B>,
    linear42: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule6<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization19 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear35 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear36 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization20 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear37 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear38 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization21 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear39 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear40 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization22 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear41 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear42 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization19,
            linear35,
            linear36,
            layernormalization20,
            linear37,
            linear38,
            layernormalization21,
            linear39,
            linear40,
            layernormalization22,
            linear41,
            linear42,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add18_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization19_out1 = {
            let dtype = add18_out1.clone().dtype();
            self.layernormalization19
                .forward(add18_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear35_out1 = self.linear35.forward(layernormalization19_out1);
        let gelu9_out1 = burn::tensor::activation::gelu(linear35_out1);
        let linear36_out1 = self.linear36.forward(gelu9_out1);
        let add19_out1 = add18_out1.add(linear36_out1);
        let layernormalization20_out1 = {
            let dtype = add19_out1.clone().dtype();
            self.layernormalization20
                .forward(add19_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose74_out1 = layernormalization20_out1.permute([1, 0, 2]);
        let linear37_out1 = self.linear37.forward(transpose74_out1);
        let reshape103_out1 = linear37_out1.reshape(concat4_out1);
        let unsqueeze10_out1: Tensor<B, 5> = reshape103_out1.unsqueeze_dims::<5>(&[0]);
        let transpose75_out1 = unsqueeze10_out1.permute([3, 1, 2, 0, 4]);
        let squeeze11_out1 = transpose75_out1.squeeze_dims::<4>(&[-2]);
        let gather28_out1 = {
            let sliced = squeeze11_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather29_out1 = {
            let sliced = squeeze11_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather30_out1 = {
            let sliced = squeeze11_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape104_out1 = gather28_out1.reshape(concat5_out1);
        let transpose76_out1 = reshape104_out1.permute([1, 0, 2]);
        let reshape105_out1 = gather29_out1.reshape(concat5_out1);
        let transpose77_out1 = reshape105_out1.permute([1, 0, 2]);
        let reshape106_out1 = gather30_out1.reshape(concat5_out1);
        let transpose78_out1 = reshape106_out1.permute([1, 0, 2]);
        let reshape107_out1 = transpose76_out1.reshape(concat6_out1);
        let reshape108_out1 = transpose77_out1.reshape(concat6_out1);
        let reshape109_out1 = transpose78_out1.reshape(concat6_out1);
        let shape11_out1: [i64; 4] = {
            let axes = &reshape108_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice28_out1: [i64; 1] = shape11_out1[3..4].try_into().unwrap();
        let slice29_out1: [i64; 1] = shape11_out1[2..3].try_into().unwrap();
        let slice30_out1: [i64; 2] = shape11_out1[0..2].try_into().unwrap();
        let concat27_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice29_out1[..], &slice28_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape110_out1 = reshape108_out1.reshape(concat27_out1);
        let transpose79_out1 = reshape110_out1.permute([0, 2, 1]);
        let concat28_out1: [i64; 4usize] =
            [&slice30_out1[..], &slice28_out1[..], &slice29_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape111_out1 = transpose79_out1.reshape(concat28_out1);
        let mul21_out1 = reshape107_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul22_out1 = reshape111_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul47_out1 = mul21_out1.matmul(mul22_out1);
        let softmax10_out1 = burn::tensor::activation::softmax(matmul47_out1, 3);
        let matmul48_out1 = softmax10_out1.matmul(reshape109_out1);
        let transpose80_out1 = matmul48_out1.permute([2, 0, 1, 3]);
        let reshape112_out1 = transpose80_out1.reshape(concat9_out1);
        let linear38_out1 = self.linear38.forward(reshape112_out1);
        let reshape113_out1 = linear38_out1.reshape(concat10_out1);
        let transpose81_out1 = reshape113_out1.permute([1, 0, 2]);
        let add20_out1 = add19_out1.add(transpose81_out1);
        let layernormalization21_out1 = {
            let dtype = add20_out1.clone().dtype();
            self.layernormalization21
                .forward(add20_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear39_out1 = self.linear39.forward(layernormalization21_out1);
        let gelu10_out1 = burn::tensor::activation::gelu(linear39_out1);
        let linear40_out1 = self.linear40.forward(gelu10_out1);
        let add21_out1 = add20_out1.add(linear40_out1);
        let layernormalization22_out1 = {
            let dtype = add21_out1.clone().dtype();
            self.layernormalization22
                .forward(add21_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose82_out1 = layernormalization22_out1.permute([1, 0, 2]);
        let linear41_out1 = self.linear41.forward(transpose82_out1);
        let reshape114_out1 = linear41_out1.reshape(concat4_out1);
        let unsqueeze11_out1: Tensor<B, 5> = reshape114_out1.unsqueeze_dims::<5>(&[0]);
        let transpose83_out1 = unsqueeze11_out1.permute([3, 1, 2, 0, 4]);
        let squeeze12_out1 = transpose83_out1.squeeze_dims::<4>(&[-2]);
        let gather31_out1 = {
            let sliced = squeeze12_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather32_out1 = {
            let sliced = squeeze12_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather33_out1 = {
            let sliced = squeeze12_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape115_out1 = gather31_out1.reshape(concat5_out1);
        let transpose84_out1 = reshape115_out1.permute([1, 0, 2]);
        let reshape116_out1 = gather32_out1.reshape(concat5_out1);
        let transpose85_out1 = reshape116_out1.permute([1, 0, 2]);
        let reshape117_out1 = gather33_out1.reshape(concat5_out1);
        let transpose86_out1 = reshape117_out1.permute([1, 0, 2]);
        let reshape118_out1 = transpose84_out1.reshape(concat6_out1);
        let reshape119_out1 = transpose85_out1.reshape(concat6_out1);
        let reshape120_out1 = transpose86_out1.reshape(concat6_out1);
        let shape12_out1: [i64; 4] = {
            let axes = &reshape119_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice31_out1: [i64; 1] = shape12_out1[3..4].try_into().unwrap();
        let slice32_out1: [i64; 1] = shape12_out1[2..3].try_into().unwrap();
        let slice33_out1: [i64; 2] = shape12_out1[0..2].try_into().unwrap();
        let concat29_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice32_out1[..], &slice31_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape121_out1 = reshape119_out1.reshape(concat29_out1);
        let transpose87_out1 = reshape121_out1.permute([0, 2, 1]);
        let concat30_out1: [i64; 4usize] =
            [&slice33_out1[..], &slice31_out1[..], &slice32_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape122_out1 = transpose87_out1.reshape(concat30_out1);
        let mul23_out1 = reshape118_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul24_out1 =
            reshape122_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul52_out1 = mul23_out1.matmul(mul24_out1);
        let softmax11_out1 = burn::tensor::activation::softmax(matmul52_out1, 3);
        let matmul53_out1 = softmax11_out1.matmul(reshape120_out1);
        let transpose88_out1 = matmul53_out1.permute([2, 0, 1, 3]);
        let reshape123_out1 = transpose88_out1.reshape(concat9_out1);
        let linear42_out1 = self.linear42.forward(reshape123_out1);
        let reshape124_out1 = linear42_out1.reshape(concat10_out1);
        let transpose89_out1 = reshape124_out1.permute([1, 0, 2]);
        let add22_out1 = add21_out1.add(transpose89_out1);
        add22_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule7<B: Backend> {
    layernormalization23: LayerNorm<B>,
    linear43: Linear<B>,
    linear44: Linear<B>,
    layernormalization24: LayerNorm<B>,
    linear45: Linear<B>,
    linear46: Linear<B>,
    layernormalization25: LayerNorm<B>,
    linear47: Linear<B>,
    linear48: Linear<B>,
    layernormalization26: LayerNorm<B>,
    linear49: Linear<B>,
    linear50: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule7<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization23 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear43 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear44 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization24 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear45 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear46 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization25 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear47 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear48 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization26 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear49 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear50 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization23,
            linear43,
            linear44,
            layernormalization24,
            linear45,
            linear46,
            layernormalization25,
            linear47,
            linear48,
            layernormalization26,
            linear49,
            linear50,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add22_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization23_out1 = {
            let dtype = add22_out1.clone().dtype();
            self.layernormalization23
                .forward(add22_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear43_out1 = self.linear43.forward(layernormalization23_out1);
        let gelu11_out1 = burn::tensor::activation::gelu(linear43_out1);
        let linear44_out1 = self.linear44.forward(gelu11_out1);
        let add23_out1 = add22_out1.add(linear44_out1);
        let layernormalization24_out1 = {
            let dtype = add23_out1.clone().dtype();
            self.layernormalization24
                .forward(add23_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose90_out1 = layernormalization24_out1.permute([1, 0, 2]);
        let linear45_out1 = self.linear45.forward(transpose90_out1);
        let reshape125_out1 = linear45_out1.reshape(concat4_out1);
        let unsqueeze12_out1: Tensor<B, 5> = reshape125_out1.unsqueeze_dims::<5>(&[0]);
        let transpose91_out1 = unsqueeze12_out1.permute([3, 1, 2, 0, 4]);
        let squeeze13_out1 = transpose91_out1.squeeze_dims::<4>(&[-2]);
        let gather34_out1 = {
            let sliced = squeeze13_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather35_out1 = {
            let sliced = squeeze13_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather36_out1 = {
            let sliced = squeeze13_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape126_out1 = gather34_out1.reshape(concat5_out1);
        let transpose92_out1 = reshape126_out1.permute([1, 0, 2]);
        let reshape127_out1 = gather35_out1.reshape(concat5_out1);
        let transpose93_out1 = reshape127_out1.permute([1, 0, 2]);
        let reshape128_out1 = gather36_out1.reshape(concat5_out1);
        let transpose94_out1 = reshape128_out1.permute([1, 0, 2]);
        let reshape129_out1 = transpose92_out1.reshape(concat6_out1);
        let reshape130_out1 = transpose93_out1.reshape(concat6_out1);
        let reshape131_out1 = transpose94_out1.reshape(concat6_out1);
        let shape13_out1: [i64; 4] = {
            let axes = &reshape130_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice34_out1: [i64; 1] = shape13_out1[3..4].try_into().unwrap();
        let slice35_out1: [i64; 1] = shape13_out1[2..3].try_into().unwrap();
        let slice36_out1: [i64; 2] = shape13_out1[0..2].try_into().unwrap();
        let concat31_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice35_out1[..], &slice34_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape132_out1 = reshape130_out1.reshape(concat31_out1);
        let transpose95_out1 = reshape132_out1.permute([0, 2, 1]);
        let concat32_out1: [i64; 4usize] =
            [&slice36_out1[..], &slice34_out1[..], &slice35_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape133_out1 = transpose95_out1.reshape(concat32_out1);
        let mul25_out1 = reshape129_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul26_out1 = reshape133_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul57_out1 = mul25_out1.matmul(mul26_out1);
        let softmax12_out1 = burn::tensor::activation::softmax(matmul57_out1, 3);
        let matmul58_out1 = softmax12_out1.matmul(reshape131_out1);
        let transpose96_out1 = matmul58_out1.permute([2, 0, 1, 3]);
        let reshape134_out1 = transpose96_out1.reshape(concat9_out1);
        let linear46_out1 = self.linear46.forward(reshape134_out1);
        let reshape135_out1 = linear46_out1.reshape(concat10_out1);
        let transpose97_out1 = reshape135_out1.permute([1, 0, 2]);
        let add24_out1 = add23_out1.add(transpose97_out1);
        let layernormalization25_out1 = {
            let dtype = add24_out1.clone().dtype();
            self.layernormalization25
                .forward(add24_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear47_out1 = self.linear47.forward(layernormalization25_out1);
        let gelu12_out1 = burn::tensor::activation::gelu(linear47_out1);
        let linear48_out1 = self.linear48.forward(gelu12_out1);
        let add25_out1 = add24_out1.add(linear48_out1);
        let layernormalization26_out1 = {
            let dtype = add25_out1.clone().dtype();
            self.layernormalization26
                .forward(add25_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose98_out1 = layernormalization26_out1.permute([1, 0, 2]);
        let linear49_out1 = self.linear49.forward(transpose98_out1);
        let reshape136_out1 = linear49_out1.reshape(concat4_out1);
        let unsqueeze13_out1: Tensor<B, 5> = reshape136_out1.unsqueeze_dims::<5>(&[0]);
        let transpose99_out1 = unsqueeze13_out1.permute([3, 1, 2, 0, 4]);
        let squeeze14_out1 = transpose99_out1.squeeze_dims::<4>(&[-2]);
        let gather37_out1 = {
            let sliced = squeeze14_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather38_out1 = {
            let sliced = squeeze14_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather39_out1 = {
            let sliced = squeeze14_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape137_out1 = gather37_out1.reshape(concat5_out1);
        let transpose100_out1 = reshape137_out1.permute([1, 0, 2]);
        let reshape138_out1 = gather38_out1.reshape(concat5_out1);
        let transpose101_out1 = reshape138_out1.permute([1, 0, 2]);
        let reshape139_out1 = gather39_out1.reshape(concat5_out1);
        let transpose102_out1 = reshape139_out1.permute([1, 0, 2]);
        let reshape140_out1 = transpose100_out1.reshape(concat6_out1);
        let reshape141_out1 = transpose101_out1.reshape(concat6_out1);
        let reshape142_out1 = transpose102_out1.reshape(concat6_out1);
        let shape14_out1: [i64; 4] = {
            let axes = &reshape141_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice37_out1: [i64; 1] = shape14_out1[3..4].try_into().unwrap();
        let slice38_out1: [i64; 1] = shape14_out1[2..3].try_into().unwrap();
        let slice39_out1: [i64; 2] = shape14_out1[0..2].try_into().unwrap();
        let concat33_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice38_out1[..], &slice37_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape143_out1 = reshape141_out1.reshape(concat33_out1);
        let transpose103_out1 = reshape143_out1.permute([0, 2, 1]);
        let concat34_out1: [i64; 4usize] =
            [&slice39_out1[..], &slice37_out1[..], &slice38_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape144_out1 = transpose103_out1.reshape(concat34_out1);
        let mul27_out1 = reshape140_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul28_out1 =
            reshape144_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul62_out1 = mul27_out1.matmul(mul28_out1);
        let softmax13_out1 = burn::tensor::activation::softmax(matmul62_out1, 3);
        let matmul63_out1 = softmax13_out1.matmul(reshape142_out1);
        let transpose104_out1 = matmul63_out1.permute([2, 0, 1, 3]);
        let reshape145_out1 = transpose104_out1.reshape(concat9_out1);
        let linear50_out1 = self.linear50.forward(reshape145_out1);
        let reshape146_out1 = linear50_out1.reshape(concat10_out1);
        let transpose105_out1 = reshape146_out1.permute([1, 0, 2]);
        let add26_out1 = add25_out1.add(transpose105_out1);
        add26_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule8<B: Backend> {
    layernormalization27: LayerNorm<B>,
    linear51: Linear<B>,
    linear52: Linear<B>,
    layernormalization28: LayerNorm<B>,
    linear53: Linear<B>,
    linear54: Linear<B>,
    layernormalization29: LayerNorm<B>,
    linear55: Linear<B>,
    linear56: Linear<B>,
    layernormalization30: LayerNorm<B>,
    linear57: Linear<B>,
    linear58: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule8<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization27 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear51 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear52 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization28 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear53 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear54 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization29 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear55 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear56 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization30 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear57 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear58 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization27,
            linear51,
            linear52,
            layernormalization28,
            linear53,
            linear54,
            layernormalization29,
            linear55,
            linear56,
            layernormalization30,
            linear57,
            linear58,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add26_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization27_out1 = {
            let dtype = add26_out1.clone().dtype();
            self.layernormalization27
                .forward(add26_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear51_out1 = self.linear51.forward(layernormalization27_out1);
        let gelu13_out1 = burn::tensor::activation::gelu(linear51_out1);
        let linear52_out1 = self.linear52.forward(gelu13_out1);
        let add27_out1 = add26_out1.add(linear52_out1);
        let layernormalization28_out1 = {
            let dtype = add27_out1.clone().dtype();
            self.layernormalization28
                .forward(add27_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose106_out1 = layernormalization28_out1.permute([1, 0, 2]);
        let linear53_out1 = self.linear53.forward(transpose106_out1);
        let reshape147_out1 = linear53_out1.reshape(concat4_out1);
        let unsqueeze14_out1: Tensor<B, 5> = reshape147_out1.unsqueeze_dims::<5>(&[0]);
        let transpose107_out1 = unsqueeze14_out1.permute([3, 1, 2, 0, 4]);
        let squeeze15_out1 = transpose107_out1.squeeze_dims::<4>(&[-2]);
        let gather40_out1 = {
            let sliced = squeeze15_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather41_out1 = {
            let sliced = squeeze15_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather42_out1 = {
            let sliced = squeeze15_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape148_out1 = gather40_out1.reshape(concat5_out1);
        let transpose108_out1 = reshape148_out1.permute([1, 0, 2]);
        let reshape149_out1 = gather41_out1.reshape(concat5_out1);
        let transpose109_out1 = reshape149_out1.permute([1, 0, 2]);
        let reshape150_out1 = gather42_out1.reshape(concat5_out1);
        let transpose110_out1 = reshape150_out1.permute([1, 0, 2]);
        let reshape151_out1 = transpose108_out1.reshape(concat6_out1);
        let reshape152_out1 = transpose109_out1.reshape(concat6_out1);
        let reshape153_out1 = transpose110_out1.reshape(concat6_out1);
        let shape15_out1: [i64; 4] = {
            let axes = &reshape152_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice40_out1: [i64; 1] = shape15_out1[3..4].try_into().unwrap();
        let slice41_out1: [i64; 1] = shape15_out1[2..3].try_into().unwrap();
        let slice42_out1: [i64; 2] = shape15_out1[0..2].try_into().unwrap();
        let concat35_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice41_out1[..], &slice40_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape154_out1 = reshape152_out1.reshape(concat35_out1);
        let transpose111_out1 = reshape154_out1.permute([0, 2, 1]);
        let concat36_out1: [i64; 4usize] =
            [&slice42_out1[..], &slice40_out1[..], &slice41_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape155_out1 = transpose111_out1.reshape(concat36_out1);
        let mul29_out1 = reshape151_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul30_out1 = reshape155_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul67_out1 = mul29_out1.matmul(mul30_out1);
        let softmax14_out1 = burn::tensor::activation::softmax(matmul67_out1, 3);
        let matmul68_out1 = softmax14_out1.matmul(reshape153_out1);
        let transpose112_out1 = matmul68_out1.permute([2, 0, 1, 3]);
        let reshape156_out1 = transpose112_out1.reshape(concat9_out1);
        let linear54_out1 = self.linear54.forward(reshape156_out1);
        let reshape157_out1 = linear54_out1.reshape(concat10_out1);
        let transpose113_out1 = reshape157_out1.permute([1, 0, 2]);
        let add28_out1 = add27_out1.add(transpose113_out1);
        let layernormalization29_out1 = {
            let dtype = add28_out1.clone().dtype();
            self.layernormalization29
                .forward(add28_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear55_out1 = self.linear55.forward(layernormalization29_out1);
        let gelu14_out1 = burn::tensor::activation::gelu(linear55_out1);
        let linear56_out1 = self.linear56.forward(gelu14_out1);
        let add29_out1 = add28_out1.add(linear56_out1);
        let layernormalization30_out1 = {
            let dtype = add29_out1.clone().dtype();
            self.layernormalization30
                .forward(add29_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose114_out1 = layernormalization30_out1.permute([1, 0, 2]);
        let linear57_out1 = self.linear57.forward(transpose114_out1);
        let reshape158_out1 = linear57_out1.reshape(concat4_out1);
        let unsqueeze15_out1: Tensor<B, 5> = reshape158_out1.unsqueeze_dims::<5>(&[0]);
        let transpose115_out1 = unsqueeze15_out1.permute([3, 1, 2, 0, 4]);
        let squeeze16_out1 = transpose115_out1.squeeze_dims::<4>(&[-2]);
        let gather43_out1 = {
            let sliced = squeeze16_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather44_out1 = {
            let sliced = squeeze16_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather45_out1 = {
            let sliced = squeeze16_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape159_out1 = gather43_out1.reshape(concat5_out1);
        let transpose116_out1 = reshape159_out1.permute([1, 0, 2]);
        let reshape160_out1 = gather44_out1.reshape(concat5_out1);
        let transpose117_out1 = reshape160_out1.permute([1, 0, 2]);
        let reshape161_out1 = gather45_out1.reshape(concat5_out1);
        let transpose118_out1 = reshape161_out1.permute([1, 0, 2]);
        let reshape162_out1 = transpose116_out1.reshape(concat6_out1);
        let reshape163_out1 = transpose117_out1.reshape(concat6_out1);
        let reshape164_out1 = transpose118_out1.reshape(concat6_out1);
        let shape16_out1: [i64; 4] = {
            let axes = &reshape163_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice43_out1: [i64; 1] = shape16_out1[3..4].try_into().unwrap();
        let slice44_out1: [i64; 1] = shape16_out1[2..3].try_into().unwrap();
        let slice45_out1: [i64; 2] = shape16_out1[0..2].try_into().unwrap();
        let concat37_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice44_out1[..], &slice43_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape165_out1 = reshape163_out1.reshape(concat37_out1);
        let transpose119_out1 = reshape165_out1.permute([0, 2, 1]);
        let concat38_out1: [i64; 4usize] =
            [&slice45_out1[..], &slice43_out1[..], &slice44_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape166_out1 = transpose119_out1.reshape(concat38_out1);
        let mul31_out1 = reshape162_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul32_out1 =
            reshape166_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul72_out1 = mul31_out1.matmul(mul32_out1);
        let softmax15_out1 = burn::tensor::activation::softmax(matmul72_out1, 3);
        let matmul73_out1 = softmax15_out1.matmul(reshape164_out1);
        let transpose120_out1 = matmul73_out1.permute([2, 0, 1, 3]);
        let reshape167_out1 = transpose120_out1.reshape(concat9_out1);
        let linear58_out1 = self.linear58.forward(reshape167_out1);
        let reshape168_out1 = linear58_out1.reshape(concat10_out1);
        let transpose121_out1 = reshape168_out1.permute([1, 0, 2]);
        let add30_out1 = add29_out1.add(transpose121_out1);
        add30_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule9<B: Backend> {
    layernormalization31: LayerNorm<B>,
    linear59: Linear<B>,
    linear60: Linear<B>,
    layernormalization32: LayerNorm<B>,
    linear61: Linear<B>,
    linear62: Linear<B>,
    layernormalization33: LayerNorm<B>,
    linear63: Linear<B>,
    linear64: Linear<B>,
    layernormalization34: LayerNorm<B>,
    linear65: Linear<B>,
    linear66: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule9<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization31 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear59 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear60 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization32 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear61 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear62 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization33 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear63 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear64 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization34 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear65 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear66 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization31,
            linear59,
            linear60,
            layernormalization32,
            linear61,
            linear62,
            layernormalization33,
            linear63,
            linear64,
            layernormalization34,
            linear65,
            linear66,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add30_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization31_out1 = {
            let dtype = add30_out1.clone().dtype();
            self.layernormalization31
                .forward(add30_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear59_out1 = self.linear59.forward(layernormalization31_out1);
        let gelu15_out1 = burn::tensor::activation::gelu(linear59_out1);
        let linear60_out1 = self.linear60.forward(gelu15_out1);
        let add31_out1 = add30_out1.add(linear60_out1);
        let layernormalization32_out1 = {
            let dtype = add31_out1.clone().dtype();
            self.layernormalization32
                .forward(add31_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose122_out1 = layernormalization32_out1.permute([1, 0, 2]);
        let linear61_out1 = self.linear61.forward(transpose122_out1);
        let reshape169_out1 = linear61_out1.reshape(concat4_out1);
        let unsqueeze16_out1: Tensor<B, 5> = reshape169_out1.unsqueeze_dims::<5>(&[0]);
        let transpose123_out1 = unsqueeze16_out1.permute([3, 1, 2, 0, 4]);
        let squeeze17_out1 = transpose123_out1.squeeze_dims::<4>(&[-2]);
        let gather46_out1 = {
            let sliced = squeeze17_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather47_out1 = {
            let sliced = squeeze17_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather48_out1 = {
            let sliced = squeeze17_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape170_out1 = gather46_out1.reshape(concat5_out1);
        let transpose124_out1 = reshape170_out1.permute([1, 0, 2]);
        let reshape171_out1 = gather47_out1.reshape(concat5_out1);
        let transpose125_out1 = reshape171_out1.permute([1, 0, 2]);
        let reshape172_out1 = gather48_out1.reshape(concat5_out1);
        let transpose126_out1 = reshape172_out1.permute([1, 0, 2]);
        let reshape173_out1 = transpose124_out1.reshape(concat6_out1);
        let reshape174_out1 = transpose125_out1.reshape(concat6_out1);
        let reshape175_out1 = transpose126_out1.reshape(concat6_out1);
        let shape17_out1: [i64; 4] = {
            let axes = &reshape174_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice46_out1: [i64; 1] = shape17_out1[3..4].try_into().unwrap();
        let slice47_out1: [i64; 1] = shape17_out1[2..3].try_into().unwrap();
        let slice48_out1: [i64; 2] = shape17_out1[0..2].try_into().unwrap();
        let concat39_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice47_out1[..], &slice46_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape176_out1 = reshape174_out1.reshape(concat39_out1);
        let transpose127_out1 = reshape176_out1.permute([0, 2, 1]);
        let concat40_out1: [i64; 4usize] =
            [&slice48_out1[..], &slice46_out1[..], &slice47_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape177_out1 = transpose127_out1.reshape(concat40_out1);
        let mul33_out1 = reshape173_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul34_out1 = reshape177_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul77_out1 = mul33_out1.matmul(mul34_out1);
        let softmax16_out1 = burn::tensor::activation::softmax(matmul77_out1, 3);
        let matmul78_out1 = softmax16_out1.matmul(reshape175_out1);
        let transpose128_out1 = matmul78_out1.permute([2, 0, 1, 3]);
        let reshape178_out1 = transpose128_out1.reshape(concat9_out1);
        let linear62_out1 = self.linear62.forward(reshape178_out1);
        let reshape179_out1 = linear62_out1.reshape(concat10_out1);
        let transpose129_out1 = reshape179_out1.permute([1, 0, 2]);
        let add32_out1 = add31_out1.add(transpose129_out1);
        let layernormalization33_out1 = {
            let dtype = add32_out1.clone().dtype();
            self.layernormalization33
                .forward(add32_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear63_out1 = self.linear63.forward(layernormalization33_out1);
        let gelu16_out1 = burn::tensor::activation::gelu(linear63_out1);
        let linear64_out1 = self.linear64.forward(gelu16_out1);
        let add33_out1 = add32_out1.add(linear64_out1);
        let layernormalization34_out1 = {
            let dtype = add33_out1.clone().dtype();
            self.layernormalization34
                .forward(add33_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose130_out1 = layernormalization34_out1.permute([1, 0, 2]);
        let linear65_out1 = self.linear65.forward(transpose130_out1);
        let reshape180_out1 = linear65_out1.reshape(concat4_out1);
        let unsqueeze17_out1: Tensor<B, 5> = reshape180_out1.unsqueeze_dims::<5>(&[0]);
        let transpose131_out1 = unsqueeze17_out1.permute([3, 1, 2, 0, 4]);
        let squeeze18_out1 = transpose131_out1.squeeze_dims::<4>(&[-2]);
        let gather49_out1 = {
            let sliced = squeeze18_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather50_out1 = {
            let sliced = squeeze18_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather51_out1 = {
            let sliced = squeeze18_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape181_out1 = gather49_out1.reshape(concat5_out1);
        let transpose132_out1 = reshape181_out1.permute([1, 0, 2]);
        let reshape182_out1 = gather50_out1.reshape(concat5_out1);
        let transpose133_out1 = reshape182_out1.permute([1, 0, 2]);
        let reshape183_out1 = gather51_out1.reshape(concat5_out1);
        let transpose134_out1 = reshape183_out1.permute([1, 0, 2]);
        let reshape184_out1 = transpose132_out1.reshape(concat6_out1);
        let reshape185_out1 = transpose133_out1.reshape(concat6_out1);
        let reshape186_out1 = transpose134_out1.reshape(concat6_out1);
        let shape18_out1: [i64; 4] = {
            let axes = &reshape185_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice49_out1: [i64; 1] = shape18_out1[3..4].try_into().unwrap();
        let slice50_out1: [i64; 1] = shape18_out1[2..3].try_into().unwrap();
        let slice51_out1: [i64; 2] = shape18_out1[0..2].try_into().unwrap();
        let concat41_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice50_out1[..], &slice49_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape187_out1 = reshape185_out1.reshape(concat41_out1);
        let transpose135_out1 = reshape187_out1.permute([0, 2, 1]);
        let concat42_out1: [i64; 4usize] =
            [&slice51_out1[..], &slice49_out1[..], &slice50_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape188_out1 = transpose135_out1.reshape(concat42_out1);
        let mul35_out1 = reshape184_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul36_out1 =
            reshape188_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul82_out1 = mul35_out1.matmul(mul36_out1);
        let softmax17_out1 = burn::tensor::activation::softmax(matmul82_out1, 3);
        let matmul83_out1 = softmax17_out1.matmul(reshape186_out1);
        let transpose136_out1 = matmul83_out1.permute([2, 0, 1, 3]);
        let reshape189_out1 = transpose136_out1.reshape(concat9_out1);
        let linear66_out1 = self.linear66.forward(reshape189_out1);
        let reshape190_out1 = linear66_out1.reshape(concat10_out1);
        let transpose137_out1 = reshape190_out1.permute([1, 0, 2]);
        let add34_out1 = add33_out1.add(transpose137_out1);
        add34_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule10<B: Backend> {
    layernormalization35: LayerNorm<B>,
    linear67: Linear<B>,
    linear68: Linear<B>,
    layernormalization36: LayerNorm<B>,
    linear69: Linear<B>,
    linear70: Linear<B>,
    layernormalization37: LayerNorm<B>,
    linear71: Linear<B>,
    linear72: Linear<B>,
    layernormalization38: LayerNorm<B>,
    linear73: Linear<B>,
    linear74: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule10<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization35 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear67 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear68 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization36 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear69 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear70 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization37 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear71 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear72 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization38 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear73 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear74 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization35,
            linear67,
            linear68,
            layernormalization36,
            linear69,
            linear70,
            layernormalization37,
            linear71,
            linear72,
            layernormalization38,
            linear73,
            linear74,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add34_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization35_out1 = {
            let dtype = add34_out1.clone().dtype();
            self.layernormalization35
                .forward(add34_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear67_out1 = self.linear67.forward(layernormalization35_out1);
        let gelu17_out1 = burn::tensor::activation::gelu(linear67_out1);
        let linear68_out1 = self.linear68.forward(gelu17_out1);
        let add35_out1 = add34_out1.add(linear68_out1);
        let layernormalization36_out1 = {
            let dtype = add35_out1.clone().dtype();
            self.layernormalization36
                .forward(add35_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose138_out1 = layernormalization36_out1.permute([1, 0, 2]);
        let linear69_out1 = self.linear69.forward(transpose138_out1);
        let reshape191_out1 = linear69_out1.reshape(concat4_out1);
        let unsqueeze18_out1: Tensor<B, 5> = reshape191_out1.unsqueeze_dims::<5>(&[0]);
        let transpose139_out1 = unsqueeze18_out1.permute([3, 1, 2, 0, 4]);
        let squeeze19_out1 = transpose139_out1.squeeze_dims::<4>(&[-2]);
        let gather52_out1 = {
            let sliced = squeeze19_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather53_out1 = {
            let sliced = squeeze19_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather54_out1 = {
            let sliced = squeeze19_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape192_out1 = gather52_out1.reshape(concat5_out1);
        let transpose140_out1 = reshape192_out1.permute([1, 0, 2]);
        let reshape193_out1 = gather53_out1.reshape(concat5_out1);
        let transpose141_out1 = reshape193_out1.permute([1, 0, 2]);
        let reshape194_out1 = gather54_out1.reshape(concat5_out1);
        let transpose142_out1 = reshape194_out1.permute([1, 0, 2]);
        let reshape195_out1 = transpose140_out1.reshape(concat6_out1);
        let reshape196_out1 = transpose141_out1.reshape(concat6_out1);
        let reshape197_out1 = transpose142_out1.reshape(concat6_out1);
        let shape19_out1: [i64; 4] = {
            let axes = &reshape196_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice52_out1: [i64; 1] = shape19_out1[3..4].try_into().unwrap();
        let slice53_out1: [i64; 1] = shape19_out1[2..3].try_into().unwrap();
        let slice54_out1: [i64; 2] = shape19_out1[0..2].try_into().unwrap();
        let concat43_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice53_out1[..], &slice52_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape198_out1 = reshape196_out1.reshape(concat43_out1);
        let transpose143_out1 = reshape198_out1.permute([0, 2, 1]);
        let concat44_out1: [i64; 4usize] =
            [&slice54_out1[..], &slice52_out1[..], &slice53_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape199_out1 = transpose143_out1.reshape(concat44_out1);
        let mul37_out1 = reshape195_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul38_out1 = reshape199_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul87_out1 = mul37_out1.matmul(mul38_out1);
        let softmax18_out1 = burn::tensor::activation::softmax(matmul87_out1, 3);
        let matmul88_out1 = softmax18_out1.matmul(reshape197_out1);
        let transpose144_out1 = matmul88_out1.permute([2, 0, 1, 3]);
        let reshape200_out1 = transpose144_out1.reshape(concat9_out1);
        let linear70_out1 = self.linear70.forward(reshape200_out1);
        let reshape201_out1 = linear70_out1.reshape(concat10_out1);
        let transpose145_out1 = reshape201_out1.permute([1, 0, 2]);
        let add36_out1 = add35_out1.add(transpose145_out1);
        let layernormalization37_out1 = {
            let dtype = add36_out1.clone().dtype();
            self.layernormalization37
                .forward(add36_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear71_out1 = self.linear71.forward(layernormalization37_out1);
        let gelu18_out1 = burn::tensor::activation::gelu(linear71_out1);
        let linear72_out1 = self.linear72.forward(gelu18_out1);
        let add37_out1 = add36_out1.add(linear72_out1);
        let layernormalization38_out1 = {
            let dtype = add37_out1.clone().dtype();
            self.layernormalization38
                .forward(add37_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose146_out1 = layernormalization38_out1.permute([1, 0, 2]);
        let linear73_out1 = self.linear73.forward(transpose146_out1);
        let reshape202_out1 = linear73_out1.reshape(concat4_out1);
        let unsqueeze19_out1: Tensor<B, 5> = reshape202_out1.unsqueeze_dims::<5>(&[0]);
        let transpose147_out1 = unsqueeze19_out1.permute([3, 1, 2, 0, 4]);
        let squeeze20_out1 = transpose147_out1.squeeze_dims::<4>(&[-2]);
        let gather55_out1 = {
            let sliced = squeeze20_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather56_out1 = {
            let sliced = squeeze20_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather57_out1 = {
            let sliced = squeeze20_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape203_out1 = gather55_out1.reshape(concat5_out1);
        let transpose148_out1 = reshape203_out1.permute([1, 0, 2]);
        let reshape204_out1 = gather56_out1.reshape(concat5_out1);
        let transpose149_out1 = reshape204_out1.permute([1, 0, 2]);
        let reshape205_out1 = gather57_out1.reshape(concat5_out1);
        let transpose150_out1 = reshape205_out1.permute([1, 0, 2]);
        let reshape206_out1 = transpose148_out1.reshape(concat6_out1);
        let reshape207_out1 = transpose149_out1.reshape(concat6_out1);
        let reshape208_out1 = transpose150_out1.reshape(concat6_out1);
        let shape20_out1: [i64; 4] = {
            let axes = &reshape207_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice55_out1: [i64; 1] = shape20_out1[3..4].try_into().unwrap();
        let slice56_out1: [i64; 1] = shape20_out1[2..3].try_into().unwrap();
        let slice57_out1: [i64; 2] = shape20_out1[0..2].try_into().unwrap();
        let concat45_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice56_out1[..], &slice55_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape209_out1 = reshape207_out1.reshape(concat45_out1);
        let transpose151_out1 = reshape209_out1.permute([0, 2, 1]);
        let concat46_out1: [i64; 4usize] =
            [&slice57_out1[..], &slice55_out1[..], &slice56_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape210_out1 = transpose151_out1.reshape(concat46_out1);
        let mul39_out1 = reshape206_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul40_out1 =
            reshape210_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul92_out1 = mul39_out1.matmul(mul40_out1);
        let softmax19_out1 = burn::tensor::activation::softmax(matmul92_out1, 3);
        let matmul93_out1 = softmax19_out1.matmul(reshape208_out1);
        let transpose152_out1 = matmul93_out1.permute([2, 0, 1, 3]);
        let reshape211_out1 = transpose152_out1.reshape(concat9_out1);
        let linear74_out1 = self.linear74.forward(reshape211_out1);
        let reshape212_out1 = linear74_out1.reshape(concat10_out1);
        let transpose153_out1 = reshape212_out1.permute([1, 0, 2]);
        let add38_out1 = add37_out1.add(transpose153_out1);
        add38_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule11<B: Backend> {
    layernormalization39: LayerNorm<B>,
    linear75: Linear<B>,
    linear76: Linear<B>,
    layernormalization40: LayerNorm<B>,
    linear77: Linear<B>,
    linear78: Linear<B>,
    layernormalization41: LayerNorm<B>,
    linear79: Linear<B>,
    linear80: Linear<B>,
    layernormalization42: LayerNorm<B>,
    linear81: Linear<B>,
    linear82: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule11<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization39 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear75 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear76 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization40 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear77 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear78 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization41 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear79 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear80 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization42 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear81 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear82 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            layernormalization39,
            linear75,
            linear76,
            layernormalization40,
            linear77,
            linear78,
            layernormalization41,
            linear79,
            linear80,
            layernormalization42,
            linear81,
            linear82,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add38_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> Tensor<B, 3> {
        let layernormalization39_out1 = {
            let dtype = add38_out1.clone().dtype();
            self.layernormalization39
                .forward(add38_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear75_out1 = self.linear75.forward(layernormalization39_out1);
        let gelu19_out1 = burn::tensor::activation::gelu(linear75_out1);
        let linear76_out1 = self.linear76.forward(gelu19_out1);
        let add39_out1 = add38_out1.add(linear76_out1);
        let layernormalization40_out1 = {
            let dtype = add39_out1.clone().dtype();
            self.layernormalization40
                .forward(add39_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose154_out1 = layernormalization40_out1.permute([1, 0, 2]);
        let linear77_out1 = self.linear77.forward(transpose154_out1);
        let reshape213_out1 = linear77_out1.reshape(concat4_out1);
        let unsqueeze20_out1: Tensor<B, 5> = reshape213_out1.unsqueeze_dims::<5>(&[0]);
        let transpose155_out1 = unsqueeze20_out1.permute([3, 1, 2, 0, 4]);
        let squeeze21_out1 = transpose155_out1.squeeze_dims::<4>(&[-2]);
        let gather58_out1 = {
            let sliced = squeeze21_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather59_out1 = {
            let sliced = squeeze21_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather60_out1 = {
            let sliced = squeeze21_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape214_out1 = gather58_out1.reshape(concat5_out1);
        let transpose156_out1 = reshape214_out1.permute([1, 0, 2]);
        let reshape215_out1 = gather59_out1.reshape(concat5_out1);
        let transpose157_out1 = reshape215_out1.permute([1, 0, 2]);
        let reshape216_out1 = gather60_out1.reshape(concat5_out1);
        let transpose158_out1 = reshape216_out1.permute([1, 0, 2]);
        let reshape217_out1 = transpose156_out1.reshape(concat6_out1);
        let reshape218_out1 = transpose157_out1.reshape(concat6_out1);
        let reshape219_out1 = transpose158_out1.reshape(concat6_out1);
        let shape21_out1: [i64; 4] = {
            let axes = &reshape218_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice58_out1: [i64; 1] = shape21_out1[3..4].try_into().unwrap();
        let slice59_out1: [i64; 1] = shape21_out1[2..3].try_into().unwrap();
        let slice60_out1: [i64; 2] = shape21_out1[0..2].try_into().unwrap();
        let concat47_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice59_out1[..], &slice58_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape220_out1 = reshape218_out1.reshape(concat47_out1);
        let transpose159_out1 = reshape220_out1.permute([0, 2, 1]);
        let concat48_out1: [i64; 4usize] =
            [&slice60_out1[..], &slice58_out1[..], &slice59_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape221_out1 = transpose159_out1.reshape(concat48_out1);
        let mul41_out1 = reshape217_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul42_out1 = reshape221_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul97_out1 = mul41_out1.matmul(mul42_out1);
        let softmax20_out1 = burn::tensor::activation::softmax(matmul97_out1, 3);
        let matmul98_out1 = softmax20_out1.matmul(reshape219_out1);
        let transpose160_out1 = matmul98_out1.permute([2, 0, 1, 3]);
        let reshape222_out1 = transpose160_out1.reshape(concat9_out1);
        let linear78_out1 = self.linear78.forward(reshape222_out1);
        let reshape223_out1 = linear78_out1.reshape(concat10_out1);
        let transpose161_out1 = reshape223_out1.permute([1, 0, 2]);
        let add40_out1 = add39_out1.add(transpose161_out1);
        let layernormalization41_out1 = {
            let dtype = add40_out1.clone().dtype();
            self.layernormalization41
                .forward(add40_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear79_out1 = self.linear79.forward(layernormalization41_out1);
        let gelu20_out1 = burn::tensor::activation::gelu(linear79_out1);
        let linear80_out1 = self.linear80.forward(gelu20_out1);
        let add41_out1 = add40_out1.add(linear80_out1);
        let layernormalization42_out1 = {
            let dtype = add41_out1.clone().dtype();
            self.layernormalization42
                .forward(add41_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose162_out1 = layernormalization42_out1.permute([1, 0, 2]);
        let linear81_out1 = self.linear81.forward(transpose162_out1);
        let reshape224_out1 = linear81_out1.reshape(concat4_out1);
        let unsqueeze21_out1: Tensor<B, 5> = reshape224_out1.unsqueeze_dims::<5>(&[0]);
        let transpose163_out1 = unsqueeze21_out1.permute([3, 1, 2, 0, 4]);
        let squeeze22_out1 = transpose163_out1.squeeze_dims::<4>(&[-2]);
        let gather61_out1 = {
            let sliced = squeeze22_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather62_out1 = {
            let sliced = squeeze22_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather63_out1 = {
            let sliced = squeeze22_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape225_out1 = gather61_out1.reshape(concat5_out1);
        let transpose164_out1 = reshape225_out1.permute([1, 0, 2]);
        let reshape226_out1 = gather62_out1.reshape(concat5_out1);
        let transpose165_out1 = reshape226_out1.permute([1, 0, 2]);
        let reshape227_out1 = gather63_out1.reshape(concat5_out1);
        let transpose166_out1 = reshape227_out1.permute([1, 0, 2]);
        let reshape228_out1 = transpose164_out1.reshape(concat6_out1);
        let reshape229_out1 = transpose165_out1.reshape(concat6_out1);
        let reshape230_out1 = transpose166_out1.reshape(concat6_out1);
        let shape22_out1: [i64; 4] = {
            let axes = &reshape229_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice61_out1: [i64; 1] = shape22_out1[3..4].try_into().unwrap();
        let slice62_out1: [i64; 1] = shape22_out1[2..3].try_into().unwrap();
        let slice63_out1: [i64; 2] = shape22_out1[0..2].try_into().unwrap();
        let concat49_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice62_out1[..], &slice61_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape231_out1 = reshape229_out1.reshape(concat49_out1);
        let transpose167_out1 = reshape231_out1.permute([0, 2, 1]);
        let concat50_out1: [i64; 4usize] =
            [&slice63_out1[..], &slice61_out1[..], &slice62_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape232_out1 = transpose167_out1.reshape(concat50_out1);
        let mul43_out1 = reshape228_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul44_out1 =
            reshape232_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul102_out1 = mul43_out1.matmul(mul44_out1);
        let softmax21_out1 = burn::tensor::activation::softmax(matmul102_out1, 3);
        let matmul103_out1 = softmax21_out1.matmul(reshape230_out1);
        let transpose168_out1 = matmul103_out1.permute([2, 0, 1, 3]);
        let reshape233_out1 = transpose168_out1.reshape(concat9_out1);
        let linear82_out1 = self.linear82.forward(reshape233_out1);
        let reshape234_out1 = linear82_out1.reshape(concat10_out1);
        let transpose169_out1 = reshape234_out1.permute([1, 0, 2]);
        let add42_out1 = add41_out1.add(transpose169_out1);
        add42_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule12<B: Backend> {
    layernormalization43: LayerNorm<B>,
    linear83: Linear<B>,
    linear84: Linear<B>,
    layernormalization44: LayerNorm<B>,
    linear85: Linear<B>,
    linear86: Linear<B>,
    layernormalization45: LayerNorm<B>,
    linear87: Linear<B>,
    linear88: Linear<B>,
    layernormalization46: LayerNorm<B>,
    linear89: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule12<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization43 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear83 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear84 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization44 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear85 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear86 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization45 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear87 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear88 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization46 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear89 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        Self {
            layernormalization43,
            linear83,
            linear84,
            layernormalization44,
            linear85,
            linear86,
            layernormalization45,
            linear87,
            linear88,
            layernormalization46,
            linear89,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add42_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
    ) -> (Tensor<B, 4>, Tensor<B, 3>) {
        let layernormalization43_out1 = {
            let dtype = add42_out1.clone().dtype();
            self.layernormalization43
                .forward(add42_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear83_out1 = self.linear83.forward(layernormalization43_out1);
        let gelu21_out1 = burn::tensor::activation::gelu(linear83_out1);
        let linear84_out1 = self.linear84.forward(gelu21_out1);
        let add43_out1 = add42_out1.add(linear84_out1);
        let layernormalization44_out1 = {
            let dtype = add43_out1.clone().dtype();
            self.layernormalization44
                .forward(add43_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose170_out1 = layernormalization44_out1.permute([1, 0, 2]);
        let linear85_out1 = self.linear85.forward(transpose170_out1);
        let reshape235_out1 = linear85_out1.reshape(concat4_out1);
        let unsqueeze22_out1: Tensor<B, 5> = reshape235_out1.unsqueeze_dims::<5>(&[0]);
        let transpose171_out1 = unsqueeze22_out1.permute([3, 1, 2, 0, 4]);
        let squeeze23_out1 = transpose171_out1.squeeze_dims::<4>(&[-2]);
        let gather64_out1 = {
            let sliced = squeeze23_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather65_out1 = {
            let sliced = squeeze23_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather66_out1 = {
            let sliced = squeeze23_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape236_out1 = gather64_out1.reshape(concat5_out1);
        let transpose172_out1 = reshape236_out1.permute([1, 0, 2]);
        let reshape237_out1 = gather65_out1.reshape(concat5_out1);
        let transpose173_out1 = reshape237_out1.permute([1, 0, 2]);
        let reshape238_out1 = gather66_out1.reshape(concat5_out1);
        let transpose174_out1 = reshape238_out1.permute([1, 0, 2]);
        let reshape239_out1 = transpose172_out1.reshape(concat6_out1);
        let reshape240_out1 = transpose173_out1.reshape(concat6_out1);
        let reshape241_out1 = transpose174_out1.reshape(concat6_out1);
        let shape23_out1: [i64; 4] = {
            let axes = &reshape240_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice64_out1: [i64; 1] = shape23_out1[3..4].try_into().unwrap();
        let slice65_out1: [i64; 1] = shape23_out1[2..3].try_into().unwrap();
        let slice66_out1: [i64; 2] = shape23_out1[0..2].try_into().unwrap();
        let concat51_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice65_out1[..], &slice64_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape242_out1 = reshape240_out1.reshape(concat51_out1);
        let transpose175_out1 = reshape242_out1.permute([0, 2, 1]);
        let concat52_out1: [i64; 4usize] =
            [&slice66_out1[..], &slice64_out1[..], &slice65_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape243_out1 = transpose175_out1.reshape(concat52_out1);
        let mul45_out1 = reshape239_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul46_out1 = reshape243_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul107_out1 = mul45_out1.matmul(mul46_out1);
        let softmax22_out1 = burn::tensor::activation::softmax(matmul107_out1, 3);
        let matmul108_out1 = softmax22_out1.matmul(reshape241_out1);
        let transpose176_out1 = matmul108_out1.permute([2, 0, 1, 3]);
        let reshape244_out1 = transpose176_out1.reshape(concat9_out1);
        let linear86_out1 = self.linear86.forward(reshape244_out1);
        let reshape245_out1 = linear86_out1.reshape(concat10_out1);
        let transpose177_out1 = reshape245_out1.permute([1, 0, 2]);
        let add44_out1 = add43_out1.add(transpose177_out1);
        let layernormalization45_out1 = {
            let dtype = add44_out1.clone().dtype();
            self.layernormalization45
                .forward(add44_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear87_out1 = self.linear87.forward(layernormalization45_out1);
        let gelu22_out1 = burn::tensor::activation::gelu(linear87_out1);
        let linear88_out1 = self.linear88.forward(gelu22_out1);
        let add45_out1 = add44_out1.add(linear88_out1);
        let layernormalization46_out1 = {
            let dtype = add45_out1.clone().dtype();
            self.layernormalization46
                .forward(add45_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose178_out1 = layernormalization46_out1.permute([1, 0, 2]);
        let linear89_out1 = self.linear89.forward(transpose178_out1);
        let reshape246_out1 = linear89_out1.reshape(concat4_out1);
        let unsqueeze23_out1: Tensor<B, 5> = reshape246_out1.unsqueeze_dims::<5>(&[0]);
        let transpose179_out1 = unsqueeze23_out1.permute([3, 1, 2, 0, 4]);
        let squeeze24_out1 = transpose179_out1.squeeze_dims::<4>(&[-2]);
        let gather67_out1 = {
            let sliced = squeeze24_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather68_out1 = {
            let sliced = squeeze24_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather69_out1 = {
            let sliced = squeeze24_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape247_out1 = gather67_out1.reshape(concat5_out1);
        let transpose180_out1 = reshape247_out1.permute([1, 0, 2]);
        let reshape248_out1 = gather68_out1.reshape(concat5_out1);
        let transpose181_out1 = reshape248_out1.permute([1, 0, 2]);
        let reshape249_out1 = gather69_out1.reshape(concat5_out1);
        let transpose182_out1 = reshape249_out1.permute([1, 0, 2]);
        let reshape250_out1 = transpose180_out1.reshape(concat6_out1);
        let reshape251_out1 = transpose181_out1.reshape(concat6_out1);
        let reshape252_out1 = transpose182_out1.reshape(concat6_out1);
        let shape24_out1: [i64; 4] = {
            let axes = &reshape251_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice67_out1: [i64; 1] = shape24_out1[3..4].try_into().unwrap();
        let slice68_out1: [i64; 1] = shape24_out1[2..3].try_into().unwrap();
        let slice69_out1: [i64; 2] = shape24_out1[0..2].try_into().unwrap();
        let concat53_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice68_out1[..], &slice67_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape253_out1 = reshape251_out1.reshape(concat53_out1);
        let transpose183_out1 = reshape253_out1.permute([0, 2, 1]);
        let concat54_out1: [i64; 4usize] =
            [&slice69_out1[..], &slice67_out1[..], &slice68_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape254_out1 = transpose183_out1.reshape(concat54_out1);
        let mul47_out1 = reshape250_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul48_out1 =
            reshape254_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul112_out1 = mul47_out1.matmul(mul48_out1);
        let softmax23_out1 = burn::tensor::activation::softmax(matmul112_out1, 3);
        let matmul113_out1 = softmax23_out1.matmul(reshape252_out1);
        (matmul113_out1, add45_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule13<B: Backend> {
    linear90: Linear<B>,
    layernormalization47: LayerNorm<B>,
    linear91: Linear<B>,
    linear92: Linear<B>,
    layernormalization48: LayerNorm<B>,
    linear93: Linear<B>,
    linear94: Linear<B>,
    layernormalization49: LayerNorm<B>,
    linear95: Linear<B>,
    linear96: Linear<B>,
    layernormalization50: LayerNorm<B>,
    linear97: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule13<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let linear90 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization47 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear91 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear92 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization48 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear93 = LinearConfig::new(1024, 3072).with_bias(true).init(device);
        let linear94 = LinearConfig::new(1024, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization49 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear95 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear96 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let layernormalization50 = LayerNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear97 = LinearConfig::new(1024, 768).with_bias(false).init(device);
        Self {
            linear90,
            layernormalization47,
            linear91,
            linear92,
            layernormalization48,
            linear93,
            linear94,
            layernormalization49,
            linear95,
            linear96,
            layernormalization50,
            linear97,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        matmul113_out1: Tensor<B, 4>,
        concat9_out1: [i64; 2],
        concat10_out1: [i64; 3],
        add45_out1: Tensor<B, 3>,
        concat4_out1: [i64; 4],
        concat5_out1: [i64; 3],
        concat6_out1: [i64; 4],
        constant298_out1: [i64; 1],
        constant226_out1: Tensor<B, 1>,
        shape1_out1: [i64; 1],
    ) -> Tensor<B, 2> {
        let transpose184_out1 = matmul113_out1.permute([2, 0, 1, 3]);
        let reshape255_out1 = transpose184_out1.reshape(concat9_out1);
        let linear90_out1 = self.linear90.forward(reshape255_out1);
        let reshape256_out1 = linear90_out1.reshape(concat10_out1);
        let transpose185_out1 = reshape256_out1.permute([1, 0, 2]);
        let add46_out1 = add45_out1.add(transpose185_out1);
        let layernormalization47_out1 = {
            let dtype = add46_out1.clone().dtype();
            self.layernormalization47
                .forward(add46_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear91_out1 = self.linear91.forward(layernormalization47_out1);
        let gelu23_out1 = burn::tensor::activation::gelu(linear91_out1);
        let linear92_out1 = self.linear92.forward(gelu23_out1);
        let add47_out1 = add46_out1.add(linear92_out1);
        let layernormalization48_out1 = {
            let dtype = add47_out1.clone().dtype();
            self.layernormalization48
                .forward(add47_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let transpose186_out1 = layernormalization48_out1.permute([1, 0, 2]);
        let linear93_out1 = self.linear93.forward(transpose186_out1);
        let reshape257_out1 = linear93_out1.reshape(concat4_out1);
        let unsqueeze24_out1: Tensor<B, 5> = reshape257_out1.unsqueeze_dims::<5>(&[0]);
        let transpose187_out1 = unsqueeze24_out1.permute([3, 1, 2, 0, 4]);
        let squeeze25_out1 = transpose187_out1.squeeze_dims::<4>(&[-2]);
        let gather70_out1 = {
            let sliced = squeeze25_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather71_out1 = {
            let sliced = squeeze25_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather72_out1 = {
            let sliced = squeeze25_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape258_out1 = gather70_out1.reshape(concat5_out1);
        let transpose188_out1 = reshape258_out1.permute([1, 0, 2]);
        let reshape259_out1 = gather71_out1.reshape(concat5_out1);
        let transpose189_out1 = reshape259_out1.permute([1, 0, 2]);
        let reshape260_out1 = gather72_out1.reshape(concat5_out1);
        let transpose190_out1 = reshape260_out1.permute([1, 0, 2]);
        let reshape261_out1 = transpose188_out1.reshape(concat6_out1);
        let reshape262_out1 = transpose189_out1.reshape(concat6_out1);
        let reshape263_out1 = transpose190_out1.reshape(concat6_out1);
        let shape25_out1: [i64; 4] = {
            let axes = &reshape262_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice70_out1: [i64; 1] = shape25_out1[3..4].try_into().unwrap();
        let slice71_out1: [i64; 1] = shape25_out1[2..3].try_into().unwrap();
        let slice72_out1: [i64; 2] = shape25_out1[0..2].try_into().unwrap();
        let concat55_out1: [i64; 3usize] =
            [&constant298_out1[..], &slice71_out1[..], &slice70_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape264_out1 = reshape262_out1.reshape(concat55_out1);
        let transpose191_out1 = reshape264_out1.permute([0, 2, 1]);
        let concat56_out1: [i64; 4usize] =
            [&slice72_out1[..], &slice70_out1[..], &slice71_out1[..]]
                .concat()
                .try_into()
                .unwrap();
        let reshape265_out1 = transpose191_out1.reshape(concat56_out1);
        let mul49_out1 = reshape261_out1
            .mul((constant226_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul50_out1 =
            reshape265_out1.mul((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let matmul117_out1 = mul49_out1.matmul(mul50_out1);
        let softmax24_out1 = burn::tensor::activation::softmax(matmul117_out1, 3);
        let matmul118_out1 = softmax24_out1.matmul(reshape263_out1);
        let transpose192_out1 = matmul118_out1.permute([2, 0, 1, 3]);
        let reshape266_out1 = transpose192_out1.reshape(concat9_out1);
        let linear94_out1 = self.linear94.forward(reshape266_out1);
        let reshape267_out1 = linear94_out1.reshape(concat10_out1);
        let transpose193_out1 = reshape267_out1.permute([1, 0, 2]);
        let add48_out1 = add47_out1.add(transpose193_out1);
        let layernormalization49_out1 = {
            let dtype = add48_out1.clone().dtype();
            self.layernormalization49
                .forward(add48_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear95_out1 = self.linear95.forward(layernormalization49_out1);
        let gelu24_out1 = burn::tensor::activation::gelu(linear95_out1);
        let linear96_out1 = self.linear96.forward(gelu24_out1);
        let add49_out1 = add48_out1.add(linear96_out1);
        let layernormalization50_out1 = {
            let dtype = add49_out1.dtype();
            self.layernormalization50
                .forward(add49_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let gather73_out1 = {
            let sliced = layernormalization50_out1.slice(s![.., 0, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let linear97_out1 = self.linear97.forward(gather73_out1);
        let reducel21_out1 = {
            let input_dtype = linear97_out1.clone().dtype();
            linear97_out1
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
        let constant315_out1: [i64; 1] = [768i64];
        let concat57_out1: [i64; 2usize] = [&shape1_out1[..], &constant315_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let expand2_out1 = {
            let onnx_shape: [i64; 2usize] = concat57_out1;
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
        let div1_out1 = linear97_out1.div(expand2_out1);
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
    submodule8: Submodule8<B>,
    submodule9: Submodule9<B>,
    submodule10: Submodule10<B>,
    submodule11: Submodule11<B>,
    submodule12: Submodule12<B>,
    submodule13: Submodule13<B>,
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
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 2> {
        let (
            add2_out1,
            concat4_out1,
            concat5_out1,
            concat6_out1,
            constant298_out1,
            constant226_out1,
            concat9_out1,
            concat10_out1,
            shape1_out1,
        ) = self.submodule1.forward(pixel_values);
        let add6_out1 = self.submodule2.forward(
            add2_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add10_out1 = self.submodule3.forward(
            add6_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add14_out1 = self.submodule4.forward(
            add10_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add18_out1 = self.submodule5.forward(
            add14_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add22_out1 = self.submodule6.forward(
            add18_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add26_out1 = self.submodule7.forward(
            add22_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add30_out1 = self.submodule8.forward(
            add26_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add34_out1 = self.submodule9.forward(
            add30_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add38_out1 = self.submodule10.forward(
            add34_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let add42_out1 = self.submodule11.forward(
            add38_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let (matmul113_out1, add45_out1) = self.submodule12.forward(
            add42_out1,
            concat4_out1.clone(),
            concat5_out1.clone(),
            concat6_out1.clone(),
            constant298_out1.clone(),
            constant226_out1.clone(),
            concat9_out1.clone(),
            concat10_out1.clone(),
        );
        let div1_out1 = self.submodule13.forward(
            matmul113_out1,
            concat9_out1,
            concat10_out1,
            add45_out1,
            concat4_out1,
            concat5_out1,
            concat6_out1,
            constant298_out1,
            constant226_out1,
            shape1_out1,
        );
        div1_out1
    }
}
