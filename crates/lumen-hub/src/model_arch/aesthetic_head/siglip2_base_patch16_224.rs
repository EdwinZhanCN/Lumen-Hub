// Generated from ONNX "onnx/aesthetic-head-siglip2-base-patch16-224/aesthetic.prepared.onnx" by burn-onnx
use burn::prelude::*;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    linear5: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}


extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/Volumes/CodeBase/Projects/Lumen-Hub/target/release/build/lumen-convert-0036f7b8de134cd6/out/aesthetic_head/siglip2_base_patch16_224/aesthetic.bpk",
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model.load_from(&mut store).expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(768, 1024)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear2 = LinearConfig::new(1024, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear3 = LinearConfig::new(128, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear4 = LinearConfig::new(64, 16)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear5 = LinearConfig::new(16, 1)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            linear1,
            linear2,
            linear3,
            linear4,
            linear5,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, image_features: Tensor<B, 2>) -> Tensor<B, 1> {
        let reducel21_out1 = {
            let input_dtype = image_features.clone().dtype();
            image_features
                .clone()
                .square()
                .sum_dim(1usize)
                .cast(burn::tensor::DType::F32)
                .sqrt()
                .cast(input_dtype)
        };
        let clip1_out1 = {
            let __clip_min = 0.0000009999999974752427f64;
            reducel21_out1.clamp_min(__clip_min)
        };
        let div1_out1 = image_features.div(clip1_out1);
        let linear1_out1 = self.linear1.forward(div1_out1);
        let relu1_out1 = burn::tensor::activation::relu(linear1_out1);
        let linear2_out1 = self.linear2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(linear2_out1);
        let linear3_out1 = self.linear3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(linear3_out1);
        let linear4_out1 = self.linear4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(linear4_out1);
        let linear5_out1 = self.linear5.forward(relu4_out1);
        let squeeze1_out1 = linear5_out1.squeeze_dims::<1>(&[-1]);
        squeeze1_out1
    }
}
