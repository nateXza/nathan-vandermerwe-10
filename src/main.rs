use burn::tensor::backend::AutodiffBackend;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};
use burn::train::{RegressionOutput, TrainOutput, TrainStep};
use burn::optim::SgdConfig; // Simpler optimizer for regression
use burn::data::dataloader::{Dataset, DataLoader, batcher::Batcher};
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use rand::Rng;
use textplots::{Chart, Plot, Shape};

type B = NdArray<f32>;
type BA = Autodiff<NdArray<f32>>;

// Model Definition
#[derive(Module, Debug)]
pub struct LinearRegression<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LinearRegression<B> {
    pub fn new() -> Self {
        let device = Default::default();
        let linear = LinearConfig::new(1, 1).init::<B>(&device);
        Self { linear }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

// Training Data and Dataset
#[derive(Clone)]
struct RegressionDataset {
    data: Vec<(f32, f32)>,
}

impl RegressionDataset {
    fn new(n_samples: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let x = rng.gen_range(-1.0..1.0);
            let noise = rng.gen_range(-0.2..0.2);
            let y = 2.0 * x + 1.0 + noise;
            data.push((x, y));
        }
        Self { data }
    }

    fn split(self, train_ratio: f32) -> (Self, Self) {
        let n_train = (self.data.len() as f32 * train_ratio) as usize;
        let mut train_data = self.data[..n_train].to_vec();
        let mut val_data = self.data[n_train..].to_vec();
        train_data.shrink_to_fit();
        val_data.shrink_to_fit();
        (Self { data: train_data }, Self { data: val_data })
    }
}

impl<B: Backend> Dataset<(Tensor<B, 2>, Tensor<B, 2>)> for RegressionDataset {
    fn get(&self, index: usize) -> Option<(Tensor<B, 2>, Tensor<B, 2>)> {
        self.data.get(index).map(|&(x, y)| {
            let x_tensor = Tensor::<B, 2>::from_floats([[x]]);
            let y_tensor = Tensor::<B, 2>::from_floats([[y]]);
            (x_tensor, y_tensor)
        })
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// Batcher
struct RegressionBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<(Tensor<B, 2>, Tensor<B, 2>), (Tensor<B, 2>, Tensor<B, 2>)> for RegressionBatcher<B> {
    fn batch(&self, items: Vec<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let xs = items.iter().map(|(x, _)| x.clone()).collect::<Vec<_>>();
        let ys = items.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>();
        (Tensor::cat(xs, 0), Tensor::cat(ys, 0))
    }
}

// Training Step
impl<B: AutodiffBackend> TrainStep<Tensor<B, 2>, RegressionOutput<B>> for LinearRegression<B> {
    fn step(&self, input: Tensor<B, 2>) -> TrainOutput<RegressionOutput<B>> {
        let targets = input.clone() * 2.0 + 1.0;
        let output = self.forward(input.clone());
        let diff = output.clone() - targets.clone();
        let loss_per_sample = diff.powf_scalar(2.0);
        let loss = loss_per_sample.clone().mean();
        TrainOutput::new(
            self,
            loss,
            RegressionOutput {
                loss: loss_per_sample,
                output,
                targets,
            },
        )
    }
}

fn main() {
    // Data generation and split
    let dataset = RegressionDataset::new(1000);
    let (train_dataset, val_dataset) = dataset.split(0.8);

    // Dataloaders
    let train_batcher = RegressionBatcher::<BA> { device: Default::default() };
    let val_batcher = RegressionBatcher::<B> { device: Default::default() };
    let dataloader_train = DataLoader::new(train_dataset, train_batcher, 64);
    let dataloader_val = DataLoader::new(val_dataset, val_batcher, 64);

    // Model and optimizer
    let model = LinearRegression::<BA>::new();
    let optimizer = SgdConfig { learning_rate: 1e-3 }.init();

    // Training
    let learner = LearnerBuilder::new(".")
        .devices(vec![Default::default()])
        .num_epochs(100)
        .with_metric(LossMetric::new())
        .build(model, optimizer);

    let trained_model = learner.fit(dataloader_train, dataloader_val);

    // Test predictions
    let test_dataset = RegressionDataset::new(50);
    let test_data: Vec<_> = test_dataset.data.iter().map(|&(x, _)| x).collect();
    let x_test = Tensor::<B, 2>::from_floats([test_data.clone()]);
    let predictions = trained_model.forward(x_test);
    let y_pred = predictions.into_data().to_vec().unwrap();
    let plot_points: Vec<(f32, f32)> = test_data.into_iter().zip(y_pred.into_iter()).collect();

    // Visualization
    println!("Predicted vs Actual (y = 2x + 1):");
    Chart::new(120, 60, -1.0, 1.0)
        .lineplot(&Shape::Points(&plot_points))
        .display();

    // Print learned parameters
    let weights = trained_model.linear.weight.into_data().to_vec().unwrap()[0];
    let bias = trained_model.linear.bias.unwrap().into_data().to_vec().unwrap()[0];
    println!("Learned parameters: y = {:.2}x + {:.2}", weights, bias);
}