use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

/// The supported precision calculation methods.
#[allow(missing_docs)]
pub enum AverageMethod {
    Micro,
}

/// Define a default implementation for the derive macro to use. Ultimately, this sets default averaging method used during precision calculation
impl Default for AverageMethod {
    fn default() -> Self {
        AverageMethod::Micro
    }
}

/// The precision metric.
#[derive(Default)]
pub struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    avg_method: AverageMethod,
}

/// The [precision metric](PrecisionMetric) input type.
#[derive(new)]
pub struct PrecisionInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> PrecisionMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// overrides the default average method.
    pub fn with_avg_method(mut self, method: AverageMethod) -> Self {
        match method {
            AverageMethod::Micro => {
                self.avg_method = AverageMethod::Micro;
            }
            // TODO: fill other calculation methods
            _ => {
                unimplemented!("You provided an unsupported average method; please refer to the official docs for the supported options.");
            }
        }
        self
    }
}

impl<B: Backend> Metric for PrecisionMetric<B> {
    const NAME: &'static str = "Precision";

    type Input = PrecisionInput<B>;

    fn update(&mut self, input: &PrecisionInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, _n_classes] = input.outputs.dims();

        let targets = input.targets.clone().to_device(&B::Device::default());
        let outputs = input
            .outputs
            .clone()
            .argmax(1)
            .to_device(&B::Device::default())
            .reshape([batch_size]);

        // Determine which precision average method to use.
        let precision: f64 = match self.avg_method {
            AverageMethod::Micro => {
                calc_micro_precision(&outputs, &targets, batch_size, &_n_classes)
            }
            // TODO: fill other calculation methods
        };

        self.state.update(
            100.0 * precision,
            batch_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for PrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

fn calc_micro_precision<B: Backend>(
    outputs: &Tensor<B, 1, Int>,
    targets: &Tensor<B, 1, Int>,
    batch_size: usize,
    _n_classes: &usize,
) -> f64 {
    let true_positive_count;
    let false_positive_count;

    match _n_classes {
        2 => {
            // No `logical_and` tensor op... but we know that when both are positives the sum is 2
            true_positive_count = outputs
                .clone()
                .add(targets.clone())
                .equal_elem(2)
                .int()
                .sum()
                .into_scalar()
                .elem::<f64>();

            false_positive_count = outputs
                .clone()
                .mul_scalar(2) // set every predicted positive to 2
                .add(targets.clone())
                .equal_elem(2) // predicted positive (2) + target negative (0)
                .int()
                .sum()
                .into_scalar()
                .elem::<f64>();
        }
        /* In the multiclass case any correct prediction is a true positive and any incorrect one is a false positive. This and the fact that we reshape the tensor to be the
        batch_size number of rows, simplifies the calculation.
        */
        _ => {
            true_positive_count = outputs
                .clone()
                .equal(targets.clone())
                .int()
                .sum()
                .into_scalar()
                .elem::<f64>();

            false_positive_count = batch_size as f64 - true_positive_count;
        }
    }

    true_positive_count / (true_positive_count + false_positive_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_micro_precision_binary() {
        let device = Default::default();
        let mut metric = PrecisionMetric::<TestBackend>::new();
        let input = PrecisionInput::new(
            Tensor::from_data(
                [
                    [0.4, 0.6], // 1 = fp
                    [1.0, 0.0], // 0 = tn
                    [0.8, 0.2], // 0 = fn
                    [0.3, 0.7], // 1 = tp
                    [0.8, 0.2], // 0 = fn
                ],
                &device,
            ),
            Tensor::from_data([0, 0, 1, 1, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(50.0, metric.value());
    }

    #[test]
    fn test_micro_precision_multi() {
        let device = Default::default();
        let mut metric = PrecisionMetric::<TestBackend>::new();
        let input = PrecisionInput::new(
            Tensor::from_data(
                [
                    [1.0, 0.0, 0.0, 0.0], // 0
                    [0.0, 1.0, 0.0, 0.0], // 1
                    [0.0, 0.0, 1.0, 0.0], // 2
                    [0.0, 0.0, 0.0, 1.0], // 3
                ],
                &device,
            ),
            Tensor::from_data([0, 1, 2, 0], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(75.0, metric.value());
    }
}
