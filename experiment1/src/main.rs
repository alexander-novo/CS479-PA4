#![feature(try_blocks)]
use core::fmt;
use std::{
    fmt::Display,
    path::{Path, PathBuf},
};

use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use linfa::{
    prelude::ToConfusionMatrix,
    traits::{Fit, Predict},
    Dataset,
};
use linfa_bayes::GaussianNb;
use linfa_svm::{Svm, SvmParams};
use ndarray::{s, Array, Axis, Ix1, Ix2};
use ndarray_csv::Array2Reader;

#[derive(Parser)]
struct Args {
    #[arg(default_value = "genderdata/48_60")]
    data_path: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    for fold in 1..=3 {
        println!("Fold {fold}");

        // Get an iterator of all of the SVM paramater combinations we want to try
        let params = get_params();
        // Get our fold of data
        let Fold { train, val, test } = get_fold(&args.data_path, fold)?;

        // Print table header
        println!("KType Param Cost #SV   Acc");

        let Some((best_acc, best_params, model)) = params
            // Train an SVM on each set of parameters. Pass the parameters forward so we can print them with the trained model performance.
            .flat_map(|params| -> Result<(Svm<f64, bool>, Params)> {
                try { (SvmParams::from(params).fit(&train)?, params) }
            })
            // Evaluate each model on the validation data. Pass along the model, its validation accuracy, and the params used to create it
            .map(|(svm, params)| {
                let pred = svm.predict(&val);
                let cm = pred.confusion_matrix(&val).unwrap();

                println!("{} {:3} {:.3}", params, svm.nsupport(), cm.accuracy(),);

                (cm.accuracy(), params, svm)
            })
			// Keep the model with the highest validation accuracy
            .reduce(|x, y| if x.0 > y.0 { x } else { y })
			else {unreachable!()};

        // Evaluate the best model on the test set
        let pred = model.predict(&test);
        let cm = pred.confusion_matrix(&test)?;

        println!(
            "Best model {best_params} with validation accuracy {best_acc} and test accuracy {}",
            cm.accuracy()
        );

        // Train a bayesian classifier, and evaluate on test set to compare performance
        let model_bayes = GaussianNb::params().fit(&train)?;
        let pred = model_bayes.predict(&test);
        let cm = pred.confusion_matrix(&test)?;

        println!("Bayesian Classifier accuracy: {}", cm.accuracy());

        println!();
    }

    Ok(())
}

/// Load a single dataset from a pair of PCA files.
/// `data_path` should point to an SSV file containing the eigen coefficients of the data, where each line is a single data point of floating point numbers.
/// `label_path` should point to an SSV file containing the labels of each data point. It should be a single line and the labels should be unsigned integers, which can fit in one byte.
///
/// # Errors
///
/// This function will return an error if the files can't be read from or their contents can't be deserialized into a 2D array of the appropriate data types.
fn load_dataset(
    data_path: impl AsRef<Path>,
    label_path: impl AsRef<Path>,
) -> Result<(Array<f64, Ix2>, Array<u8, Ix1>)> {
    let data = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path(data_path)?
        .deserialize_array2_dynamic::<f64>()?;

    // Only take the first 30 features
    let data = data.slice(s![.., ..30]);

    // let data = data.t();
    let data = data.as_standard_layout().to_owned();

    let targets = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path(label_path)?
        .deserialize_array2_dynamic::<u8>()?
        .slice_move(s![0, ..]);

    Ok((data, targets))
}

/// A struct which holds all of the datasets for a particular fold.
struct Fold {
    train: Dataset<f64, bool, Ix1>,
    val: Dataset<f64, bool, Ix1>,
    test: Dataset<f64, bool, Ix1>,
}

/// Get all of the datasets belonging to a particular fold.
///
/// # Panics
///
/// Panics if the training data is empty.
///
/// # Errors
///
/// This function will return an error if a dataset can't be loaded (see [`load_dataset`]).
fn get_fold(path: impl AsRef<Path>, fold: u8) -> Result<Fold> {
    let (train_data, train_labels) = load_dataset(
        path.as_ref().join(format!("trPCA_0{fold}.txt")),
        path.as_ref().join(format!("TtrPCA_0{fold}.txt")),
    )?;

    let (val_data, val_labels) = load_dataset(
        path.as_ref().join(format!("valPCA_0{fold}.txt")),
        path.as_ref().join(format!("TvalPCA_0{fold}.txt")),
    )?;

    let (test_data, test_labels) = load_dataset(
        path.as_ref().join(format!("tsPCA_0{fold}.txt")),
        path.as_ref().join(format!("TtsPCA_0{fold}.txt")),
    )?;

    let min = train_data.map_axis(Axis(0), |feature| {
        feature.iter().copied().reduce(f64::min).unwrap()
    });
    let max = train_data.map_axis(Axis(0), |feature| {
        feature.iter().copied().reduce(f64::max).unwrap()
    });

    let train_data = 2.0 * (train_data - &min) / (&max - &min) - 1.0;
    let val_data = 2.0 * (val_data - &min) / (&max - &min) - 1.0;
    let test_data = 2.0 * (test_data - &min) / (&max - &min) - 1.0;

    Ok(Fold {
        train: Dataset::new(train_data, train_labels.map(|x| *x == 1)),
        val: Dataset::new(val_data, val_labels.map(|x| *x == 1)),
        test: Dataset::new(test_data, test_labels.map(|x| *x == 1)),
    })
}

/// The kernel-specific parameters we will be looping through for SVM training
#[derive(Clone, Copy)]
enum KernelParams {
    Rbf { gamma: f64 },
    Poly { degree: u32 },
}

/// The parameters we will be looping through for SVM training
#[derive(Clone, Copy)]
struct Params {
    cost: f64,
    kernel: KernelParams,
}

impl From<Params> for SvmParams<f64, bool> {
    fn from(val: Params) -> Self {
        use KernelParams::{Poly, Rbf};
        let mut re = SvmParams::new().pos_neg_weights(val.cost, val.cost);

        match val.kernel {
            Rbf { gamma } => {
                re = re.gaussian_kernel(1.0 / gamma);
            }
            Poly { degree } => {
                re = re.polynomial_kernel(0.0, degree as f64);
            }
        };

        re
    }
}

impl Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use KernelParams::{Poly, Rbf};
        match self.kernel {
            Rbf { gamma } => write!(f, "rbf   {:5} {:4}", gamma, self.cost),
            Poly { degree } => write!(f, "poly  {:5} {:4}", degree, self.cost),
        }
    }
}

/// Return an iterator which iterates over all combinations of [`Params`] we will be using in this experiment.
fn get_params() -> impl Iterator<Item = Params> {
    const COSTS: [f64; 4] = [0.1, 1.0, 10.0, 100.0];
    const GAMMAS: [f64; 4] = [0.1, 1.0, 10.0, 100.0];
    const DEGREES: [u32; 3] = [1, 2, 3];

    let rbf_params = COSTS
        .iter()
        .copied()
        .cartesian_product(GAMMAS)
        .map(|(cost, gamma)| Params {
            cost,
            kernel: KernelParams::Rbf { gamma },
        });

    let poly_params = COSTS
        .iter()
        .copied()
        .cartesian_product(DEGREES)
        .map(|(cost, degree)| Params {
            cost,
            kernel: KernelParams::Poly { degree },
        });

    rbf_params.chain(poly_params)
}
