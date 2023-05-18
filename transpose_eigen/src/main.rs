#![feature(try_blocks)]
use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use clap::Parser;
use ndarray::Array2;
use ndarray_csv::{Array2Reader, Array2Writer};

// A list of files which should be read in and transposed
const TARGETS: [&str; 9] = [
    "trPCA_01.txt",
    "trPCA_02.txt",
    "trPCA_03.txt",
    "tsPCA_01.txt",
    "tsPCA_02.txt",
    "tsPCA_03.txt",
    "valPCA_01.txt",
    "valPCA_02.txt",
    "valPCA_03.txt",
];

// TODO: Update this doc comment when we eventually get the ability to put macros into attributes, so it lists the targets automatically
#[derive(Parser)]
/// Fixes some of the originally given input files containing eigen coefficients to be in row-order rather than in column-order.
struct Args {
    /// The directory containing all of the files needing to be transposed
    root_path: PathBuf,
    /// The directory where the output files will be written to
    out_path: PathBuf,
}

fn main() -> Result<()> {
    // Get command line arguments
    let args = Args::parse();

    // Make sure output directory exists
    fs::create_dir_all(&args.out_path)?;

    // Transpose each file
    TARGETS.iter().for_each(|t| transpose_file(t, &args));

    Ok(())
}

/// Load a single csv file, transpose it, and output it as another csv file. `target` is the name of the file, and it should reside in `args.root_path`. A new file of the same name will be created in `args.out_path`, which must already exist.
/// It is assumed that the csv file is space-separated and has no headers.
fn transpose_file(target: impl AsRef<Path>, args: &Args) {
    let in_path = args.root_path.join(&target);

    // Load the input file into an array
    let data: Result<Array2<f32>> = try {
        csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b' ')
            .from_path(&in_path)?
            .deserialize_array2_dynamic::<f32>()?
    };

    let data =
        data.unwrap_or_else(|e| panic!("Couldn't read data from `{}`:\n{e}", in_path.display()));

    // Transpose
    let data = data.t();
    // Must be converted into the standard memory layout, since transposing changes that, and serialize_array2 expects it
    let data = data.as_standard_layout().into_owned();

    let out_path = args.out_path.join(&target);

    // Output the resulting array
    let result: Result<()> = try {
        csv::WriterBuilder::new()
            .has_headers(false)
            .delimiter(b' ')
            .from_path(&out_path)?
            .serialize_array2(&data)?
    };

    result.unwrap_or_else(|e| panic!("Couldn't write data to `{}`:\n{e}", out_path.display()));
}
