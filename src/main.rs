// Rust implementation of the algorithm described in the paper:
// Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics
// through Simulated Annealing
// Justin Matejka and George Fitzmaurice
// ACM CHI 2017
// The paper, video, and associated code and datasets can be found on the Autodesk Research website:
// https://www.autodesk.com/research/publications/same-stats-different-graphs
// Interactive demo: https://bqplot.github.io/bqplot-gallery/

use clap::Parser;
use gnuplot::AxesCommon;
use gnuplot::Fix;
use gnuplot::{Caption, Color, Figure, Graph};
use kdam::tqdm;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::Normal;
use std::io::Write;

use datasaurust::shapes::*;
use datasaurust::types::*;
use datasaurust::optim::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the initial dataset
    #[arg(short, long, default_value = "data/seed_datasets/Datasaurus_data.csv")]
    dataset: String,

    // Name of the output file with a default value
    #[arg(short, long, default_value = "output")]
    output: String,

    /// Number of iterations
    #[arg(short, long, default_value_t = 4000000)]
    num_iterations: u32,

    // define a boolean flag to enable plotting
    #[arg(short, long, default_value_t = false)]
    plot: bool,

    // define a boolean flag to save plots
    #[arg(short, long, default_value_t = false)]
    save_plots: bool,

    // define a boolean flag to use uniform sampling
    #[arg(short, long, default_value_t = false)]
    uniform: bool,

    // define a boolean flag to use gaussian sampling
    #[arg(short, long, default_value_t = false)]
    gaussian: bool,

    // log interval
    #[arg(short, long, default_value_t = 10000)]
    log_interval: u32,

    // Number of decimals that are constant
    #[arg(long, default_value_t = 2)]
    decimals: i32,

    // For the plots, number of digits that change
    #[arg(long, default_value_t = 5)]
    n_digits: usize,

    // Min distance allowed between point and line segments
    #[arg(long, default_value_t = 1.0)]
    allowed_distance: f32,

    // Desired shape
    #[arg(long, default_value = "cat")]
    shape: String,

    // Random seed when using Gaussian
    // TODO: make the rest deterministic
    #[arg(long, default_value_t = 42)]
    seed: u64,

    // Min temperature
    #[arg(long, default_value_t = 0.0001)]
    min_temperature: f64,

    // Max temperature
    #[arg(long, default_value_t = 0.4)]
    max_temperature: f64,
}

fn main() {
    // let args = Args::parse_args_default_or_exit();
    let args = Args::parse();

    let num_iterations = args.num_iterations;

    let mut data: Data;
    let offset_x: f32 = -55.33 + 7.04;
    let offset_y: f32 = -50.36 + 20.23;
    // let offset_x: f32 = 0.0;
    // let offset_y: f32 = 0.0;

    // Fixed boundaries for the data
    let x_bounds = (-20.0 + offset_x, 130.0 + offset_x);
    let y_bounds = (-10.0 + offset_y, 145.0 + offset_y);

    if args.uniform {
        println!("Using uniform sampling");

        let n_points = 1000;
        let x_bounds_sample = (20.0, 80.0);
        let y_bounds_sample = (20.0, 80.0);
        // Sample n_points uniformly from the bounds
        let mut rng = rand::thread_rng();

        data = Data {
            x: vec![0.0; n_points],
            y: vec![0.0; n_points],
        };

        for i in 0..n_points {
            data.x[i] = rng.gen_range(x_bounds_sample.0..x_bounds_sample.1);
            data.y[i] = rng.gen_range(y_bounds_sample.0..y_bounds_sample.1);
        }
    } else if args.gaussian {
        println!("Using gaussian sampling");

        let n_points = 800;
        let mean_x = 55.0 + offset_x;
        let mean_y = 50.0 + offset_y;
        let std_x = 16.0;
        let std_y = 20.0;

        data = Data {
            x: vec![0.0; n_points],
            y: vec![0.0; n_points],
        };

        // Sample n points using 2 Gaussians
        // use a fixed seed for reproducibility
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

        let normal_x = Normal::new(mean_x, std_x).unwrap();
        let normal_y = Normal::new(mean_y, std_y).unwrap();

        for i in 0..n_points {
            let x = rng.sample::<f32, _>(normal_x);
            let y = rng.sample::<f32, _>(normal_y);

            // Clip the values to the bounds
            let x = x.max(1.0 + offset_x).min(98.0 + offset_x);
            let y = y.max(1.0 + offset_y).min(98.0 + offset_y);

            data.x[i] = x;
            data.y[i] = y;
        }

        // Print stats
        let (mean_x, mean_y, std_x, std_y) = compute_stats(&data);
        println!("Mean x: {}, Mean y: {}", mean_x, mean_y);
        println!("Std x: {}, Std y: {}", std_x, std_y);

        // let desired_std_y = 19.94;
        // // Loop over the random seed until we get the desired std
        // let mut seed = args.seed;
        // let mut current_std_y = compute_stats(&data).3;

        // while (current_std_y - desired_std_y).abs() > 0.005 {
        //     let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        //     let normal_x = Normal::new(mean_x, std_x).unwrap();
        //     let normal_y = Normal::new(mean_y, std_y).unwrap();

        //     for i in 0..n_points {
        //         let x = rng.sample::<f32, _>(normal_x);
        //         let y = rng.sample::<f32, _>(normal_y);

        //         // Clip the values to the bounds
        //         let x = x.max(1.0 + offset_x).min(98.0 + offset_x);
        //         let y = y.max(1.0 + offset_y).min(98.0 + offset_y);

        //         data.x[i] = x;
        //         data.y[i] = y;
        //     }

        //     current_std_y = compute_stats(&data).3;
        //     println!("std_y: {}", current_std_y);
        //     seed += 1;
        // }
        // println!("Seed: {}", seed - 1);
        // // Exit
        // return;
    } else {
        data = read_data(args.dataset.as_str());
    }

    // Min/Max temperature
    let min_temperature: f64 = args.min_temperature;
    let max_temperature: f64 = args.max_temperature;

    let decimals: i32 = args.decimals;

    // Print info every n iterations
    let log_interval = args.log_interval;
    // For the plots, number of digits that change
    let n_digits = args.n_digits;

    let initial_data = Data {
        x: data.x.clone(),
        y: data.y.clone(),
    };

    // Do a copy of the initial data
    let mut best_data = Data {
        x: data.x.clone(),
        y: data.y.clone(),
    };

    let mut fg = Figure::new();
    let show_plot = args.plot;

    // Create log directory if it doesn't exist
    let log_folder = format!("./logs/{}", args.shape);

    if !std::path::Path::new(&log_folder).exists() {
        std::fs::create_dir(&log_folder).unwrap();
    }

    let fixed_lines = get_shape(args.shape.as_str(), offset_x, offset_y);

    for i in tqdm!(0..num_iterations) {
        // for i in 0..num_iterations {
        // Compute the current temperature using a linear schedule
        // let temperature = max_temperature
        //     - (max_temperature - min_temperature) * (i as f64 / num_iterations as f64);

        // Compute the current temperature using a quadratic schedule
        let temperature = min_temperature
            + (max_temperature - min_temperature)
                * ease_in_out_quad((num_iterations - i) as f64 / num_iterations as f64);

        // Perturb the data
        data = perturb_data(
            &best_data,
            temperature,
            args.allowed_distance,
            &fixed_lines,
            x_bounds,
            y_bounds,
        );

        // Check that after the perturbation the
        // statistics of the data are still within the bounds
        if is_error_still_ok(&data, &initial_data, decimals) {
            best_data = Data {
                x: data.x.clone(),
                y: data.y.clone(),
            };
        }

        // Plot the data using gnuplot
        if i % log_interval == 0 && show_plot {
            fg.clear_axes();
            // Display stats using labels
            let stats = compute_stats(&best_data);

            // retrieve the constant and variable part of the mean
            let (mean_x, x_digits) = get_digits(stats.0, decimals, n_digits);

            // Do the same for the other stats
            let (mean_y, y_digits) = get_digits(stats.1, decimals, n_digits);
            let (std_x, std_x_digits) = get_digits(stats.2, decimals, n_digits);
            let (std_y, std_y_digits) = get_digits(stats.3, decimals, n_digits);

            let n_decimals = decimals as usize;
            let indent = 11 + (decimals as usize);

            let label_x_pos = 0.32;
            let label_y_pos = 0.95;

            fg.axes2d()
                .set_title("Datasaurust", &[])
                .set_legend(Graph(0.5), Graph(0.9), &[], &[])
                .set_x_label("X", &[])
                .set_y_label("Y", &[])
                // set max and min values for the axes
                .set_x_range(Fix(x_bounds.0 as f64), Fix(x_bounds.1 as f64))
                .set_y_range(Fix(y_bounds.0 as f64), Fix(y_bounds.1 as f64))
                .points(
                    best_data.x.iter(),
                    best_data.y.iter(),
                    // change the pointtype and pointsize and opacity
                    &[
                        Caption(""),
                        gnuplot::PointSymbol('O'),
                        gnuplot::PointSize(1.5),
                        Color("black"),
                    ],
                    // &[Caption(""), Color("black")],
                    // &[Caption("Best data"), Color("black")],
                )
                .label(
                    format!("X Mean: {:.decimals$}", mean_x, decimals = n_decimals).as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("black")],
                )
                .label(
                    format!(
                        "{:indent$}{:0<n_digits$}",
                        "",
                        x_digits,
                        indent = indent,
                        n_digits = n_digits
                    )
                    .as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("grey")],
                )
                .label(
                    format!("Y Mean: {:.decimals$}", mean_y, decimals = n_decimals).as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos - 0.06),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("black")],
                )
                .label(
                    format!(
                        "{:indent$}{:0<n_digits$}",
                        "",
                        y_digits,
                        indent = indent,
                        n_digits = n_digits
                    )
                    .as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos - 0.06),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("grey")],
                )
                .label(
                    format!("X SD  : {:.decimals$}", std_x, decimals = n_decimals).as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos - 0.12),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("black")],
                )
                .label(
                    format!(
                        "{:indent$}{:0<n_digits$}",
                        "",
                        std_x_digits,
                        indent = indent,
                        n_digits = n_digits
                    )
                    .as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos - 0.12),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("grey")],
                )
                .label(
                    format!("Y SD  : {:.decimals$}", std_y, decimals = n_decimals).as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos - 0.18),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("black")],
                )
                .label(
                    format!(
                        "{:indent$}{:0<n_digits$}",
                        "",
                        std_y_digits,
                        indent = indent,
                        n_digits = n_digits
                    )
                    .as_str(),
                    Graph(label_x_pos),
                    Graph(label_y_pos - 0.18),
                    &[gnuplot::Font("Monospace", 16.), gnuplot::TextColor("grey")],
                );

            if args.save_plots {
                let frame_idx: u32 = i / log_interval;
                let frame_name = format!("{}/{:0>6}.png", log_folder, frame_idx);
                fg.save_to_png(&frame_name, 640, 480).unwrap();
            } else {
                fg.show_and_keep_running().unwrap();
            }
        }

        // Print the data statistic every n iterations
        // if i % log_interval == 0 {
        //     let stats = compute_stats(&best_data);
        //     println!(
        //         "Iteration: {}, Temperature: {}, Mean: ({}, {}), Std: ({}, {})",
        //         i, temperature, stats.0, stats.1, stats.2, stats.3,
        //     );
        // }
    }

    // Write the best data to a csv file
    let mut output = std::fs::File::create(format!("{}/{}.csv", log_folder, args.output)).unwrap();
    for (x, y) in best_data.x.iter().zip(best_data.y.iter()) {
        writeln!(output, "{},{}", x, y).unwrap();
    }
}
