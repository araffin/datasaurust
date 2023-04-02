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

// Compute statistics for the data
fn compute_stats(data: &Data) -> (f32, f32, f32, f32) {
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut std_x = 0.0;
    let mut std_y = 0.0;

    for i in 0..data.x.len() {
        mean_x += data.x[i];
        mean_y += data.y[i];
    }

    mean_x /= data.x.len() as f32;
    mean_y /= data.y.len() as f32;

    for i in 0..data.x.len() {
        std_x += (data.x[i] - mean_x).powi(2);
        std_y += (data.y[i] - mean_y).powi(2);
    }

    std_x = (std_x / data.x.len() as f32).sqrt();
    std_y = (std_y / data.y.len() as f32).sqrt();

    (mean_x, mean_y, std_x, std_y)
}

// Get the part that doesn't change in the number (until decimals)
// and the one that does (after decimals)
fn get_digits(number: f32, decimals: i32, n_digits: usize) -> (f32, i32) {
    let constant_part = number * 10.0_f32.powi(decimals);
    let constant_part = constant_part.floor() / 10.0_f32.powi(decimals);
    let variable_part = number - constant_part;
    let variable_part = variable_part * 10.0_f32.powi(decimals + n_digits as i32);
    let digits = variable_part.floor();
    (constant_part, digits as i32)
}

fn is_error_within_tolerance(stat1: f32, stat2: f32, decimals: i32) -> bool {
    // Floor first to avoid rounding issue when computing the difference
    let stat1 = (stat1 * 10.0_f32.powi(decimals)).floor();
    let stat2 = (stat2 * 10.0_f32.powi(decimals)).floor();
    let diff = (stat1 - stat2).abs();
    diff == 0.0
}

// checks to see if the statistics are still within the acceptable bounds
fn is_error_still_ok(data1: &Data, data2: &Data, decimals: i32) -> bool {
    let (mean_x1, mean_y1, std_x1, std_y1) = compute_stats(data1);
    let (mean_x2, mean_y2, std_x2, std_y2) = compute_stats(data2);

    let mean_x_ok = is_error_within_tolerance(mean_x1, mean_x2, decimals);
    let mean_y_ok = is_error_within_tolerance(mean_y1, mean_y2, decimals);
    let std_x_ok = is_error_within_tolerance(std_x1, std_x2, decimals);
    let std_y_ok = is_error_within_tolerance(std_y1, std_y2, decimals);

    mean_x_ok && mean_y_ok && std_x_ok && std_y_ok
}

// Calculate the minimum distance between a point and a line segment
fn min_distance_segment(point: (f32, f32), line: Line) -> f32 {
    let (x1, y1) = line.0;
    let (x2, y2) = line.1;
    let (x0, y0) = point;

    // Calculate the distance between the point and the line
    let numerator = ((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1).abs();
    let denominator = ((y2 - y1).powi(2) + (x2 - x1).powi(2)).sqrt();

    // If the point is outside the line segment, return the distance to the closest endpoint
    let distance = numerator / denominator;

    let (x1, y1) = line.0;
    let (x2, y2) = line.1;
    let (x0, y0) = point;

    let x1x0 = x1 - x0;
    let x2x0 = x2 - x0;
    let y1y0 = y1 - y0;
    let y2y0 = y2 - y0;

    // If the point is inside the line segment, return the distance to the line
    if x1x0 * x2x0 < 0.0 || y1y0 * y2y0 < 0.0 {
        return distance;
    }

    // If the point is outside the line segment, return the distance to the closest endpoint
    let d1 = (x1x0.powi(2) + y1y0.powi(2)).sqrt();
    let d2 = (x2x0.powi(2) + y2y0.powi(2)).sqrt();

    if d1 < d2 {
        d1
    } else {
        d2
    }
}

// This function does one round of perturbation
// using simulated annealing
fn perturb_data(
    data: &Data,
    temperature: f64,
    allowed_distance: f32,
    fixed_lines: &[Line],
    x_bounds: (f32, f32),
    y_bounds: (f32, f32),
) -> Data {
    // Create a new data struct to store the perturbed data
    let mut new_data = Data {
        x: data.x.clone(),
        y: data.y.clone(),
    };

    // Standard deviation for the gaussian noise
    let std_dev = 0.1;

    // Choose a random point to perturb
    let index: usize = rand::thread_rng().gen_range(0..data.x.len());

    // This is the simulated annealing step
    // Allow the point to move further away from the line
    // if the temperature is high
    let allow_worse_objective = rand::thread_rng().gen_bool(temperature);

    // Compute the distance too all segments and
    // find the minimum distance
    let min_distance_old = fixed_lines
        .iter()
        .map(|line| min_distance_segment((new_data.x[index], new_data.y[index]), *line))
        .fold(f32::INFINITY, |acc, x| acc.min(x));

    let normal = Normal::new(0.0, std_dev).unwrap();

    loop {
        // perturb the x and y coordinates of the point
        // using gaussian noise
        let delta_x = rand::thread_rng().sample::<f32, _>(normal);
        let delta_y = rand::thread_rng().sample::<f32, _>(normal);

        let x = new_data.x[index] + delta_x;
        let y = new_data.y[index] + delta_y;

        // Compute min distance for the new point
        let min_distance_new = fixed_lines
            .iter()
            .map(|line| min_distance_segment((x, y), *line))
            .fold(f32::INFINITY, |acc, x| acc.min(x));

        let in_bounds = x >= x_bounds.0 && x <= x_bounds.1 && y >= y_bounds.0 && y <= y_bounds.1;
        // Check if the new point is close enough to the line
        let close_enough = min_distance_new < allowed_distance;

        // Check if the new distance is smaller than the old distance
        // or if the temperature is high enough to allow worse objective
        if (min_distance_new < min_distance_old || allow_worse_objective || close_enough)
            && in_bounds
        {
            new_data.x[index] = x;
            new_data.y[index] = y;
            return new_data;
        }
    }
}

// Function that reads the data from the csv file
fn read_data(filename: &str) -> Data {
    // Parse the csv file
    let input = std::fs::read_to_string(filename).unwrap();

    let initial_data = input
        .lines()
        .map(|line| {
            let mut iter = line.split(',');
            let x = iter.next().unwrap().parse::<f32>().unwrap();
            let y = iter.next().unwrap().parse::<f32>().unwrap();
            (x, y)
        })
        .collect::<Vec<(f32, f32)>>();

    // Convert the data into a Data struct
    Data {
        x: initial_data.iter().map(|(x, _)| *x).collect(),
        y: initial_data.iter().map(|(_, y)| *y).collect(),
    }

    // Create a new plot using gnuplot
    // let mut fg = Figure::new();
    // fg.axes2d()
    //     .set_title("Datasaurus", &[])
    //     .set_legend(Graph(0.5), Graph(0.9), &[], &[])
    //     .set_x_label("X", &[])
    //     .set_y_label("Y", &[])
    //     .points(
    //         initial_data.iter().map(|(x, _)| *x),
    //         initial_data.iter().map(|(_, y)| *y),
    //         &[Caption("Initial data"), Color("black")],
    //     );
    // fg.show().unwrap();
}

// Port from pytweening
// A quadratic tween function that accelerates, reaches the midpoint, and then decelerates.
// a "s-shaped" curve
fn ease_in_out_quad(t: f64) -> f64 {
    if t < 0.5 {
        2.0 * t.powi(2)
    } else {
        let tmp = t * 2.0 - 1.0;
        -0.5 * (tmp * (tmp - 2.0) - 1.0)
    }
}

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
                    format!("X Mean: 0{:.decimals$}", mean_x, decimals = n_decimals).as_str(),
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
