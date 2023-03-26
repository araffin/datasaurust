// Rust implementation of the algorithm described in the paper:
// Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics
// through Simulated Annealing
// Justin Matejka and George Fitzmaurice
// ACM CHI 2017
// The paper, video, and associated code and datasets can be found on the Autodesk Research website:
// http://www.autodeskresearch.com/papers/samestats

use clap::Parser;
use gnuplot::AxesCommon;
use gnuplot::Fix;
use gnuplot::{Caption, Color, Figure, Graph};
use kdam::tqdm;
use rand::Rng;
use rand_distr::Normal;
use std::io::Write;

// Struct that's going to represent the data
struct Data {
    x: Vec<f32>,
    y: Vec<f32>,
}

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

    std_x = std_x.sqrt();
    std_y = std_y.sqrt();

    (mean_x, mean_y, std_x, std_y)
}

// checks to see if the statistics are still within the acceptable bounds
fn is_error_still_ok(data1: &Data, data2: &Data, decimals: i32) -> bool {
    let (mean_x1, mean_y1, std_x1, std_y1) = compute_stats(data1);
    let (mean_x2, mean_y2, std_x2, std_y2) = compute_stats(data2);

    // Compute the difference between the two sets of statistics
    let mean_x_diff = (mean_x1 - mean_x2).abs();
    let mean_y_diff = (mean_y1 - mean_y2).abs();
    let std_x_diff = (std_x1 - std_x2).abs();
    let std_y_diff = (std_y1 - std_y2).abs();

    // Round the values to the specified number of decimals
    let mean_x_diff = (mean_x_diff * 10.0_f32.powi(decimals)).round() / 10.0_f32.powi(decimals);
    let mean_y_diff = (mean_y_diff * 10.0_f32.powi(decimals)).round() / 10.0_f32.powi(decimals);
    let std_x_diff = (std_x_diff * 10.0_f32.powi(decimals)).round() / 10.0_f32.powi(decimals);
    let std_y_diff = (std_y_diff * 10.0_f32.powi(decimals)).round() / 10.0_f32.powi(decimals);

    if mean_x_diff == 0.0 && mean_y_diff == 0.0 && std_x_diff == 0.0 && std_y_diff == 0.0 {
        return true;
    }

    false
}

// Calculate the minimum distance between a point and a line segment
fn min_distance_segment(point: (f32, f32), line: ((f32, f32), (f32, f32))) -> f32 {
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
fn perturb_data(data: &Data, temperature: f64) -> Data {
    // Create a new data struct to store the perturbed data
    let mut new_data = Data {
        x: data.x.clone(),
        y: data.y.clone(),
    };

    // Fixed boundaries for the data
    let x_bounds = (0.0, 100.0);
    let y_bounds = (0.0, 100.0);

    // Standard deviation for the gaussian noise
    let std_dev = 0.1;

    let allowed_distance = 2.0;

    // Choose a random point to perturb
    let index: usize = rand::thread_rng().gen_range(0..data.x.len());

    // This is the simulated annealing step
    // Allow the point to move further away from the line
    // if the temperature is high
    let allow_worse_objective = rand::thread_rng().gen_bool(temperature);

    // Segments defining a cross
    // let fixed_lines = vec![((20.0, 0.0), (100.0, 100.0)), ((20.0, 100.0), (100.0, 0.0))];

    // Segments defining a cat
    let fixed_lines = vec![
        ((34.84, 87.62), (43.20, 73.93)),
        ((43.20, 73.93), (56.88, 76.43)),
        ((56.88, 76.43), (70.32, 73.57)),
        ((70.32, 73.57), (81.84, 86.07)),
        ((81.84, 86.07), (86.64, 56.79)),
        ((86.64, 56.79), (85.92, 33.21)),
        ((85.92, 33.21), (74.88, 10.00)),
        ((74.88, 10.00), (60.24, 3.57)),
        ((60.24, 3.57), (42.48, 7.50)),
        ((42.48, 7.50), (35.04, 17.86)),
        ((35.04, 17.86), (29.52, 31.07)),
        ((29.52, 31.07), (28.56, 48.21)),
        ((28.56, 48.21), (29.04, 68.93)),
        ((29.04, 68.93), (34.84, 87.62)),
        ((57.88, 36.55), (56.64, 27.86)),
        ((56.64, 27.86), (50.16, 22.86)),
        ((50.16, 22.86), (45.12, 28.21)),
        ((57.16, 29.40), (60.96, 23.57)),
        ((60.96, 23.57), (66.96, 23.93)),
        ((66.96, 23.93), (70.56, 27.50)),
        ((46.71, 58.59), (46.70, 54.19)),
        ((46.70, 54.19), (46.71, 58.59)),
        ((66.85, 57.97), (66.83, 54.19)),
        ((66.83, 54.19), (66.85, 57.97)),
    ];

    // Segments defining a cat silhouette
    // let fixed_lines = vec![
    //     ((22.87, 46.36), (27.94, 63.86)),
    //     ((27.94, 63.86), (38.84, 80.95)),
    //     ((38.84, 80.95), (49.81, 93.42)),
    //     ((49.81, 93.42), (59.57, 95.59)),
    //     ((59.57, 95.59), (58.88, 85.30)),
    //     ((58.88, 85.30), (56.08, 72.34)),
    //     ((56.08, 72.34), (54.40, 68.27)),
    //     ((67.25, 87.29), (80.10, 93.59)),
    //     ((80.10, 93.59), (75.19, 72.98)),
    //     ((75.19, 72.98), (78.51, 66.20)),
    //     ((78.51, 66.20), (79.22, 43.04)),
    //     ((79.22, 43.04), (80.13, 38.58)),
    //     ((80.13, 38.58), (82.26, 34.38)),
    //     ((82.26, 34.38), (81.62, 29.12)),
    //     ((81.62, 29.12), (71.70, 18.02)),
    //     ((71.70, 18.02), (56.09, 19.75)),
    //     ((60.97, 81.55), (75.19, 72.98)),
    //     ((59.56, 19.27), (61.48, 1.01)),
    // ];

    // Segments defining a dog
    // let fixed_lines = vec![
    //     ((33.94, 68.38), (32.23, 59.31)),
    //     ((32.23, 59.31), (31.99, 43.60)),
    //     ((31.99, 43.60), (34.82, 31.32)),
    //     ((34.82, 31.32), (40.61, 22.52)),
    //     ((40.61, 22.52), (50.18, 17.21)),
    //     ((50.18, 17.21), (63.24, 17.60)),
    //     ((63.24, 17.60), (74.58, 24.78)),
    //     ((74.58, 24.78), (79.28, 35.03)),
    //     ((79.28, 35.03), (82.28, 33.21)),
    //     ((82.28, 33.21), (87.30, 35.58)),
    //     ((87.30, 35.58), (89.49, 43.97)),
    //     ((89.49, 43.97), (89.47, 57.12)),
    //     ((89.47, 57.12), (85.84, 70.72)),
    //     ((85.84, 70.72), (76.61, 82.35)),
    //     ((76.61, 82.35), (63.18, 87.43)),
    //     ((63.18, 87.43), (40.34, 86.04)),
    //     ((40.34, 86.04), (28.44, 72.76)),
    //     ((28.44, 72.76), (23.65, 57.93)),
    //     ((23.65, 57.93), (23.18, 42.47)),
    //     ((23.18, 42.47), (26.20, 36.53)),
    //     ((26.20, 36.53), (30.63, 33.14)),
    //     ((30.63, 33.14), (34.53, 33.71)),
    //     ((56.60, 43.65), (56.22, 35.58)),
    //     ((56.22, 35.58), (50.38, 29.80)),
    //     ((50.38, 29.80), (43.65, 31.80)),
    //     ((43.65, 31.80), (41.89, 39.84)),
    //     ((56.22, 35.58), (61.84, 29.71)),
    //     ((61.84, 29.71), (67.63, 30.86)),
    //     ((67.63, 30.86), (70.33, 35.56)),
    //     ((70.33, 35.56), (71.07, 41.55)),
    //     ((45.03, 60.05), (45.16, 54.23)),
    //     ((67.52, 59.27), (67.52, 54.31)),
    //     ((79.28, 35.03), (80.61, 55.40)),
    //     ((80.61, 55.40), (76.70, 69.06)),
    // ];

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

    // define a boolean flag to use uniform sampling
    #[arg(short, long, default_value_t = false)]
    uniform: bool,

    // define a boolean flag to use gaussian sampling
    #[arg(short, long, default_value_t = false)]
    gaussian: bool,

    // log interval
    #[arg(short, long, default_value_t = 10000)]
    log_interval: u32,

    // Number of decimals
    #[arg(long, default_value_t = 2)]
    decimals: i32,
}

fn main() {
    // let args = Args::parse_args_default_or_exit();
    let args = Args::parse();

    let num_iterations = args.num_iterations;

    let mut data: Data;

    if args.uniform {
        println!("Using uniform sampling");

        let n_points = 1000;
        let x_bounds = (20.0, 80.0);
        let y_bounds = (20.0, 80.0);
        // Sample n_points uniformly from the bounds
        let mut rng = rand::thread_rng();

        data = Data {
            x: vec![0.0; n_points],
            y: vec![0.0; n_points],
        };

        for i in 0..n_points {
            data.x[i] = rng.gen_range(x_bounds.0..x_bounds.1);
            data.y[i] = rng.gen_range(y_bounds.0..y_bounds.1);
        }
    } else if args.gaussian {
        println!("Using gaussian sampling");

        let n_points = 800;
        let mean_x = 55.0;
        let mean_y = 50.0;
        let std_x = 16.0;
        let std_y = 20.0;

        data = Data {
            x: vec![0.0; n_points],
            y: vec![0.0; n_points],
        };

        // Sample n points using 2 Gaussians
        let mut rng = rand::thread_rng();
        let normal_x = Normal::new(mean_x, std_x).unwrap();
        let normal_y = Normal::new(mean_y, std_y).unwrap();

        for i in 0..n_points {
            let x = rng.sample::<f32, _>(normal_x);
            let y = rng.sample::<f32, _>(normal_y);

            // Clip the values to the bounds
            let x = x.max(1.0).min(98.0);
            let y = y.max(1.0).min(98.0);

            data.x[i] = x;
            data.y[i] = y;
        }
    } else {
        data = read_data(args.dataset.as_str());
    }

    // Min/Max temperature
    let min_temperature: f64 = 0.00001;
    let max_temperature: f64 = 0.5;

    let decimals: i32 = args.decimals;

    // Print info every n iterations
    let log_interval = args.log_interval;

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
        data = perturb_data(&best_data, temperature);

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
            fg.axes2d()
                .set_title("Datasaurus", &[])
                .set_legend(Graph(0.5), Graph(0.9), &[], &[])
                .set_x_label("X", &[])
                .set_y_label("Y", &[])
                // set max and min values for the axes
                .set_x_range(Fix(0.0), Fix(100.0))
                .set_y_range(Fix(0.0), Fix(100.0))
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
                );
            fg.show_and_keep_running().unwrap();
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
    let mut output = std::fs::File::create(format!("./logs/{}.csv", args.output)).unwrap();
    for (x, y) in best_data.x.iter().zip(best_data.y.iter()) {
        writeln!(output, "{},{}", x, y).unwrap();
    }
}
