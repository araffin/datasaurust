// Rust implementation of the algorithm described in the paper:
// Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics
// through Simulated Annealing
// Justin Matejka and George Fitzmaurice
// ACM CHI 2017
// The paper, video, and associated code and datasets can be found on the Autodesk Research website:
// http://www.autodeskresearch.com/papers/samestats

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

    // let x1x2 = x1 - x2;
    // let y1y2 = y1 - y2;

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
#[allow(dead_code)]
fn ease_in_out_quad(t: f64) -> f64 {
    if t < 0.5 {
        2.0 * t.powi(2)
    } else {
        let tmp = t.powi(2) - 1.0;
        -0.5 * (tmp * (tmp - 2.0) - 1.0)
    }
}

fn main() {
    let num_iterations = 4000000;
    let mut data = read_data("../seed_datasets/Datasaurus_data.csv");

    // Min/Max temperature
    let min_temperature = 0.0001;
    let max_temperature = 0.5;

    let decimals = 2;

    // Print info every 10000 iterations
    let log_interval = 10000;

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
    let show_plot = true;

    for i in tqdm!(0..num_iterations) {
        // Compute the current temperature using a linear schedule
        let temperature = max_temperature
            - (max_temperature - min_temperature) * (i as f64 / num_iterations as f64);

        // Compute the current temperature using a quadratic schedule
        // let temperature = min_temperature
        //     + (max_temperature - min_temperature) * ease_in_out_quad(i as f64 / num_iterations as f64);

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
                    &[Caption(""), Color("black")],
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
    let mut output = std::fs::File::create("../logs/best_data.csv").unwrap();
    for (x, y) in best_data.x.iter().zip(best_data.y.iter()) {
        writeln!(output, "{},{}", x, y).unwrap();
    }
}
