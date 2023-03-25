use gnuplot::AxesCommon;
use gnuplot::{Caption, Color, Figure, Graph};
use kdam::tqdm;
use rand::Rng;
use rand_distr::Normal;

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
fn perturb_data(data: &Data, temperature: f32) -> Data {
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
    let allow_worse_objective = rand::thread_rng().gen_bool(temperature as f64);

    // List of fixed segments that defines a triangle
    let fixed_lines = vec![
        ((0.0, 0.0), (50.0, 100.0)),
        ((50.0, 100.0), (100.0, 100.0)),
        ((100.0, 100.0), (0.0, 0.0)),
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
fn read_data() -> Data {
    // Parse the csv file
    let input = std::fs::read_to_string("../seed_datasets/Datasaurus_data.csv").unwrap();

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

fn main() {
    let num_iterations = 1000000;
    let mut data = read_data();

    // Min/Max temperature
    let min_temperature = 0.0001;
    let max_temperature = 0.4;

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
    let show_plot = false;

    for i in tqdm!(0..num_iterations) {
        // Compute the current temperature using a linear schedule
        let temperature = max_temperature
            - (max_temperature - min_temperature) * (i as f32 / num_iterations as f32);

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

        // Plot the data every 1000 iterations
        // using gnuplot
        if i % log_interval == 0 && show_plot {
            fg.clear_axes();
            fg.axes2d()
                .set_title("Datasaurus", &[])
                .set_legend(Graph(0.5), Graph(0.9), &[], &[])
                .set_x_label("X", &[])
                .set_y_label("Y", &[])
                .points(
                    best_data.x.iter(),
                    best_data.y.iter(),
                    &[Caption("Best data"), Color("black")],
                );
            fg.show_and_keep_running().unwrap();
        }
    }

    // Write the best data to a csv file
    let mut output = std::fs::File::create("./logs/best_data.csv").unwrap();
    for (x, y) in best_data.x.iter().zip(best_data.y.iter()) {
        writeln!(output, "{},{}", x, y).unwrap();
    }
}
