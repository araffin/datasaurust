use rand::Rng;
use rand_distr::Normal;

use crate::types::*;

/// Compute the statistics of the data
/// Returns the mean and standard deviation of x and y
///
/// # Example
///
/// ```
/// use datasaurust::types::*;
/// use datasaurust::optim::*;
///
/// let data = Data {
///     x: vec![1.0, 2.0, 3.0, 4.0, 5.0],
///     y: vec![1.0, 2.0, 3.0, 4.0, 5.0],
/// };
///
/// let (mean_x, mean_y, std_x, std_y) = compute_stats(&data);
/// assert_eq!(mean_x, 3.0);
/// assert_eq!(mean_y, 3.0);
/// ```
pub fn compute_stats(data: &Data) -> (f32, f32, f32, f32) {
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

    // unbiased estimator of the variance
    // use n instead of n-1 to have biased estimator
    std_x = (std_x / (data.x.len() - 1) as f32).sqrt();
    std_y = (std_y / (data.y.len() - 1) as f32).sqrt();

    (mean_x, mean_y, std_x, std_y)
}

/// Get the part that doesn't change in the number (until decimals)
/// and the one that does (after decimals)
///
/// # Example
///
/// ```
/// use datasaurust::types::*;
/// use datasaurust::optim::*;
///
/// let (constant_part, variable_part) = get_digits(21.2345, 2, 2);
/// assert_eq!(constant_part, 21.23);
/// assert_eq!(variable_part, 45);
/// ```
pub fn get_digits(number: f32, decimals: i32, n_digits: usize) -> (f32, i32) {
    let constant_part = number * 10.0_f32.powi(decimals);
    let constant_part = constant_part.floor() / 10.0_f32.powi(decimals);
    let variable_part = number - constant_part;
    let variable_part = variable_part * 10.0_f32.powi(decimals + n_digits as i32);
    let digits = variable_part.floor();
    (constant_part, digits as i32)
}

/// Check if the error is within the acceptable bounds
///
/// # Example
///
/// ```
/// use datasaurust::types::*;
/// use datasaurust::optim::*;
///
/// let stat1 = 21.2345;
/// let stat2 = 21.2346;
/// let decimals = 2;
/// let ok = is_error_within_tolerance(stat1, stat2, decimals);
/// assert_eq!(ok, true);
/// let stat1 = 21.231;
/// let stat2 = 21.241;
/// let decimals = 2;
/// let ok = is_error_within_tolerance(stat1, stat2, decimals);
/// assert_eq!(ok, false);
/// ```
pub fn is_error_within_tolerance(stat1: f32, stat2: f32, decimals: i32) -> bool {
    // Floor first to avoid rounding issue when computing the difference
    let stat1 = (stat1 * 10.0_f32.powi(decimals)).floor();
    let stat2 = (stat2 * 10.0_f32.powi(decimals)).floor();
    let diff = (stat1 - stat2).abs();
    diff == 0.0
}

/// Checks to see if the statistics are still within the acceptable bounds
pub fn is_error_still_ok(data1: &Data, data2: &Data, decimals: i32) -> bool {
    let (mean_x1, mean_y1, std_x1, std_y1) = compute_stats(data1);
    let (mean_x2, mean_y2, std_x2, std_y2) = compute_stats(data2);

    let mean_x_ok = is_error_within_tolerance(mean_x1, mean_x2, decimals);
    let mean_y_ok = is_error_within_tolerance(mean_y1, mean_y2, decimals);
    let std_x_ok = is_error_within_tolerance(std_x1, std_x2, decimals);
    let std_y_ok = is_error_within_tolerance(std_y1, std_y2, decimals);

    mean_x_ok && mean_y_ok && std_x_ok && std_y_ok
}

/// Calculate the minimum distance between a point and a line segment
///
/// # Example
///
/// ```
/// use datasaurust::types::*;
/// use datasaurust::optim::*;
///
/// let point = (0.0, 0.0);
/// let line = ((1.0, 1.0), (2.0, 2.0));
/// let distance = min_distance_segment(point, line);
/// assert_eq!(distance, 1.4142135);
/// ```
pub fn min_distance_segment(point: (f32, f32), line: Line) -> f32 {
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
pub fn perturb_data(
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
pub fn read_data(filename: &str) -> Data {
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

/// Port from pytweening
/// A quadratic tween function that accelerates, reaches the midpoint, and then decelerates.
/// a "s-shaped" curve
///
/// # Example
///
/// ```
/// use datasaurust::optim::ease_in_out_quad;
///
/// assert_eq!(ease_in_out_quad(0.0), 0.0);
/// assert_eq!(ease_in_out_quad(0.5), 0.5);
/// assert_eq!(ease_in_out_quad(1.0), 1.0);
/// ```
pub fn ease_in_out_quad(t: f64) -> f64 {
    if t < 0.5 {
        2.0 * t.powi(2)
    } else {
        let tmp = t * 2.0 - 1.0;
        -0.5 * (tmp * (tmp - 2.0) - 1.0)
    }
}
