fn part1() {
    // Parse the input
    let input = std::fs::read_to_string("input.txt").unwrap();

    let mut max_calories = 0;
    let mut current_calories = 0;

    for line in input.lines() {
        // Try to parse an int from the line
        // and catch the error otherwise
        match line.parse::<i32>() {
            Ok(n) => {
                current_calories += n;
            }
            Err(_) => {
                if current_calories >= max_calories {
                    max_calories = current_calories;
                }
                current_calories = 0;
                continue;
            }
        };
    }
    // Print the max_calories
    println!("max_calories={}", max_calories);
}

fn part2() {
    // Parse the input
    let input = std::fs::read_to_string("input.txt").unwrap();

    let mut current_calories = 0;
    // Create a vector of integers to store the calories
    let mut calories: Vec<i32> = Vec::new();

    for line in input.lines() {
        // Try to parse an int from the line
        // and catch the error otherwise
        match line.parse::<i32>() {
            Ok(n) => {
                current_calories += n;
            }
            Err(_) => {
                calories.push(current_calories);
                current_calories = 0;
                continue;
            }
        };
    }
    // Sort the calories vector
    calories.sort();
    // Print the top 3
    println!(
        "top 3: {}, {}, {}",
        calories[calories.len() - 1],
        calories[calories.len() - 2],
        calories[calories.len() - 3]
    );

    // Sum the top 3
    let sum = calories.iter().rev().take(3).sum::<i32>();
    println!("sum={}", sum);
}

fn main() {
    part1();
    part2();
}
