// Struct that's going to represent the data
#[derive(Debug, PartialEq, Clone)]
pub struct Data {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

/// Struct that represents a segment of a line
pub type Line = ((f32, f32), (f32, f32));
