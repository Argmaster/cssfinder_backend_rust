// Copyright 2023 Krzysztof Wisniewski <argmaster.world@gmail.com>
//
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the “Software”), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be included in all copies
// or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


use ndarray::Array2;
use num::Complex;
use serde_derive::Deserialize;
use serde_json;
use std::fs::File;
use std::hint::black_box;
use std::io::prelude::*;
use std::time::Duration;

mod naive;
mod shared;

use naive::RustBackend;
use shared::AlgoMode;

#[derive(Deserialize)]
struct ComplexDict {
    real: f32,
    imag: f32,
}

fn run() {
    // Read the JSON file
    let mut file = File::open("complex_array.json").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Unable to read the file");

    // Deserialize the JSON string
    let complex_dicts: Vec<ComplexDict> = serde_json::from_str(&contents).unwrap();

    // Convert the Vec<ComplexDict> to Vec<Complex<f32>>
    let complex_vec: Vec<Complex<f32>> = complex_dicts
        .into_iter()
        .map(|d| Complex::new(d.real, d.imag))
        .collect();

    // Define the shape of the 2D array
    let nrows = 64; // Adjust this to the desired number of rows
    let ncols = complex_vec.len() / nrows;

    // Create an ndarray::Array2<Complex<f32>> from the Vec<Complex<f32>>
    let complex_array: Array2<Complex<f32>> =
        Array2::from_shape_vec((nrows, ncols), complex_vec).unwrap();

    let mut backend =
        RustBackend::<f32>::new(&complex_array, 2, 6, AlgoMode::FSnQd, 0.4);
    use std::time::Instant;
    let now = Instant::now();
    {
        backend.run_epoch(10000, 0); // 77000
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}

fn run_stack() {
    use num_traits::Zero;
    const N: usize = 64; // Matrix size

    // Define the two matrices using stack-allocated arrays
    let a: [[Complex<f32>; N]; N] = black_box([[Complex::<f32>::new(1.0, 0.0); N]; N]);
    let b: [[Complex<f32>; N]; N] = black_box([[Complex::<f32>::new(0.0, 1.0); N]; N]);

    // Initialize the result matrix with zeros
    let mut result: [[Complex<f32>; N]; N] = [[Complex::<f32>::zero(); N]; N];

    // Perform the matrix multiplication
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result[j][i] = result[j][i] + a[i][k] * b[j][k];
            }
        }
    }

    let mut result2: [[Complex<f32>; N]; N] = [[Complex::<f32>::zero(); N]; N];

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result2[j][i] = result2[j][i] + result[i][k] * b[j][k];
            }
        }
    }

    black_box(result);
}
fn run_heap() {
    use num_traits::Zero;
    const N: usize = 64; // Matrix size

    // Define the two matrices using stack-allocated arrays
    let a: Vec<Vec<Complex<f32>>> =
        black_box(vec![vec![Complex::<f32>::new(1.0, 0.0); N]; N]);
    let b: Vec<Vec<Complex<f32>>> =
        black_box(vec![vec![Complex::<f32>::new(0.0, 1.0); N]; N]);

    // Initialize the result matrix with zeros
    let mut result: [[Complex<f32>; N]; N] = [[Complex::<f32>::zero(); N]; N];

    // Perform the matrix multiplication
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result[j][i] = result[j][i] + a[k][i] * b[k][j];
            }
        }
    }

    let mut result2: [[Complex<f32>; N]; N] = [[Complex::<f32>::zero(); N]; N];

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result2[i][j] = result2[i][j] + result[k][i] * b[k][j];
            }
        }
    }

    black_box(result);
}

fn run_nd() {
    const N: usize = 64; // Matrix size
    let a = ndarray::Array2::<Complex<f32>>::eye(N);
    let b = ndarray::Array2::<Complex<f32>>::eye(N);

    let out = a.dot(&b);
    let out2 = out.dot(&b);
    black_box(out2);
}

fn main() {
    let elapsed = measure(1000, run_stack);
    println!("Stack elapsed: {:.2?}", elapsed);
    let elapsed = measure(1000, run_heap);
    println!("Heap elapsed: {:.2?}", elapsed);
    let elapsed = measure(1000, run_nd);
    println!("ND elapsed: {:.2?}", elapsed);
}

fn measure(reps: u32, operation: fn()) -> Duration {
    use std::time::Instant;
    let now = Instant::now();
    for _ in 0..reps {
        operation();
        black_box(());
    }
    now.elapsed() / reps
}
