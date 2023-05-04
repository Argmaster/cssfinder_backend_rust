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


use std::{
    hint::black_box,
    time::{Duration, Instant},
    usize,
};

use ndarray as nd;
use num::Complex;
use num_traits::Float;

use serde::Serialize;
use std::fs;
use std::thread;

const STACK_SIZE: usize = 256 * 1024 * 1024;


pub fn main() {
    // Spawn thread with explicit stack size
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}


#[derive(Serialize)]
struct Data {
    sizes: Vec<usize>,
    results_slow: Vec<u128>,
    results_slow_2: Vec<u128>,
    results_fast: Vec<u128>,
    results_fast_2: Vec<u128>,
    results_fast_3: Vec<u128>,
}


fn run() {
    let mut sizes = Vec::<usize>::new();
    let mut results_slow = Vec::<u128>::new();
    let mut results_slow_2 = Vec::<u128>::new();
    let mut results_fast = Vec::<u128>::new();
    let mut results_fast_2 = Vec::<u128>::new();
    let mut results_fast_3 = Vec::<u128>::new();

    for i in (8..96).step_by(8) {
        let (time_slow, time_slow_2, time_fast, time_fast_2, time_fast_3) =
            run_benchmark_product(i);
        sizes.push(i);
        results_slow.push(time_slow.as_nanos());
        results_slow_2.push(time_slow_2.as_nanos());
        results_fast.push(time_fast.as_nanos());
        results_fast_2.push(time_fast_2.as_nanos());
        results_fast_3.push(time_fast_3.as_nanos());
    }
    println!("{:?}", sizes);
    println!("{:?}", results_slow);
    println!("{:?}", results_slow_2);
    println!("{:?}", results_fast);
    println!("{:?}", results_fast_2);
    println!("{:?}", results_fast_3);

    let data = Data {
        sizes,
        results_slow,
        results_slow_2,
        results_fast,
        results_fast_2,
        results_fast_3,
    };
    let content = serde_json::to_string(&data).unwrap();
    let result = fs::write("results.json", content);
    match result {
        Ok(_) => println!("File written successfully."),
        Err(e) => println!("Error writing file: {}", e),
    }
}


fn run_benchmark_product(
    size: usize,
) -> (Duration, Duration, Duration, Duration, Duration) {
    println!("Matrix size {} ({})", size, size * size * 8);

    let lhs = nd::Array2::<Complex<f64>>::from_elem(
        (size, size),
        Complex::<f64>::new(0f64, 0f64),
    );
    let rhs = nd::Array2::<Complex<f64>>::from_elem(
        (size, size),
        Complex::<f64>::new(0f64, 0f64),
    );
    black_box(&lhs);
    black_box(&rhs);

    let reps = 20_000;

    let time_slow = Instant::now();
    {
        let mut total = 0f64;
        for _ in 0..reps {
            total = total + product_slow(&lhs.view(), &rhs.view());
            black_box(&lhs);
            black_box(&rhs);
            black_box(total);
        }
        black_box(total);
        println!("{}", total);
    }
    let time_slow_e = time_slow.elapsed();

    let time_slow_2 = Instant::now();
    {
        let mut total = 0f64;
        for _ in 0..reps {
            total = total + product_slow_2(&lhs.view(), &rhs.view());
            black_box(&lhs);
            black_box(&rhs);
            black_box(total);
        }
        black_box(total);
        println!("{}", total);
    }
    let time_slow_2_e = time_slow_2.elapsed();

    let time_fast = Instant::now();
    {
        let mut total = 0f64;
        for _ in 0..reps {
            total = total + product_fast(&lhs, &rhs);
            black_box(&lhs);
            black_box(&rhs);
            black_box(total);
        }
        black_box(total);
        println!("{}", total);
    }
    let time_fast_e = time_fast.elapsed();

    let time_fast_2 = Instant::now();
    {
        let mut total = 0f64;
        for _ in 0..reps {
            total = total + product_fast_2(&lhs, &rhs);
            black_box(&lhs);
            black_box(&rhs);
            black_box(total);
        }
        black_box(total);
        println!("{}", total);
    }
    let time_fast_2_e = time_fast_2.elapsed();

    let time_fast_3 = Instant::now();
    {
        let mut total = 0f64;
        for _ in 0..reps {
            total = total + product_fast_3(&lhs, &rhs);
            black_box(&lhs);
            black_box(&rhs);
            black_box(total);
        }
        black_box(total);
        println!("{}", total);
    }
    let time_fast_3_e = time_fast_3.elapsed();

    return (
        time_slow_e / reps,
        time_slow_2_e / reps,
        time_fast_e / reps,
        time_fast_2_e / reps,
        time_fast_3_e / reps,
    );
}


fn product_slow<T>(
    lhs: &nd::ArrayView2<Complex<T>>,
    rhs: &nd::ArrayView2<Complex<T>>,
) -> T
where
    T: Float + 'static,
{
    lhs.dot(rhs).diag().mapv(|x| x.re).sum()
}


fn product_maybe_slow_4<T>(
    lhs: &nd::ArrayView2<Complex<T>>,
    rhs: &nd::ArrayView2<Complex<T>>,
) -> T
where
    T: Float + 'static,
{
    let n = lhs.dim().0;
    let mut sum: T = T::zero();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                sum = sum + (lhs[[i, k]] * rhs[[k, j]]).re;
            }
        }
    }
    sum
}

fn product_slow_2<T>(
    lhs: &nd::ArrayView2<Complex<T>>,
    rhs: &nd::ArrayView2<Complex<T>>,
) -> T
where
    T: Float + 'static,
{
    let n = lhs.dim().0;

    let mut out = nd::Array2::<Complex<T>>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let mut dot_product: Complex<T> = Complex::new(T::zero(), T::zero());
            for k in 0..n {
                dot_product = dot_product + lhs[[i, k]] * rhs[[k, j]];
            }
            out[[i, j]] = dot_product;
        }
    }

    let mut sum: T = T::zero();
    for i in 0..n {
        sum = sum + out[[i, i]].re;
    }

    sum
}


fn product_fast<T>(lhs: &nd::Array2<Complex<T>>, rhs: &nd::Array2<Complex<T>>) -> T
where
    T: Float + std::fmt::Debug + 'static,
{
    let square_matrix_size = lhs.dim().0;
    let mut result = T::zero();

    for i in 0..square_matrix_size {
        for k in 0..square_matrix_size {
            result = result + (lhs[[i, k]] * rhs[[k, i]]).re;
        }
    }
    result
}


fn product_fast_2<T>(lhs: &nd::Array2<Complex<T>>, rhs: &nd::Array2<Complex<T>>) -> T
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let square_matrix_size = lhs.dim().0;
    let mut result = T::zero();

    let a = lhs.map(|v| v.re);
    let b = rhs.map(|v| v.re);

    for i in 0..square_matrix_size {
        for k in 0..square_matrix_size {
            result = result + a[[i, k]] * b[[k, i]];
        }
    }
    result
}


fn product_fast_3<T>(lhs: &nd::Array2<Complex<T>>, rhs: &nd::Array2<Complex<T>>) -> T
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let square_matrix_size = lhs.dim().0;
    let mut result = T::zero();


    for i in 0..square_matrix_size {
        for k in 0..square_matrix_size {
            result = result + lhs[[k, i]].re * rhs[[k, i]].re;
        }
    }
    result
}
