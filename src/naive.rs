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

use ndarray as nd;
use ndarray_rand::rand_distr::StandardNormal;
use num::Complex;
use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub fn product<T>(
    lhs: &nd::ArrayView2<Complex<T>>,
    rhs: &nd::ArrayView2<Complex<T>>,
) -> T
where
    T: Float + 'static,
{
    lhs.dot(rhs).diag().mapv(|x| x.re).sum()
}

pub fn normalize<T>(vec: &nd::ArrayView1<Complex<T>>) -> nd::Array1<Complex<T>>
where
    T: Float + 'static,
{
    let divisor = vec.dot(&vec.mapv(|x| x.conj())).re.sqrt();
    nd::Zip::from(vec).map_collect(|x| x / divisor)
}

pub fn project<T>(a: &nd::ArrayView1<Complex<T>>) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
{
    let length = a.len();
    let b = a.mapv(|x| x.conj()).into_shape((length, 1)).unwrap();
    // .zip(mtx1_conj.outer_iter()).map_collect(|(x, y)| x * y)
    let a1 = a.into_shape((1, length)).unwrap();

    b.dot(&a1).reversed_axes()
}

pub fn kronecker<T>(
    a: &nd::ArrayView2<Complex<T>>,
    b: &nd::ArrayView2<Complex<T>>,
) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
{
    let ddd1 = a.dim().0;
    let ddd2 = b.dim().0;

    let output_shape = (ddd1 * ddd2, ddd1 * ddd2);

    let mut out_mtx = nd::Array::zeros((ddd1, ddd2, ddd1, ddd2));

    for ((i1, j1), x) in a.indexed_iter() {
        for ((i2, j2), y) in b.indexed_iter() {
            out_mtx[[i1, i2, j1, j2]] = x * y;
        }
    }

    out_mtx.into_shape(output_shape).unwrap()
}

pub fn rotate<T>(
    rho2: &nd::ArrayView2<Complex<T>>,
    unitary: &nd::ArrayView2<Complex<T>>,
) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
{
    let unitary_conj_transpose = unitary.mapv(|x| x.conj()).reversed_axes();
    let rho2a = rho2.dot(&unitary_conj_transpose);
    unitary.dot(&rho2a)
}

pub fn get_random_haar_1d<T>(depth: usize) -> nd::Array1<Complex<T>>
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    let normal =
        Normal::<T>::new(T::from(0.0).unwrap(), T::from(1.0).unwrap()).unwrap();

    let rng_real = rand::thread_rng();
    let real = rng_real
        .sample_iter(&normal)
        .take(depth)
        .collect::<Vec<T>>();

    let rng_imaginary = rand::thread_rng();
    let imaginary = rng_imaginary
        .sample_iter(&normal)
        .take(depth)
        .collect::<Vec<T>>();

    let out = real
        .into_iter()
        .zip(imaginary.into_iter())
        .map(|(r, i)| Complex::new(r, i));

    nd::Array1::from_iter(out)
}

pub fn get_random_haar_2d<T>(depth: usize, quantity: usize) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    let normal =
        Normal::<T>::new(T::from(0.0).unwrap(), T::from(1.0).unwrap()).unwrap();

    let rng_real = rand::thread_rng();
    let real = rng_real
        .sample_iter(&normal)
        .take(quantity * depth)
        .collect::<Vec<T>>();

    let real_array = nd::Array::from_shape_vec((quantity, depth), real).unwrap();

    let rng_imaginary = rand::thread_rng();
    let imaginary = rng_imaginary
        .sample_iter(&normal)
        .take(quantity * depth)
        .collect::<Vec<T>>();

    let imaginary_array =
        nd::Array::from_shape_vec((quantity, depth), imaginary).unwrap();

    let mut out = nd::Array::zeros((quantity, depth));

    nd::Zip::from(&mut out)
        .and(&real_array)
        .and(&imaginary_array)
        .for_each(|cell, &r, &i| *cell = Complex::new(r, i));

    out
}

pub fn expand_d_fs<T>(
    value: &nd::ArrayView2<Complex<T>>,
    depth: usize,
    quantity: usize,
    idx: usize,
) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
{
    let depth_1 = depth.pow(idx as u32);
    let identity_1 = nd::Array::eye(depth_1);

    let depth_2 = depth.pow((quantity - idx - 1) as u32);
    let identity_2 = nd::Array::eye(depth_2);

    let kronecker_1 = kronecker(&identity_1.view(), &value.view());
    let kronecker_2 = kronecker(&kronecker_1.view(), &identity_2.view());

    kronecker_2
}

pub fn random_unitary_d_fs<T>(
    depth: usize,
    quantity: usize,
    idx: usize,
) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    let value = _random_unitary_d_fs::<T>(depth);
    let mtx = expand_d_fs(&value.view(), depth, quantity, idx);
    mtx
}

pub fn _random_unitary_d_fs<T>(depth: usize) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    let random_mtx = get_random_haar_2d(depth, 1);
    let identity_mtx = nd::Array2::<Complex<T>>::eye(depth);

    let value = _value::<T>();
    let rand_mul = random_mtx.mapv(|x| value * x);

    rand_mul + identity_mtx
}

pub fn _value<T>() -> Complex<T>
where
    T: Float + 'static,
{
    use std::f64::consts::PI;

    let real = T::from(0.01 * PI).unwrap();
    let imaginary = T::from(0.01 * PI).unwrap();

    Complex::new(real.cos(), imaginary.sin() - T::one())
}

pub fn random_d_fs<T>(depth: usize, quantity: usize) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    let rand_vectors = get_random_haar_2d::<T>(depth, quantity);

    let mut vector = normalize(&rand_vectors.slice(nd::s![0, ..]).view())
        .into_shape((depth, 1))
        .unwrap();
    let mut width = depth;

    for i in 1..quantity {
        let idx_vector = normalize(&rand_vectors.slice(nd::s![i, ..]).view())
            .into_shape((1, depth))
            .unwrap();

        width *= depth;

        // dot on column vector and row vector gives a matrix.
        let outer_product = vector.dot(&idx_vector);

        // flatten into vector, with every iteration vector gets longer.
        vector = outer_product.into_shape((width, 1)).unwrap();
    }

    project(&vector.view().into_shape((width,)).unwrap())
}
