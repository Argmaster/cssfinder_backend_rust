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

pub fn normalize<T>(mtx: &nd::ArrayView2<Complex<T>>) -> nd::Array2<Complex<T>>
where
    T: Float + 'static,
{
    let mtx2 = mtx.dot(&mtx.mapv(|x| x.conj()));
    nd::Zip::from(mtx)
        .and(&mtx2)
        .map_collect(|x, y| x / y.re.sqrt())
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
