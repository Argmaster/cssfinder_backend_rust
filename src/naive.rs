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

use std::any;
use std::f64::consts::PI;
use std::fmt;
use std::ops::Sub;

use ndarray as nd;
use num::Complex;
use num_traits::Float;
use rand::Rng;

use crate::shared::AlgoMode;

pub fn product<T>(lhs: &nd::Array2<Complex<T>>, rhs: &nd::Array2<Complex<T>>) -> T
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
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

pub fn normalize<T>(vec: &nd::Array1<Complex<T>>) -> nd::Array1<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let divisor = vec.dot(&vec.mapv(|x| x.conj())).re.sqrt();
    nd::Zip::from(vec).map_collect(|x| x / divisor)
}

pub fn project<T>(a: &nd::Array1<Complex<T>>) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let length = a.len();
    let b = a.mapv(|x| x.conj()).into_shape((length, 1)).unwrap();
    // .zip(mtx1_conj.outer_iter()).map_collect(|(x, y)| x * y)
    let a1 = a.clone().into_shape((1, length)).unwrap();

    b.dot(&a1).reversed_axes()
}

pub fn kronecker<T>(
    a: &nd::Array2<Complex<T>>,
    b: &nd::Array2<Complex<T>>,
) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
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
    rho2: &nd::Array2<Complex<T>>,
    unitary: &nd::Array2<Complex<T>>,
) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let unitary_conj_transpose = unitary.mapv(|x| x.conj()).reversed_axes();
    let rho2a = rho2.dot(&unitary_conj_transpose);
    unitary.dot(&rho2a)
}

pub fn get_random_haar_1d<T>(depth: usize) -> nd::Array1<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let normal =
        rand_distr::Uniform::<T>::new(T::from(0.0).unwrap(), T::from(1.0).unwrap());

    let mut rng_real = rand::thread_rng();

    let real = (&mut rng_real)
        .sample_iter(&normal)
        .take(depth)
        .collect::<Vec<T>>();

    let imaginary = (&mut rng_real)
        .sample_iter(&normal)
        .take(depth)
        .collect::<Vec<T>>();

    nd::Zip::from(&real).and(&imaginary).map_collect(|r, i| {
        let in_exp = T::from(2.0f64).unwrap() * T::from(PI).unwrap() * *r;
        let c_r = Complex::<T>::new(T::zero(), in_exp);
        let c_i = (-T::ln(*i)).sqrt();

        Complex::<T>::exp(c_r) * c_i
    })
}

fn apply_symmetries<T>(
    state: &nd::Array2<Complex<T>>,
    symmetries: &Vec<Vec<nd::Array2<Complex<T>>>>,
) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let mut output = state.to_owned();

    for row in symmetries {
        for symmetry in row {
            output = rotate(&output, symmetry) + output;
        }
    }
    output
}

//   ██████     ███████    ███████            ███    ███     ██████     ██████     ███████
//   ██   ██    ██         ██                 ████  ████    ██    ██    ██   ██    ██
//   ██   ██    █████      ███████            ██ ████ ██    ██    ██    ██   ██    █████
//   ██   ██    ██              ██            ██  ██  ██    ██    ██    ██   ██    ██
//   ██████     ██         ███████            ██      ██     ██████     ██████     ███████

pub fn expand_d_fs<T>(
    value: &nd::Array2<Complex<T>>,
    depth: usize,
    quantity: usize,
    idx: usize,
) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let depth_1 = depth.pow(idx as u32);
    let identity_1 = nd::Array::eye(depth_1);

    let depth_2 = depth.pow((quantity - idx - 1) as u32);
    let identity_2 = nd::Array::eye(depth_2);

    let kronecker_1 = kronecker(&identity_1, value);
    kronecker(&kronecker_1, &identity_2)
}

pub fn random_unitary_d_fs<T>(
    depth: usize,
    quantity: usize,
    idx: usize,
) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let value = _random_unitary_d_fs::<T>(depth);
    expand_d_fs(&value, depth, quantity, idx)
}

pub fn _random_unitary_d_fs<T>(depth: usize) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let random_mtx = get_random_haar_1d(depth).into_shape((depth, 1)).unwrap();
    let identity_mtx = nd::Array2::<Complex<T>>::eye(depth);

    let value = _value::<T>();
    let rand_mul = random_mtx.mapv(|x| value * x);

    rand_mul + identity_mtx
}

pub fn _value<T>() -> Complex<T>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let real = T::from(0.01 * PI).unwrap();
    let imaginary = T::from(0.01 * PI).unwrap();

    Complex::new(real.cos() - T::one(), imaginary.sin())
}

pub fn random_d_fs<T>(depth: usize, quantity: usize) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let vector = normalize(&get_random_haar_1d(depth));
    let mut vector_2d;

    vector_2d = vector.into_shape((depth, 1)).unwrap();

    let mut vector_width = depth;

    for _ in 1..quantity {
        let rand_vector = get_random_haar_1d(depth);
        let normalized_rand_vector = normalize(&rand_vector);

        let normalized_rand_vector_2d =
            normalized_rand_vector.into_shape((1, depth)).unwrap();

        let matrix = vector_2d.dot(&normalized_rand_vector_2d);
        vector_width *= depth;

        vector_2d = matrix.into_shape((vector_width, 1)).unwrap();
    }
    project(&vector_2d.into_shape((vector_width,)).unwrap())
}

pub fn optimize_d_fs<T>(
    new_state: &nd::Array2<Complex<T>>,
    visibility_state: &nd::Array2<Complex<T>>,
    depth: usize,
    quantity: usize,
    updates_count: usize,
) -> nd::Array2<Complex<T>>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    let mut product_2_3 = product(new_state, visibility_state);
    let mut unitary = random_unitary_d_fs(depth, quantity, 0);
    let mut rotated_2 = rotate(new_state, &unitary);

    for idx in 0..updates_count {
        let idx_mod = idx % quantity;
        unitary = random_unitary_d_fs(depth, quantity, idx_mod);

        rotated_2 = rotate(new_state, &unitary);

        let mut product_rot2_3 = product(&rotated_2, visibility_state);

        if product_2_3 > product_rot2_3 {
            unitary = unitary.mapv(|x| x.conj()).t().to_owned();
            rotated_2 = rotate(new_state, &unitary);
        }

        while product_2_3 > product_rot2_3 {
            product_2_3 = product_rot2_3;
            rotated_2 = rotate(&rotated_2, &unitary);

            product_rot2_3 = product(&rotated_2, visibility_state);
        }
    }

    rotated_2
}

/*
 ████   ███   ████ █   █ █████ █    █ ████      ████ █      ███   ████  ████
 █   █ █   █ █     █  █  █     ██   █ █   █    █     █     █   █ █     █
 ████  █████ █     ███   ███   █ █  █ █   █    █     █     █████  ███   ███
 █   █ █   █ █     █  █  █     █  █ █ █   █    █     █     █   █     █     █
 ████  █   █  ████ █   █ █████ █   ██ ████      ████ █████ █   █ ████  ████
*/

#[derive(Clone)]
pub struct RustBackend<T> {
    initial: nd::Array2<Complex<T>>,
    depth: usize,
    quantity: usize,

    visibility: nd::Array2<Complex<T>>,
    intermediate: nd::Array2<Complex<T>>,
    visibility_reduced: nd::Array2<Complex<T>>,

    symmetries: Option<Vec<Vec<nd::Array2<Complex<T>>>>>,
    projection: Option<nd::Array2<Complex<T>>>,

    aa4: T,
    aa6: T,
    dd1: T,

    corrections: Vec<(usize, usize, T)>,
    // Specified at the very bottom to match construction argument order. It can not
    // be passed during construction before `optimize_callback` as it uses match on mode
    // the mode otherwise would be moved, thus requiring a clone.
    mode: AlgoMode,
}

impl<T> fmt::Debug for RustBackend<T>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RustBackend<{}>(depth: {}, quantity: {}, state-size: {})",
            any::type_name::<T>(),
            self.depth,
            self.quantity,
            self.initial.dim().0
        )
    }
}

impl<T> RustBackend<T>
where
    T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
{
    pub fn new(
        initial: &nd::Array2<Complex<T>>,
        depth: usize,
        quantity: usize,
        mode: AlgoMode,
        visibility: T,
    ) -> Self
    where
        T: Float + std::fmt::Debug + rand_distr::uniform::SampleUniform + 'static,
    {
        let visibility_matrix =
            RustBackend::create_visibility_matrix(initial, visibility);
        let intermediate_matrix =
            RustBackend::create_intermediate_state(&visibility_matrix);

        let visibility_reduced =
            visibility_matrix.view().sub(&intermediate_matrix.view());

        let aa4 =
            T::from(2).unwrap() * product(&visibility_matrix, &intermediate_matrix);
        let aa6 = product(&intermediate_matrix, &intermediate_matrix);
        let dd1 = product(&intermediate_matrix, &visibility_reduced);

        RustBackend {
            initial: initial.to_owned(),

            depth,
            quantity,

            visibility: visibility_matrix,
            intermediate: intermediate_matrix,
            visibility_reduced,

            symmetries: None,
            projection: None,

            corrections: vec![],

            aa4,
            aa6,
            dd1,

            mode,
        }
    }

    fn create_visibility_matrix(
        initial: &nd::Array2<Complex<T>>,
        visibility: T,
    ) -> nd::Array2<Complex<T>> {
        let length_of_first_axis = initial.dim().0;
        let array_size_complex =
            Complex::<T>::new(T::from(length_of_first_axis).unwrap(), T::zero());

        let vis_state = initial.mapv(|x| x * visibility);
        let identity = nd::Array2::<Complex<T>>::eye(length_of_first_axis);

        let inverted_vis = T::one() - visibility;
        let inv_vis_ident = nd::Zip::from(&identity).map_collect(|i| i * inverted_vis);

        nd::Zip::from(&vis_state)
            .and(&inv_vis_ident)
            .map_collect(|v, iv| v + iv / array_size_complex)
    }

    fn create_intermediate_state(
        visibility_matrix: &nd::Array2<Complex<T>>,
    ) -> nd::Array2<Complex<T>> {
        let length_of_first_axis = visibility_matrix.dim().0;
        let mut intermediate_state = nd::Array2::<Complex<T>>::zeros((
            length_of_first_axis,
            length_of_first_axis,
        ));

        for i in 0..length_of_first_axis {
            intermediate_state[[i, i]] = visibility_matrix[[i, i]];
        }

        intermediate_state
    }

    pub fn set_symmetries(&mut self, symmetries: Vec<Vec<nd::Array2<Complex<T>>>>) {
        self.symmetries = Some(symmetries);
    }

    pub fn get_state(&self) -> &nd::Array2<Complex<T>> {
        &self.intermediate
    }

    pub fn get_corrections(&self) -> &Vec<(usize, usize, T)> {
        &self.corrections
    }

    pub fn run_epoch(&mut self, iterations: i64, epoch_index: usize) {
        let depth = self.depth;
        let quantity = self.quantity;
        let epochs = 20 * depth * depth * quantity;

        for iteration_index in 0..iterations {
            let alternative_state = match self.mode {
                AlgoMode::FSnQd => random_d_fs(depth, quantity),
                AlgoMode::SBiPa => panic!("Mode 'SBiPa' is currently not supported."),
                AlgoMode::G3PaE3qD => {
                    panic!("Mode 'G3PaE3qD' is currently not supported.")
                }
                AlgoMode::G4PaE3qD => {
                    panic!("Mode 'G4PaE3qD' is currently not supported.")
                }
            };

            if product(&alternative_state, &self.visibility_reduced) > self.dd1 {
                self.update_state(
                    &alternative_state,
                    iterations,
                    epoch_index,
                    epochs,
                    iteration_index,
                );
            }
        }
    }

    fn update_state(
        &mut self,
        alternative_state: &ndarray::Array2<Complex<T>>,
        iterations: i64,
        epoch_index: usize,
        epochs: usize,
        iteration_index: i64,
    ) {
        let depth = self.depth;
        let quantity = self.quantity;
        let literal_two = T::from(2).unwrap();

        let optimized_state = match self.mode {
            AlgoMode::FSnQd => optimize_d_fs(
                alternative_state,
                &self.visibility_reduced,
                depth,
                quantity,
                epochs,
            ),
            AlgoMode::SBiPa => panic!("Mode 'SBiPa' is currently not supported."),
            AlgoMode::G3PaE3qD => {
                panic!("Mode 'G3PaE3qD' is currently not supported.")
            }
            AlgoMode::G4PaE3qD => {
                panic!("Mode 'G4PaE3qD' is currently not supported.")
            }
        };

        if let Some(ref symmetries) = self.symmetries {
            self.intermediate = apply_symmetries(&self.intermediate, symmetries);
        }
        if let Some(ref projection) = self.projection {
            self.intermediate = rotate(&self.intermediate, projection);
        }

        let aa3 = product(&optimized_state, &optimized_state);
        let aa2 = literal_two * product(&self.visibility, &optimized_state);
        let aa5 = literal_two * product(&self.intermediate, &optimized_state);

        let bb2 = -self.aa4 + aa2 + aa5 - (literal_two * aa3);
        let bb3 = self.aa6 - aa5 + aa3;
        let cc1 = -bb2 / (literal_two * bb3);

        if T::zero() <= cc1 && cc1 <= T::one() {
            let cc1_inverse = T::one() - cc1;
            self.intermediate = self.intermediate.mapv(|x| x * cc1)
                + alternative_state.mapv(|x| x * cc1_inverse);

            self.visibility_reduced = &self.visibility - &self.intermediate;

            self.aa4 = literal_two * product(&self.visibility, &self.intermediate);
            self.aa6 = product(&self.intermediate, &self.intermediate);
            self.dd1 = self.aa4 / literal_two - self.aa6;

            self.corrections.push((
                (epoch_index * iterations as usize + iteration_index as usize + 1),
                self.corrections.len() + 1,
                product(&self.visibility_reduced, &self.visibility_reduced),
            ));
        }
    }
}
