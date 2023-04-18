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

use pyo3::{prelude::*, types::PyDict};

mod naive;
mod shared;

/// A Python module implemented in Rust.
#[pymodule]
fn cssfinder_backend_rust(py: Python, m: &PyModule) -> PyResult<()> {
    register_complex64(py, m)?;
    register_complex128(py, m)?;

    m.add("__version__", "0.1.0")?;

    #[pyfunction]
    fn export_backend(py: Python) -> &PyDict {
        Python::with_gil(|_py| {
            let cssfinder_cssfproject =
                PyModule::import(py, "cssfinder.cssfproject").unwrap();
            let precision_enum = cssfinder_cssfproject.getattr("Precision").unwrap();

            let cssfinder_backend_rust_module =
                PyModule::import(py, "cssfinder_backend_rust").unwrap();

            let backend_class_f64 = cssfinder_backend_rust_module
                .getattr("complex128")
                .unwrap()
                .getattr("NaiveRustBackendF64")
                .unwrap();

            let backends_dict = PyDict::new(py);

            backends_dict
                .set_item(
                    ("rust_naive", precision_enum.getattr("DOUBLE").unwrap()),
                    backend_class_f64,
                )
                .unwrap();

            let backend_class_f32 = cssfinder_backend_rust_module
                .getattr("complex64")
                .unwrap()
                .getattr("NaiveRustBackendF32")
                .unwrap();

            backends_dict
                .set_item(
                    ("rust_naive", precision_enum.getattr("SINGLE").unwrap()),
                    backend_class_f32,
                )
                .unwrap();

            backends_dict
        })
    }
    m.add_function(wrap_pyfunction!(export_backend, m)?)?;

    Ok(())
}

fn register_complex128(py: Python, parent: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "complex128")?;

    module.add_function(wrap_pyfunction!(complex128::product, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::normalize, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::project, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::kronecker, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::rotate, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::get_random_haar_1d, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::expand_d_fs, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::random_unitary_d_fs, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::random_d_fs, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::optimize_d_fs, parent)?)?;
    module.add_function(wrap_pyfunction!(complex128::noop, parent)?)?;

    module.add_class::<complex128::NaiveRustBackendF64>()?;

    parent.add_submodule(module)?;

    Ok(())
}

fn register_complex64(py: Python, parent: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "complex64")?;

    module.add_class::<complex64::NaiveRustBackendF32>()?;

    parent.add_submodule(module)?;

    Ok(())
}

mod complex128 {
    use num::Complex;
    use numpy as np;
    use pyo3::prelude::*;

    #[pyfunction]
    pub fn product(
        _py: Python,
        a: np::PyReadonlyArray2<Complex<f64>>,
        b: np::PyReadonlyArray2<Complex<f64>>,
    ) -> PyResult<f64> {
        let array_1 = a.as_array();
        let array_2 = b.as_array();
        Ok(super::naive::product(
            &array_1.to_owned(),
            &array_2.to_owned(),
        ))
    }

    #[pyfunction]
    pub fn normalize<'py>(
        py: Python<'py>,
        a: np::PyReadonlyArray1<Complex<f64>>,
    ) -> &'py np::PyArray1<Complex<f64>> {
        let array_1 = a.as_array();
        let array_2 = super::naive::normalize(&array_1.to_owned());
        let array_out = np::PyArray::from_owned_array(py, array_2);
        array_out
    }

    #[pyfunction]
    pub fn project<'py>(
        py: Python<'py>,
        a: np::PyReadonlyArray1<Complex<f64>>,
    ) -> &'py np::PyArray2<Complex<f64>> {
        let array_1 = a.as_array();
        let array_2 = super::naive::project(&array_1.to_owned());
        let array_out = np::PyArray::from_owned_array(py, array_2);
        array_out
    }

    #[pyfunction]
    pub fn kronecker<'py>(
        py: Python<'py>,
        a: np::PyReadonlyArray2<Complex<f64>>,
        b: np::PyReadonlyArray2<Complex<f64>>,
    ) -> &'py np::PyArray2<Complex<f64>> {
        let array_1 = a.as_array();
        let array_2 = b.as_array();
        let array_3 = super::naive::kronecker(&array_1.to_owned(), &array_2.to_owned());
        let array_out = np::PyArray::from_owned_array(py, array_3);
        array_out
    }

    #[pyfunction]
    pub fn rotate<'py>(
        py: Python<'py>,
        a: np::PyReadonlyArray2<Complex<f64>>,
        b: np::PyReadonlyArray2<Complex<f64>>,
    ) -> &'py np::PyArray2<Complex<f64>> {
        let array_3 =
            super::naive::rotate(&a.as_array().to_owned(), &b.as_array().to_owned());
        let array_out = np::PyArray::from_owned_array(py, array_3);
        array_out
    }

    #[pyfunction]
    pub fn get_random_haar_1d(py: Python, a: usize) -> &np::PyArray1<Complex<f64>> {
        let array_3 = super::naive::get_random_haar_1d(a);
        let array_out = np::PyArray::from_owned_array(py, array_3);
        array_out
    }

    #[pyfunction]
    pub fn expand_d_fs<'py>(
        py: Python<'py>,
        value: np::PyReadonlyArray2<Complex<f64>>,
        depth: usize,
        quantity: usize,
        idx: usize,
    ) -> &'py np::PyArray2<Complex<f64>> {
        let array_3 = super::naive::expand_d_fs(
            &value.as_array().to_owned(),
            depth,
            quantity,
            idx,
        );
        let array_out = np::PyArray::from_owned_array(py, array_3);
        array_out
    }

    #[pyfunction]
    pub fn random_unitary_d_fs(
        py: Python,
        depth: usize,
        quantity: usize,
        idx: usize,
    ) -> &np::PyArray2<Complex<f64>> {
        let array_3 = super::naive::random_unitary_d_fs(depth, quantity, idx);
        let array_out = np::PyArray::from_owned_array(py, array_3);
        array_out
    }

    #[pyfunction]
    pub fn random_d_fs(
        py: Python,
        depth: usize,
        quantity: usize,
    ) -> &np::PyArray2<Complex<f64>> {
        let array_3 = super::naive::random_d_fs(depth, quantity);
        let array_out = np::PyArray::from_owned_array(py, array_3);
        array_out
    }

    #[pyfunction]
    pub fn optimize_d_fs<'py>(
        py: Python<'py>,
        new_state: np::PyReadonlyArray2<Complex<f64>>,
        visibility_state: np::PyReadonlyArray2<Complex<f64>>,
        depth: usize,
        quantity: usize,
        updates_count: usize,
    ) -> &'py np::PyArray2<Complex<f64>> {
        let array_out = super::naive::optimize_d_fs(
            &new_state.as_array().to_owned(),
            &visibility_state.as_array().to_owned(),
            depth,
            quantity,
            updates_count,
        );
        let array_out_py = np::PyArray::from_owned_array(py, array_out);
        array_out_py
    }

    #[pyfunction]
    pub fn noop(_py: Python) -> PyResult<()> {
        use ndarray as nd;
        let a = nd::array!([1, 2, 3]);
        let b = nd::array!([3, 2, 1]);

        let a_len = a.len();
        let b_len = b.len();

        let a1 = a.into_shape((a_len, 1)).unwrap();
        let b1 = b.into_shape((1, b_len)).unwrap();

        let c = b1.dot(&a1).into_shape(a_len * b_len);

        println!("{:?}", c);

        Ok(())
    }

    #[pyclass]
    pub struct NaiveRustBackendF64 {
        backend: super::naive::RustBackend<f64>,
    }

    #[pymethods]
    impl NaiveRustBackendF64 {
        #[new]
        fn new(
            initial: np::PyReadonlyArray2<Complex<f64>>,
            depth: usize,
            quantity: usize,
            mode: super::shared::AlgoMode,
            visibility: f64,
            is_debug: Option<bool>,
        ) -> Self {
            let state_array = initial.as_array();
            assert!(is_debug.unwrap_or(false) || !is_debug.unwrap_or(false));

            let backend = crate::naive::RustBackend::<f64>::new(
                &state_array.to_owned(),
                depth,
                quantity,
                mode,
                visibility,
            );

            NaiveRustBackendF64 { backend }
        }

        fn set_symmetries(
            &mut self,
            symmetries: Vec<Vec<np::PyReadonlyArray2<Complex<f64>>>>,
        ) {
            use ndarray as nd;

            let symmetries_local = symmetries
                .into_iter()
                .map(|inner_vec| {
                    inner_vec
                        .into_iter()
                        .map(|pyarray| {
                            let array_ref = pyarray.as_array();
                            let array: nd::Array2<Complex<f64>> = array_ref.to_owned();
                            array
                        })
                        .collect()
                })
                .collect();
            self.backend.set_symmetries(symmetries_local);
        }

        fn set_projection(&mut self, projection: np::PyReadonlyArray2<Complex<f64>>) {
            println!("{:?}", projection);
        }

        fn get_state<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<&'py np::PyArray2<Complex<f64>>> {
            let array_out = self.backend.get_state();
            Ok(np::PyArray::from_owned_array(py, array_out.to_owned()))
        }

        fn get_corrections(&self) -> PyResult<Vec<(usize, usize, f64)>> {
            Ok(self.backend.get_corrections().to_owned())
        }

        fn get_corrections_count(&self) -> PyResult<usize> {
            Ok(self.backend.get_corrections().len())
        }

        fn run_epoch(&mut self, iterations: i64, epoch_index: usize) {
            self.backend.run_epoch(iterations, epoch_index)
        }
    }
}

mod complex64 {
    use num::Complex;
    use numpy as np;
    use pyo3::prelude::*;

    #[pyclass]
    pub struct NaiveRustBackendF32 {
        backend: super::naive::RustBackend<f32>,
    }

    #[pymethods]
    impl NaiveRustBackendF32 {
        #[new]
        fn new(
            initial: np::PyReadonlyArray2<Complex<f64>>,
            depth: usize,
            quantity: usize,
            mode: super::shared::AlgoMode,
            visibility: f32,
            is_debug: Option<bool>,
        ) -> Self {
            let state_array = initial
                .as_array()
                .mapv(|x| Complex::<f32>::new(x.re as f32, x.im as f32));
            assert!(is_debug.unwrap_or(false) || !is_debug.unwrap_or(false));

            let backend = crate::naive::RustBackend::<f32>::new(
                &state_array,
                depth,
                quantity,
                mode,
                visibility,
            );

            NaiveRustBackendF32 { backend }
        }

        fn set_symmetries(
            &mut self,
            symmetries: Vec<Vec<np::PyReadonlyArray2<Complex<f32>>>>,
        ) {
            use ndarray as nd;

            let symmetries_local = symmetries
                .into_iter()
                .map(|inner_vec| {
                    inner_vec
                        .into_iter()
                        .map(|pyarray| {
                            let array_ref = pyarray.as_array();
                            let array: nd::Array2<Complex<f32>> = array_ref.to_owned();
                            array
                        })
                        .collect()
                })
                .collect();
            self.backend.set_symmetries(symmetries_local);
        }

        fn set_projection(&mut self, projection: np::PyReadonlyArray2<Complex<f32>>) {
            println!("{:?}", projection);
        }

        fn get_state<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<&'py np::PyArray2<Complex<f32>>> {
            let array_out = self.backend.get_state();
            Ok(np::PyArray::from_owned_array(py, array_out.to_owned()))
        }

        fn get_corrections(&self) -> PyResult<Vec<(usize, usize, f32)>> {
            Ok(self.backend.get_corrections().to_owned())
        }

        fn get_corrections_count(&self) -> PyResult<usize> {
            Ok(self.backend.get_corrections().len())
        }

        fn run_epoch(&mut self, iterations: i64, epoch_index: usize) {
            self.backend.run_epoch(iterations, epoch_index)
        }
    }
}
