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
                            let array: nd::Array2<Complex<f32>> = array_ref.mapv(|x| {
                                Complex::<f32>::new(x.re as f32, x.im as f32)
                            });
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
