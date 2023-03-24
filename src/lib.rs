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

use pyo3::prelude::*;

mod pure;

/// A Python module implemented in Rust.
#[pymodule]
fn cssfinder_backend_rust(py: Python, m: &PyModule) -> PyResult<()> {
    register_complex128(py, m)?;
    m.add("__version__", "0.1.0")?;

    Ok(())
}

fn register_complex128(py: Python, parent: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "complex128")?;

    parent.add_function(wrap_pyfunction!(complex128::product, parent)?)?;
    parent.add_function(wrap_pyfunction!(complex128::normalize, parent)?)?;

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
        Ok(super::pure::product(&array_1, &array_2))
    }

    #[pyfunction]
    pub fn normalize<'py>(
        py: Python<'py>,
        a: np::PyReadonlyArray2<Complex<f64>>,
    ) -> &'py np::PyArray2<Complex<f64>> {
        let array_1 = a.as_array();
        let array_2 = super::pure::normalize(&array_1);
        let array_out = np::PyArray::from_owned_array(py, array_2);
        array_out
    }
}
