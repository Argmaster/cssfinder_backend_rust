# Changelog

NOTE: cssfinder_backend_numpy follows the [semver](https://semver.org/)
versioning standard.

### 0.1.1 - 27 April 2023

- Changed set_symmetries function so now takes in a vector of vectors of
  `PyReadonlyArray2<Complex<f64>>` instead of `PyReadonlyArray2<Complex<f32>>`.
  The input `f64` arrays are converted to `f32` arrays for processing.
- Modified src/naive.rs, `apply_symmetries` to add normalization and trace
  computation to the output matrix.
- Upgraded cssfinder-backend-rust from version 0.1.0 to 0.1.1.
- Upgraded getrandom from version 0.2.8 to 0.2.9.
- Upgraded libc from version 0.2.140 to 0.2.142.
- Upgraded matrixmultiply from version 0.3.2 to 0.3.3.
- Upgraded proc-macro2 from version 1.0.54 to 1.0.56.
- Upgraded pyo3 from version 0.18.2 to 0.18.3.
- Upgraded pyo3-build-config from version 0.18.2 to 0.18.3.
- Upgraded pyo3-ffi from version 0.18.2 to 0.18.3.
- Upgraded pyo3-macros from version 0.18.2 to 0.18.3.
- Upgraded pyo3-macros-backend from version 0.18.2 to 0.18.3.

### 0.1.0 - 20 April 2023

- Added `complex64` module with `NaiveRustBackendF32` class as `rust_naive`
  with precision `SINGLE`
- Added `complex128` module with `NaiveRustBackendF64` class as `rust_naive`
  with precision `DOUBLE`
