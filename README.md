# CSSFinder Rust Backend

Implementation of CSSFinder in Rust Programming Language.

## Installing

To install CSSFinder NumPy Backend from PyPI, use `pip` in terminal:

```
pip install cssfinder_backend_rust
```

If you want to use development version, traverse `Development` and `Packaging`
sections below.

## Development

Both `Rust` (`>=1.65`) and `Python` (`>=3.8`) are required. Additionally, for
comfortable development experience, `poetry` is recomended, but it is not used
for deployment. To deploy code, `maturin` package is used.

- Install `poetry` use `pip install poetry==1.4.0`
- Open dev shell `poetry shell`
- Install development requirements `poetry install --sync`
- Build package `poe build`
- Install pre-commit hooks `poe install-hooks`
- Run pre-commit hooks `poe run-hooks`
