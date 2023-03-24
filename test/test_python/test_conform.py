from typing import ClassVar

import cssfinder_backend_numpy.numpy._complex128 as numpy_c128
import cssfinder_backend_rust as rust_c128
import numpy as np
from cssfinder_backend_numpy.impl import Implementation


class ValidateConformance:
    this: ClassVar[Implementation]
    reference: ClassVar[Implementation]
    dtype: ClassVar[np.dtype]

    def setup_class(self) -> None:
        np.random.seed(0)
        self.lhs = (
            np.random.random((32, 32)) + 1j * np.random.random((32, 32))
        ).astype(self.dtype)
        self.rhs = (
            np.random.random((32, 32)) + 1j * np.random.random((32, 32))
        ).astype(self.dtype)

    def test_product(self) -> None:
        """Validate product return value."""

        a = self.reference.product(self.lhs, self.rhs)
        b = self.this.product(self.lhs, self.rhs)

        assert (a - b) < 1e-9

    def test_normalize(self) -> None:
        """Validate normalize return value."""

        a = self.reference.normalize(self.lhs)
        b = self.this.normalize(self.lhs)

        print(self.lhs)
        print(a)
        print(b)

        assert ((a - b) < 1e-9).conj().all()


class TestComplex128(ValidateConformance):
    this = rust_c128
    reference = numpy_c128
    dtype = np.complex128
