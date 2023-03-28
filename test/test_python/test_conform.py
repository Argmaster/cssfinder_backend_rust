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

        s = 5

        self.lhs_mtx = (
            np.random.random((s, s)) + 1j * np.random.random((s, s))
        ).astype(self.dtype)

        self.rhs_mtx = (
            np.random.random((s, s)) + 1j * np.random.random((s, s))
        ).astype(self.dtype)

        self.lhs_vec = (np.random.random((s,)) + 1j * np.random.random((s,))).astype(
            self.dtype
        )

        self.rhs_vec = (np.random.random((s,)) + 1j * np.random.random((s,))).astype(
            self.dtype
        )

    def test_product(self) -> None:
        """Validate product return value."""

        a = self.reference.product(self.lhs_mtx, self.rhs_mtx)
        b = self.this.product(self.lhs_mtx, self.rhs_mtx)

        print(self.lhs_mtx)
        print(self.rhs_mtx)
        print("Reference", a)
        print("This", b)

        assert (a - b) < 1e-9

    def test_normalize(self) -> None:
        """Validate normalize return value."""

        a = self.reference.normalize(self.lhs_mtx)
        b = self.this.normalize(self.lhs_mtx)

        print(self.lhs_mtx)
        print("Reference", a)
        print("This", b)

        assert a.shape == b.shape
        assert ((a - b) < 1e-9).conj().all()

    def test_project(self) -> None:
        """Validate vector projection."""

        a = self.reference.project(self.lhs_vec)
        b = self.this.project(self.lhs_vec)

        print(self.lhs_vec)
        print("Reference", a)
        print("This", b)

        assert a.shape == b.shape
        assert ((a - b) < 1e-9).conj().all(), (a, b)

    def test_kronecker(self) -> None:
        """Validate vector projection."""

        a = self.reference.kronecker(self.lhs_mtx, self.rhs_mtx)
        b = self.this.kronecker(self.lhs_mtx, self.rhs_mtx)

        print(self.lhs_vec)
        print("Reference", a)
        print("This", b)

        assert a.shape == b.shape
        assert ((a - b) < 1e-9).conj().all(), (a, b)

    def test_rotate(self) -> None:
        """Validate vector projection."""

        a = self.reference.rotate(self.lhs_mtx, self.rhs_mtx)
        b = self.this.rotate(self.lhs_mtx, self.rhs_mtx)

        print(self.lhs_vec)
        print("Reference", a)
        print("This", b)

        assert a.shape == b.shape
        assert ((a - b) < 1e-9).conj().all(), (a, b)

class TestComplex128(ValidateConformance):
    this = rust_c128
    reference = numpy_c128
    dtype = np.complex128
