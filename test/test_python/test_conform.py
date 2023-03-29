from typing import ClassVar

import cssfinder_backend_numpy.numpy._complex128 as numpy_c128
import cssfinder_backend_rust as rust_c128
import numpy as np
from cssfinder_backend_numpy.impl import Implementation
from matplotlib import pyplot as plt


class ValidateConformance:
    this: ClassVar[Implementation]
    reference: ClassVar[Implementation]
    dtype: ClassVar[np.dtype]

    def setup_class(self) -> None:
        np.random.seed(0)

        self.size = 5

        self.lhs_mtx = (
            np.random.random((self.size, self.size))
            + 1j * np.random.random((self.size, self.size))
        ).astype(self.dtype)

        self.rhs_mtx = (
            np.random.random((self.size, self.size))
            + 1j * np.random.random((self.size, self.size))
        ).astype(self.dtype)

        self.lhs_vec = (
            np.random.random((self.size,)) + 1j * np.random.random((self.size,))
        ).astype(self.dtype)

        self.rhs_vec = (
            np.random.random((self.size,)) + 1j * np.random.random((self.size,))
        ).astype(self.dtype)

        self.limit = 1e-6

    def test_product(self) -> None:
        """Validate product return value."""

        a = self.reference.product(self.lhs_mtx, self.rhs_mtx)
        b = self.this.product(self.lhs_mtx, self.rhs_mtx)

        print(self.lhs_mtx)
        print(self.rhs_mtx)
        print("Reference", a)
        print("This", b)

        assert np.abs(a - b) < self.limit

    def test_normalize(self) -> None:
        """Validate normalize return value."""

        a = self.reference.normalize(self.lhs_mtx)
        b = self.this.normalize(self.lhs_mtx)

        print(self.lhs_mtx)
        print("Reference", a)
        print("This", b)

        assert a.shape == b.shape
        assert (np.abs(a - b) < self.limit).conj().all()

    def test_project(self) -> None:
        """Validate vector projection."""

        a = self.reference.project(self.lhs_vec)
        b = self.this.project(self.lhs_vec)

        print(self.lhs_vec)
        print("Reference\n", a.round(2))
        print("This\n", b.round(2))
        print("Diff\n", np.abs(a - b).round(2))

        assert a.shape == b.shape
        assert (np.abs(a - b) < self.limit).conj().all()

    def test_kronecker(self) -> None:
        """Validate vector projection."""

        a = self.reference.kronecker(self.lhs_mtx, self.rhs_mtx)
        b = self.this.kronecker(self.lhs_mtx, self.rhs_mtx)

        print(self.lhs_vec)
        print("Reference", a)
        print("This", b)

        assert a.shape == b.shape
        assert (np.abs(a - b) < self.limit).conj().all()

    def test_rotate(self) -> None:
        """Validate vector projection."""

        a = self.reference.rotate(self.lhs_mtx, self.rhs_mtx)
        b = self.this.rotate(self.lhs_mtx, self.rhs_mtx)

        print(self.lhs_vec)
        print("Reference", a)
        print("This", b)

        assert a.shape == b.shape
        assert (np.abs(a - b) < self.limit).conj().all()

    def test_get_random_haar_1d(self) -> None:
        """Validate vector sampling."""

        reference = np.array(
            [
                np.abs(self.reference.get_random_haar_1d(self.size).sum())
                for _ in range(100_000)
            ]
        )
        this = np.array(
            [
                np.abs(self.this.get_random_haar_1d(self.size).sum())
                for _ in range(100_000)
            ]
        )

        right = max(reference.max(), this.max())

        plt.hist(reference, 6, alpha=0.5, range=(0, right), color="blue")
        plt.hist(this, 6, alpha=0.5, range=(0, right), color="red")
        # ; plt.show()

        assert reference.shape == this.shape


class TestComplex128(ValidateConformance):
    this = rust_c128
    reference = numpy_c128
    dtype = np.complex128
