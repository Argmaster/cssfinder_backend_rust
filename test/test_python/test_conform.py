from typing import ClassVar

import cssfinder_backend_numpy.numpy._complex128 as numpy_c128
import cssfinder_backend_rust as rust_backend
import numpy as np
from cssfinder.algorithm.backend.base import BackendBase
from cssfinder.cssfproject import AlgoMode
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

        self.mtx32 = (
            np.random.random((32, 32)) + 1j * np.random.random((32, 32))
        ).astype(self.dtype)

        self.limit = 1e-6

    def test_product(self) -> None:
        """Validate product return value."""
        assert (self.lhs_mtx != self.rhs_mtx).all()

        a = self.reference.product(self.lhs_mtx, self.rhs_mtx)
        b = self.this.product(self.lhs_mtx, self.rhs_mtx)

        print(self.lhs_mtx)
        print(self.rhs_mtx)
        print("Reference", a)
        print("This", b)

        assert np.abs(a - b) < self.limit

    def test_normalize(self) -> None:
        """Validate normalize return value."""

        a = self.reference.normalize(self.lhs_vec)
        b = self.this.normalize(self.lhs_vec)

        print(self.lhs_vec)
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
        print("Reference:\n", a.round(2))
        print("This:\n", b.round(2))

        assert a.shape == b.shape
        assert (np.abs(a - b) < self.limit).conj().all()

    def test_get_random_haar_1d(self) -> None:
        """Validate vector sampling."""

        reference = np.array(
            [
                self.reference.get_random_haar_1d(self.size)
                for _ in range(1_000)
            ]
        ).flatten()
        this = np.array(
            [
                self.this.get_random_haar_1d(self.size)
                for _ in range(1_000)
            ]
        ).flatten()

        plt.hist(reference, 100, alpha=0.5, range=(-2.0, 6.0), color="blue")
        plt.hist(this, 100, alpha=0.5, range=(-2.0, 6.0), color="red")
        # ; plt.show()

        assert reference.shape == this.shape

    def test_expand_d_fs(self) -> None:
        """Validate expand_d_fs method."""

        reference = self.reference.expand_d_fs(self.mtx32, 5, 2, 0)
        this = self.this.expand_d_fs(self.mtx32, 5, 2, 0)

        print("Reference:\n", reference.round(2))
        print("This:\n", this.round(2))

        assert reference.shape == this.shape
        assert (reference == this).all()

    def test_random_unitary_d_fs(self) -> None:
        """Validate vector sampling."""

        reference = self.reference.random_unitary_d_fs(5, 2, 0)
        this = self.this.random_unitary_d_fs(5, 2, 0)

        print("Reference:\n", reference.round(2))
        print("This:\n", this.round(2))

        # Check shape of matrices
        assert reference.shape == this.shape
        # Check layout of matrices
        assert (reference.astype(np.bool8) == this.astype(np.bool8)).all()

        print(
            reference_mean := np.array(
                [self.reference.random_unitary_d_fs(5, 2, 0) for _ in range(20_000)]
            ).mean()
        )
        print(
            this_mean := np.array(
                [self.this.random_unitary_d_fs(5, 2, 0) for _ in range(20_000)]
            ).mean()
        )

        assert reference_mean.round(2) == this_mean.round(2)

    def test_random_d_fs(self) -> None:
        reference = self.reference.random_d_fs(5, 2)
        this = self.this.random_d_fs(5, 2)
        with np.printoptions(precision=3, suppress=True, linewidth=1000):
            print("Reference:\n", reference.round(2))
            print("This:\n", this.round(2))

        # Check shape of matrices
        assert reference.shape == this.shape

        reference_multi_sample = np.array(
            [self.reference.random_d_fs(5, 2) for _ in range(20000)]
        )
        this_multi_sample = np.array(
            [self.this.random_d_fs(5, 2) for _ in range(20000)]
        )

        with np.printoptions(precision=3, suppress=True, linewidth=1000):
            print("Reference:\n", (reference_multi_sample.sum(0) > 0).astype(np.int32))
            print("This:\n", (this_multi_sample.sum(0) > 0).astype(np.int32))

        # Check if matrix is symmetric
        assert (
            np.triu(this_multi_sample.sum(0) > 0).T
            == np.tril(this_multi_sample.sum(0) > 0)
        ).all()

        print(reference_mean := reference_multi_sample.mean())
        print(this_mean := this_multi_sample.mean())

        assert reference_mean.round(2) == this_mean.round(2)

    def test_noop(self) -> None:
        self.this.noop()


class TestComplex128(ValidateConformance):
    this = rust_backend.complex128
    reference = numpy_c128
    dtype = np.complex128


class TestBackendClass:
    def get_backend_instance(self) -> BackendBase:
        return rust_backend.complex128.NaiveRustBackendF64(
            np.identity(32).astype(np.complex128),
            2,
            5,
            AlgoMode.FSnQd,
            1.0,
        )

    def test_backend_set_symmetries(self) -> None:
        instance = self.get_backend_instance()
        instance.set_symmetries([[np.identity(32).astype(np.complex128)]])

    def test_backend_set_projection(self) -> None:
        instance = self.get_backend_instance()
        instance.set_projection(np.identity(32).astype(np.complex128))

    def test_backend_get_state(self) -> None:
        instance = self.get_backend_instance()
        assert (instance.get_state() == np.identity(32).astype(np.complex128)).all()

    def test_backend_get_corrections(self) -> None:
        instance = self.get_backend_instance()
        assert instance.get_corrections() == []

    def test_backend_run_epoch(self) -> None:
        instance = self.get_backend_instance()
        instance.run_epoch(16, 0)
