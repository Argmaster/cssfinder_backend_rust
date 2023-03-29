from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


import cssfinder_backend_rust as cbr
import cssfinder_backend_numpy

a = np.identity(2).astype(dtype=np.complex128)
b = np.identity(2).astype(dtype=np.complex128).T

np.random.seed(0)

s = 3

a = ((np.random.random((s, s)) + 1j * np.random.random((s, s))) * 5).astype(np.int64)
b = ((np.random.random((s, s)) + 1j * np.random.random((s, s))) * 5).astype(np.int64)


ddd1 = len(a)
ddd2 = len(b)

output_shape = (ddd1 * ddd2, ddd1 * ddd2)
print(output_shape)

dot_0_1 = np.tensordot(a, b, 0)
print(dot_0_1.shape)
print("dot_0_1")
for row in dot_0_1.tolist():
    for row2 in row:
        print(row2)

out_mtx = np.swapaxes(dot_0_1, 1, 2)
print(out_mtx.shape)
print("out_mtx")
for row in out_mtx.tolist():
    for row2 in row:
        print(row2)

retval = out_mtx.reshape(output_shape).astype(np.complex64, copy=False)
print(retval.shape)

# start = time.perf_counter()
# c = cbr.product(a, b)
# stop = time.perf_counter()
# print(c, stop - start)
#
#
# def product(
#     matrix1: npt.NDArray[np.complex64], matrix2: npt.NDArray[np.complex64]
# ) -> np.float32:
#     """Calculate scalar product of two matrices."""
#     retval = np.trace(np.dot(matrix1, matrix2)).real
#
#     return retval  # type: ignore[no-any-return]
#
# start = time.perf_counter()
# product(a, b)
# stop = time.perf_counter()
# print(product(a, b), stop - start)
