from __future__ import annotations

from typing import Union, cast

import numpy as np
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import int64
from arkouda.logger import getArkoudaLogger
from arkouda.numpy.dtypes.dtypes import NumericDTypes
from arkouda.pdarrayclass import pdarray
from arkouda.sparrayclass import create_sparray, sparray

__all__ = ["random_sparse_matrix", "sparse_matrix_matrix_mult", "create_sparse_matrix"]

logger = getArkoudaLogger(name="sparsematrix")


@typechecked
def random_sparse_matrix(
    size: int, density: float, layout: str, dtype: Union[type, str] = int64
) -> sparray:
    """
    Create a random sparse matrix with the specified number of rows and columns
    and the specified density. The density is the fraction of non-zero elements
    in the matrix. The non-zero elements are uniformly distributed random
    numbers in the range [0,1).

    Parameters
    ----------
    size : int
        The number of rows in the matrix, columns are equal to rows right now
    density : float
        The fraction of non-zero elements in the matrix
    dtype : Union[DTypes, str]
        The dtype of the elements in the matrix (default is int64)

    Returns
    -------
    sparray
        A sparse matrix with the specified number of rows and columns
        and the specified density

    Raises
    ------
    ValueError
        Raised if density is not in the range [0,1]
    """
    if density < 0.0 or density > 1.0:
        raise ValueError("density must be in the range [0,1]")

    if layout not in ["CSR", "CSC"]:
        raise ValueError("layout must be 'CSR' or 'CSC'")

    repMsg = generic_msg(
        cmd=f"random_sparse_matrix<{akdtype(dtype)},{layout}>",
        args={
            "shape": tuple([size, size]),
            "density": density,
        },
    )

    return create_sparray(repMsg)


@typechecked
def sparse_matrix_matrix_mult(A, B: sparray) -> sparray:
    """
    Multiply two sparse matrices.

    Parameters
    ----------
    A : sparray
        The left-hand sparse matrix
    B : sparray
        The right-hand sparse matrix

    Returns
    -------
    sparray
        The product of the two sparse matrices
    """
    if not (isinstance(A, sparray) and isinstance(B, sparray)):
        raise TypeError("A and B must be sparrays for sparse_matrix_matrix_mult")
    if not A.dtype == B.dtype:
        raise ValueError("A and B must have the same dtype for sparse matrix multiplication")
    repMsg = generic_msg(
        cmd=f"sparse_matrix_matrix_mult<{A.dtype}>",
        args={"arg1": A.name, "arg2": B.name},
    )

    return create_sparray(repMsg)


def create_sparse_matrix(size: int, rows: pdarray, cols: pdarray, vals: pdarray, layout: str) -> sparray:
    """
    Create a sparse matrix from three pdarrays representing the row indices,
    column indices, and values of the non-zero elements of the matrix.

    Parameters
    ----------
    rows : pdarray
        The row indices of the non-zero elements
    cols : pdarray
        The column indices of the non-zero elements
    vals : pdarray
        The values of the non-zero elements

    Returns
    -------
    sparray
        A sparse matrix with the specified row and column indices and values
    """
    if not (isinstance(rows, pdarray) and isinstance(cols, pdarray) and isinstance(vals, pdarray)):
        raise TypeError("rows, cols, and vals must be pdarrays for create_sparse_matrix")
    if not (rows.ndim == 1 and cols.ndim == 1 and vals.ndim == 1):
        raise ValueError("rows, cols, and vals must be 1D for create_sparse_matrix")
    if not (rows.size == cols.size and rows.size == vals.size):
        raise ValueError("rows, cols, and vals must have the same size for create_sparse_matrix")
    if not (rows.dtype == int64 and cols.dtype == int64):
        raise ValueError("rows and cols must have dtype int64 for create_sparse_matrix")
    if layout not in ["CSR", "CSC"]:
        raise ValueError("layout must be 'CSR' or 'CSC'")

    vals_dtype_name = cast(np.dtype, vals.dtype).name
    # check dtype for error
    if vals_dtype_name not in NumericDTypes:
        raise TypeError(f"unsupported dtype {vals.dtype}")

    shape = (size, size)
    repMsg = generic_msg(
        cmd=f"sparse_matrix_from_pdarrays<{vals.dtype},{layout}>",
        args={"rows": rows.name, "cols": cols.name, "vals": vals.name, "shape": shape},
    )

    return create_sparray(repMsg)
