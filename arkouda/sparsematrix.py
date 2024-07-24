from __future__ import annotations

from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.logger import getArkoudaLogger
from arkouda.sparrayclass import create_sparray, sparray

__all__ = ["random_sparse_matrix", "sparse_matrix_matrix_mult"]

logger = getArkoudaLogger(name="sparsematrix")


@typechecked
def random_sparse_matrix(size: int, density: float, layout: str) -> sparray:
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
        The dtype of the elements in the matrix

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
    repMsg = generic_msg(
        cmd="random_sparse_matrix",
        args={
            # "dtype": dtype,
            "size": size,
            "density": density,
            # shape : always 2D
            "layout": layout,
            # distributed ? maybe always true? maybe not for now
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
    repMsg = generic_msg(
        cmd="sparse_matrix_matrix_mult",
        args={"arg1": A.name, "arg2": B.name},
    )

    return create_sparray(repMsg)
