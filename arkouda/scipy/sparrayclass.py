from __future__ import annotations

import builtins

from typing import List, Optional, Sequence, Union, cast
from typing import cast as type_cast

import numpy as np

from typeguard import typechecked

from arkouda.logger import get_arkouda_logger
from arkouda.numpy.dtypes import NumericDTypes, dtype, int_scalars
from arkouda.numpy.pdarrayclass import create_pdarrays, pdarray


logger = get_arkouda_logger(name="sparrayclass")

__all__ = [
    "sparray",
    "create_sparray",
]


class sparray:
    """
    The class for sparse arrays. This class contains only the
    attributies of the array; the data resides on the arkouda
    server. When a server operation results in a new array, arkouda
    will create a sparray instance that points to the array data on
    the server. As such, the user should not initialize sparray
    instances directly.

    Attributes
    ----------
    name : str
        The server-side identifier for the array
    dtype : type
        The element dtype of the array
    size : int_scalars
        The size of any one dimension of the array (all dimensions are assumed to be equal sized for now)
    nnz: int_scalars
        The number of non-zero elements in the array
    ndim : int_scalars
        The rank of the array (currently only rank 2 arrays supported)
    shape : Sequence[int]
        A list or tuple containing the sizes of each dimension of the array
    layout: str
        The layout of the array ("CSR" or "CSC" are the only valid values)
    itemsize : int_scalars
        The size in bytes of each element
    """

    name: str
    dtype: type
    size: int_scalars
    nnz: int_scalars
    ndim: int_scalars
    shape: Sequence[int]
    layout: str
    itemsize: int_scalars

    def __init__(
        self,
        name: str,
        mydtype: Union[np.dtype, str],
        size: int_scalars,
        nnz: int_scalars,
        ndim: int_scalars,
        shape: Sequence[int],
        layout: str,
        itemsize: int_scalars,
        max_bits: Optional[int] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype(mydtype)
        self.size = size
        self.nnz = nnz
        self.ndim = ndim
        self.shape = shape
        self.layout = layout
        self.itemsize = itemsize
        if max_bits:
            self.max_bits = max_bits

    def __del__(self):  # pragma: no cover
        from arkouda.client import generic_msg

        try:
            logger.debug(f"deleting sparray with name {self.name}")
            generic_msg(cmd="delete", args={"name": self.name})
        except (RuntimeError, AttributeError):
            pass

    def __bool__(self) -> builtins.bool:
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous."
                "Use a.any() or a.all()"
            )
        return builtins.bool(self[0])

    def __len__(self):
        return self.nnz  # This is the number of non-zero elements in the matrix

    def __getitem__(self, key):
        raise NotImplementedError("sparray does not support __getitem__")

    def __str__(self):
        from arkouda.client import generic_msg, sparrayIterThresh

        return generic_msg(cmd="str", args={"array": self, "printThresh": sparrayIterThresh})

    # def __repr__(self):
    #     from arkouda.client import sparrayIterThresh
    #     print("Called repr")
    #     return generic_msg(cmd="repr", args={"array": self, "printThresh": sparrayIterThresh})

    """
    Converts the sparse matrix to a list of 3 pdarrays (rows, cols, vals)

    Returns
    -------
    List[ak.pdarray]
        A list of 3 pdarrays which contain the row indices, the column indices,
        and the values at the respective indices within the sparse matrix.
    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.random_sparse_matrix(100,0.2,"CSR");
    >>> a.to_pdarray()
    [array([1 1 1 ... 100 100 100]), array([17 21 29 ... 75 77 85]), array([0 0 0 ... 0 0 0])]
    """

    @typechecked
    def to_pdarray(self) -> List[pdarray]:
        from arkouda.client import generic_msg

        dtype = self.dtype
        dtype_name = cast(np.dtype, dtype).name
        # check dtype for error
        if dtype_name not in NumericDTypes:
            raise TypeError(f"unsupported dtype {dtype}")
        response_arrays = generic_msg(
            cmd=f"sparse_to_pdarrays<{self.dtype},{self.layout}>", args={"matrix": self}
        )
        array_list = create_pdarrays(type_cast(str, response_arrays))
        return array_list

    """"""

    def fill_vals(self, a: pdarray):
        from arkouda.client import generic_msg

        if self.dtype != a.dtype:
            raise ValueError("sparray and pdarray must have the same dtype for fill_vals")

        if a.ndim != 1:
            raise ValueError("pdarray must be 1D for fill_vals")

        generic_msg(
            cmd=f"fill_sparse_vals<{self.dtype},2,{self.layout},{a.dtype},1>",
            args={"matrix": self, "vals": a},
        )


# creates sparray object
#   only after:
#       all values have been checked by python module and...
#       server has created pdarray already before this is called
@typechecked
def create_sparray(rep_msg: str, max_bits=None) -> sparray:
    """
    Return a sparray instance pointing to an array created by the arkouda server.
    The user should not call this function directly.

    Parameters
    ----------
    rep_msg : str
        space-delimited string containing the sparray name, datatype, size
        dimension, shape,and itemsize

    Returns
    -------
    sparray
        A sparray with the same attributes as on the server

    Raises
    ------
    ValueError
        If there's an error in parsing the rep_msg parameter into the six
        values needed to create the pdarray instance
    RuntimeError
        Raised if a server-side error is thrown in the process of creating
        the pdarray instance
    """
    try:
        fields = rep_msg.split()
        name = fields[1]
        mydtype = fields[2]
        size = int(fields[3])
        nnz = int(fields[4])
        ndim = int(fields[5])

        if fields[6] == "[]":
            shape = []
        else:
            trailing_comma_offset = -2 if fields[6][len(fields[6]) - 2] == "," else -1
            shape = [int(el) for el in fields[6][1:trailing_comma_offset].split(",")]
        layout = fields[7]
        itemsize = int(fields[8])
    except Exception as e:
        raise ValueError(e)
    logger.debug(
        f"created Chapel sparse array with name: {name} dtype: {mydtype} ndim: {ndim} "
        + f"nnz:{nnz} shape: {shape} layout: {layout} itemsize: {itemsize}"
    )
    return sparray(
        name,
        dtype(mydtype),
        size,
        nnz,
        ndim,
        shape,
        layout,
        itemsize,
        max_bits,  # type: ignore
    )
