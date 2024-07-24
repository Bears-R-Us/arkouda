from __future__ import annotations

import builtins
from functools import reduce
from typing import Optional, Sequence, Union

import numpy as np
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.dtypes import dtype, int_scalars
from arkouda.logger import getArkoudaLogger

logger = getArkoudaLogger(name="sparrayclass")


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
    dtype : dtype
        The element type of the array
    size : int_scalars
        The size of any one dimension of the array (all dimensions are assumed to be equal sized for now)
    ndim : int_scalars
        The rank of the array (currently only rank 2 arrays supported)
    shape : Sequence[int]
        A list or tuple containing the sizes of each dimension of the array
    layout: str
        The layout of the array ("CSR" or "CSC" are the only valid values)
    itemsize : int_scalars
        The size in bytes of each element
    """

    def __init__(
        self,
        name: str,
        mydtype: Union[np.dtype, str],
        size: int_scalars,
        ndim: int_scalars,
        shape: Sequence[int],
        layout: str,
        itemsize: int_scalars,
        max_bits: Optional[int] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype(mydtype)
        self.size = size
        self.ndim = ndim
        self.shape = shape
        self.layout = layout
        self.itemsize = itemsize
        if max_bits:
            self.max_bits = max_bits

    def __del__(self):
        try:
            logger.debug(f"deleting pdarray with name {self.name}")
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
        return reduce(lambda x, y: x * y, self.shape)

    def __getitem__(self, key):
        raise NotImplementedError("sparray does not support __getitem__")

    # def __str__(self): # This won't work out of the box for sparrays need to add this in later
    #     from arkouda.client import pdarrayIterThresh

    #     return generic_msg(cmd="str", args={"array": self, "printThresh": pdarrayIterThresh})

    # def __repr__(self):
    #     from arkouda.client import pdarrayIterThresh

    #     return generic_msg(cmd="repr", args={"array": self, "printThresh": pdarrayIterThresh})


# creates sparray object
#   only after:
#       all values have been checked by python module and...
#       server has created pdarray already before this is called
@typechecked
def create_sparray(repMsg: str, max_bits=None) -> sparray:
    """
    Return a sparray instance pointing to an array created by the arkouda server.
    The user should not call this function directly.

    Parameters
    ----------
    repMsg : str
        space-delimited string containing the sparray name, datatype, size
        dimension, shape,and itemsize

    Returns
    -------
    sparray
        A sparray with the same attributes as on the server

    Raises
    -----
    ValueError
        If there's an error in parsing the repMsg parameter into the six
        values needed to create the pdarray instance
    RuntimeError
        Raised if a server-side error is thrown in the process of creating
        the pdarray instance
    """
    try:
        fields = repMsg.split()
        name = fields[1]
        mydtype = fields[2]
        size = int(fields[3])
        ndim = int(fields[4])

        if fields[5] == "[]":
            shape = []
        else:
            trailing_comma_offset = -2 if fields[5][len(fields[5]) - 2] == "," else -1
            shape = [int(el) for el in fields[5][1:trailing_comma_offset].split(",")]
        layout = fields[6]
        itemsize = int(fields[7])
    except Exception as e:
        raise ValueError(e)
    logger.debug(
        f"created Chapel sparse array with name: {name} dtype: {mydtype} ndim: {ndim} "
        + f"shape: {shape} layout: {layout} itemsize: {itemsize}"
    )
    return sparray(name, dtype(mydtype), size, ndim, shape, layout, itemsize, max_bits)
