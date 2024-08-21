from typing import cast as type_cast

from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.dtypes import (
    int64 as ak_int64,
    float64 as ak_float64,
    bool_ as ak_bool,
    uint64 as ak_uint64,
)

NUMERIC_TYPES = [ak_int64, ak_float64, ak_bool, ak_uint64]

__all__ = ["floor"]


@typechecked
def floor(pda: pdarray) -> pdarray:
    """
    Return the element-wise floor of the array.

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing floor values of the input array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Examples
    --------
    >>> ak.floor(ak.linspace(1.1,5.5,5))
    array([1, 2, 3, 4, 5])
    """
    repMsg = generic_msg(
        cmd=f"efunc{pda.ndim}D",
        args={
            "func": "floor",
            "array": pda,
        },
    )
    return create_pdarray(type_cast(str, repMsg))
