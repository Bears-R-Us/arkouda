from typeguard import typechecked
from typing import Callable, Union, Optional

from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray
from arkouda.pdarraycreation import create_pdarray
import cloudpickle
import base64
import numpy as np
from arkouda.numpy.dtypes import dtype


__all__ = [
    "apply",
]


# TODO: it would be nice to typecheck that func takes and returns arr.dtype,
# but that likely requires making pdarray generic over its dtype
@typechecked
def apply(
    arr: pdarray,
    func: Union[Callable, str],
    result_dtype: Optional[Union[np.dtype, str]] = None,
) -> pdarray:
    """
    Apply a python function to a pdarray. The function should take 1 argument
    and return a new value. The function will then be called on each element in
    the pdarray.

    For example,
    >>> apply(ak.array([1, 2, 3]), lambda x: x+1)

    Or,
    >>> import math
    >>> arr = ak.randint(0, 10, 10_000)
    >>> def times_pi(x):
            return x*math.pi
    >>> ak.apply(arr, times_pi, "float64")

    Warning: This function is experimental and may not work as expected.
    Known limitations:
    - The use of some python modules inside of the function may crash the server.
    Known modules know to crash the server are:
        - numpy
        - pandas
        - matplotlib
    - Any python modules used inside of the function must be installed on the server.

    Parameters
    ----------
    arr : pdarray
        The pdarray to which the function is applied

    func : Union[Callable, str]

        The function to apply to the array. This can be a callable function or
        a string, but either way it should take a single argument and return a
        single value. If a string, it should be a lambda function that takes a
        single argument, e.g. "lambda x,: x+1".

    result_dtype : Optional[Union[np.dtype, str]]

        The dtype of the resulting pdarray. If None, the dtype of the resulting
        pdarray will be the same as the input pdarray. If a string, it should be
        a valid numpy dtype string, e.g. "float64". If a numpy dtype, it should
        be a valid numpy dtype object, e.g. np.float64. This is not supported
        for functions passed as strings.

    Returns
    -------
    pdarray
        The pdarray resulting from applying the function to the input array
    """

    if result_dtype is None:
        result_type = arr.dtype
    else:
        result_type = dtype(result_dtype)

    if isinstance(func, str):
        if result_type != arr.dtype:
            raise TypeError("result_dtype must match the dtype of the input")

        repMsg = generic_msg(
            cmd=f"applyStr<{arr.dtype},{arr.ndim}>",
            args={"x": arr, "funcStr": func},
        )
        return create_pdarray(repMsg)
    elif callable(func):
        pickleData = cloudpickle.dumps(func)
        pickleDataStr = base64.b64encode(pickleData).decode("utf-8")
        repMsg = generic_msg(
            cmd=f"applyPickle<{arr.dtype},{arr.ndim},{result_type}>",
            args={"x": arr, "pickleData": pickleDataStr},
        )
        return create_pdarray(repMsg)
    else:
        raise TypeError("func must be a string or a callable function")
