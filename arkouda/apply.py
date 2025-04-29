import base64
import sys
from typing import Callable, Optional, Union, cast

import cloudpickle
import numpy as np
from typeguard import typechecked

from arkouda.client import generic_msg, get_config
from arkouda.numpy.dtypes import dtype
from arkouda.numpy.pdarrayclass import parse_single_value, pdarray
from arkouda.numpy.pdarraycreation import create_pdarray

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
    Apply a python function to a pdarray.

    The function should take one argument
    and return a new value. The function will then be called on each element in
    the pdarray.

    For example,
    >>> ak.apply(ak.array([1, 2, 3]), lambda x: x+1)

    Or,
    >>> import math
    >>> arr = ak.randint(0, 10, 10_000)
    >>> def times_pi(x):
            return x*math.pi
    >>> ak.apply(arr, times_pi, "float64")

    Warning: This function is experimental and may not work as expected.
    Known limitations:
    - Any python modules used inside of the function must be installed on the server.

    Parameters
    ----------
    arr : pdarray
        The pdarray to which the function is applied

    func : Union[Callable, str]

        The function to apply to the array. This can be a callable function or
        a string, but either way it should take a single argument and return a
        single value. If a string, it should be a lambda function that takes a
        single argument, e.g. "lambda x,: x+1". Note the dangling comma after
        the argument, this is required for string functions.

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
    if getattr(apply, "is_apply_supported", None) is None:
        res = generic_msg("isPythonModuleSupported")
        is_supported = parse_single_value(cast(str, res))
        setattr(apply, "is_apply_supported", is_supported)

    if not getattr(apply, "is_apply_supported", False):
        raise RuntimeError(
            "The apply module is not supported by the version of Chapel " + "this server was built with."
        )

    vers_supported = getattr(apply, "is_version_supported", None)
    if vers_supported is None:
        interp_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        cmd_res = generic_msg("isVersionSupported", args={"versionString": interp_version})
        vers_supported = parse_single_value(cast(str, cmd_res))
        setattr(apply, "is_version_supported", vers_supported)
    if not vers_supported:
        interp_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        server_version = get_config()["pythonVersion"]
        raise RuntimeError(
            f"The current Python interpreter version ({interp_version}) "
            + f"does not match the server ({server_version})."
        )

    if not getattr(apply, "is_initialized", False):
        generic_msg("initPythonInterpreters")
        setattr(apply, "is_initialized", True)

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
