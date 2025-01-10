from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray
from arkouda.pdarraycreation import create_pdarray

__all__ = [
    "apply",
]

@typechecked
def apply(arr: pdarray, func: str) -> pdarray:
    """
    Apply a lambda function to a pdarray. The lambda function should be a string

    For example,
    >>> apply(ak.array([1, 2, 3]), "lambda x,: x+1")

    Parameters
    ----------
    arr : pdarray
        The pdarray to which the function is applied

    func : str

        The lambda function to apply to the array. This should be a string that
        has the following format: "lambda: x,: ...", where the "..." can be a
        valid python expression.

    Returns
    -------
    pdarray
        The pdarray resulting from applying the lambda function to the input array
    """

    repMsg = generic_msg(cmd=f"apply<{arr.dtype},{arr.ndim}>",
                            args={"x" : arr, "funcStr" : func})
    return create_pdarray(repMsg)
