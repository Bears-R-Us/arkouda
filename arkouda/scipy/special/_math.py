from typing import Union
from warnings import warn

import numpy as np

from arkouda.numpy import log
from arkouda.numpy.pdarrayclass import pdarray

__all__ = [
    "xlogy",
]


def xlogy(x: Union[pdarray, np.float64], y: pdarray):
    """
    Computes x * log(y).

    Parameters
    ----------
    x : pdarray or np.float64
        x must have a datatype that is castable to float64
    y : pdarray

    Returns
    -------
    arkouda.numpy.pdarrayclass.pdarray

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.scipy.special import xlogy
    >>> xlogy( ak.array([1, 2, 3, 4]),  ak.array([5,6,7,8]))
    array([1.6094379124341003 3.5835189384561099 5.8377304471659395 8.317766166719343])
    >>> xlogy( 5.0, ak.array([1, 2, 3, 4]))
    array([0.00000000000000000 3.4657359027997265 5.4930614433405491 6.9314718055994531])

    """
    if not isinstance(x, (np.float64, pdarray)) and np.can_cast(x, np.float64):
        x = np.float64(x)

    if isinstance(x, pdarray) and isinstance(y, pdarray):
        if x.size == y.size:
            return x * log(y)
        else:
            msg = "x and y must have the same size."
            warn(msg, UserWarning)
            return None
    elif isinstance(x, np.float64) and isinstance(y, pdarray):
        return x * log(y)
    else:
        msg = "x and y must both be pdarrays or x must be castable to float64 and y must be a pdarray."
        warn(msg, UserWarning)
        return None
