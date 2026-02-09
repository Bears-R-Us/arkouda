from typing import Iterable, Tuple, Union

from numpy import ndarray

from arkouda.numpy.dtypes import all_scalars, is_supported_dtype
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.strings import Strings


__all__ = ["shape"]


def shape(a: Union[pdarray, Strings, all_scalars]) -> Tuple:
    """
    Return the shape of an array.

    Parameters
    ----------
    a : pdarray, Strings, or all_scalars
        Input array.

    Returns
    -------
    Tuple
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.shape(ak.eye(3,2))
    (3, 2)
    >>> ak.shape([[1, 3]])
    (1, 2)
    >>> ak.shape([0])
    (1,)
    >>> ak.shape(0)
    ()

    """
    if isinstance(a, (pdarray, Strings, ndarray, Iterable)) and not isinstance(a, str):
        if isinstance(a, (pdarray, Strings, ndarray)):
            result = a.shape
        else:
            from arkouda import array

            result = array(a).shape
        return result
    elif is_supported_dtype(a):
        return ()
    else:
        raise TypeError("shape requires type pdarray, ndarray, Iterable, or numeric scalar.")
