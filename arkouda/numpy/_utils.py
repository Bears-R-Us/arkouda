from typing import Iterable, Tuple, Union

from numpy import ndarray

from arkouda.numpy.dtypes import all_scalars, isSupportedDType
from arkouda.pdarrayclass import pdarray
from arkouda.strings import Strings

__all__ = ["shape"]


def shape(a: Union[pdarray, Strings, all_scalars]) -> Tuple:
    """
    Return the shape of an array.

    Parameters
    ----------
    a : pdarray
        Input array.

    Returns
    -------
    shape : tuple of ints
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
        try:
            result = a.shape
        except AttributeError:
            from arkouda import array

            result = array(a).shape
        return result
    elif isSupportedDType(a):
        return ()
    else:
        raise TypeError("shape requires type pdarray, ndarray, Iterable, or numeric scalar.")
