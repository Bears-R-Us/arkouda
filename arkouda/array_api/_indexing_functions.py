from __future__ import annotations

from ._array_object import Array
from ._dtypes import _integer_dtypes

import arkouda as ak

def take(x: Array, indices: Array, /, *, axis: Optional[int] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.take <numpy.take>`.

    See its docstring for more information.
    """
    raise ValueError("take not implemented")
