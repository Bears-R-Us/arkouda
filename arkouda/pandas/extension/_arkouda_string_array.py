from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar
from typing import cast as type_cast

import numpy as np

from numpy import ndarray
from pandas.api.extensions import ExtensionArray

from arkouda.numpy.dtypes import str_
from arkouda.pandas.extension import ArkoudaArray

from ._arkouda_extension_array import ArkoudaExtensionArray
from ._dtypes import ArkoudaStringDtype


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
else:
    Strings = TypeVar("Strings")

__all__ = ["ArkoudaStringArray"]


class ArkoudaStringArray(ArkoudaExtensionArray, ExtensionArray):
    """
    Arkouda-backed string pandas ExtensionArray.

    Ensures the underlying data is an Arkouda ``Strings`` object. Accepts existing
    ``Strings`` or converts from NumPy arrays and Python sequences of strings.

    Parameters
    ----------
    data : Strings | ndarray | Sequence[Any] | ArkoudaStringArray
        Input to wrap or convert.
        - If ``Strings``, used directly.
        - If NumPy/sequence, converted via ``ak.array``.
        - If another ``ArkoudaStringArray``, its backing ``Strings`` is reused.

    Raises
    ------
    TypeError
        If ``data`` cannot be converted to Arkouda ``Strings``.

    Attributes
    ----------
    default_fill_value : str
        Sentinel used when filling missing values (default: "").
    """

    default_fill_value: str = ""

    def __init__(self, data: Strings | ndarray | Sequence[Any] | "ArkoudaStringArray"):
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.numpy.strings import Strings

        if isinstance(data, ArkoudaStringArray):
            self._data = data._data
            return

        if isinstance(data, (np.ndarray, list, tuple)):
            data = type_cast(Strings, ak_array(data, dtype="str_"))

        if not isinstance(data, Strings):
            raise TypeError(f"Expected arkouda.Strings, got {type(data).__name__}")

        self._data = data

    @property
    def dtype(self):
        return ArkoudaStringDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        from arkouda.numpy.pdarraycreation import array as ak_array

        return cls(ak_array(scalars))

    def __getitem__(self, key):
        result = self._data[key]
        if np.isscalar(key):
            if hasattr(result, "to_ndarray"):
                return result.to_ndarray()[()]
            else:
                return result
        return ArkoudaStringArray(result)

    def astype(self, dtype, copy: bool = False):
        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=copy)
        # Let pandas do the rest locally
        return self.to_ndarray().astype(dtype, copy=copy)

    def isna(self):
        from arkouda.numpy.pdarraycreation import zeros

        return zeros(self._data.size, dtype="bool")

    def __eq__(self, other):
        """
        Elementwise equality for string arrays using pandas ExtensionArray semantics.
        Returns ArkoudaArray of booleans.
        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraycreation import array as ak_array

        # Case 1: ArkoudaStringArray
        if isinstance(other, ArkoudaStringArray):
            if len(self) != len(other):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other._data)

        # Case 2: arkouda pdarray (should contain encoded string indices)
        if isinstance(other, pdarray):
            if other.size not in (1, len(self)):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other)

        # Case 3: scalar (string or bytes)
        if isinstance(other, (str, str_)):
            return ArkoudaArray(self._data == other)

        # Case 4: numpy array or Python sequence
        if isinstance(other, (list, tuple, np.ndarray)):
            other_ak = ak_array(other)
            if other_ak.size == 1:
                return ArkoudaArray(self._data == other_ak[0])
            if other_ak.size != len(self):
                raise ValueError("Lengths must match for elementwise comparison")
            return ArkoudaArray(self._data == other_ak)

        # Case 5: unsupported type
        return NotImplemented

    def __repr__(self):
        return f"ArkoudaStringArray({self._data})"

    def _not_implemented(self, name: str):
        raise NotImplementedError(f"`{name}` is not implemented for Arkouda-backed arrays yet.")

    def all(self, *args, **kwargs):
        self._not_implemented("all")

    def any(self, *args, **kwargs):
        self._not_implemented("any")

    def argpartition(self, *args, **kwargs):
        self._not_implemented("argpartition")

    def byteswap(self, *args, **kwargs):
        self._not_implemented("byteswap")

    def choose(self, *args, **kwargs):
        self._not_implemented("choose")

    def clip(self, *args, **kwargs):
        self._not_implemented("clip")

    def compress(self, *args, **kwargs):
        self._not_implemented("compress")

    def conj(self, *args, **kwargs):
        self._not_implemented("conj")

    def conjugate(self, *args, **kwargs):
        self._not_implemented("conjugate")

    def cumprod(self, *args, **kwargs):
        self._not_implemented("cumprod")

    def cumsum(self, *args, **kwargs):
        self._not_implemented("cumsum")

    def diagonal(self, *args, **kwargs):
        self._not_implemented("diagonal")

    def dot(self, *args, **kwargs):
        self._not_implemented("dot")

    def dump(self, *args, **kwargs):
        self._not_implemented("dump")

    def dumps(self, *args, **kwargs):
        self._not_implemented("dumps")

    def fill(self, *args, **kwargs):
        self._not_implemented("fill")

    def flatten(self, *args, **kwargs):
        self._not_implemented("flatten")

    def getfield(self, *args, **kwargs):
        self._not_implemented("getfield")

    def item(self, *args, **kwargs):
        self._not_implemented("item")

    def max(self, *args, **kwargs):
        self._not_implemented("max")

    def mean(self, *args, **kwargs):
        self._not_implemented("mean")

    def min(self, *args, **kwargs):
        self._not_implemented("min")

    def nonzero(self, *args, **kwargs):
        self._not_implemented("nonzero")

    def partition(self, *args, **kwargs):
        self._not_implemented("partition")

    def prod(self, *args, **kwargs):
        self._not_implemented("prod")

    def put(self, *args, **kwargs):
        self._not_implemented("put")

    def resize(self, *args, **kwargs):
        self._not_implemented("resize")

    def round(self, *args, **kwargs):
        self._not_implemented("round")

    def setfield(self, *args, **kwargs):
        self._not_implemented("setfield")

    def setflags(self, *args, **kwargs):
        self._not_implemented("setflags")

    def sort(self, *args, **kwargs):
        self._not_implemented("sort")

    def std(self, *args, **kwargs):
        self._not_implemented("std")

    def sum(self, *args, **kwargs):
        self._not_implemented("sum")

    def swapaxes(self, *args, **kwargs):
        self._not_implemented("swapaxes")

    def to_device(self, *args, **kwargs):
        self._not_implemented("to_device")

    def tobytes(self, *args, **kwargs):
        self._not_implemented("tobytes")

    def tofile(self, *args, **kwargs):
        self._not_implemented("tofile")

    def trace(self, *args, **kwargs):
        self._not_implemented("trace")

    def var(self, *args, **kwargs):
        self._not_implemented("var")
