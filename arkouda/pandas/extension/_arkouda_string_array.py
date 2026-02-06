from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar, Union, overload
from typing import cast as type_cast

import numpy as np

from numpy import ndarray
from numpy.typing import NDArray
from pandas import StringDtype as pd_StringDtype
from pandas.api.extensions import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype

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

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve one or more string values.

        Parameters
        ----------
        key : Any
            Positional indexer. Supports:
            * scalar integer positions
            * slice objects
            * NumPy integer arrays (signed/unsigned)
            * NumPy boolean masks
            * Python lists of integers / booleans
            * Arkouda pdarray indexers (int / uint / bool)

        Returns
        -------
        Any
            A Python string for scalar access, or a new ArkoudaStringArray
            for non-scalar indexers.

        Raises
        ------
        TypeError
            If ``key`` is a NumPy array with an unsupported dtype (for example,
            a floating point or object dtype).

        Examples
        --------
        Basic scalar access:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaStringArray
        >>> arr = ArkoudaStringArray(ak.array(["a", "b", "c", "d"]))
        >>> arr[1]
        np.str_('b')

        Negative indexing:

        >>> arr[-1]
        np.str_('d')

        Slice indexing (returns a new ArkoudaStringArray):

        >>> arr[1:3]
        ArkoudaStringArray(['b', 'c'])

        NumPy integer array indexing:

        >>> idx = np.array([0, 2], dtype=np.int64)
        >>> arr[idx]
        ArkoudaStringArray(['a', 'c'])

        NumPy boolean mask:

        >>> mask = np.array([True, False, True, False])
        >>> arr[mask]
        ArkoudaStringArray(['a', 'c'])

        Arkouda integer indexer:

        >>> ak_idx = ak.array([3, 1])
        >>> arr[ak_idx]
        ArkoudaStringArray(['d', 'b'])

        Empty indexer returns an empty ArkoudaStringArray:

        >>> empty_idx = np.array([], dtype=np.int64)
        >>> arr[empty_idx]
        ArkoudaStringArray([])
        """
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.numpy.strings import Strings

        # Normalize NumPy indexers to Arkouda pdarrays, mirroring ArkoudaArray.__getitem__
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = ak_array(key)
            elif key.dtype.kind in {"i"}:
                # signed integer
                key = ak_array(key, dtype="int64")
            elif key.dtype.kind in {"u"}:
                # unsigned integer
                key = ak_array(key, dtype="uint64")
            else:
                raise TypeError(f"Unsupported numpy index type {key.dtype}")

        result = self._data[key]

        # Scalar access: return a plain Python str (or scalar) instead of a Strings object
        if np.isscalar(key):
            return result

        # Non-scalar: expect an Arkouda Strings, wrap it
        if isinstance(result, Strings):
            return ArkoudaStringArray(result)

        # Fallback: if Arkouda returned something array-like but not Strings,
        # materialize via ak.array and wrap again as Strings.
        return ArkoudaStringArray(ak_array(result))

    # docstr-coverage:excused `typing-only overload stub`
    @overload
    def astype(self, dtype: np.dtype[Any], copy: bool = True) -> NDArray[Any]: ...

    # docstr-coverage:excused `typing-only overload stub`
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = True) -> ExtensionArray: ...

    # docstr-coverage:excused `typing-only overload stub`
    @overload
    def astype(self, dtype: Any, copy: bool = True) -> Union[ExtensionArray, NDArray[Any]]: ...

    def astype(
        self,
        dtype: Any,
        copy: bool = True,
    ) -> Union[ExtensionArray, NDArray[Any]]:
        """
        Cast to a specified dtype.

        Casting rules:

        * If ``dtype`` requests ``object``, returns a NumPy ``NDArray[Any]`` of dtype
          ``object`` containing the string values.
        * If ``dtype`` is a string dtype (e.g. pandas ``StringDtype``, NumPy unicode,
          or Arkouda string dtype), returns an ``ArkoudaStringArray``. If ``copy=True``,
          attempts to copy the underlying Arkouda ``Strings`` data.
        * For all other dtypes, casts the underlying Arkouda ``Strings`` using
          ``Strings.astype`` and returns an Arkouda-backed ``ArkoudaExtensionArray``
          constructed from the result.

        Parameters
        ----------
        dtype : Any
            Target dtype. May be a NumPy dtype, pandas dtype, or Arkouda dtype.
        copy : bool
            Whether to force a copy when the result is an ``ArkoudaStringArray``.
            Default is True.

        Returns
        -------
        Union[ExtensionArray, NDArray[Any]]
            The cast result. Returns a NumPy array only when casting to ``object``;
            otherwise returns an Arkouda-backed ExtensionArray.

        Examples
        --------
        Casting to a string dtype returns an Arkouda-backed string array:

        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaStringArray
        >>> s = ArkoudaStringArray(ak.array(["a", "b", "c"]))
        >>> out = s.astype("string")
        >>> out is s
        False

        Forcing a copy when casting to a string dtype returns a new array:

        >>> out2 = s.astype("string", copy=True)
        >>> out2 is s
        False
        >>> out2.to_ndarray()
        array(['a', 'b', 'c'], dtype='<U1')

        Casting to ``object`` materializes the data to a NumPy array:

        >>> s.astype(object)
        array(['a', 'b', 'c'], dtype=object)

        Casting to a non-string dtype uses Arkouda to cast the underlying strings
        and returns an Arkouda-backed ExtensionArray:

        >>> s_num = ArkoudaStringArray(ak.array(["1", "2", "3"]))
        >>> a = s_num.astype("int64")
        >>> a.to_ndarray()
        array([1, 2, 3])

        NumPy and pandas dtype objects are also accepted:

        >>> import numpy as np
        >>> a = s_num.astype(np.dtype("float64"))
        >>> a.to_ndarray()
        array([1., 2., 3.])
        """
        from arkouda.numpy._typing._typing import is_string_dtype_hint
        from arkouda.numpy.dtypes import dtype as ak_dtype

        # --- 1) ExtensionDtype branch first (satisfies overload #2) ---
        if isinstance(dtype, ExtensionDtype):
            if hasattr(dtype, "numpy_dtype"):
                dtype = dtype.numpy_dtype

            if isinstance(dtype, pd_StringDtype) or is_string_dtype_hint(dtype):
                if not copy:
                    return self
                data = self._data.copy() if hasattr(self._data, "copy") else self._data
                return type_cast(ExtensionArray, type(self)(data))

            dtype = ak_dtype(dtype)
            casted = self._data.astype(dtype)
            return type_cast(ExtensionArray, ArkoudaExtensionArray._from_sequence(casted))

        # --- 2) object -> numpy (satisfies overload #1 / general) ---
        if dtype in (object, np.object_, "object", np.dtype("O")):
            return self.to_ndarray().astype(object, copy=False)

        # string targets -> stay string EA
        if isinstance(dtype, pd_StringDtype) or is_string_dtype_hint(dtype):
            if not copy:
                return self
            data = self._data.copy() if hasattr(self._data, "copy") else self._data
            return type(self)(data)

        dtype = ak_dtype(dtype)
        casted = self._data.astype(dtype)
        return ArkoudaExtensionArray._from_sequence(casted)

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
