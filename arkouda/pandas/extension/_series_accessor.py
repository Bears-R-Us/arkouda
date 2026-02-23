"""
Series pandas accessor for Arkouda-backed data.

This module defines the ``.ak`` accessor for :class:`pandas.Series`,
enabling seamless conversion between:

* standard NumPy-backed pandas Series
* pandas Series backed by :class:`~arkouda.pandas.extension.ArkoudaExtensionArray`
  (zero-copy, distributed Arkouda representation)
* legacy Arkouda server-side arrays (pdarray, Strings, Categorical)

The API parallels the Index/MultiIndex accessor:

- ``s.ak.to_ak()``
    Convert a pandas Series to an Arkouda-backed Series using
    :class:`ArkoudaExtensionArray`.

- ``s.ak.collect()``
    Materialize an Arkouda-backed Series into a normal NumPy-backed Series.

- ``s.ak.to_ak_legacy()``
    Convert to a legacy Arkouda array object (pdarray, Strings, Categorical).

- ``pd.Series.ak.from_ak_legacy(ak_arr)``
    Build a pandas Series backed by Arkouda arrays, without materializing.

- ``s.ak.is_arkouda``
    Inspect whether the Series is currently Arkouda-backed.

All operations avoid materializing to NumPy unless explicitly requested.
"""

from __future__ import annotations

from typing import Any, Union

import pandas as pd

from pandas import Index as pd_Index
from pandas.api.extensions import register_series_accessor

from arkouda.pandas.extension import ArkoudaExtensionArray, ArkoudaIndexAccessor
from arkouda.pandas.groupbyclass import GroupBy
from arkouda.pandas.series import Series as ak_Series


# ---------------------------------------------------------------------------
# Helpers: wrapping and unwrapping
# ---------------------------------------------------------------------------


def _ak_arr_to_pandas_ea(akarr: Any) -> ArkoudaExtensionArray:
    """
    Wrap a legacy Arkouda array (pdarray, Strings, Categorical)
    into an ArkoudaExtensionArray without NumPy materialization.
    """
    return ArkoudaExtensionArray._from_sequence(akarr)


def _pandas_series_to_ak_array(s: pd.Series) -> Any:
    """
    Convert a pandas Series (NumPy or EA-backed) into an Arkouda array.

    If already Arkouda-backed, peel off the '_data' member.
    """
    from arkouda.numpy.pdarraycreation import array as ak_array

    arr = s.array

    if isinstance(arr, ArkoudaExtensionArray):
        akcol = getattr(arr, "_data", None)
        if akcol is None:
            raise TypeError("Arkouda-backed Series array does not expose '_data'")
        return akcol

    return ak_array(s)


def _ak_array_to_pandas_series(
    akarr: Any,
    name: str | None = None,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Wrap a legacy Arkouda array into a pandas Series backed by ArkoudaExtensionArray.

    If `index` is not provided, construct an Arkouda-backed default index.
    If `index` is provided and not Arkouda-backed, convert it.
    """
    from arkouda.numpy.pdarraycreation import arange as ak_arange
    from arkouda.pandas.extension import ArkoudaExtensionArray, ArkoudaIndexAccessor

    ea = _ak_arr_to_pandas_ea(akarr)

    if index is None:
        index = pd.Index(ArkoudaExtensionArray._from_sequence(ak_arange(len(ea))))

    if not ArkoudaIndexAccessor(index).is_arkouda:
        index = ArkoudaIndexAccessor(index).to_ak()

    return pd.Series(ea, index=index, name=name)


# ---------------------------------------------------------------------------
# Series accessor
# ---------------------------------------------------------------------------


@register_series_accessor("ak")
class ArkoudaSeriesAccessor:
    """
    Arkouda-backed Series accessor.

    Provides a symmetric API to the Index accessor for Series-level
    conversion and materialization.

    Parameters
    ----------
    pandas_obj : pd.Series
        The Series this accessor wraps.

    Examples
    --------
    >>> import pandas as pd
    >>> import arkouda as ak
    >>> s = pd.Series([1, 2, 3], name="nums")

    Convert to Arkouda-backed:

    >>> ak_s = s.ak.to_ak()
    >>> ak_s.ak.is_arkouda
    True

    Materialize back:

    >>> restored = ak_s.ak.collect()
    >>> restored.equals(s)
    True

    Convert to legacy Arkouda:

    >>> ak_arr = s.ak.to_ak_legacy()
    >>> type(ak_arr)
    <class 'arkouda.pandas.series.Series'>

    """

    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    # ------------------------------------------------------------------
    # Distributed / local conversion
    # ------------------------------------------------------------------

    def to_ak(self) -> pd.Series:
        """
        Convert this pandas Series into an Arkouda-backed Series.

        This method produces a pandas ``Series`` whose underlying storage uses
        :class:`~arkouda.pandas.extension.ArkoudaExtensionArray`, meaning the
        data reside on the Arkouda server rather than in local NumPy buffers.
        The conversion is zero-copy with respect to NumPy: data are only
        materialized if the original Series is NumPy-backed.

        The returned Series preserves the original index (including index names)
        and the original Series ``name``.

        Returns
        -------
        pd.Series
            A Series backed by an :class:`ArkoudaExtensionArray`, referencing
            Arkouda server-side arrays. The resulting Series retains the original
            index and name.

        Notes
        -----
        * If the Series is already Arkouda-backed, this method returns a new
          Series that is semantically equivalent and still Arkouda-backed.
        * If the Series is NumPy-backed, values are transferred to Arkouda
          server-side arrays via ``ak.array``.
        * No NumPy-side materialization occurs when converting an already
          Arkouda-backed Series.

        Examples
        --------
        Basic numeric conversion:

        >>> import pandas as pd
        >>> import arkouda as ak
        >>> s = pd.Series([1, 2, 3], name="nums")
        >>> s_ak = s.ak.to_ak()
        >>> type(s_ak.array)
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>
        >>> s_ak.tolist()
        [np.int64(1), np.int64(2), np.int64(3)]

        Preserving the index and name:

        >>> idx = pd.Index([10, 20, 30], name="id")
        >>> s = pd.Series([100, 200, 300], index=idx, name="values")
        >>> s_ak = s.ak.to_ak()
        >>> s_ak.name
        'values'
        >>> s_ak.index.name
        'id'

        String data:

        >>> s = pd.Series(["red", "blue", "green"], name="colors")
        >>> s_ak = s.ak.to_ak()
        >>> s_ak.tolist()
        [np.str_('red'), np.str_('blue'), np.str_('green')]

        Idempotence (calling ``to_ak`` repeatedly stays Arkouda-backed):

        >>> s_ak2 = s_ak.ak.to_ak()
        >>> s_ak2.ak.is_arkouda
        True
        >>> s_ak2.tolist() == s_ak.tolist()
        True
        """
        if self.is_arkouda:
            return self._obj

        idx: Union[None, pd_Index] = self._obj.index
        if isinstance(idx, pd_Index):
            idx = ArkoudaIndexAccessor(self._obj.index).to_ak()

        akarr = _pandas_series_to_ak_array(self._obj)
        ea = _ak_arr_to_pandas_ea(akarr)

        return pd.Series(ea, index=idx, name=self._obj.name)

    def collect(self) -> pd.Series:
        """
        Materialize this Series back to a NumPy-backed pandas Series.

        Returns
        -------
        pd.Series
            A NumPy-backed Series.

        Examples
        --------
        >>> s = pd.Series([1,2,3]).ak.to_ak()
        >>> out = s.ak.collect()
        >>> type(out.array)
        <class 'pandas...NumpyExtensionArray'>
        """
        s = self._obj
        arr = s.array

        values = arr.to_numpy()
        idx = ArkoudaIndexAccessor(s.index).collect()

        return pd.Series(values, index=idx, name=s.name)

    # ------------------------------------------------------------------
    # Legacy Arkouda conversions
    # ------------------------------------------------------------------

    def to_ak_legacy(self) -> ak_Series:
        """
        Convert this Series into a legacy Arkouda Series.

        Returns
        -------
        ak_Series
            The legacy Arkouda Series..

        Examples
        --------
        >>> import pandas as pd
        >>> s = pd.Series([10,20,30])
        >>> ak_arr = s.ak.to_ak_legacy()
        >>> type(ak_arr)
        <class 'arkouda.pandas.series.Series'>
        """
        idx = ArkoudaIndexAccessor(self._obj.index).to_ak_legacy()

        return ak_Series(_pandas_series_to_ak_array(self._obj), index=idx, name=self._obj.name)

    @staticmethod
    def from_ak_legacy(akarr: Any, name: str | None = None) -> pd.Series:
        """
        Construct an Arkouda-backed pandas Series directly from a legacy Arkouda array.

        This performs zero-copy wrapping using ArkoudaExtensionArray and does
        not materialize data.

        Parameters
        ----------
        akarr : Any
            A legacy Arkouda array (pdarray, Strings, or Categorical).
        name : str | None
            Optional. Name of the resulting Series.

        Returns
        -------
        pd.Series
            A pandas Series backed by ArkoudaExtensionArray.

        Examples
        --------
        >>> import arkouda as ak
        >>> import pandas as pd

        Basic example with a legacy ``pdarray``:

        >>> ak_arr = ak.arange(5)
        >>> s = pd.Series.ak.from_ak_legacy(ak_arr, name="values")
        >>> s
        0    0
        1    1
        2    2
        3    3
        4    4
        Name: values, dtype: int64

        The underlying data remain on the Arkouda server:

        >>> type(s._values)
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>

        Using a legacy ``Strings`` object:

        >>> ak_str = ak.array(["a", "b", "c"])
        >>> s_str = pd.Series.ak.from_ak_legacy(ak_str, name="letters")
        >>> s_str
        0    a
        1    b
        2    c
        Name: letters, dtype: string

        Using a legacy ``Categorical``:

        >>> ak_cat = ak.Categorical(ak.array(["red", "blue", "red"]))
        >>> s_cat = pd.Series.ak.from_ak_legacy(ak_cat, name="color")
        >>> s_cat
        0     red
        1    blue
        2     red
        Name: color, dtype: category

        No NumPy copies are made—the Series is a zero-copy wrapper over
        Arkouda server-side arrays.
        """
        return _ak_array_to_pandas_series(akarr, name=name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_arkouda(self) -> bool:
        """
        Return whether the underlying Series is Arkouda-backed.

        A Series is Arkouda-backed if its underlying storage uses
        :class:`ArkoudaExtensionArray`.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.ak.is_arkouda
        False

        >>> ak_s = s.ak.to_ak()
        >>> ak_s.ak.is_arkouda
        True
        """
        arr = getattr(self._obj, "array", None)
        idx_arr = self._obj.index.values
        return isinstance(arr, ArkoudaExtensionArray) and isinstance(idx_arr, ArkoudaExtensionArray)

    # ------------------------------------------------------------------
    # Legacy delegation: thin wrappers over ak.Series
    # ------------------------------------------------------------------

    def locate(self, key: object) -> pd.Series:
        """
        Lookup values by index label on the Arkouda server.

        This is a thin wrapper around the legacy :meth:`arkouda.pandas.series.Series.locate`
        method. It converts the pandas Series to a legacy Arkouda ``ak.Series``,
        performs the locate operation on the server, and wraps the result back into
        an Arkouda-backed pandas Series (ExtensionArray-backed) without NumPy
        materialization.

        Parameters
        ----------
        key : object
            Lookup key or keys. Interpreted in the same way as the legacy Arkouda
            ``Series.locate`` method. This may be:
            - a scalar
            - a list/tuple of scalars
            - an Arkouda ``pdarray``
            - an Arkouda ``Index`` / ``MultiIndex``
            - an Arkouda ``Series`` (special case: preserves key index)

        Returns
        -------
        pd.Series
            A pandas Series backed by Arkouda ExtensionArrays containing the located
            values. The returned Series remains distributed (no NumPy materialization)
            and is sorted by index.

        Notes
        -----
        * This method is Arkouda-specific; pandas does not define ``Series.locate``.
        * If ``key`` is a pandas Index/MultiIndex, consider converting it via
          ``key.ak.to_ak_legacy()`` before calling ``locate`` for the most direct path.

        Examples
        --------
        >>> import arkouda as ak
        >>> import pandas as pd
        >>> s = pd.Series([10, 20, 30], index=pd.Index([1, 2, 3])).ak.to_ak()
        >>> out = s.ak.locate([3, 1])
        >>> out.tolist()
        [np.int64(10), np.int64(30)]
        """
        # Lift the pandas Series into a legacy Arkouda Series
        aks = self.to_ak_legacy()

        # Normalize common pandas key types to legacy Arkouda where possible
        # (mirrors the “thin wrapper” approach used in the Index accessor).
        if isinstance(key, (pd.Index, pd.MultiIndex)):
            key = ArkoudaIndexAccessor(key).to_ak_legacy()
        elif isinstance(key, pd.Series):
            # For pandas Series keys, convert to legacy ak.Series to preserve key.index semantics.
            key = ArkoudaSeriesAccessor(key).to_ak_legacy()

        out_ak = aks.locate(key)

        # Wrap result into an Arkouda-backed pandas Series
        # (keep the returned index/name from the legacy result).
        idx = ArkoudaIndexAccessor(out_ak.index.to_pandas()).to_ak()  # preserve names/levels
        return _ak_array_to_pandas_series(out_ak.values, name=out_ak.name).set_axis(idx)

    @staticmethod
    def _from_return_msg(rep_msg: str) -> pd.Series:
        """
        Construct a pandas Series from a legacy Arkouda return message.

        This mirrors :meth:`ArkoudaIndexAccessor.from_return_msg`. It calls the legacy
        ``ak.Series.from_return_msg`` constructor, then immediately wraps the resulting
        Arkouda arrays back into a pandas Series backed by Arkouda ExtensionArrays.

        Parameters
        ----------
        rep_msg : str
            The ``+ delimited`` string message returned by Arkouda server operations
            that construct a Series.

        Returns
        -------
        pd.Series
            A pandas Series backed by Arkouda ExtensionArrays (no NumPy materialization).
        """
        ak_s = ak_Series.from_return_msg(rep_msg)

        # Wrap values and index without materializing
        out = _ak_array_to_pandas_series(ak_s.values, name=ak_s.name)

        # Prefer wrapping the legacy index via the Index accessor helper path.
        # We need a pandas Index/MultiIndex backed by EAs.
        pd_idx = ArkoudaIndexAccessor.from_ak_legacy(ak_s.index)
        return pd.Series(out.array, index=pd_idx, name=ak_s.name)

    def groupby(self) -> GroupBy:
        """
        Return an Arkouda GroupBy object for this Series, without materializing.

        Returns
        -------
        GroupBy

        Raises
        ------
        TypeError
            Returns TypeError if Series is not arkouda backed.

        Examples
        --------
        >>> import arkouda as ak
        >>> import pandas as pd
        >>> s = pd.Series([80, 443, 80]).ak.to_ak()
        >>> g = s.ak.groupby()
        >>> keys, counts = g.size()
        """
        if not self.is_arkouda:
            raise TypeError("Series must be Arkouda-backed. Call .ak.to_ak() first.")

        arr = self._obj.array
        akcol = getattr(arr, "_data", None)
        if akcol is None:
            raise TypeError("Arkouda-backed Series array does not expose '_data'")

        return GroupBy(akcol)

    def argsort(
        self,
        *,
        ascending: bool = True,
        **kwargs: object,
    ) -> pd.Series:
        """
        Return the integer indices that would sort the Series values.

        This mirrors ``pandas.Series.argsort`` but returns an Arkouda-backed
        pandas Series (distributed), not a NumPy-backed result.

        Parameters
        ----------
        ascending : bool
            Sort values in ascending order if True, descending order if False.
            Default is True.
        **kwargs : object
            Additional keyword arguments.

            Supported keyword arguments
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            na_position : {"first", "last"}, default "last"
                Where to place NaN values in the sorted result. Currently only
                applied for floating-point ``pdarray`` data; for ``Strings`` and
                ``Categorical`` it has no effect.

        Returns
        -------
        pd.Series
            An Arkouda-backed Series of integer permutation indices. The returned
            Series has the same index as the original.

        Raises
        ------
        TypeError
            If the Series is not Arkouda-backed, or the underlying dtype does not
            support sorting.
        ValueError
            If ``na_position`` is not "first" or "last".
        """
        from arkouda.numpy import argsort as ak_argsort
        from arkouda.numpy.numeric import isnan as ak_isnan
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.pdarraysetops import concatenate
        from arkouda.numpy.strings import Strings
        from arkouda.numpy.util import is_float
        from arkouda.pandas.categorical import Categorical

        if not self.is_arkouda:
            raise TypeError("argsort() requires an Arkouda-backed Series.")

        na_position = kwargs.pop("na_position", "last")
        if na_position not in {"first", "last"}:
            raise ValueError("na_position must be 'first' or 'last'.")

        akcol = _pandas_series_to_ak_array(self._obj)

        if not isinstance(akcol, (pdarray, Strings, Categorical)):
            raise TypeError(f"Unsupported argsort dtype: {type(akcol)}")

        perm = ak_argsort(akcol, ascending=ascending)

        if is_float(akcol):
            is_nan = ak_isnan(akcol)[perm]
            if na_position == "last":
                perm = concatenate([perm[~is_nan], perm[is_nan]])
            else:
                perm = concatenate([perm[is_nan], perm[~is_nan]])

        return _ak_array_to_pandas_series(
            perm,
            name=str(self._obj.name),
            index=self._obj.index,
        )
