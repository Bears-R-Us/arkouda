"""
Index and MultiIndex pandas accessors for Arkouda-backed indices.

This module defines the ``.ak`` accessor for :class:`pandas.Index` and
:class:`pandas.MultiIndex`, enabling seamless conversion between:

* standard NumPy-backed pandas index objects
* pandas indexes backed by :class:`~arkouda.pandas.extension.ArkoudaExtensionArray`
  (zero-copy, distributed Arkouda representation)
* legacy Arkouda ``ak.Index`` and ``ak.MultiIndex`` objects

The goal of this accessor is to make pandas Index/MultiIndex behave naturally
with data stored on the Arkouda server, allowing users to switch between local
and distributed representations without changing their analytical workflow.

The accessor mirrors the behavior of the DataFrame ``.ak`` accessor and provides
a symmetric conversion API:

- ``idx.ak.to_ak()``
    Convert a pandas Index/MultiIndex to an Arkouda-backed pandas object
    using :class:`ArkoudaExtensionArray`.

- ``idx.ak.collect()``
    Materialize an Arkouda-backed Index/MultiIndex into a NumPy-backed
    pandas object.

- ``idx.ak.to_ak_legacy()``
    Convert to legacy Arkouda ``ak.Index`` or ``ak.MultiIndex`` objects.

- ``idx.ak.from_ak_legacy(akidx)``
    Wrap a legacy Arkouda index object back into an Arkouda-backed pandas
    Index/MultiIndex.

- ``idx.ak.is_arkouda``
    Inspect whether the index is currently backed by Arkouda.

All operations avoid materializing data to NumPy unless explicitly required
(e.g., via :meth:`collect`). MultiIndex levels are handled symmetrically through
zero-copy wrapping of each Arkouda level.

Examples
--------
Index conversion:

>>> import pandas as pd
>>> import arkouda as ak
>>> idx = pd.Index([10, 20, 30], name="nums")
>>> ak_idx = idx.ak.to_ak()
>>> ak_idx.ak.is_arkouda
True
>>> restored = ak_idx.ak.collect()
>>> restored.equals(idx)
True
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from pandas.api.extensions import register_index_accessor

from arkouda.index import Index as ak_Index
from arkouda.index import MultiIndex as ak_MultiIndex
from arkouda.pandas.extension import ArkoudaExtensionArray


# ---------------------------------------------------------------------------
# Helper conversion functions
# ---------------------------------------------------------------------------


def _pandas_index_to_ak(index: Union[pd.Index, pd.MultiIndex]) -> Union[ak_Index, ak_MultiIndex]:
    """
    Convert a pandas Index or MultiIndex into a legacy Arkouda Index/MultiIndex.

    This mirrors the behavior of :class:`arkouda.index.Index` and
    :class:`arkouda.index.MultiIndex` constructors, which already know how to
    consume pandas Index / MultiIndex, including categorical levels.

    Parameters
    ----------
    index : Union[pd.Index, pd.MultiIndex]

    Returns
    -------
    Union[ak_Index, ak_MultiIndex]

    Raises
    ------
    ValueError
        If a pandas RangeIndex has an invalid step (e.g., step == 0).
    """
    from arkouda.numpy.pdarraycreation import arange as ak_arange

    # MultiIndex: delegate directly to ak.MultiIndex, which handles
    # pandas.MultiIndex in its constructor.
    if isinstance(index, pd.MultiIndex):
        # Preserve names when constructing the ak.MultiIndex
        return ak_MultiIndex(index, names=index.names)

    # IMPORTANT: RangeIndex is virtual (start/stop/step). Converting it via any path
    # that materializes values can blow up the client for huge sizes.
    # Create it directly on the Arkouda server instead.
    if isinstance(index, pd.RangeIndex):
        start, stop, step = index.start, index.stop, index.step
        if step == 0:
            raise ValueError("RangeIndex step cannot be 0")
        if start == 0 and step == 1:
            arr = ak_arange(stop)
        else:
            arr = ak_arange(start, stop, step)
        return ak_Index(arr, name=index.name)

    # Single-level Index: ak.Index already knows how to consume pandas.Index,
    # including CategoricalIndex.
    return ak_Index(index, name=index.name)


def _ak_index_to_pandas_no_copy(akidx: Union[ak_Index, ak_MultiIndex]) -> Union[pd.Index, pd.MultiIndex]:
    """
    Wrap a legacy Arkouda ``Index`` or ``MultiIndex`` into pandas objects
    without materializing to NumPy.

    Each Arkouda column (pdarray, Strings, Categorical) is wrapped in an
    :class:`ArkoudaExtensionArray` ExtensionArray, producing a pandas
    ``Index`` or ``MultiIndex`` whose data remain on the Arkouda server.

    Parameters
    ----------
    akidx : Union[ak_Index, ak_MultiIndex]
        The legacy Arkouda index object to wrap.

    Returns
    -------
    Union[pd.Index, pd.MultiIndex]
        A pandas index object backed by :class:`ArkoudaExtensionArray`
        instances.

    Raises
    ------
    TypeError
        If ``akidx`` is not an instance of ``ak_Index`` or ``ak_MultiIndex``.
    """
    # MultiIndex: wrap each level
    if isinstance(akidx, ak_MultiIndex):
        arrays = [ArkoudaExtensionArray._from_sequence(level) for level in akidx.levels]
        return pd.MultiIndex.from_arrays(arrays, names=list(akidx.names))

    # Single-level Index
    if isinstance(akidx, ak_Index):
        ea = ArkoudaExtensionArray._from_sequence(akidx.values)
        return pd.Index(ea, name=akidx.name)

    raise TypeError(f"Expected ak.Index or ak.MultiIndex, got {type(akidx)!r}")


# ---------------------------------------------------------------------------
# Index accessor
# ---------------------------------------------------------------------------


@register_index_accessor("ak")
class ArkoudaIndexAccessor:
    """
    Arkouda-backed index accessor for pandas ``Index`` and ``MultiIndex``.

    This accessor provides methods for converting between:

    * NumPy-backed pandas indexes
    * pandas indexes backed by :class:`ArkoudaExtensionArray` (zero-copy EA mode)
    * legacy Arkouda ``ak.Index`` and ``ak.MultiIndex`` objects

    The ``.ak`` namespace mirrors the DataFrame accessor, providing a consistent
    interface for distributed index operations. All conversions avoid unnecessary
    NumPy materialization unless explicitly requested via :meth:`collect`.

    Parameters
    ----------
    pandas_obj : Union[pd.Index, pd.MultiIndex]
        The pandas ``Index`` or ``MultiIndex`` instance that this accessor wraps.

    Notes
    -----
    * ``to_ak`` → pandas object, Arkouda-backed (ExtensionArrays).
    * ``to_ak_legacy`` → legacy Arkouda index objects.
    * ``collect`` → NumPy-backed pandas object.
    * ``is_arkouda`` → reports whether the index is Arkouda-backed.

    Examples
    --------
    Basic single-level Index conversion:

    >>> import pandas as pd
    >>> import arkouda as ak
    >>> idx = pd.Index([10, 20, 30], name="vals")

    Convert to Arkouda-backed:

    >>> ak_idx = idx.ak.to_ak()
    >>> ak_idx.ak.is_arkouda
    True

    Materialize back:

    >>> restored = ak_idx.ak.collect()
    >>> restored.equals(idx)
    True

    Convert to legacy Arkouda:

    >>> ak_legacy = idx.ak.to_ak_legacy()
    >>> type(ak_legacy)
    <class 'arkouda.pandas.index.Index'>

    MultiIndex conversion:

    >>> arrays = [[1, 1, 2], ["red", "blue", "red"]]
    >>> midx = pd.MultiIndex.from_arrays(arrays, names=["num", "color"])
    >>> ak_midx = midx.ak.to_ak()
    >>> ak_midx.ak.is_arkouda
    True
    """

    def __init__(self, pandas_obj: Union[pd.Index, pd.MultiIndex]):
        self._obj = pandas_obj

    # ------------------------------------------------------------------
    # Distributed / local conversion
    # ------------------------------------------------------------------

    @staticmethod
    def from_ak_legacy(akidx: Union[ak_Index, ak_MultiIndex]) -> Union[pd.Index, pd.MultiIndex]:
        """
        Convert a legacy Arkouda ``ak.Index`` or ``ak.MultiIndex`` into a
        pandas Index/MultiIndex backed by Arkouda ExtensionArrays.

        This is the index analogue of ``df.ak.from_ak_legacy_ea()``: it performs a
        zero-copy-style wrapping of Arkouda server-side arrays into
        :class:`ArkoudaExtensionArray` objects, producing a pandas Index or
        MultiIndex whose levels remain distributed on the Arkouda server.

        No materialization to NumPy occurs.

        Parameters
        ----------
        akidx : Union[ak_Index, ak_MultiIndex]
            The legacy Arkouda Index or MultiIndex to wrap.

        Returns
        -------
        Union[pd.Index, pd.MultiIndex]
            A pandas index object whose underlying data are
            :class:`ArkoudaExtensionArray` instances referencing the Arkouda
            server-side arrays.

        Notes
        -----
        * ``ak.Index`` → ``pd.Index`` with Arkouda-backed values.
        * ``ak.MultiIndex`` → ``pd.MultiIndex`` where each level is backed by
          an :class:`ArkoudaExtensionArray`.
        * This function does not validate whether the input is already wrapped;
          callers should ensure the argument is a legacy Arkouda index object.

        Examples
        --------
        >>> import arkouda as ak
        >>> import pandas as pd

        Wrap a legacy ``ak.Index`` into a pandas ``Index`` without copying:

        >>> ak_idx = ak.Index(ak.arange(5))
        >>> pd_idx = pd.Index.ak.from_ak_legacy(ak_idx)
        >>> pd_idx
        Index([0, 1, 2, 3, 4], dtype='int64')

        The resulting index stores its values on the Arkouda server:

        >>> type(pd_idx.array)
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>

        MultiIndex example:

        >>> ak_lvl1 = ak.array(['a', 'a', 'b', 'b'])
        >>> ak_lvl2 = ak.array([1, 2, 1, 2])
        >>> ak_mi = ak.MultiIndex([ak_lvl1, ak_lvl2], names=['letter', 'number'])

        >>> pd_mi = pd.Index.ak.from_ak_legacy(ak_mi)
        >>> pd_mi
        MultiIndex([('a', 1),
                    ('a', 2),
                    ('b', 1),
                    ('b', 2)],
                   names=['letter', 'number'])

        Each level is backed by an Arkouda ExtensionArray and remains distributed:

        >>> [type(level._data) for level in pd_mi.levels]
        [<class 'arkouda.pandas.extension._arkouda_string_array.ArkoudaStringArray'>,
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>]

        No NumPy materialization occurs; the underlying data stay on the Arkouda server.
        """
        return _ak_index_to_pandas_no_copy(akidx)

    def to_ak(self) -> Union[pd.Index, pd.MultiIndex]:
        """
        Convert this pandas Index or MultiIndex to an Arkouda-backed index.

        Unlike :meth:`to_ak_legacy`, which returns a legacy Arkouda Index object,
        this method returns a *pandas* Index or MultiIndex whose data reside
        on the Arkouda server and are wrapped in
        :class:`ArkoudaExtensionArray` ExtensionArrays.

        The conversion is zero-copy with respect to NumPy: no materialization
        to local NumPy arrays occurs.

        Returns
        -------
        Union[pd.Index, pd.MultiIndex]
            An Index whose underlying data live on the Arkouda server.

        Examples
        --------
        Convert a simple Index to Arkouda-backed form:

        >>> import pandas as pd
        >>> import arkouda as ak
        >>> idx = pd.Index([10, 20, 30], name="values")
        >>> ak_idx = idx.ak.to_ak()
        >>> type(ak_idx.array)
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>

        Round-trip back to NumPy-backed pandas objects:

        >>> restored = ak_idx.ak.collect()
        >>> restored.equals(idx)
        True

        """
        akidx = _pandas_index_to_ak(self._obj)
        return _ak_index_to_pandas_no_copy(akidx)

    def collect(self) -> Union[pd.Index, pd.MultiIndex]:
        """
        Materialize this Index or MultiIndex back to a plain NumPy-backed
        pandas index.

        Returns
        -------
        Union[pd.Index, pd.MultiIndex]
            An Index whose underlying data are plain NumPy arrays.

        Raises
        ------
        TypeError
            If the index is Arkouda-backed but does not expose the expected
            ``_data`` attribute, or if the index type is unsupported.

        Examples
        --------
        Single-level Index round-trip:

        >>> import pandas as pd
        >>> import arkouda as ak
        >>> idx = pd.Index([1, 2, 3], name="x")
        >>> ak_idx = idx.ak.to_ak()
        >>> np_idx = ak_idx.ak.collect()
        >>> np_idx
        Index([1, 2, 3], dtype='int64', name='x')
        >>> np_idx.equals(idx)
        True

        Behavior when already NumPy-backed (no-op except shallow copy):

        >>> plain = pd.Index([10, 20, 30])
        >>> plain2 = plain.ak.collect()
        >>> plain2.equals(plain)
        True

        Verifying that Arkouda-backed values materialize to NumPy:

        >>> ak_idx = pd.Index([5, 6, 7]).ak.to_ak()
        >>> type(ak_idx.array)
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>
        >>> out = ak_idx.ak.collect()
        >>> type(out.array)
        <class 'pandas...NumpyExtensionArray'>
        """
        idx = self._obj

        # --------------------------------------------------------------
        # Single-level Index
        # --------------------------------------------------------------
        if isinstance(idx, pd.Index) and not isinstance(idx, pd.MultiIndex):
            arr = idx.array

            # Arkouda-backed: peel out Arkouda column and materialize
            if isinstance(arr, ArkoudaExtensionArray):
                akcol = getattr(arr, "_data", None)
                if akcol is None:
                    raise TypeError("Arkouda-backed index array does not expose '_data'")
                values = akcol.to_ndarray()
            else:
                # Already NumPy-backed (or other EA with .to_numpy)
                values = idx.to_numpy()

            return pd.Index(values, name=idx.name)

        # --------------------------------------------------------------
        # MultiIndex
        # --------------------------------------------------------------
        if isinstance(idx, pd.MultiIndex):
            # Materialize full tuples; works for both Arkouda-backed and plain.
            tuples = list(idx.to_list())
            return pd.MultiIndex.from_tuples(tuples, names=list(idx.names))

        raise TypeError(f"Unsupported index type for collect(): {type(idx)!r}")

    # ------------------------------------------------------------------
    # Legacy Arkouda conversions
    # ------------------------------------------------------------------
    def to_ak_legacy(self) -> Union[ak_Index, ak_MultiIndex]:
        """
        Convert this pandas Index or MultiIndex into a legacy Arkouda
        ``ak.Index`` or ``ak.MultiIndex`` object.

        This is the index analogue of ``df.ak.to_ak_legacy()``, returning the
        *actual* Arkouda index objects on the server, rather than a pandas
        wrapper backed by :class:`ArkoudaExtensionArray`.

        The conversion is zero-copy with respect to NumPy: values are transferred
        directly into Arkouda arrays without materializing to local NumPy.

        Returns
        -------
        Union[ak_Index, ak_MultiIndex]
            A legacy Arkouda Index/MultiIndex whose data live on the Arkouda server.

        Examples
        --------
        Convert a simple pandas Index into a legacy Arkouda Index:

        >>> import pandas as pd
        >>> import arkouda as ak
        >>> idx = pd.Index([10, 20, 30], name="numbers")
        >>> ak_idx = idx.ak.to_ak_legacy()
        >>> type(ak_idx)
        <class 'arkouda.pandas.index.Index'>
        >>> ak_idx.name
        'numbers'
        """
        return _pandas_index_to_ak(self._obj)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_arkouda(self) -> bool:
        """
        Return whether the underlying Index is Arkouda-backed.

        An Index or MultiIndex is considered Arkouda-backed if its underlying
        storage uses :class:`ArkoudaExtensionArray`. This applies to both
        single-level and multi-level indices.

        Returns
        -------
        bool
            True if the Index/MultiIndex is backed by Arkouda server-side
            arrays, False otherwise.

        Examples
        --------
        NumPy-backed Index:

        >>> import pandas as pd
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.ak.is_arkouda
        False

        Arkouda-backed single-level Index:

        >>> import arkouda as ak
        >>> ak_idx = pd.Index([10, 20, 30]).ak.to_ak()
        >>> ak_idx.ak.is_arkouda
        True

        Arkouda-backed MultiIndex:

        >>> arrays = [[1, 1, 2], ["a", "b", "a"]]
        >>> midx = pd.MultiIndex.from_arrays(arrays)
        >>> ak_midx = midx.ak.to_ak()
        >>> ak_midx.ak.is_arkouda
        True

        """
        if hasattr(self._obj, "levels"):
            for level in self._obj.levels:
                values = level.array if hasattr(level, "_data") else None
                if not isinstance(values, ArkoudaExtensionArray):
                    return False
            return True
        elif hasattr(self._obj, "array"):
            arr = getattr(self._obj, "array", None)
            return isinstance(arr, ArkoudaExtensionArray)
        else:
            return False
