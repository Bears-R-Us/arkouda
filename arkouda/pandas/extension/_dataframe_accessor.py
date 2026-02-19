from typing import TYPE_CHECKING, Dict, List, Optional, TypeVar, Union

import pandas as pd

from pandas import DataFrame as pd_DataFrame
from pandas.api.extensions import register_dataframe_accessor
from pandas.api.types import CategoricalDtype
from typeguard import typechecked

from arkouda.numpy.pdarrayclass import pdarray
from arkouda.pandas.categorical import Categorical
from arkouda.pandas.dataframe import DataFrame as ak_DataFrame

from . import ArkoudaExtensionArray
from ._arkouda_array import ArkoudaArray
from ._dtypes import _ArkoudaBaseDtype


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
else:
    Strings = TypeVar("Strings")

__all__ = ["ArkoudaDataFrameAccessor"]


def _looks_like_ak_col(obj: object) -> bool:
    """
    Return whether an object appears to be a valid Arkouda column type.

    This function checks for the core column types used by the legacy
    ``arkouda.pandas.dataframe.DataFrame`` API, namely:

    * ``pdarray`` for numeric/boolean data
    * ``Strings`` for string columns
    * ``Categorical`` for categorical data

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
        True if ``obj`` is an instance of one of the recognized Arkouda
        column classes (``pdarray``, ``Strings``, ``Categorical``);
        False otherwise.
    """
    from arkouda.numpy.strings import Strings

    return isinstance(obj, (pdarray, Strings, Categorical))


def _extract_ak_from_ea(ea: object) -> object:
    """
    Extract the underlying Arkouda column from an Arkouda-backed ExtensionArray.

    Parameters
    ----------
    ea : object
        A pandas ExtensionArray-like object presumed to wrap Arkouda
        column data.

    Returns
    -------
    object
        The underlying Arkouda column.

    Raises
    ------
    TypeError
        If the ExtensionArray does not expose its Arkouda-backed data
        through a known attribute or the extracted object is not a valid
        Arkouda column.
    """
    if hasattr(ea, "_data"):
        col = getattr(ea, "_data", None)
        if _looks_like_ak_col(col):
            return col
    raise TypeError("Arkouda EA does not expose an Arkouda column via a known attribute/method.")


def _is_arkouda_series(s: pd.Series) -> bool:
    """
    Return whether a pandas Series is backed by an Arkouda ExtensionArray.

    This function checks two conditions:

    1. Whether the Series has an Arkouda-specific dtype
       (a subclass of ``_ArkoudaBaseDtype``).
    2. Whether the underlying array (``s.array``) is an instance of
       :class:`ArkoudaArray`.

    If either condition is true, the Series is considered Arkouda-backed.

    Parameters
    ----------
    s : pd.Series
        The Series to inspect.

    Returns
    -------
    bool
        True if the Series is Arkouda-backed; False otherwise.
    """
    # dtype check first; fallback to EA instance name if helpful
    if isinstance(getattr(s, "dtype", None), _ArkoudaBaseDtype):
        return True
    return isinstance(getattr(s, "array", None), ArkoudaArray)


def _series_to_akcol_no_copy(s: pd.Series) -> object:
    """
    Extract the underlying Arkouda column from an Arkouda-backed Series.

    Parameters
    ----------
    s : pd.Series
        Series expected to be backed by an :class:`ArkoudaArray`.

    Returns
    -------
    object
        The underlying Arkouda column.

    Raises
    ------
    TypeError
        If the Series is not Arkouda-backed or does not expose an Arkouda column.
    """
    if not _is_arkouda_series(s):
        raise TypeError(
            f"Column '{s.name}' is not Arkouda-backed (dtype={s.dtype!r}). "
            "Wrap columns with ArkoudaArray before calling df.ak.merge."
        )
    return _extract_ak_from_ea(s.array)


def _akcol_to_series(
    name: str,
    akcol: object,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Wrap an Arkouda column into a pandas Series using ArkoudaExtensionArray,
    ensuring the index is Arkouda-backed (including the default index).
    """
    from arkouda.numpy.pdarraycreation import arange as ak_arange
    from arkouda.pandas.extension import ArkoudaIndexAccessor  # avoids `index.ak` for mypy

    ea = ArkoudaExtensionArray._from_sequence(akcol)

    if index is None:
        n = len(ea)
        index_ea = ArkoudaExtensionArray._from_sequence(ak_arange(n))
        index = pd.Index(index_ea)

    # mypy-safe: call accessor class directly (pandas stubs don't know `.ak`)
    acc = ArkoudaIndexAccessor(index)
    if not acc.is_arkouda:
        index = acc.to_ak()

    return pd.Series(ea, index=index, name=name)


def _df_to_akdf_no_copy(df: pd.DataFrame) -> ak_DataFrame:
    """
    Convert a pandas DataFrame to an ak_DataFrame without copying data.

    All columns in ``df`` must be Arkouda-backed (recognized by
    :func:`_is_arkouda_series`). The underlying Arkouda columns are
    extracted and used to construct a legacy :class:`ak_DataFrame`.

    Parameters
    ----------
    df : pd.DataFrame
        Input pandas DataFrame whose columns are all Arkouda-backed.

    Returns
    -------
    ak_DataFrame
        Legacy Arkouda DataFrame built from the underlying Arkouda columns.
    """
    cols = {}
    for name in df.columns:
        s = df[name]
        cols[name] = _series_to_akcol_no_copy(s)
    return ak_DataFrame(cols)


def _akdf_to_pandas_no_copy(akdf: ak_DataFrame) -> pd.DataFrame:
    """
    Convert an ak_DataFrame back to pandas with Arkouda ExtensionArrays.

    No NumPy/Python conversion is performed. Each Arkouda column is wrapped
    into a pandas Series backed by :class:`ArkoudaArray`.

    Parameters
    ----------
    akdf : ak_DataFrame
        Legacy Arkouda DataFrame whose columns will be wrapped into
        Arkouda-backed pandas Series.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame with :class:`ArkoudaArray`-backed columns.
    """
    cols = {}
    for name in akdf.columns:
        cols[name] = _akcol_to_series(name, akdf[name])
    return pd.DataFrame(cols)


@register_dataframe_accessor("ak")
class ArkoudaDataFrameAccessor:
    """
    Arkouda DataFrame accessor.

    Allows ``df.ak`` access to Arkouda-backed operations.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def from_ak_legacy(akdf: ak_DataFrame) -> pd_DataFrame:
        """
        Convert a legacy Arkouda ``DataFrame`` into a pandas ``DataFrame``
        backed by Arkouda ExtensionArrays.

        This is the zero-copy-ish counterpart to :meth:`to_ak_legacy`.
        Instead of materializing columns into NumPy arrays, this function
        wraps each underlying Arkouda server-side array in the appropriate
        ``ArkoudaExtensionArray`` subclass (``ArkoudaArray``,
        ``ArkoudaStringArray``, or ``ArkoudaCategoricalArray``).
        The resulting pandas ``DataFrame`` therefore keeps all data on the
        Arkouda server, enabling scalable operations without transferring
        data to the Python client.

        Parameters
        ----------
        akdf : ak_DataFrame
            A legacy Arkouda ``DataFrame`` (``arkouda.pandas.dataframe.DataFrame``)
            whose columns are Arkouda objects (``pdarray``, ``Strings``,
            or ``Categorical``).

        Returns
        -------
        pd_DataFrame
            A pandas ``DataFrame`` in which each column is an Arkouda-backed
            ExtensionArray—typically one of:

            * :class:`ArkoudaArray`
            * :class:`ArkoudaStringArray`
            * :class:`ArkoudaCategoricalArray`

            No materialization to NumPy occurs.
            All column data remain server-resident.

        Notes
        -----
        * This function performs a **zero-copy** conversion for the underlying
          Arkouda arrays (server-side). Only lightweight Python wrappers are
          created.
        * The resulting pandas ``DataFrame`` can interoperate with most pandas
          APIs that support extension arrays.
        * Round-tripping through ``to_ak_legacy()`` and
          ``from_ak_legacy()`` preserves Arkouda semantics.

        Examples
        --------
        Basic conversion
        ~~~~~~~~~~~~~~~~
        >>> import arkouda as ak
        >>> akdf = ak.DataFrame({"a": ak.arange(5), "b": ak.array([10,11,12,13,14])})

        >>> pdf = pd.DataFrame.ak.from_ak_legacy(akdf)
        >>> pdf
           a   b
        0  0  10
        1  1  11
        2  2  12
        3  3  13
        4  4  14

        Columns stay Arkouda-backed
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        >>> type(pdf["a"].array)
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>

        >>> pdf["a"].array._data
        array([0 1 2 3 4])

        No NumPy materialization occurs
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        >>> pdf["a"].values    # pandas always materializes .values
        ArkoudaArray([0 1 2 3 4])

        But the underlying column is still Arkouda:
        >>> pdf["a"].array._data
        array([0 1 2 3 4])

        Categorical and Strings columns work as well
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        >>> akdf2 = ak.DataFrame({
        ...     "s": ak.array(["a","b","a"]),
        ...     "c": ak.Categorical(ak.array(["e","f","g"]))
        ... })
        >>> pdf2 = pd.DataFrame.ak.from_ak_legacy(akdf2)

        >>> type(pdf2["s"].array)
        <class 'arkouda.pandas.extension._arkouda_string_array.ArkoudaStringArray'>

        >>> type(pdf2["c"].array)
        <class 'arkouda.pandas.extension._arkouda_categorical_array.ArkoudaCategoricalArray'>

        """
        return _akdf_to_pandas_no_copy(akdf)

    def to_ak(self) -> pd_DataFrame:
        """
        Convert this pandas DataFrame to an Arkouda-backed pandas DataFrame.

        Each column of the original pandas DataFrame is materialized to the
        Arkouda server via :func:`ak.array` and wrapped in an
        :class:`ArkoudaArray` ExtensionArray. The result is still a
        *pandas* DataFrame, but all column data reside on the Arkouda server
        and behave according to the Arkouda ExtensionArray API.

        This method does **not** return a legacy :class:`ak_DataFrame`.
        For that (server-side DataFrame structure), use :meth:`to_ak_legacy`.

        Returns
        -------
        pd_DataFrame
            A pandas DataFrame whose columns are Arkouda-backed
            :class:`ArkoudaArray` objects.

        Examples
        --------
        Convert a plain pandas DataFrame to an Arkouda-backed one:

        >>> import pandas as pd
        >>> import arkouda as ak
        >>> df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        >>> akdf = df.ak.to_ak()
        >>> type(akdf)
         <class 'pandas...DataFrame'>

        The columns are now Arkouda ExtensionArrays:

        >>> isinstance(akdf["x"].array, ArkoudaArray)
        True
        >>> akdf["x"].tolist()
        [np.int64(1), np.int64(2), np.int64(3)]

        Arkouda operations work directly on the columns:

        >>> akdf["x"].array._data + 10
        array([11 12 13])

        Converting back to a NumPy-backed DataFrame:

        >>> akdf_numpy = akdf.ak.collect()
        >>> akdf_numpy
           x  y
        0  1  a
        1  2  b
        2  3  c
        """
        from arkouda.numpy.pdarraycreation import array as ak_array
        from arkouda.pandas.extension import ArkoudaIndexAccessor

        idx = ArkoudaIndexAccessor(self._obj.index).to_ak()

        cols = {}
        for name, col in self._obj.items():
            if isinstance(col.array, ArkoudaExtensionArray):
                cols[name] = col.array
            else:
                cols[name] = ArkoudaExtensionArray._from_sequence(ak_array(col.values))
        return pd_DataFrame(cols, index=idx)

    def collect(self) -> pd_DataFrame:
        """
        Materialize an Arkouda-backed pandas DataFrame into a NumPy-backed one.

        This operation retrieves each Arkouda-backed column from the server
        using ``to_ndarray()`` and constructs a standard pandas DataFrame whose
        columns are plain NumPy ``ndarray`` objects. The returned DataFrame
        has no dependency on Arkouda.

        Returns
        -------
        pd_DataFrame
            A pandas DataFrame with NumPy-backed columns.

        Examples
        --------
        Converting an Arkouda-backed DataFrame into a NumPy-backed one:

        >>> import pandas as pd
        >>> import arkouda as ak
        >>> from arkouda.pandas.extension import ArkoudaDataFrameAccessor

        Create a pandas DataFrame and convert it to Arkouda-backed form:

        >>> df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        >>> akdf = df.ak.to_ak()

        ``akdf`` is still a pandas DataFrame, but its columns live on Arkouda:

        >>> type(akdf["x"].array)
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>

        Now fully materialize it to local NumPy arrays:

        >>> collected = akdf.ak.collect()
        >>> collected
           x  y
        0  1  a
        1  2  b
        2  3  c

        The columns are now NumPy arrays:

        >>> type(collected["x"].values)
        <class 'numpy.ndarray'>
        """
        cols = {name: self._obj[name].values.to_ndarray() for name in self._obj.columns}
        return pd_DataFrame(cols)

    def to_ak_legacy(self) -> ak_DataFrame:
        """
        Convert this pandas DataFrame into the legacy :class:`arkouda.DataFrame`.

        This method performs a *materializing* conversion of a pandas DataFrame
        into the legacy Arkouda DataFrame structure. Every column is converted
        to Arkouda server-side data:

        * Python / NumPy numeric and boolean arrays become :class:`pdarray`.
        * String columns become Arkouda string arrays (``Strings``).
        * Pandas categoricals become Arkouda ``Categorical`` objects.
        * The result is a legacy :class:`ak_DataFrame` whose columns all reside
          on the Arkouda server.

        This differs from :meth:`to_ak`, which creates Arkouda-backed
        ExtensionArrays but retains a pandas.DataFrame structure.

        Returns
        -------
        ak_DataFrame
            The legacy Arkouda DataFrame with all columns materialized
            onto the Arkouda server.

        Examples
        --------
        Convert a plain pandas DataFrame to a legacy Arkouda DataFrame:

        >>> import pandas as pd
        >>> import arkouda as ak
        >>> df = pd.DataFrame({
        ...     "i": [1, 2, 3],
        ...     "s": ["a", "b", "c"],
        ...     "c": pd.Series(["low", "low", "high"], dtype="category"),
        ... })
        >>> akdf = df.ak.to_ak_legacy()
        >>> type(akdf)
        <class 'arkouda.pandas.dataframe.DataFrame'>

        Columns have the appropriate Arkouda types:

        >>> from arkouda.numpy.pdarrayclass import pdarray
        >>> from arkouda.numpy.strings import Strings
        >>> from arkouda.pandas.categorical import Categorical
        >>> isinstance(akdf["i"], pdarray)
        True
        >>> isinstance(akdf["s"], Strings)
        True
        >>> isinstance(akdf["c"], Categorical)
        True

        Values round-trip through the conversion:

        >>> akdf["i"].tolist()
        [1, 2, 3]
        """
        from arkouda.numpy.pdarraycreation import array as ak_array

        cols: Dict[str, Union[pdarray, Strings, Categorical]] = {}

        for name, s in self._obj.items():
            values = s.values

            # Strings
            if pd.api.types.is_string_dtype(s.dtype) or s.dtype == "string":
                cols[name] = ak_array(values)

            # Pandas Categorical
            elif isinstance(s.dtype, CategoricalDtype):
                cat = s.astype("category")
                cat = pd.Categorical(cat)
                cols[name] = Categorical(cat)

            # Everything else: convert with ak.array()
            else:
                cols[name] = ak_array(values)

        return ak_DataFrame(cols)

    def _assert_all_arkouda(self, df: pd.DataFrame, side: str) -> None:
        """
        Validate that all columns in the given DataFrame are Arkouda-backed.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.
        side : str
            Label indicating which operand is being checked
            (for example, ``"left"`` or ``"right"``).

        Raises
        ------
        TypeError
            If ``df`` is not a pandas DataFrame or if any column
            in ``df`` is not Arkouda-backed.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{side} must be a pandas.DataFrame; got {type(df).__name__}. "
                "If you already have ak.DataFrame, call left_ak.merge(right_ak, ...)."
            )
        bad = [c for c in df.columns if not _is_arkouda_series(df[c])]
        if bad:
            raise TypeError(
                f"All columns in the {side} DataFrame must be Arkouda ExtensionArrays. "
                f"Non-Arkouda columns: {bad}"
            )

    @typechecked
    def merge(
        self,
        right: pd.DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        left_suffix: str = "_x",
        right_suffix: str = "_y",
        convert_ints: bool = True,
        sort: bool = True,
    ) -> pd.DataFrame:
        """
        Merge two Arkouda-backed pandas DataFrames using Arkouda's join.

        Parameters
        ----------
        right : pd.DataFrame
            Right-hand DataFrame to merge with ``self._obj``. All columns must
            be Arkouda-backed ExtensionArrays.
        on : Optional[Union[str, List[str]]]
            Column name(s) to join on. Must be present in both left and right
            DataFrames. If not provided and neither ``left_on`` nor ``right_on``
            is set, the intersection of column names in left and right is used.
            Default is None.
        left_on : Optional[Union[str, List[str]]]
            Column name(s) from the left DataFrame to use as join keys. Must be
            used together with ``right_on``. If provided, ``on`` is ignored for
            the left side.  Default is None
        right_on : Optional[Union[str, List[str]]]
            Column name(s) from the right DataFrame to use as join keys. Must be
            used together with ``left_on``. If provided, ``on`` is ignored for
            the right side. Default is None
        how : str
            Type of merge to be performed. One of ``'left'``, ``'right'``,
            ``'inner'``, or ``'outer'``. Default is 'inner'.
        left_suffix : str
            Suffix to apply to overlapping column names from the left frame that
            are not part of the join keys. Default is '_x'.
        right_suffix : str
            Suffix to apply to overlapping column names from the right frame that
            are not part of the join keys.Default is '_y'.
        convert_ints : bool
            Whether to allow Arkouda to upcast integer columns as needed
            (for example, to accommodate missing values) during the merge.
            Default is True.
        sort : bool
            Whether to sort the join keys in the output. Default is True.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame whose columns are :class:`ArkoudaArray`
            ExtensionArrays. All column data remain on the Arkouda server.

        Raises
        ------
        TypeError
            If ``right`` is not a :class:`pandas.DataFrame` or if any column in
            the left or right DataFrame is not Arkouda-backed.
        """
        if not isinstance(right, pd.DataFrame):
            raise TypeError("`right` must be a pandas.DataFrame")

        # Ensure both sides are Arkouda-backed
        self._assert_all_arkouda(self._obj, "left")
        self._assert_all_arkouda(right, "right")

        # Lift to ak.DataFrame (zero-copy-ish)
        left_ak = _df_to_akdf_no_copy(self._obj)
        right_ak = _df_to_akdf_no_copy(right)

        from arkouda.pandas.dataframe import merge

        # Delegate to the ak.DataFrame merge
        out_ak = merge(
            left_ak,
            right_ak,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            convert_ints=convert_ints,
            sort=sort,
        )

        # Wrap back into pandas with Arkouda EAs
        return _akdf_to_pandas_no_copy(out_ak)
