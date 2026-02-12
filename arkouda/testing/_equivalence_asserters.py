from __future__ import annotations

# Third Party
import numpy as np
import pandas as pd

from numpy import bool_, floating, integer, str_

# First Party
from arkouda import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    SegArray,
    Series,
    Strings,
    array,
    pdarray,
)
from arkouda.testing import (
    assert_almost_equal,
    assert_arkouda_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)


DEBUG = True

__all__ = [
    "assert_almost_equivalent",
    "assert_arkouda_array_equivalent",
    "assert_equivalent",
    "assert_frame_equivalent",
    "assert_index_equivalent",
    "assert_series_equivalent",
]


def _convert_to_arkouda(obj):
    """
    Convert a NumPy or pandas object to an Arkouda object.

    This function attempts to convert a supported NumPy or pandas object
    (including arrays, Series, DataFrames, Index types, or categoricals)
    into an Arkouda-compatible equivalent.

    Parameters
    ----------
    obj : object
        A NumPy or pandas object to convert. Must be one of the supported types
        including np.ndarray, pd.Series, pd.DataFrame, pd.Index, pd.Categorical,
        or their Arkouda equivalents.

    Returns
    -------
    object
        An Arkouda object of the same logical structure as the input.

    Raises
    ------
    TypeError
        If the input object is not a recognized Arkouda, NumPy, or pandas type.

    Examples
    --------
    >>> import pandas as pd
    >>> import arkouda as ak
    >>> from arkouda.testing._equivalence_asserters import _convert_to_arkouda
    >>> _convert_to_arkouda(pd.Series([1, 2, 3]))
    0    1
    1    2
    2    3
    dtype: int64

    """
    if isinstance(
        obj,
        (
            DataFrame,
            Series,
            Index,
            MultiIndex,
            SegArray,
            Categorical,
            Strings,
            pdarray,
            str_,
            integer,
            floating,
            bool_,
            bool,
            float,
        ),
    ):
        return obj

    if not isinstance(
        obj,
        (pd.MultiIndex, pd.Index, pd.Series, pd.DataFrame, pd.Categorical, np.ndarray),
    ):
        raise TypeError(f"obj must be an arkouda, numpy or pandas object, but was type: {type(obj)}")

    if isinstance(obj, pd.MultiIndex):
        return MultiIndex(obj)
    elif isinstance(obj, pd.Index):
        return Index(obj)
    elif isinstance(obj, pd.Series):
        return Series(obj)
    elif isinstance(obj, pd.DataFrame):
        return DataFrame(obj)
    elif isinstance(obj, pd.Categorical):
        return Categorical(obj)
    elif isinstance(obj, np.ndarray):
        return array(
            obj if obj.flags.c_contiguous else np.ascontiguousarray(obj)
        )  # required for some multi-dim cases
    return None


def assert_almost_equivalent(
    left,
    right,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> None:
    """
    Check that two objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    If the objects are pandas or numpy objects, they are converted to Arkouda objects.
    Then assert_almost_equal is applied to the result.

    Parameters
    ----------
    left : object
        First object to compare.
    right : object
        Second object to compare.
    rtol : float
        Relative tolerance. Default is 1e-5.
    atol : float
        Absolute tolerance. Default is 1e-8.

    Raises
    ------
    TypeError
        If either input is not a supported numeric-like type.

    Warning
    -------
    This function cannot be used on pdarrays of size > ak.core.client.maxTransferBytes
    because it converts pdarrays to numpy arrays and calls np.allclose.

    See Also
    --------
    assert_almost_equal

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.testing import assert_almost_equivalent
    >>> assert_almost_equivalent(0.123456, 0.123457, rtol=1e-4)

    """
    __tracebackhide__ = not DEBUG

    assert_almost_equal(
        _convert_to_arkouda(left),
        _convert_to_arkouda(right),
        rtol=rtol,
        atol=atol,
    )


def assert_index_equivalent(
    left: Index | pd.Index,
    right: Index | pd.Index,
    exact: bool = True,
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "Index",
) -> None:
    """
    Check that two Index objects are equal.

    If the objects are pandas Index, they are converted to Arkouda Index.
    Then assert_index_equal is applied to the result.

    Parameters
    ----------
    left : Index or pd.Index
        First Index to compare.
    right : Index or pd.Index
        Second Index to compare.
    exact : bool
        Whether to check that class, dtype, and inferred type are identical. Default is True.
    check_names : bool
        Whether to check the names attribute. Default is True.
    check_exact : bool
        Whether to compare values exactly. Default is True.
    check_categorical : bool
        Whether to compare internal Categoricals exactly. Default is True.
    check_order : bool
        Whether to require identical order in index values. Default is True.
    rtol : float
        Relative tolerance used when check_exact is False. Default is 1e-5.
    atol : float
        Absolute tolerance used when check_exact is False. Default is 1e-8.
    obj : str
        Object name used in error messages. Default is "Index".

    Raises
    ------
    TypeError
        If either input is not an Index or pd.Index.

    See Also
    --------
    assert_index_equal

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import testing as tm
    >>> import pandas as pd
    >>> a = ak.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> tm.assert_index_equivalent(a, b)

    """
    __tracebackhide__ = not DEBUG

    if not isinstance(left, (Index, pd.Index)) or not isinstance(right, (Index, pd.Index)):
        raise TypeError(
            f"left and right must be type arkouda.Index, or pandas.Index.  "
            f"Instead types were {type(left)} and {type(right)}"
        )

    assert_index_equal(
        _convert_to_arkouda(left),
        _convert_to_arkouda(right),
        exact=exact,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        check_order=check_order,
        rtol=rtol,
        atol=atol,
        obj=obj,
    )


def assert_arkouda_array_equivalent(
    left: pdarray | Strings | Categorical | SegArray | np.ndarray | pd.Categorical,
    right: pdarray | Strings | Categorical | SegArray | np.ndarray | pd.Categorical,
    check_dtype: bool = True,
    err_msg=None,
    check_same=None,
    obj: str = "pdarray",
    index_values=None,
) -> None:
    """
     Check that two Arkouda-compatible arrays are equal.

    Supported types include numpy arrays, pandas Categorical, and Arkouda arrays.

    Parameters
    ----------
    left : pdarray, Strings, Categorical, SegArray, np.ndarray, or pd.Categorical
        First array to compare.
    right : pdarray, Strings, Categorical, SegArray, np.ndarray, or pd.Categorical
        Second array to compare.
    check_dtype : bool
        Whether to verify that dtypes match. Default is True.
    err_msg : str or None
        Optional message to display on failure.
    check_same : None or {"copy", "same"}
        Whether to ensure identity or separation in memory. Default is None.
    obj : str
        Object label for error messages. Default is "pdarray".
    index_values : Index or pdarray, optional
        Shared index used in error output. Default is None.

    Raises
    ------
    TypeError
        If either input is not a supported array type.

    See Also
    --------
    assert_arkouda_array_equal

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import Strings
    >>> from arkouda.testing import assert_arkouda_array_equivalent
    >>> a = ak.array([1, 2, 3])
    >>> b = ak.array([1, 2, 3])
    >>> assert_arkouda_array_equivalent(a, b)
    >>> s1 = ak.array(['x', 'y'])
    >>> s2 = ak.array(['x', 'y'])
    >>> assert_arkouda_array_equivalent(s1, s2)

    """
    __tracebackhide__ = not DEBUG

    if not isinstance(
        left, (np.ndarray, pd.Categorical, pdarray, Strings, Categorical, SegArray)
    ) or not isinstance(right, (np.ndarray, pd.Categorical, pdarray, Strings, Categorical, SegArray)):
        raise TypeError(
            f"left and right must be type np.ndarray, pdarray, Strings, "
            f"Categorical, or SegArray.  "
            f"Instead types were {type(left)} and {type(right)}"
        )

    assert_arkouda_array_equal(
        _convert_to_arkouda(left),
        _convert_to_arkouda(right),
        check_dtype=check_dtype,
        err_msg=err_msg,
        check_same=check_same,
        obj=obj,
        index_values=index_values,
    )


def assert_series_equivalent(
    left: Series | pd.Series,
    right: Series | pd.Series,
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "Series",
    *,
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    """
    Check that two Series are equal.

    This function compares two Series and raises an assertion if they differ.
    pandas Series are converted to Arkouda equivalents before comparison.
    The comparison can be customized using the provided keyword arguments.

    Parameters
    ----------
    left : Series or pd.Series
        First Series to compare.
    right : Series or pd.Series
        Second Series to compare.
    check_dtype : bool
        Whether to check that dtypes are identical. Default is True.
    check_index_type : bool
        Whether to check that index class, dtype, and inferred type are identical. Default is True.
    check_series_type : bool
        Whether to check that the Series class is identical. Default is True.
    check_names : bool
        Whether to check that the Series and Index name attributes are identical. Default is True.
    check_exact : bool
        Whether to compare numbers exactly. Default is False.
    check_categorical : bool
        Whether to compare internal Categoricals exactly. Default is True.
    check_category_order : bool
        Whether to compare category order in internal Categoricals. Default is True.
    rtol : float
        Relative tolerance used when check_exact is False. Default is 1e-5.
    atol : float
        Absolute tolerance used when check_exact is False. Default is 1e-8.
    obj : str
        Object name used in error messages. Default is "Series".
    check_index : bool
        Whether to check index equivalence. If False, only values are compared. Default is True.
    check_like : bool
        If True, ignore the order of the index. Must be False if check_index is False.
        Note: identical labels must still correspond to the same data. Default is False.

    Raises
    ------
    TypeError
        If either input is not a Series or pd.Series.

    See Also
    --------
    assert_series_equal

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import testing as tm
    >>> import pandas as pd
    >>> a = ak.Series([1, 2, 3, 4])
    >>> b = pd.Series([1, 2, 3, 4])
    >>> tm.assert_series_equivalent(a, b)

    """
    __tracebackhide__ = not DEBUG

    if not isinstance(left, (Series, pd.Series)) or not isinstance(right, (Series, pd.Series)):
        raise TypeError(
            f"left and right must be type arkouda.pandas.Series or pandas.pandas.Series.  "
            f"Instead types were {type(left)} and {type(right)}."
        )

    assert_series_equal(
        _convert_to_arkouda(left),
        _convert_to_arkouda(right),
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_series_type=check_series_type,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        check_category_order=check_category_order,
        rtol=rtol,
        atol=atol,
        obj=obj,
        check_index=check_index,
        check_like=check_like,
    )


def assert_frame_equivalent(
    left: DataFrame | pd.DataFrame,
    right: DataFrame | pd.DataFrame,
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_column_type: bool = True,
    check_frame_type: bool = True,
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_like: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "DataFrame",
) -> None:
    """
    Check that two DataFrames are equal.

    This function compares two DataFrames and raises an assertion if they differ.
    It is intended primarily for use in unit tests. pandas DataFrames are converted to
    Arkouda equivalents before comparison.

    Parameters
    ----------
    left : DataFrame or pd.DataFrame
        First DataFrame to compare.
    right : DataFrame or pd.DataFrame
        Second DataFrame to compare.
    check_dtype : bool
        Whether to check that dtypes are identical. Default is True.
    check_index_type : bool
        Whether to check that index class, dtype, and inferred type are identical. Default is True.
    check_column_type : bool
        Whether to check that column class, dtype, and inferred type are identical. Default is True.
    check_frame_type : bool
        Whether to check that the DataFrame class is identical. Default is True.
    check_names : bool
        Whether to check that the index and column names are identical. Default is True.
    check_exact : bool
        Whether to compare values exactly. Default is True.
    check_categorical : bool
        Whether to compare internal categoricals exactly. Default is True.
    check_like : bool
        Whether to ignore the order of index and columns. Labels must still match their data. /
        Default is False.
    rtol : float
        Relative tolerance used when check_exact is False. Default is 1e-5.
    atol : float
        Absolute tolerance used when check_exact is False. Default is 1e-8.
    obj : str
        Object name used in error messages. Default is "DataFrame".

    Raises
    ------
    TypeError
        If either input is not a DataFrame or pd.DataFrame.

    See Also
    --------
    assert_frame_equal

    Examples
    --------
    >>> import arkouda as ak
    >>> import pandas as pd
    >>> from arkouda.testing import assert_frame_equivalent
    >>> df1 = ak.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    Fails because dtypes are different:
    >>> assert_frame_equivalent(df1, df2)  # doctest: +SKIP

    """
    __tracebackhide__ = not DEBUG

    if not isinstance(left, (DataFrame, pd.DataFrame)) or not isinstance(
        right, (DataFrame, pd.DataFrame)
    ):
        raise TypeError(
            f"left and right must be type arkouda.pandas.DataFrame or pandas.DataFrame.  "
            f"Instead types were {type(left)} and {type(right)}."
        )

    assert_frame_equal(
        _convert_to_arkouda(left),
        _convert_to_arkouda(right),
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_frame_type=check_frame_type,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        check_like=check_like,
        rtol=rtol,
        atol=atol,
        obj=obj,
    )


def assert_equivalent(left, right, **kwargs) -> None:
    """
    Dispatch to the appropriate assertion function depending on object types.

    Parameters
    ----------
    left : Any
        First object to compare. Type determines which assertion function is used.
    right : Any
        Second object to compare.
    **kwargs : dict
        Keyword arguments passed to the specific assertion function.

    Raises
    ------
    AssertionError
        If values are not equivalent.

    Examples
    --------
    >>> import arkouda as ak
    >>> import pandas as pd
    >>> from arkouda.testing import assert_equivalent
    >>> ak_series = ak.Series([1, 2, 3])
    >>> pd_series = pd.Series([1, 2, 3])
    >>> assert_equivalent(ak_series, pd_series)

    """
    __tracebackhide__ = not DEBUG

    if isinstance(left, (Index, pd.Index)):
        assert_index_equivalent(left, right, **kwargs)
    elif isinstance(left, (Series, pd.Series)):
        assert_series_equivalent(left, right, **kwargs)
    elif isinstance(left, (DataFrame, pd.DataFrame)):
        assert_frame_equivalent(left, right, **kwargs)
    elif isinstance(left, (pdarray, np.ndarray, Strings, Categorical, pd.Categorical, SegArray)):
        assert_arkouda_array_equivalent(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        assert_almost_equivalent(left, right)
