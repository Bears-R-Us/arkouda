from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import bool_, floating, integer, str_

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
    Convert a numpy or pandas object to an arkouda object.
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
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    If the objects are pandas or numpy objects, they are converted to arkouda objects.
    Then assert_almost_equal is applied to the result.

    Parameters
    ----------
    left : object
    right : object
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.

    Warning
    -------
    This function cannot be used on pdarray of size > ak.client.maxTransferBytes
    because it converts pdarrays to numpy arrays and calls np.allclose.

    See Also
    --------
    assert_almost_equal
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
    Check that left and right Index are equal.

    If the objects are pandas.Index, they are converted to arkouda.Index.
    Then assert_almost_equal is applied to the result.

    Parameters
    ----------
    left : Index or pandas.Index
    right : Index or pandas.Index
    exact : True
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_names : bool, default True
        Whether to check the names attribute.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Index'
        Specify object name being compared, internally used to show appropriate
        assertion message.

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
    Check that 'np.array', 'pd.Categorical', 'ak.pdarray', 'ak.Strings',
    'ak.Categorical', or 'ak.SegArray' is equivalent.

    np.nparray's and pd.Categorical's will be converted to the arkouda equivalent.
    Then assert_arkouda_pdarray_equal will be applied to the result.

    Parameters
    ----------
    left, right : np.ndarray, pd.Categorical, arkouda.pdarray or arkouda.numpy.Strings or \
    arkouda.Categorical
        The two arrays to be compared.
    check_dtype : bool, default True
        Check dtype if both a and b are ak.pdarray or np.ndarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | arkouda.pdarray, default None
        optional index (shared by both left and right), used in output.

    See Also
    --------
    assert_arkouda_array_equal
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
    Check that left and right Series are equal.

    pd.Series's will be converted to the arkouda equivalent.
    Then assert_series_equal will be applied to the result.

    Parameters
    ----------
    left : Series or pd.Series
    right : Series or pd.Series
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool, default True
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.
    check_like : bool, default False
        If True, ignore the order of the index. Must be False if check_index is False.
        Note: same labels must be with the same data.

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
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. It is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    pd.DataFrame's will be converted to the arkouda equivalent.
    Then assert_frame_equal will be applied to the result.

    Parameters
    ----------
    left : DataFrame or pd.DataFrame
        First DataFrame to compare.
    right : DataFrame or pd.DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool, default = True
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_frame_equal

    Examples
    --------
    >>> import arkouda as ak
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from arkouda.testing import assert_frame_equivalent
    >>> import pandas as pd
    >>> df1 = ak.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
    >>> assert_frame_equivalent(df1, df1)

    """
    __tracebackhide__ = not DEBUG

    if not isinstance(left, (DataFrame, pd.DataFrame)) or not isinstance(
        right, (DataFrame, pd.DataFrame)
    ):
        raise TypeError(
            f"left and right must be type arkouda.DataFrame or pandas.DataFrame.  "
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
    Wrapper for tm.assert_*_equivalent to dispatch to the appropriate test function.

    Parameters
    ----------
    left : Index, pd.Index, Series, pd.Series, DataFrame, pd.DataFrame, \
Strings, Categorical, pd.Categorical, SegArray, pdarray, np.ndarray
        The first item to be compared.

    right : Index, pd.Index, Series, pd.Series, DataFrame, pd.DataFrame, \
Strings, Categorical, pd.Categorical, SegArray, pdarray, np.ndarray
        The second item to be compared.

    **kwargs
        All keyword arguments are passed through to the underlying assert method.

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
