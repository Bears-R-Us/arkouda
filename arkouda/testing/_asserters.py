from __future__ import annotations

from typing import Literal, NoReturn, cast

import numpy as np

from pandas.api.types import is_bool, is_number
from pandas.io.formats.printing import pprint_thing  # type: ignore

from arkouda import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    SegArray,
    Series,
    Strings,
    argsort,
    array,
    pdarray,
    sort,
)
from arkouda.numpy.pdarrayclass import sum as aksum
from arkouda.numpy.util import is_numeric


DEBUG = True

__all__ = [
    "assert_almost_equal",
    "assert_arkouda_array_equal",
    "assert_arkouda_pdarray_equal",
    "assert_arkouda_segarray_equal",
    "assert_arkouda_strings_equal",
    "assert_attr_equal",
    "assert_categorical_equal",
    "assert_class_equal",
    "assert_contains_all",
    "assert_copy",
    "assert_dict_equal",
    "assert_equal",
    "assert_frame_equal",
    "assert_index_equal",
    "assert_is_sorted",
    "assert_series_equal",
]


def assert_almost_equal(
    left,
    right,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> None:
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

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

    """
    if isinstance(left, Index):
        assert_index_equal(
            left,
            right,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, Series):
        assert_series_equal(
            left,
            right,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, DataFrame):
        assert_frame_equal(
            left,
            right,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    else:
        # Other sequences.
        if is_number(left) and is_number(right):
            # Do not compare numeric classes, like np.float64 and float.
            pass
        elif is_bool(left) and is_bool(right):
            # Do not compare bool classes, like np.bool_ and bool.
            pass
        else:
            if isinstance(left, pdarray) or isinstance(right, pdarray):
                obj = "pdarray"
            else:
                obj = "Input"
            assert_class_equal(left, right, obj=obj)

        if isinstance(left, pdarray) and isinstance(right, pdarray):
            assert np.allclose(
                left.to_ndarray(),
                right.to_ndarray(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        else:
            assert np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)


def _check_isinstance(left, right, cls) -> None:
    """
    Ensure that both objects are instances of the expected type.

    Helper method for our assert_* methods.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.

    """
    cls_name = cls.__name__

    if not isinstance(left, cls):
        raise AssertionError(f"{cls_name} Expected type {cls}, found {type(left)} instead")
    if not isinstance(right, cls):
        raise AssertionError(f"{cls_name} Expected type {cls}, found {type(right)} instead")


def assert_dict_equal(left, right, compare_keys: bool = True) -> None:
    """
    Assert that two dictionaries are equal.

    Values must be arkouda objects.

    Parameters
    ----------
    left, right: dict
        The dictionaries to be compared.
    compare_keys : bool
        Whether to compare the keys. Defaults to True.
        If False, only the values are compared.

    """
    _check_isinstance(left, right, dict)

    left_keys = frozenset(left.keys())
    right_keys = frozenset(right.keys())

    if compare_keys:
        assert left_keys == right_keys

    for k in left_keys:
        assert_almost_equal(left[k], right[k])

    return None


def assert_index_equal(
    left: Index,
    right: Index,
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

    Parameters
    ----------
    left : Index
        The first Index to compare.
    right : Index
        The second Index to compare.
    exact : bool
        Whether to check that the Index class, dtype, and inferred_type
        are identical. Defaults to True.
    check_names : bool
        Whether to check the `name` attribute. Defaults to True.
    check_exact : bool
        Whether to compare numbers exactly. Defaults to True.
    check_categorical : bool
        Whether to compare internal Categorical values exactly. Defaults to True.
    check_order : bool
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
        Defaults to True.
    rtol : float
        Relative tolerance. Only used when `check_exact` is False.
        Defaults to 1e-5.
    atol : float
        Absolute tolerance. Only used when `check_exact` is False.
        Defaults to 1e-8.
    obj : str
        A name for the object being compared, used in assertion messages.
        Defaults to 'Index'.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import testing as tm
    >>> a = ak.Index([1, 2, 3])
    >>> b = ak.Index([1, 2, 3])
    >>> tm.assert_index_equal(a, b)

    """
    __tracebackhide__ = not DEBUG

    def _check_types(left, right, obj: str = "Index") -> None:
        if not exact:
            return

        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal("inferred_type", left, right, obj=obj)

        # Skip exact dtype checking when `check_categorical` is False
        if isinstance(left.dtype, Categorical) and isinstance(right.dtype, Categorical):
            if check_categorical:
                assert_attr_equal("dtype", left, right, obj=obj)
                assert_index_equal(left.categories, right.categories, exact=exact)
            return

        assert_attr_equal("dtype", left, right, obj=obj)

    # instance validation
    _check_isinstance(left, right, Index)

    # class / dtype comparison
    _check_types(left, right, obj=obj)

    # level comparison
    if left.nlevels != right.nlevels:
        msg1 = f"{obj} levels are different"
        msg2 = f"{left.nlevels}, {left}"
        msg3 = f"{right.nlevels}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{obj} length are different"
        msg2 = f"{len(left)}, {left}"
        msg3 = f"{len(right)}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # If order doesn't matter then sort the index entries
    if not check_order:
        left = left[left.argsort()]
        right = right[right.argsort()]

    # MultiIndex special comparison for little-friendly error messages
    if isinstance(left, MultiIndex):
        right = cast(MultiIndex, right)

        for level in range(left.nlevels):
            lobj = f"MultiIndex level [{level}]"
            try:
                # try comparison on levels/codes to avoid densifying MultiIndex
                assert_index_equal(
                    Index(left.levels[level]),
                    Index(right.levels[level]),
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)

                assert_index_equal(
                    llevel,
                    rlevel,
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
            # get_level_values may change dtype
            _check_types(left.levels[level], right.levels[level], obj=obj)

    # skip exact index checking when `check_categorical` is False
    # differed from pandas due to unintuitive pandas behavior.
    elif check_exact is True or not is_numeric(left) or not is_numeric(right):
        if not left.equals(right):
            if isinstance(left, list) and isinstance(right, list):
                mismatch = np.array(left) != np.array(right)
            else:
                mismatch = left != right

            diff = aksum(mismatch) * 100.0 / len(left)
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right)
    else:
        # @TODO Use new ak.allclose function
        assert_almost_equal(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            check_dtype=exact,
            obj=obj,
            lobj=left,
            robj=right,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("names", left, right, obj=obj)

    if check_categorical:
        if isinstance(left, Categorical) or isinstance(right, Categorical):
            assert_categorical_equal(left.values, right.values, obj=f"{obj} category")


def assert_class_equal(left, right, exact: bool = True, obj: str = "Input") -> None:
    """Check classes are equal."""
    __tracebackhide__ = not DEBUG

    def repr_class(x):
        if isinstance(x, Index):
            # return Index as it is to include values in the error message
            return x

        return type(x).__name__

    if type(left) is type(right):
        return

    msg = f"{obj} classes are different"
    raise_assert_detail(obj, msg, repr_class(left), repr_class(right))


def assert_attr_equal(attr: str, left, right, obj: str = "Attributes") -> None:
    """
    Check that attributes are equal. Both objects must have the given attribute.

    Parameters
    ----------
    attr : str
        The name of the attribute being compared.
    left : object
        The first object to compare.
    right : object
        The second object to compare.
    obj : str
        A name for the object being compared, used in assertion messages.
        Defaults to 'Attributes'.

    """
    __tracebackhide__ = not DEBUG

    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)

    if left_attr is right_attr:
        return None

    try:
        result = left_attr == right_attr
    except TypeError:
        result = False
    if (left_attr is None) ^ (right_attr is None):
        result = False
    elif not isinstance(result, bool):
        result = result.all()

    if not result:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)
    return None


def assert_is_sorted(seq) -> None:
    """Assert that the sequence is sorted."""
    if isinstance(seq, (Index, Series)):
        seq = seq.values

    # sorting does not change precisions
    assert_arkouda_array_equal(seq, sort(array(seq)))


def assert_categorical_equal(
    left,
    right,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = "Categorical",
) -> None:
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
        The first Categorical to compare.
    right : Categorical
        The second Categorical to compare.
    check_dtype : bool
        Whether to check that the integer dtype of the codes is the same.
        Defaults to True.
    check_category_order : bool
        Whether to compare the order of the categories (which implies identical integer codes).
        If False, only the resulting values are compared. The `ordered` attribute is
        always checked. Defaults to True.
    obj : str
        A name for the object being compared, used in assertion messages.
        Defaults to 'Categorical'.

    """
    _check_isinstance(left, right, Categorical)

    exact = True

    if check_category_order:
        assert_index_equal(
            Index(left.categories),
            Index(right.categories),
            obj=f"{obj}.categories",
            exact=exact,
        )
        assert_arkouda_array_equal(left.codes, right.codes, check_dtype=check_dtype, obj=f"{obj}.codes")
    else:
        try:
            # @TODO replace with Index.sort_values
            lc = Index(
                left.categories[argsort(left.categories)]
            )  # .sort_values()  # left.sort().categories
            rc = Index(
                right.categories[argsort(right.categories)]
            )  # .sort_values()  # right.sort().categories
        except TypeError:
            # e.g. '<' not supported between instances of 'int' and 'str'
            lc, rc = Index(left.categories), Index(right.categories)
        assert_index_equal(lc, rc, obj=f"{obj}.categories", exact=exact)
        # @TODO Replace with Index.take
        assert_index_equal(
            Index(left.categories[left.codes]),
            Index(right.categories[right.codes]),
            obj=f"{obj}.values",
            exact=exact,
        )

    # @TODO uncomment when Categorical.ordered is added
    # assert_attr_equal("ordered", left, right, obj=obj)


def raise_assert_detail(
    obj, message, left, right, diff=None, first_diff=None, index_values=None
) -> NoReturn:
    __tracebackhide__ = not DEBUG

    msg = f"""{obj} are different

{message}"""

    if isinstance(index_values, Index):
        index_values = index_values.values.to_ndarray()

    if isinstance(index_values, pdarray):
        index_values = index_values.to_ndarray()

    if isinstance(index_values, np.ndarray):
        msg += f"\n[index]: {pprint_thing(index_values)}"

    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (Categorical, Strings, pdarray)):
        left = repr(left)

    if isinstance(right, pdarray):
        right = right.to_ndarray()
    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif isinstance(right, (Categorical, Strings)):
        right = repr(right)

    msg += f"""
[left]:  {left}
[right]: {right}"""

    if diff is not None:
        msg += f"\n[diff]: {diff}"

    if first_diff is not None:
        msg += f"\n{first_diff}"

    raise AssertionError(msg)


def assert_arkouda_pdarray_equal(
    left: pdarray,
    right: pdarray,
    check_dtype: bool = True,
    err_msg=None,
    check_same=None,
    obj: str = "pdarray",
    index_values=None,
) -> None:
    """
    Check that two Arkouda pdarray objects are equivalent.

    Parameters
    ----------
    left : pdarray
        The first array to compare.
    right : pdarray
        The second array to compare.
    check_dtype : bool
        Whether to check dtype if both arrays are pdarrays. Defaults to True.
    err_msg : str or None
        Custom assertion message to display on failure. Defaults to None.
    check_same : {'copy', 'same'} or None
        If not None, asserts whether `left` and `right` share the same memory:
        - 'copy': assert they do **not** share memory
        - 'same': assert they **do** share memory
        Defaults to None.
    obj : str
        A name for the object being compared, used in assertion messages.
        Defaults to 'pdarray'.
    index_values : Index or pdarray or None
        Optional index shared by both arrays, used to enhance output on failure.
        Defaults to None.

    """
    __tracebackhide__ = not DEBUG

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an ak.pdarray
    _check_isinstance(left, right, pdarray)

    assert left.ndim == right.ndim, (
        f"left dimension {left.ndim} does not match right dimension {right.ndim}."
    )
    assert left.size == right.size, f"left size {left.size} does not match right size {right.size}."
    if left.shape:
        assert left.shape == right.shape, (
            f"left shape {left.shape} does not match right shape {right.shape}."
        )
    else:
        assert (
            isinstance(left.shape, tuple)
            and isinstance(right.shape, tuple)
            and len(left.shape) == 0
            and len(right.shape) == 0
        ), f"left shape {left.shape} does not match right shape {right.shape}."

    assert len(left) == len(right), (
        f"Arrays were not same size.  left had length {len(left)} and right had length {len(right)}"
    )

    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{repr(left_base)} is not {repr(right_base)}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{repr(left_base)} is {repr(right_base)}")

    def _raise(left: pdarray, right: pdarray, err_msg):
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(obj, f"{obj} shapes are different", left.shape, right.shape)

            diff = aksum(left != right)

            diff = diff * 100.0 / float(left.size)
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    from arkouda import all as akall
    from arkouda.numpy.dtypes import bigint, dtype

    # compare shape and values
    # @TODO use ak.allclose
    if isinstance(left, pdarray) and isinstance(right, pdarray) and left.dtype == dtype(bigint):
        if not akall(left == right):
            _raise(left, right, err_msg)
    elif not np.allclose(left.to_ndarray(), right.to_ndarray(), atol=0, rtol=0, equal_nan=True):
        _raise(left, right, err_msg)

    if check_dtype:
        if isinstance(left, pdarray) and isinstance(right, pdarray):
            assert_attr_equal("dtype", left, right, obj=obj)


def assert_arkouda_segarray_equal(
    left: SegArray,
    right: SegArray,
    check_dtype: bool = True,
    err_msg=None,
    check_same=None,
    obj: str = "segarray",
) -> None:
    """
    Check that two Arkouda SegArray objects are equivalent.

    Parameters
    ----------
    left : SegArray
        The first SegArray to compare.
    right : SegArray
        The second SegArray to compare.
    check_dtype : bool
        Whether to check dtype if both arrays contain pdarrays. Defaults to True.
    err_msg : str or None
        Custom assertion message. Defaults to None.
    check_same : {'copy', 'same'} or None
        If not None, asserts whether `left` and `right` share the same memory.
        - 'copy': assert that they do **not** share memory.
        - 'same': assert that they **do** share memory.
        Defaults to None.
    obj : str
        Name of the object being compared (used in assertion messages).
        Defaults to 'segarray'.

    """
    __tracebackhide__ = not DEBUG

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an ak.SegArray
    _check_isinstance(left, right, SegArray)

    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{repr(left_base)} is not {repr(right_base)}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{repr(left_base)} is {repr(right_base)}")

    if check_dtype:
        if isinstance(left, SegArray) and isinstance(right, SegArray):
            assert_attr_equal("dtype", left, right, obj=obj)

    assert_arkouda_pdarray_equal(
        left.values,
        right.values,
        check_dtype=check_dtype,
        err_msg=err_msg,
        check_same=check_same,
        obj="segarray values",
        index_values=None,
    )

    assert_arkouda_pdarray_equal(
        left.segments,
        right.segments,
        check_dtype=True,
        err_msg=None,
        check_same=None,
        obj="segarray segments",
        index_values=None,
    )


def assert_arkouda_strings_equal(
    left,
    right,
    err_msg=None,
    check_same=None,
    obj: str = "Strings",
    index_values=None,
) -> None:
    """
    Check that two `ak.Strings` arrays are equivalent.

    Parameters
    ----------
    left : Strings
        The first Strings object to compare.
    right : Strings
        The second Strings object to compare.
    err_msg : str or None
        Custom assertion message. Defaults to None.
    check_same : {'copy', 'same'} or None
        If not None, assert whether `left` and `right` share the same memory.
        - 'copy': assert that they do **not** share memory
        - 'same': assert that they **do** share memory
        Defaults to None.
    obj : str
        A name for the object being compared, used in assertion messages.
        Defaults to 'Strings'.
    index_values : Index or pdarray or None
        Optional index shared by both arrays, used in output. Defaults to None.

    """
    __tracebackhide__ = not DEBUG

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an ak.pdarray
    _check_isinstance(left, right, Strings)

    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{repr(left_base)} is not {repr(right_base)}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{repr(left_base)} is {repr(right_base)}")

    def _raise(left: Strings, right: Strings, err_msg):
        if err_msg is None:
            diff = aksum(left != right)
            diff = diff * 100.0 / float(left.size)
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    if left.shape != right.shape:
        raise_assert_detail(obj, f"{obj} shapes are different", left.shape, right.shape)

    if not aksum(left != right) == 0:
        _raise(left, right, err_msg)


def assert_arkouda_array_equal(
    left: pdarray | Strings | Categorical | SegArray,
    right: pdarray | Strings | Categorical | SegArray,
    check_dtype: bool = True,
    err_msg=None,
    check_same=None,
    obj: str = "pdarray",
    index_values=None,
) -> None:
    """
    Check that two Arkouda arrays are equivalent. Supports pdarray, Strings,
    Categorical, and SegArray.

    Parameters
    ----------
    left : pdarray or Strings or Categorical or SegArray
        The first array to compare.
    right : pdarray or Strings or Categorical or SegArray
        The second array to compare.
    check_dtype : bool
        Whether to check dtype if both `left` and `right` are ak.pdarray.
        Defaults to True.
    err_msg : str or None
        Custom assertion message, if provided. Defaults to None.
    check_same : {'copy', 'same'} or None
        If not None, assert whether `left` and `right` share the same memory.
        - `'copy'`: assert that they do **not** share memory.
        - `'same'`: assert that they **do** share memory.
        Defaults to None.
    obj : str
        Object name used in assertion error messages. Defaults to 'pdarray'.
    index_values : Index or pdarray or None
        Optional index shared by both `left` and `right`, used to enhance
        output in error messages. Defaults to None.

    """
    assert_class_equal(left, right)

    if isinstance(left, Strings) and isinstance(right, Strings):
        assert_arkouda_strings_equal(
            left,
            right,
            err_msg=err_msg,
            check_same=check_same,
            obj=obj,
            index_values=index_values,
        )
    elif isinstance(left, Categorical) and isinstance(right, Categorical):
        assert_arkouda_array_equal(
            left.categories[left.codes],
            right.categories[right.codes],
            check_dtype=check_dtype,
            err_msg=err_msg,
            check_same=check_same,
            obj=obj,
            index_values=index_values,
        )
    elif isinstance(left, SegArray) and isinstance(right, SegArray):
        assert_arkouda_segarray_equal(
            left,
            right,
            check_dtype=check_dtype,
            err_msg=err_msg,
            check_same=check_same,
            obj=obj,
        )
    elif isinstance(left, pdarray) and isinstance(right, pdarray):
        assert_arkouda_pdarray_equal(
            left,
            right,
            check_dtype=check_dtype,
            err_msg=err_msg,
            check_same=check_same,
            obj=obj,
            index_values=index_values,
        )
    else:
        raise TypeError(
            "assert_arkouda_array_equal can only compare arrays of matching type: "
            "pdarray | Strings | Categorical | SegArray"
        )


# This could be refactored to use the NDFrame.equals method
def assert_series_equal(
    left,
    right,
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

    Parameters
    ----------
    left : Series
        First Series to compare.
    right : Series
        Second Series to compare.
    check_dtype : bool
        Whether to check the Series dtype is identical. Defaults to True.
    check_index_type : bool
        Whether to check the Index class, dtype, and inferred_type are identical. Defaults to True.
    check_series_type : bool
        Whether to check that the Series class is identical. Defaults to True.
    check_names : bool
        Whether to check the Series and Index `name` attribute. Defaults to True.
    check_exact : bool
        Whether to compare numbers exactly. Defaults to False.
    check_categorical : bool
        Whether to compare internal Categoricals exactly. Defaults to True.
    check_category_order : bool
        Whether to compare the category order of internal Categoricals. Defaults to True.
    rtol : float
        Relative tolerance. Only used when `check_exact` is False. Defaults to 1e-5.
    atol : float
        Absolute tolerance. Only used when `check_exact` is False. Defaults to 1e-8.
    obj : str
        Name of the object being compared, used in assertion messages. Defaults to 'Series'.
    check_index : bool
        Whether to check index equivalence. If False, only the values are compared. Defaults to True.
    check_like : bool
        If True, ignore the order of the index.
        Must be False if `check_index` is False.
        Note: same labels must be with the same data. Defaults to False.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import testing as tm
    >>> a = ak.Series([1, 2, 3, 4])
    >>> b = ak.Series([1, 2, 3, 4])
    >>> tm.assert_series_equal(a, b)

    """
    __tracebackhide__ = not DEBUG

    if not check_index and check_like:
        raise ValueError("check_like must be False if check_index is False")

    # instance validation
    _check_isinstance(left, right, Series)

    if check_series_type:
        assert_class_equal(left, right, obj=obj)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{len(left)}, {left.index}"
        msg2 = f"{len(right)}, {right.index}"
        raise_assert_detail(obj, "Series length are different", msg1, msg2)

    if check_index:
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_exact=check_exact,
            check_categorical=check_categorical,
            check_order=not check_like,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )

    if check_like:
        # @TODO use Series.reindex_like
        left = left[right.index.values]

    if check_dtype:
        # We want to skip exact dtype checking when `check_categorical`
        # is False. We'll still raise if only one is a `Categorical`,
        # regardless of `check_categorical`
        if isinstance(left, Categorical) and isinstance(right, Categorical) and not check_categorical:
            pass
        else:
            assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")

    if check_exact or not is_numeric(left.values) or not is_numeric(right.values):
        assert_arkouda_array_equal(
            left.values,
            right.values,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    else:
        assert_almost_equal(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("name", left, right, obj=obj)

    if check_categorical is True:
        if isinstance(left.values, Categorical) or isinstance(right.values, Categorical):
            assert_categorical_equal(
                left.values,
                right.values,
                obj=f"{obj} category",
                check_category_order=check_category_order,
                check_dtype=check_dtype,
            )


# This could be refactored to use the NDFrame.equals method
def assert_frame_equal(
    left: DataFrame,
    right: DataFrame,
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_column_type: bool | Literal["equiv"] = "equiv",
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

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool
        Whether to check the DataFrame dtype is identical. Defaults to True.
    check_index_type : bool
        Whether to check the Index class, dtype, and inferred_type are identical.
        Defaults to True.
    check_column_type : bool or {'equiv'}
        Whether to check the column class, dtype, and inferred_type are identical.
        Passed as the ``exact`` argument of :func:`assert_index_equal`.
        Defaults to 'equiv'.
    check_frame_type : bool
        Whether to check the DataFrame class is identical. Defaults to True.
    check_names : bool
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical. Defaults to True.
    check_exact : bool
        Whether to compare numbers exactly. Defaults to False.
    check_categorical : bool
        Whether to compare internal Categoricals exactly. Defaults to True.
    check_like : bool
        If True, ignore the order of index and columns.
        Note: index labels must match their respective rows (as in columns);
        same labels must be with the same data. Defaults to False.
    rtol : float
        Relative tolerance. Only used when `check_exact` is False. Defaults to 1e-5.
    atol : float
        Absolute tolerance. Only used when `check_exact` is False. Defaults to 1e-8.
    obj : str
        A name for the object being compared, used in assertion messages.
        Defaults to 'DataFrame'.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.

    Examples
    --------
    >>> import arkouda as ak

    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from arkouda.testing import assert_frame_equal
    >>> df1 = ak.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = ak.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    df1 equals itself.
    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.
    >>> assert_frame_equal(df1, df2) # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)

    """
    __tracebackhide__ = not DEBUG

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))
        assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(obj, f"{obj} shape mismatch", f"{repr(left.shape)}", f"{repr(right.shape)}")

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=rtol,
        atol=atol,
        obj=f"{obj}.index",
    )

    if check_column_type == "equiv":
        check_column_type = True
    assert isinstance(check_column_type, bool)

    # column comparison
    assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_exact=check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=rtol,
        atol=atol,
        obj=f"{obj}.columns",
    )

    if check_like:
        # @TODO use left.reindex_like(right)
        left = left[right.index.values]

    for col in left.columns.values:
        # We have already checked that columns match, so we can do
        #  fast location-based lookups
        lcol = left[col]
        rcol = right[col]
        if not isinstance(lcol, Series):
            lcol = Series(lcol)
        if not isinstance(rcol, Series):
            rcol = Series(rcol)

        # use check_index=False, because we do not want to run
        # assert_index_equal for each column,
        # as we already checked it for the whole dataframe before.
        assert_series_equal(
            lcol,
            rcol,
            check_dtype=check_dtype,
            check_index_type=check_index_type,
            check_exact=check_exact,
            check_names=check_names,
            check_categorical=check_categorical,
            obj=f'{obj}.(column name="{col}")',
            rtol=rtol,
            atol=atol,
            check_index=False,
        )


def assert_equal(left, right, **kwargs) -> None:
    """
    Wrap tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, or pdarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.

    """
    __tracebackhide__ = not DEBUG

    if isinstance(left, Index):
        assert_index_equal(left, right, **kwargs)
    elif isinstance(left, Series):
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, (pdarray, Strings, Categorical, SegArray)):
        assert_arkouda_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        assert_almost_equal(left, right)


def assert_contains_all(iterable, dic) -> None:
    """
    Assert that a dictionary contains all the elements of an iterable.

    Parameters
    ----------
    iterable: iterable
    dic: dict

    """
    for k in iterable:
        assert k in dic, f"Did not contain item: {repr(k)}"


def assert_copy(iter1, iter2, **eql_kwargs) -> None:
    """
    Check that the elements are equal, but not the same object.

    Does not check that items in sequences are also not the same object.

    Parameters
    ----------
    iter1, iter2: iterable
        Iterables that produce elements comparable with assert_almost_equal.

    """
    for elem1, elem2 in zip(iter1, iter2):
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        msg = (
            f"Expected object {repr(type(elem1))} and object {repr(type(elem2))} to be "
            "different objects, but they were the same object."
        )
        assert elem1 is not elem2, msg
