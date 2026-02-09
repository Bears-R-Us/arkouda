from __future__ import annotations

import builtins
import json
import sys

from math import prod as maprod
from typing import TYPE_CHECKING, List, Literal, Sequence, Tuple, TypeVar, Union, cast

from typeguard import typechecked

from arkouda.client_dtypes import BitVector, IPv4, bit_vectorizer
from arkouda.infoclass import list_registry
from arkouda.numpy.dtypes import (
    _is_dtype_in_union,
    dtype,
    float_scalars,
    int_scalars,
    numeric_scalars,
)
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraysetops import unique
from arkouda.numpy.sorting import coargsort
from arkouda.numpy.timeclass import Datetime, Timedelta
from arkouda.pandas.groupbyclass import GroupBy


__all__ = [
    "attach",
    "attach_all",
    "_axis_validation",
    "broadcast_dims",
    "broadcast_shapes",
    "broadcast_arrays",
    "convert_bytes",
    "convert_if_categorical",
    "copy",
    "generic_concat",
    "get_callback",
    "identity",
    "_integer_axis_validation",
    "invert_permutation",
    "is_float",
    "is_int",
    "is_numeric",
    "is_registered",
    "map",
    "may_share_memory",
    "register",
    "register_all",
    "report_mem",
    "shares_memory",
    "sparse_sum_help",
    "unregister",
    "unregister_all",
]


if TYPE_CHECKING:
    from arkouda.categorical import Categorical
    from arkouda.client import get_config, get_mem_used
    from arkouda.numpy.pdarraycreation import arange
    from arkouda.numpy.segarray import SegArray
    from arkouda.numpy.strings import Strings
    from arkouda.pandas.index import Index
    from arkouda.pandas.series import Series
else:
    Categorical = TypeVar("Categorical")
    SegArray = TypeVar("SegArray")
    Strings = TypeVar("Strings")
    Index = TypeVar("Index")
    Series = TypeVar("Series")


def identity(x):
    return x


def get_callback(x):
    if type(x) in {Datetime, Timedelta, IPv4}:
        return type(x)
    elif hasattr(x, "_cast"):
        return x._cast
    elif isinstance(x, BitVector):
        return bit_vectorizer(width=x.width, reverse=x.reverse)
    else:
        return identity


def generic_concat(items, ordered=True):
    # this version can be called with Dataframe and Series (which have Class.concat methods)
    from arkouda.numpy.pdarraysetops import concatenate as pdarrayconcatenate

    types = {type(x) for x in items}
    if len(types) != 1:
        raise TypeError(f"Items must all have same type: {types}")
    t = types.pop()

    if t is list:
        return [x for lst in items for x in lst]

    return (
        t.concat(items, ordered=ordered)
        if hasattr(t, "concat")
        else pdarrayconcatenate(items, ordered=ordered)
    )


def report_mem(pre=""):
    cfg = get_config()
    used = get_mem_used() / (cfg["numLocales"] * cfg["physicalMemory"])
    sys.stdout.write(f"{pre} mem use: {get_mem_used() / (1024**4): .2f} TB ({used:.1%})")


@typechecked
def invert_permutation(perm: pdarray) -> pdarray:
    """
    Compute the inverse of a permutation array.

    The inverse permutation undoes the effect of the original permutation.
    For a valid permutation array `perm`, this function returns an array `inv`
    such that `inv[perm[i]] == i` for all `i`.

    Parameters
    ----------
    perm : pdarray
        A permutation of the integers `[0, N-1]`, where `N` is the length of the array.

    Returns
    -------
    pdarray
        The inverse of the input permutation.

    Raises
    ------
    ValueError
        If `perm` is not a valid permutation of the range `[0, N-1]`, such as
        containing duplicates or missing values.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import array, invert_permutation
    >>> perm = array([2, 0, 3, 1])
    >>> inv = invert_permutation(perm)
    >>> print(inv)
    [1 3 0 2]

    """
    unique_vals = unique(perm)
    if (not isinstance(unique_vals, pdarray)) or unique_vals.size != perm.size:
        raise ValueError("The array is not a permutation.")
    return coargsort([perm, arange(0, perm.size)])


def convert_if_categorical(values):
    """
    Convert a Categorical array to a Strings array for display purposes.

    If the input is a Categorical, it is converted to its string labels
    based on its codes. If not, the input is returned unchanged.

    Parameters
    ----------
    values : Categorical or any
        The input array, which may be a Categorical.

    Returns
    -------
    Strings or original type
        The string labels if `values` is a Categorical, otherwise the original input.

    Examples
    --------
    >>> import arkouda as ak

    Example with a Categorical
    >>> categories = ak.array(["apple", "banana", "cherry"])
    >>> cat = ak.Categorical(categories)
    >>> result = convert_if_categorical(cat)
    >>> print(result)
    ['apple', 'banana', 'cherry']

    Example with a non-Categorical input
    >>> values = ak.array([1, 2, 3])
    >>> result = convert_if_categorical(values)
    >>> print(result)
    [1 2 3]
    """
    from arkouda.pandas.categorical import Categorical

    if isinstance(values, Categorical):
        values = values.categories[values.codes]
    return values


def register(obj, name):
    """
    Register an Arkouda object with a user-specified name.

    This function registers the provided Arkouda object (`obj`) under a
    given name (`name`). It is backwards compatible with earlier versions
    of Arkouda.

    Parameters
    ----------
    obj : Arkouda object
        The Arkouda object to register.
    name : str
        The name to associate with the object.

    Returns
    -------
    Registered object
        The input object, now registered with the specified name.

    Raises
    ------
    AttributeError
        If `obj` does not have a `register` method.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.util import register
    >>> obj = ak.array([1, 2, 3])
    >>> registered_obj = register(obj, "my_array")
    >>> print(registered_obj)
    [1 2 3]
    >>> registered_obj.unregister()

    Example of registering a different Arkouda object
    >>> categories = ak.array(["apple", "banana", "cherry"])
    >>> cat = ak.Categorical(categories)
    >>> registered_cat = register(cat, "my_cat")
    >>> print(registered_cat)
    ['apple', 'banana', 'cherry']

    """
    return obj.register(name)


@typechecked
def attach(name: str):
    """
    Attach a previously created Arkouda object by its registered name.

    This function retrieves an Arkouda object (e.g., `pdarray`, `DataFrame`,
    `Series`, etc.) associated with a given `name`. It returns the corresponding
    object based on the type of object stored under that name.

    Parameters
    ----------
    name : str
        The name of the object to attach.

    Returns
    -------
    object
        The Arkouda object associated with the given `name`. The returned object
        could be of any supported type, such as `pdarray`, `DataFrame`, `Series`,
        etc.

    Raises
    ------
    ValueError
        If the object type in the response message does not match any known types.

    Examples
    --------
    >>> import arkouda as ak

    Attach an existing pdarray
    >>> obj = ak.array([1, 2, 3])
    >>> registered_obj = obj.register("my_array")
    >>> arr = ak.attach("my_array")
    >>> print(arr)
    [1 2 3]
    >>> registered_obj.unregister()
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarrayclass import pdarray
    from arkouda.numpy.segarray import SegArray
    from arkouda.numpy.strings import Strings
    from arkouda.pandas.categorical import Categorical
    from arkouda.pandas.dataframe import DataFrame
    from arkouda.pandas.index import Index, MultiIndex
    from arkouda.pandas.series import Series

    attachable = Union[
        pdarray,
        Strings,
        Datetime,
        Timedelta,
        IPv4,
        SegArray,
        DataFrame,
        GroupBy,
        Categorical,
    ]

    rep_msg = json.loads(cast(str, generic_msg(cmd="attach", args={"name": name})))
    rtn_obj: attachable | None = None
    if rep_msg["objType"].lower() == pdarray.objType.lower():
        rtn_obj = create_pdarray(rep_msg["create"])
    elif rep_msg["objType"].lower() == Strings.objType.lower():
        rtn_obj = Strings.from_return_msg(rep_msg["create"])
    elif rep_msg["objType"].lower() == Datetime.special_objType.lower():
        rtn_obj = Datetime(create_pdarray(rep_msg["create"]))
    elif rep_msg["objType"].lower() == Timedelta.special_objType.lower():
        rtn_obj = Timedelta(create_pdarray(rep_msg["create"]))
    elif rep_msg["objType"].lower() == IPv4.special_objType.lower():
        rtn_obj = IPv4(create_pdarray(rep_msg["create"]))
    elif rep_msg["objType"].lower() == SegArray.objType.lower():
        rtn_obj = SegArray.from_return_msg(rep_msg["create"])
    elif rep_msg["objType"].lower() == DataFrame.objType.lower():
        rtn_obj = DataFrame.from_return_msg(rep_msg["create"])
    elif rep_msg["objType"].lower() == GroupBy.objType.lower():
        rtn_obj = GroupBy.from_return_msg(rep_msg["create"])
    elif rep_msg["objType"].lower() == Categorical.objType.lower():
        rtn_obj = Categorical.from_return_msg(rep_msg["create"])
    elif (
        rep_msg["objType"].lower() == Index.objType.lower()
        or rep_msg["objType"].lower() == MultiIndex.objType.lower()
    ):
        rtn_obj = Index.from_return_msg(rep_msg["create"])
    elif rep_msg["objType"].lower() == Series.objType.lower():
        rtn_obj = Series.from_return_msg(rep_msg["create"])
    elif rep_msg["objType"].lower() == BitVector.special_objType.lower():
        rtn_obj = BitVector.from_return_msg(rep_msg["create"])

    if rtn_obj is not None:
        rtn_obj.registered_name = name
    return rtn_obj


@typechecked
def unregister(name: str) -> str:
    """
    Unregister an Arkouda object by its name.

    This function sends a request to unregister the Arkouda object associated
    with the specified `name`. It returns a response message indicating the
    success or failure of the operation.

    Parameters
    ----------
    name : str
        The name of the object to unregister.

    Returns
    -------
    str
        A message indicating the result of the unregister operation.

    Raises
    ------
    RuntimeError
        If the object associated with the given `name` does not exist or cannot
        be unregistered.

    Examples
    --------
    >>> import arkouda as ak

    Unregister an existing object
    >>> obj = ak.array([1, 2, 3])
    >>> registered_obj = obj.register("my_array")
    >>> response = ak.unregister("my_array")
    >>> print(response)
    Unregistered PDARRAY my_array

    """
    from arkouda.client import generic_msg

    rep_msg = cast(str, generic_msg(cmd="unregister", args={"name": name}))

    return rep_msg


@typechecked
def is_registered(name: str, as_component: bool = False) -> bool:
    """
    Determine if the provided name is associated with a registered Arkouda object.

    This function checks if the `name` is found in the registry of objects,
    and optionally checks if it is registered as a component of a registered object.

    Parameters
    ----------
    name : str
        The name to check for in the registry.
    as_component : bool, default=False
        When True, the function checks if the name is registered as a component
        of a registered object (rather than as a standalone object).

    Returns
    -------
    bool
        `True` if the name is found in the registry, `False` otherwise.

    Raises
    ------
    KeyError
        If the registry query encounters an issue (e.g., invalid registry data or access issues).

    Examples
    --------
    >>> import arkouda as ak

    Check if a name is registered as an object
    >>> obj = ak.array([1, 2, 3])
    >>> registered_obj = obj.register("my_array")
    >>> result = ak.is_registered("my_array")
    >>> print(result)
    True
    >>> registered_obj.unregister()

    Check if a name is registered as a component
    >>> result = ak.is_registered("my_component", as_component=True)
    >>> print(result)
    False
    """
    return name in list_registry()["Components" if as_component else "Objects"]


def register_all(data: dict):
    """
    Register all objects in the provided dictionary.

    This function iterates through the dictionary `data`, registering each object
    with its corresponding name. It is useful for batch registering multiple
    objects in Arkouda.

    Parameters
    ----------
    data : dict
        A dictionary that maps the name to register the object to the object itself.
        For example, {"MyArray": ak.array([0, 1, 2])}.

    Examples
    --------
    >>> import arkouda as ak
    >>> data = { "array1": ak.array([0, 1, 2]), "array2": ak.array([3, 4, 5]) }
    >>> ak.register_all(data)

    After calling this function, "array1" and "array2" are registered
    in Arkouda, and can be accessed by their names.
    >>> ak.unregister_all(["array1", "array2"])

    """
    for reg_name, obj in data.items():
        register(obj, reg_name)


def unregister_all(names: List[str]):
    """
    Unregister all Arkouda objects associated with the provided names.

    This function iterates through the list of `names`, unregistering each
    corresponding object from the Arkouda server.

    Parameters
    ----------
    names : List of str
        A list of registered names corresponding to Arkouda objects
        that should be unregistered.

    Examples
    --------
    >>> import arkouda as ak
    >>> data = { "array1": ak.array([0, 1, 2]), "array2": ak.array([3, 4, 5]) }
    >>> ak.register_all(data)

    After calling this function, "array1" and "array2" are registered
    in Arkouda, and can be accessed by their names.
    >>> ak.unregister_all(["array1", "array2"])

    "arr1" and "arr2" are now unregistered

    """
    for n in names:
        unregister(n)


def attach_all(names: list):
    """
    Attach to all objects registered with the provided names.

    This function returns a dictionary mapping each name in the input list
    to the corresponding Arkouda object retrieved using `attach`.

    Parameters
    ----------
    names : List of str
        A list of names corresponding to registered Arkouda objects.

    Returns
    -------
    dict
        A dictionary mapping each name to the attached Arkouda object.

    Examples
    --------
    >>> import arkouda as ak
    >>> data = { "arr1": ak.array([0, 1, 2]), "arr2": ak.array([3, 4, 5]) }
    >>> ak.register_all(data)

    Assuming "arr1" and "arr2" were previously registered
    >>> attached_objs = ak.attach_all(["arr1", "arr2"])
    >>> print(attached_objs["arr1"])
    [0 1 2]
    >>> print(type(attached_objs["arr2"]))
    <class 'arkouda.numpy.pdarrayclass.pdarray'>
    >>> ak.unregister_all(["arr1", "arr2"])
    """
    return {n: attach(n) for n in names}


def sparse_sum_help(
    idx1: pdarray,
    idx2: pdarray,
    val1: pdarray,
    val2: pdarray,
    merge: bool = True,
    percent_transfer_limit: int = 100,
) -> Tuple[pdarray, pdarray]:
    """
    Sum two sparse matrices together.

    This function returns the result of summing two sparse matrices by combining
    their indices and values. Internally, it performs the equivalent of:

        ak.GroupBy(ak.concatenate([idx1, idx2])).sum(ak.concatenate((val1, val2)))

    Parameters
    ----------
    idx1 : pdarray
        Indices for the first sparse matrix.
    idx2 : pdarray
        Indices for the second sparse matrix.
    val1 : pdarray
        Values for the first sparse matrix.
    val2 : pdarray
        Values for the second sparse matrix.
    merge : bool, default=True
        If True, the indices are combined using a merge-based workflow.
        If False, a sort-based workflow is used.
    percent_transfer_limit : int, default=100
        Only used when `merge` is True. This defines the maximum percentage of
        data allowed to move between locales during the merge. If this threshold
        is exceeded, a sort-based workflow is used instead.

    Returns
    -------
    Tuple[pdarray, pdarray]
        A tuple containing:
        - The indices of the resulting sparse matrix.
        - The summed values associated with those indices.

    Examples
    --------
    >>> import arkouda as ak
    >>> idx1 = ak.array([0, 1, 3, 4, 7, 9])
    >>> idx2 = ak.array([0, 1, 3, 6, 9])
    >>> vals1 = idx1
    >>> vals2 = ak.array([10, 11, 13, 16, 19])
    >>> ak.util.sparse_sum_help(idx1, idx2, vals1, vals2)
    (array([0 1 3 4 6 7 9]), array([10 12 16 4 16 7 28]))

    >>> ak.GroupBy(ak.concatenate([idx1, idx2])).sum(ak.concatenate((vals1, vals2)))
    (array([0 1 3 4 6 7 9]), array([10 12 16 4 16 7 28]))
    """
    from arkouda.client import generic_msg

    rep_msg = generic_msg(
        cmd="sparseSumHelp",
        args={
            "idx1": idx1,
            "idx2": idx2,
            "val1": val1,
            "val2": val2,
            "merge": merge,
            "percent_transfer_limit": percent_transfer_limit,
        },
    )
    inds, vals = cast(str, rep_msg).split("+", maxsplit=1)
    return create_pdarray(inds), create_pdarray(vals)


@typechecked
def broadcast_shapes(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Determine a broadcasted shape, given an arbitary number of shapes.

    This function implements the broadcasting rules from the Array API standard
    to compute the shape resulting from broadcasting two arrays together.

    See: https://data-apis.org/array-api/latest/API_specification/broadcasting.html#algorithm

    Parameters
    ----------
    shapes : Tuple[int, ...]
        a list or tuple of the shapes to be broadcast

    Returns
    -------
    Tuple[int, ...]
        The broadcasted shape

    Raises
    ------
    ValueError
        If the shapes are not compatible for broadcasting.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.broadcast_shapes((1,2,3),(4,1,3),(4,2,1))
    (4, 2, 3)

    """
    from numpy import broadcast_shapes as b_shapes

    try:
        return b_shapes(*shapes)
    except ValueError:
        raise ValueError(f"Found no common broadcast shape for: {shapes}")


@typechecked
def broadcast_arrays(*arrays: pdarray) -> List[pdarray]:
    """
    Broadcast arrays to a common shape.

    Parameters
    ----------
    arrays : pdarray
        The arrays to broadcast. Must be broadcastable to a common shape.

    Returns
    -------
    List
        A list whose elements are the given Arrays broadcasted to the common shape.

    Raises
    ------
    ValueError
        Raised by broadcast_to if a common shape cannot be determined.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.arange(10).reshape(1,2,5)
    >>> b = ak.arange(20).reshape(4,1,5)
    >>> c = ak.broadcast_arrays(a,b)
    >>> c[0][0,:,:]
    array([array([0 1 2 3 4]) array([5 6 7 8 9])])
    >>> c[1][:,0,0]
    array([0 5 10 15])

    """
    shapes = [a.shape for a in arrays]
    bc_shape = broadcast_shapes(*shapes)
    return [broadcast_to(a, shape=bc_shape) for a in arrays]


@typechecked
def broadcast_to(x: Union[numeric_scalars, pdarray], shape: Union[int, Tuple[int, ...]]) -> pdarray:
    """
    Broadcast the array to the specified shape.

    Parameters
    ----------
    x: int, pdarray
        The int or array to be broadcast.
    shape: int, Tuple[int, ...]
        The shape to which the array is to be broadcast.

    Notes
    -----
    If x and shape are both integers, the result has shape (shape,).
    If x is an int and shape is a tuple, the result has shape (shape,).
    if x is a pdarray and shape is an int, then if x.shape == (shape,)
        x is unchanged.  Otherwise a ValueError is raised.
    If x is a pdarray and shape is a tuple, then x is broadcast to shape, if possible.

    Returns
    -------
    pdarray
        A new array which is x broadcast to the provided shape.

    Raises
    ------
    ValueError
        Raised server-side if the broadcast fails, or client-side in the case where
        x is a pdarray, shape is an int, and x.shape != (shape,).

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.arange(5)
    >>> ak.broadcast_to(a,(2,5))
    array([array([0 1 2 3 4]) array([0 1 2 3 4])])


    """
    from arkouda.client import generic_msg
    from arkouda.numpy.dtypes import _val_isinstance_of_union
    from arkouda.numpy.pdarraycreation import full as akfull

    if _val_isinstance_of_union(x, numeric_scalars):
        assert not isinstance(x, pdarray)  # Required for mypy
        return akfull(shape, x, dtype=type(x))
    elif isinstance(x, pdarray) and isinstance(shape, int):
        if x.ndim == 1 and x.size == shape:
            return x
        else:
            raise ValueError(f"Operands could not be broadcast together: {x.shape} and {shape}")
    elif isinstance(x, pdarray) and isinstance(shape, tuple):
        try:
            return create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"broadcast<{x.dtype},{x.ndim},{len(shape)}>",
                        args={
                            "name": x,
                            "shape": shape,
                        },
                    ),
                )
            )
        except RuntimeError as e:
            raise ValueError(f"Failed to broadcast array: {e}")
    else:
        raise ValueError("Operands could not be broadcast.")


@typechecked
def broadcast_dims(sa: Sequence[int], sb: Sequence[int]) -> Tuple[int, ...]:
    """
    Determine the broadcasted shape of two arrays given their shapes.

    This function implements the broadcasting rules from the Array API standard
    to compute the shape resulting from broadcasting two arrays together.

    See: https://data-apis.org/array-api/latest/API_specification/broadcasting.html#algorithm

    Parameters
    ----------
    sa : Sequence[int]
        The shape of the first array.
    sb : Sequence[int]
        The shape of the second array.

    Returns
    -------
    Tuple[int, ...]
        The broadcasted shape resulting from combining `sa` and `sb`.

    Raises
    ------
    ValueError
        If the shapes are not compatible for broadcasting.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.util import broadcast_dims
    >>> broadcast_dims((5, 1), (1, 3))
    (5, 3)

    >>> broadcast_dims((4,), (3, 1))
    (3, 4)
    """
    n_a = len(sa)
    n_b = len(sb)
    n = max(n_a, n_b)
    shape_out = [0 for i in range(n)]

    i = n - 1
    while i >= 0:
        n1 = n_a - n + i
        n2 = n_b - n + i

        d1 = sa[n1] if n1 >= 0 else 1
        d2 = sb[n2] if n2 >= 0 else 1

        if d1 == 1:
            shape_out[i] = d2
        elif d2 == 1:
            shape_out[i] = d1
        elif d1 == d2:
            shape_out[i] = d1
        else:
            raise ValueError("Incompatible dimensions for broadcasting")

        i -= 1

    return tuple(shape_out)


def convert_bytes(nbytes: int_scalars, unit: Literal["B", "KB", "MB", "GB"] = "B") -> numeric_scalars:
    """
    Convert a number of bytes to a larger unit: KB, MB, or GB.

    Parameters
    ----------
    nbytes : int_scalars
        The number of bytes to convert.
    unit : {"B", "KB", "MB", "GB"}, default="B"
        The unit to convert to. One of {"B", "KB", "MB", "GB"}.

    Returns
    -------
    numeric_scalars
        The converted value in the specified unit.

    Raises
    ------
    ValueError
        If `unit` is not one of {"B", "KB", "MB", "GB"}.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.util import convert_bytes
    >>> convert_bytes(2048, unit="KB")
    2.0

    >>> convert_bytes(1048576, unit="MB")
    1.0

    >>> convert_bytes(1073741824, unit="GB")
    1.0

    """
    kb = 1024
    mb = kb * kb
    gb = mb * kb
    if unit == "B":
        return nbytes
    elif unit == "KB":
        return float(nbytes / kb)
    elif unit == "MB":
        return float(nbytes / mb)
    elif unit == "GB":
        return float(nbytes / gb)
    else:
        raise ValueError("Invalid unit. Must be one of {'B', 'KB', 'MB', 'GB'}")


def is_numeric(arry: Union[pdarray, Strings, Categorical, Series, Index]) -> builtins.bool:
    """
    Check if the dtype of the given array-like object is numeric.

    Parameters
    ----------
    arry : pdarray, Strings, Categorical, Series, or Index
        The object to check.

    Returns
    -------
    bool
        True if the dtype of `arry` is numeric, False otherwise.

    Examples
    --------
    >>> import arkouda as ak
    >>> data = ak.array([1, 2, 3, 4, 5])
    >>> ak.util.is_numeric(data)
    True

    >>> strings = ak.array(["a", "b", "c"])
    >>> ak.util.is_numeric(strings)
    False

    >>> from arkouda import Categorical
    >>> cat = Categorical(strings)
    >>> ak.util.is_numeric(cat)
    False
    """
    from arkouda.pandas.index import Index
    from arkouda.pandas.series import Series

    if isinstance(arry, (pdarray, Series, Index)):
        return _is_dtype_in_union(dtype(arry.dtype), numeric_scalars)
    else:
        return False


def is_float(arry: Union[pdarray, Strings, Categorical, Series, Index]) -> builtins.bool:
    """
    Check if the dtype of the given array-like object is a float type.

    Parameters
    ----------
    arry : pdarray, Strings, Categorical, Series, or Index
        The object to check.

    Returns
    -------
    bool
        True if the dtype of `arry` is a float type, False otherwise.

    Examples
    --------
    >>> import arkouda as ak
    >>> data = ak.array([1.0, 2, 3, 4, float('nan')])
    >>> ak.util.is_float(data)
    True

    >>> data2 = ak.arange(5)
    >>> ak.util.is_float(data2)
    False

    >>> strings = ak.array(["1.0", "2.0"])
    >>> ak.util.is_float(strings)
    False
    """
    from arkouda.pandas.index import Index
    from arkouda.pandas.series import Series

    if isinstance(arry, (pdarray, Series, Index)):
        return _is_dtype_in_union(dtype(arry.dtype), float_scalars)
    else:
        return False


def is_int(arry: Union[pdarray, Strings, Categorical, Series, Index]) -> builtins.bool:
    """
    Check if the dtype of the given array-like object is an integer type.

    Parameters
    ----------
    arry : pdarray, Strings, Categorical, Series, or Index
        The object to check.

    Returns
    -------
    bool
        True if the dtype of `arry` is an integer type, False otherwise.

    Examples
    --------
    >>> import arkouda as ak
    >>> data = ak.array([1.0, 2, 3, 4, float('nan')])
    >>> ak.util.is_int(data)
    False

    >>> data2 = ak.arange(5)
    >>> ak.util.is_int(data2)
    True

    >>> strings = ak.array(["1", "2"])
    >>> ak.util.is_int(strings)
    False
    """
    from arkouda.pandas.index import Index
    from arkouda.pandas.series import Series

    if isinstance(arry, (pdarray, Series, Index)):
        return _is_dtype_in_union(dtype(arry.dtype), int_scalars)
    else:
        return False


def map(
    values: Union[pdarray, Strings, Categorical], mapping: Union[dict, Series]
) -> Union[pdarray, Strings]:
    """
    Map the values of an array according to an input mapping.

    Parameters
    ----------
    values : pdarray, Strings, or Categorical
        The values to be mapped.
    mapping : dict or Series
        The mapping correspondence. A dictionary or Series that defines how
        to map the `values` array.

    Returns
    -------
    Union[pdarray, Strings]
        A new array with the values mapped by the provided mapping.
        The return type matches the type of `values`. If the input `Series`
        has Categorical values, the return type will be `Strings`.

    Raises
    ------
    TypeError
        If `mapping` is not of type `dict` or `Series`.
        If `values` is not of type `pdarray`, `Categorical`, or `Strings`.
    ValueError
        If a mapping with tuple keys has inconsistent lengths, or if a MultiIndex
        mapping has a different number of levels than the GroupBy keys.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.numpy.util import map
    >>> a = ak.array([2, 3, 2, 3, 4])
    >>> a
    array([2 3 2 3 4])
    >>> ak.util.map(a, {4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
    array([30.00000000000000000 5.00000000000000000 30.00000000000000000
    5.00000000000000000 25.00000000000000000])
    >>> s = ak.Series(ak.array(["a", "b", "c", "d"]), index=ak.array([4, 2, 1, 3]))
    >>> ak.util.map(a, s)
    array(['b', 'd', 'b', 'd', 'a'])
    """
    import numpy as np

    from arkouda import Series, array, broadcast, full
    from arkouda.numpy.pdarraysetops import in1d
    from arkouda.numpy.strings import Strings
    from arkouda.pandas.categorical import Categorical
    from arkouda.pandas.index import MultiIndex

    keys = values
    gb = GroupBy(keys, dropna=False)
    gb_keys = gb.unique_keys

    # helper: number of unique keys (works for single key or tuple-of-keys)
    nuniq = gb_keys[0].size if isinstance(gb_keys, tuple) else gb_keys.size

    # Fast-path: empty mapping => everything is missing
    if (isinstance(mapping, dict) and len(mapping) == 0) or (
        isinstance(mapping, Series) and len(mapping.index) == 0
    ):
        if not isinstance(values, (Strings, Categorical)):
            return broadcast(gb.segments, full(nuniq, np.nan, values.dtype), permutation=gb.permutation)
        else:
            return broadcast(gb.segments, full(nuniq, "null"), permutation=gb.permutation)

    if isinstance(mapping, dict):
        # Build mapping as a Series with an Index/MultiIndex (avoid rank>1 arrays)
        m_keys = list(mapping.keys())
        m_vals = list(mapping.values())

        k0 = m_keys[0]
        if isinstance(k0, tuple):
            # validate tuple keys
            if not all(isinstance(k, tuple) for k in m_keys):
                raise TypeError("Mixed key types in mapping dict (tuple and non-tuple).")
            n = len(k0)
            if not all(len(k) == n for k in m_keys):
                raise ValueError("All tuple keys in mapping dict must have the same length.")

            cols = list(zip(*m_keys))  # transpose list[tuple] -> list[level]
            idx = MultiIndex([array(col) for col in cols])
            mapping = Series(array(m_vals), index=idx)
        else:
            mapping = Series(array(m_vals), index=array(m_keys))

    if isinstance(mapping, Series):
        # Normalize mapping index keys into a "groupable" (single array OR tuple-of-arrays)
        mindex = mapping.index
        if isinstance(mindex, MultiIndex):
            mkeys = tuple(mindex.index)
        else:
            mkeys = mindex.values

        if isinstance(gb_keys, tuple) and isinstance(mkeys, tuple):
            if len(gb_keys) != len(mkeys):
                raise ValueError(
                    f"Mapping MultiIndex has {len(mkeys)} levels but GroupBy has {len(gb_keys)} keys"
                )

        # invert=True => mask is True for GroupBy unique keys that are *missing* from the mapping,
        # i.e., values that should be filled with NaN/"null".
        mask = in1d(gb_keys, mkeys, invert=True)

        # Compute extra keys + extra size without mixing tuple/non-tuple assignments
        if isinstance(gb_keys, tuple):
            xtra_keys_t = tuple(k[mask] for k in gb_keys)
            xtra_size = xtra_keys_t[0].size if len(xtra_keys_t) > 0 else 0

            if xtra_size > 0:
                nans: Union[pdarray, Strings]  # without this, mypy complains
                if not isinstance(mapping.values, (Strings, Categorical)):
                    nans = full(xtra_size, np.nan, mapping.values.dtype)
                else:
                    nans = full(xtra_size, "null")

                # Convert any categorical levels to strings, level-by-level
                xtra_keys_t = tuple(
                    k.to_strings() if isinstance(k, Categorical) else k for k in xtra_keys_t
                )

                xtra_series = Series(nans, index=MultiIndex(list(xtra_keys_t)))
                mapping = Series.concat([mapping, xtra_series])

        else:
            xtra_keys_s = gb_keys[mask]
            xtra_size = xtra_keys_s.size

            if xtra_size > 0:
                if not isinstance(mapping.values, (Strings, Categorical)):
                    nans = full(xtra_size, np.nan, mapping.values.dtype)
                else:
                    nans = full(xtra_size, "null")

                if isinstance(xtra_keys_s, Categorical):
                    xtra_keys_s = xtra_keys_s.to_strings()

                xtra_series = Series(nans, index=xtra_keys_s)
                mapping = Series.concat([mapping, xtra_series])

        # Align mapping to gb_keys
        if isinstance(gb_keys, Categorical):
            mapping = mapping[gb_keys.to_strings()]
        else:
            mapping = mapping[gb_keys]

        if isinstance(mapping.values, (pdarray, Strings)):
            return broadcast(gb.segments, mapping.values, permutation=gb.permutation)
        else:
            raise TypeError("Map values must be castable to pdarray or Strings.")
    else:
        raise TypeError("Map must be dict or arkouda.Series.")


def _infer_shape_from_size(size):
    """
    Infer the shape, number of dimensions (ndim), and full size from a given size or shape.

    This function is used in pdarray creation functions that allow a size (1D) or shape (multi-dim).
    If the input is a tuple, it is treated as a multidimensional shape.
    If the input is a single integer, it is treated as a 1D shape.

    Parameters
    ----------
    size : int or tuple of int
        The size (for 1D arrays) or shape (for multidimensional arrays) of the desired array.

    Returns
    -------
    tuple
        A tuple containing:
        - shape: The shape of the array (either an integer for 1D or a tuple for multidimensional).
        - ndim: The number of dimensions
        (1 for 1D, or the length of the shape tuple for multidimensional).
        - full_size: The total number of elements in the array
        (size for 1D or product of dimensions for multidimensional).

    Examples
    --------
    >>> import arkouda as ak
    >>> _infer_shape_from_size(5)
    (5, 1, 5)

    >>> _infer_shape_from_size((3, 4))
    ((3, 4), 2, 12)
    """
    # used in pdarray creation functions that allow a size (1D) or shape (multi-dim)
    shape: Union[int_scalars, Tuple[int_scalars, ...]] = 1
    if isinstance(size, tuple):
        shape = cast(Tuple, size)
        full_size = 1
        for s in cast(Tuple, shape):
            full_size *= s
        ndim = len(shape)
    else:
        full_size = cast(int, size)
        shape = full_size
        ndim = 1
    return shape, ndim, full_size


def _generate_test_shape(rank, size):
    """
    Generate a shape for a multi-dimensional array that is close to a given size,
    while ensuring each dimension is at least 2.

    The shape will consist of `rank` dimensions, where the product of the dimensions
    is close to the given `size`. The first `rank-1` dimensions are set to 2,
    and the last dimension is adjusted such that the product of all dimensions is
    close to the desired size.

    Parameters
    ----------
    rank : int
        The number of dimensions (rank) for the generated shape.
    size : int
        The desired total size of the multi-dimensional array.

    Returns
    -------
    tuple
        A tuple containing:
        - shape: The generated shape as a tuple of integers.
        - local_size: The product of the shape dimensions, i.e., the total size.

    Examples
    --------
    >>> import arkouda as ak
    >>> _generate_test_shape(3, 16)
    ((2, 2, 4), 16)

    >>> _generate_test_shape(4, 24)
    ((2, 2, 2, 3), 24)
    """
    # used to generate shapes of the form (2,2,...n) for testing multi-dim creation
    last_dim = max(2, size // (2 ** (rank - 1)))  # such that 2*2*..*n is close to size,
    shape = (rank - 1) * [2]  # and with the final dim at least 2.
    shape.append(last_dim)  # building "shape" really does take
    shape = tuple(shape)  # multiple steps because .append doesn't
    local_size = maprod(shape)  # have a return value
    return shape, local_size


def copy(a: Union[Strings, pdarray]) -> Union[Strings, pdarray]:
    """
    Return a deep copy of the given Arkouda object.

    Parameters
    ----------
    a : Union[Strings, pdarray]
        The object to copy.

    Returns
    -------
    Union[Strings, pdarray]
        A deep copy of the pdarray or Strings object.

    Raises
    ------
    TypeError
        If the input is not a Strings or pdarray instance.
    """
    from arkouda.numpy.strings import Strings

    if isinstance(a, (Strings, pdarray)):
        return a.copy()
    raise TypeError(f"Unsupported type for copy: {type(a)}")


def _ak_buffer_names(x):
    """
    Return a set of server-side buffer names that back `x`.

    We try to be conservative: if we recognize a container, we pull out
    all of its backing pdarrays' `.name` values.

    Supported:
      - pdarray
      - Strings  (offsets + values/bytes)
      - SegArray (segments + values)
      - Categorical (codes + categories Strings)
      - Nested containers of the above (tuples/lists/dicts)
    """
    names = set()

    # Base case: pdarray
    if hasattr(x, "name") and isinstance(getattr(x, "name"), str):
        # Heuristic: Arkouda pdarray has `.name` referring to a server object
        names.add(x.name)
        return names

    # Strings: typically has .offsets and .values (or .offsets and .bytes)
    try:
        from arkouda.strings import Strings

        if isinstance(x, Strings):
            # Some versions expose .values, older expose .bytes
            _ak_buffer_names(x.get_offsets())
            _ak_buffer_names(x.get_bytes())
            if hasattr(x, "entry"):
                names |= _ak_buffer_names(x.entry)
            return names
    except Exception:
        pass

    # SegArray: segments + values
    try:
        from arkouda.segarray import SegArray

        if isinstance(x, SegArray):
            if hasattr(x, "segments"):
                names |= _ak_buffer_names(x.segments)
            if hasattr(x, "values"):
                names |= _ak_buffer_names(x.values)
            return names
    except Exception:
        pass

    # Categorical: codes + categories (Strings)
    try:
        from arkouda.categorical import Categorical

        if isinstance(x, Categorical):
            if hasattr(x, "codes"):
                names |= _ak_buffer_names(x.codes)
            if hasattr(x, "categories"):
                names |= _ak_buffer_names(x.categories)
            if hasattr(x, "segments"):
                names |= _ak_buffer_names(x.segments)
            if hasattr(x, "permutation"):
                names |= _ak_buffer_names(x.permutation)
            return names
    except Exception:
        pass

    # Compound structures: recurse
    if isinstance(x, (list, tuple, set)):
        for xi in x:
            names |= _ak_buffer_names(xi)
        return names

    if isinstance(x, dict):
        for xi in x.values():
            names |= _ak_buffer_names(xi)
        return names

    # Unknown / unsupported type: returns empty => we assume no buffers known
    return names


def shares_memory(a, b):
    """
    Return True if `a` and `b` share any Arkouda server-side buffers.

    This is an Arkouda analogue of numpy.shares_memory with a simpler definition:
    it checks for identical backing buffer *identities* (same server object names).

    Notes
    -----
    - Because Arkouda commonly *materializes* results (rather than views),
      aliasing is rare and usually only true when objects literally reference
      the same backing buffers.
    - For compound containers (e.g., SegArray, Strings, Categorical), we check
      all of their component buffers.
    - If you introduce true view semantics in the future, teach `_ak_buffer_names`
      to surface the base buffer name(s) and view descriptors, and compare bases.
    """
    a_names = _ak_buffer_names(a)
    b_names = _ak_buffer_names(b)
    return len(a_names.intersection(b_names)) > 0


def may_share_memory(a, b):
    """
    Conservative version akin to numpy.may_share_memory.

    For now it just defers to shares_memory.

    """
    # Example conservative policy:
    # if we fail to find any buffer names for either side but recognize
    # the object as Arkouda-ish, return True to be conservative.
    a_names = _ak_buffer_names(a)
    b_names = _ak_buffer_names(b)
    if not a_names and not b_names:
        # Unknown types: be conservative if you wish; here we say False.
        return False
    return len(a_names.intersection(b_names)) > 0


# bounds_check is just called on integers, to ensure they fit in the range


def bounds_check(axis, rank):
    if axis < -rank or axis >= rank:
        return False
    else:
        return True


# adjust_negs will only be called if bounds_check passes


def adjust_negs(axis, rank):
    return axis if axis >= 0 else axis + rank


# axis validation can be called in multiple conditions.

# Some functions require the axis to be an integer (or None).  For that, we have
# _integer_axis_validation, which returns a boolean and an int (or None).


def _integer_axis_validation(axis, rank):
    if axis is None:
        return True, None
    elif isinstance(axis, int):
        if bounds_check(axis, rank):
            axis = adjust_negs(axis, rank)
            return True, axis
        else:
            return False, axis
    else:
        return False, axis


# Other functions allow the axis to be None, int, List, or Tuple.
# For that, we have the more general _axis_validation, which returns
# a boolean and a list.


def _axis_validation(axis, rank):
    if axis is None:
        return True, None

    elif isinstance(axis, int):
        if bounds_check(axis, rank):
            axis = adjust_negs(axis, rank)
            return True, [axis]
        else:
            return False, [axis]

    else:
        if isinstance(axis, list):
            axis_ = axis.copy()
        elif isinstance(axis, tuple):
            axis_ = list(axis)
        else:
            return False, axis
        valid = True
        for i in range(len(axis_)):
            if bounds_check(axis_[i], rank):
                axis_[i] = adjust_negs(axis_[i], rank)
            else:
                valid = False
        return valid, axis_
