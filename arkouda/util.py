from __future__ import annotations

import builtins
import json
from typing import TYPE_CHECKING, Sequence, Tuple, Union, cast
from warnings import warn

from typeguard import typechecked

from arkouda.categorical import Categorical
from arkouda.client import generic_msg, get_config, get_mem_used
from arkouda.client_dtypes import BitVector, BitVectorizer, IPv4
from arkouda.groupbyclass import GroupBy, broadcast
from arkouda.infoclass import list_registry
from arkouda.numpy.dtypes import (
    _is_dtype_in_union,
    dtype,
    float_scalars,
    int_scalars,
    numeric_scalars,
)
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import arange
from arkouda.pdarraysetops import unique
from arkouda.segarray import SegArray
from arkouda.sorting import coargsort
from arkouda.strings import Strings
from arkouda.timeclass import Datetime, Timedelta

if TYPE_CHECKING:
    from arkouda.index import Index
    from arkouda.series import Series


def identity(x):
    return x


def get_callback(x):
    if type(x) in {Datetime, Timedelta, IPv4}:
        return type(x)
    elif hasattr(x, "_cast"):
        return x._cast
    elif isinstance(x, BitVector):
        return BitVectorizer(width=x.width, reverse=x.reverse)
    else:
        return identity


def concatenate(items, ordered=True):
    warn(
        "This function is deprecated and will be removed in a later version of Arkouda."
        " Use arkouda.util.generic_concat(items, ordered) instead.",
        DeprecationWarning,
    )

    return generic_concat(items, ordered=ordered)


def generic_concat(items, ordered=True):
    # this version can be called with Dataframe and Series (which have Class.concat methods)
    from arkouda.pdarraysetops import concatenate as pdarrayconcatenate

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
    print(f"{pre} mem use: {get_mem_used()/(1024**4): .2f} TB ({used:.1%})")


def enrich_inplace(data, keynames, aggregations, **kwargs):
    warn(
        "This function is deprecated and will be removed in a later version of Arkouda.",
        DeprecationWarning,
    )

    # TO DO: validate reductions and values
    try:
        keys = data[keynames]
    except (KeyError, TypeError):
        keys = [data[k] for k in keynames]
    g = GroupBy(keys, **kwargs)
    for resname, (reduction, values) in aggregations.items():
        try:
            values = data[values]
        except (KeyError, TypeError):
            pass
        if reduction == "count":
            pergroupval = g.size()[1]
        else:
            pergroupval = g.aggregate(values, reduction)[1]
        data[resname] = g.broadcast(pergroupval, permute=True)


def expand(size, segs, vals):
    """
    Expand an array with values placed into the indicated segments.

    Parameters
    ----------
    size : ak.pdarray
        The size of the array to be expanded
    segs : ak.pdarray
        The indices where the values should be placed
    vals : ak.pdarray
        The values to be placed in each segment

    Returns
    -------
    pdarray
        The expanded array.

    Notes
    -----
    This function (with different order of arguments) is now in arkouda
    proper as ak.broadcast. It is retained here for backwards compatibility.

    """
    warn(
        "This function is deprecated and will be removed in a later version of Arkouda."
        " Use arkouda.broadcast(segments, values, size) instead.",
        DeprecationWarning,
    )
    return broadcast(segs, vals, size=size)


def invert_permutation(perm):
    """
    Find the inverse of a permutation array.

    Parameters
    ----------
    perm : ak.pdarray
        The permutation array.

    Returns
    -------
    ak.array
        The inverse of the permutation array.

    """
    if unique(perm).size != perm.size:
        raise ValueError("The array is not a permutation.")
    return coargsort([perm, arange(0, perm.size)])


def most_common(g, values):
    warn(
        "This function is deprecated and will be removed in a later version of Arkouda."
        " Use arkouda.GroupBy.most_common(values) instead.",
        DeprecationWarning,
    )

    return g.most_common(values)


def convert_if_categorical(values):
    """
    Convert a Categorical array to Strings for display
    """

    if isinstance(values, Categorical):
        values = values.categories[values.codes]
    return values


def register(obj, name):
    """
    Register an arkouda object with a user-specified name. Backwards compatible
    with earlier arkouda versions.
    """
    return obj.register(name)


@typechecked
def attach(name: str):
    from arkouda.dataframe import DataFrame
    from arkouda.index import Index, MultiIndex
    from arkouda.pdarrayclass import pdarray
    from arkouda.series import Series

    rep_msg = json.loads(cast(str, generic_msg(cmd="attach", args={"name": name})))
    rtn_obj = None
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
    rep_msg = cast(str, generic_msg(cmd="unregister", args={"name": name}))

    return rep_msg


@typechecked
def is_registered(name: str, as_component: bool = False) -> bool:
    """
    Determine if the name provided is associated with a registered Object

    Parameters
    ----------
    name: str
        The name to check for in the registry
    as_component: bool
        Default: False
        When True, the name will be checked to determine if it is registered as a component of
        a registered object

    Return
    -------
    bool
    """
    return name in list_registry()["Components" if as_component else "Objects"]


def register_all(data: dict):
    """
    Register all objects in the provided dictionary

    Parameters
    -----------
    data: dict
        Maps name to register the object to the object. For example, {"MyArray": ak.array([0, 1, 2])

    Returns
    --------
    None
    """
    for reg_name, obj in data.items():
        register(obj, reg_name)


def unregister_all(names: list):
    """
    Unregister all names provided

    Parameters
    -----------
    names : list
        List of names used to register objects to be unregistered

    Returns
    --------
    None
    """
    for n in names:
        unregister(n)


def attach_all(names: list):
    """
    Attach to all objects registered with the names provide

    Parameters
    -----------
    names: list
        List of names to attach to

    Returns
    --------
    dict
    """
    return {n: attach(n) for n in names}


def sparse_sum_help(idx1, idx2, val1, val2, merge=True, percent_transfer_limit=100):
    """
    Helper for summing two sparse matrices together

    Return is equivalent to
    ak.GroupBy(ak.concatenate([idx1, idx2])).sum(ak.concatenate((val1, val2)))

    Parameters
    -----------
    idx1: pdarray
        indices for the first sparse matrix
    idx2: pdarray
        indices for the second sparse matrix
    val1: pdarray
        values for the first sparse matrix
    val2: pdarray
        values for the second sparse matrix
    merge: bool
        If true the indices are combined using a merge based workflow,
        otherwise they are combine using a sort based workflow.
    percent_transfer_limit: int
        Only used when merge is true. This is the maximum percentage of the data allowed
        to be moved between locales during the merge workflow. If we would exceed this percentage,
        we fall back to using the sort based workflow.

    Returns
    --------
    (pdarray, pdarray)
        indices and values for the summed sparse matrix

    Examples
    --------
    >>> idx1 = ak.array([0, 1, 3, 4, 7, 9])
    >>> idx2 = ak.array([0, 1, 3, 6, 9])
    >>> vals1 = idx1
    >>> vals2 = ak.array([10, 11, 13, 16, 19])
    >>> ak.util.sparse_sum_help(idx1, inds2, vals1, vals2)
    (array([0 1 3 4 6 7 9]), array([10 12 16 4 16 7 28]))

    >>> ak.GroupBy(ak.concatenate([idx1, idx2])).sum(ak.concatenate((vals1, vals2)))
    (array([0 1 3 4 6 7 9]), array([10 12 16 4 16 7 28]))
    """
    repMsg = generic_msg(
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
    inds, vals = repMsg.split("+", maxsplit=1)
    return create_pdarray(inds), create_pdarray(vals)


def broadcast_dims(sa: Sequence[int], sb: Sequence[int]) -> Tuple[int, ...]:
    """
    Algorithm to determine shape of broadcasted PD array given two array shapes

    see: https://data-apis.org/array-api/latest/API_specification/broadcasting.html#algorithm
    """

    Na = len(sa)
    Nb = len(sb)
    N = max(Na, Nb)
    shapeOut = [0 for i in range(N)]

    i = N - 1
    while i >= 0:
        n1 = Na - N + i
        n2 = Nb - N + i

        d1 = sa[n1] if n1 >= 0 else 1
        d2 = sb[n2] if n2 >= 0 else 1

        if d1 == 1:
            shapeOut[i] = d2
        elif d2 == 1:
            shapeOut[i] = d1
        elif d1 == d2:
            shapeOut[i] = d1
        else:
            raise ValueError("Incompatible dimensions for broadcasting")

        i -= 1

    return tuple(shapeOut)


def convert_bytes(nbytes, unit="B"):
    """
    Convert the number of bytes to KB, MB, or GB.

    Parameters
    ----------
    unit : str, default = "B"
        Unit to return. One of {'B', 'KB', 'MB', 'GB'}.

    Returns
    -------
    int

    """
    kb = 1024
    mb = kb * kb
    gb = mb * kb
    if unit == "B":
        return nbytes
    elif unit == "KB":
        return nbytes / kb
    elif unit == "MB":
        return nbytes / mb
    elif unit == "GB":
        return nbytes / gb


def is_numeric(
    arry: Union[pdarray, Strings, Categorical, "Series", "Index"]  # noqa: F821
) -> builtins.bool:
    """
    Check if the dtype of the given array is numeric.

    Parameters:
         arry ((pdarray, Strings, Categorical)):
            The input pdarray, Strings, or Categorical object.

    Returns
    -------
    bool:
        True if the dtype of pda is numeric, False otherwise.

    Example:
        >>> import arkouda as ak
        >>> ak.connect()
        >>> data = ak.array([1, 2, 3, 4, 5])
        >>> is_numeric(data)
        True

        >>> strings = ak.array(["a", "b", "c"])
        >>> is_numeric(strings)
        False

    """
    from arkouda.index import Index
    from arkouda.series import Series

    if isinstance(arry, (pdarray, Series, Index)):
        return _is_dtype_in_union(dtype(arry.dtype), numeric_scalars)
    else:
        return False


def is_float(arry: Union[pdarray, Strings, Categorical, "Series", "Index"]):  # noqa: F821
    """
    Check if the dtype of the given array is float.

    Parameters:
         arry ((pdarray, Strings, Categorical)):
            The input pdarray, Strings, or Categorical object.

    Returns
    -------
    bool:
        True if the dtype of pda is of type float, False otherwise.

    Example:
        >>> import arkouda as ak
        >>> ak.connect()
        >>> data = ak.array([1.0, 2, 3, 4, np.nan])
        >>> is_float(data)
        True

        >>> data2 = ak.arange(5)
        >>> is_float(data2)
        False

    """
    from arkouda.index import Index
    from arkouda.series import Series

    if isinstance(arry, (pdarray, Series, Index)):
        return _is_dtype_in_union(dtype(arry.dtype), float_scalars)
    else:
        return False


def is_int(arry: Union[pdarray, Strings, Categorical, "Series", "Index"]):  # noqa: F821
    """
    Check if the dtype of the given array is int.

    Parameters
    ----------
    arry ((pdarray, Strings, Categorical)):
            The input pdarray, Strings, or Categorical object.

    Returns
    -------
    bool:
        True if the dtype of pda is of type int, False otherwise.

    Example:
    >>> import arkouda as ak
    >>> ak.connect()
    >>> data = ak.array([1.0, 2, 3, 4, np.nan])
    >>> is_int(data)
    False

    >>> data2 = ak.arange(5)
    >>> is_int(data2)
    True

    """
    from arkouda.index import Index
    from arkouda.series import Series

    if isinstance(arry, (pdarray, Series, Index)):
        return _is_dtype_in_union(dtype(arry.dtype), int_scalars)
    else:
        return False


def map(
    values: Union[pdarray, Strings, Categorical], mapping: Union[dict, "Series"]
) -> Union[pdarray, Strings]:
    """
    Map values of an array according to an input mapping.

    Parameters
    ----------
    values :  pdarray, Strings, or Categorical
        The values to be mapped.
    mapping : dict or arkouda.Series
        The mapping correspondence.

    Returns
    -------
    arkouda.pdarrayclass.pdarray or arkouda.strings.Strings
        A new array with the values mapped by the mapping correspondence.
        When the input Series has Categorical values,
        the return Series will have Strings values.
        Otherwise, the return type will match the input type.
    Raises
    ------
    TypeError
        Raised if arg is not of type dict or arkouda.Series.
        Raised if values not of type pdarray, Categorical, or Strings.
    Examples
    --------
    >>> import arkouda as ak
    >>> ak.connect()
    >>> from arkouda.util import map
    >>> a = ak.array([2, 3, 2, 3, 4])
    >>> a
    array([2 3 2 3 4])
    >>> map(a, {4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
    array([30.00000000000000000 5.00000000000000000 30.00000000000000000
    5.00000000000000000 25.00000000000000000])
    >>> s = ak.Series(ak.array(["a","b","c","d"]), index = ak.array([4,2,1,3]))
    >>> map(a, s)
    array(['b', 'b', 'd', 'd', 'a'])

    """
    import numpy as np

    from arkouda import Series, array, broadcast, full
    from arkouda.pdarraysetops import in1d

    keys = values
    gb = GroupBy(keys, dropna=False)
    gb_keys = gb.unique_keys

    if isinstance(mapping, dict):
        mapping = Series([array(list(mapping.keys())), array(list(mapping.values()))])

    if isinstance(mapping, Series):
        xtra_keys = gb_keys[in1d(gb_keys, mapping.index.values, invert=True)]

        if xtra_keys.size > 0:
            if not isinstance(mapping.values, (Strings, Categorical)):
                nans = full(xtra_keys.size, np.nan, mapping.values.dtype)
            else:
                nans = full(xtra_keys.size, "null")

            if isinstance(xtra_keys, Categorical):
                xtra_keys = xtra_keys.to_strings()

            xtra_series = Series(nans, index=xtra_keys)
            mapping = Series.concat([mapping, xtra_series])

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
