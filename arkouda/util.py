from typing import cast, Tuple, Sequence
from warnings import warn
import json

from typeguard import typechecked

from arkouda.categorical import Categorical
from arkouda.client import generic_msg, get_config, get_mem_used
from arkouda.client_dtypes import BitVector, BitVectorizer, IPv4
from arkouda.groupbyclass import GroupBy, broadcast
from arkouda.infoclass import list_registry
from arkouda.pdarrayclass import create_pdarray
from arkouda.pdarraycreation import arange
from arkouda.pdarraysetops import unique
from arkouda.segarray import SegArray
from arkouda.sorting import coargsort
from arkouda.strings import Strings
from arkouda.timeclass import Datetime, Timedelta


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
            pergroupval = g.count()[1]
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
    from arkouda.pdarrayclass import pdarray
    from arkouda.index import Index, MultiIndex
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
def unregister(name: str):
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
