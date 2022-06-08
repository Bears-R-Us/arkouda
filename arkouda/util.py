import os
import re
from typing import Mapping, Union, cast
from warnings import warn

import h5py  # type: ignore
import numpy as np  # type: ignore

from arkouda.categorical import Categorical
from arkouda.client import generic_msg, get_config, get_mem_used
from arkouda.client_dtypes import BitVector, BitVectorizer, IPv4
from arkouda.groupbyclass import GroupBy, broadcast
from arkouda.infoclass import list_symbol_table
from arkouda.pdarrayclass import RegistrationError, create_pdarray, pdarray
from arkouda.pdarraycreation import arange
from arkouda.pdarrayIO import read
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
    elif type(x) == BitVector:
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

    types = set([type(x) for x in items])
    if len(types) != 1:
        raise TypeError(f"Items must all have same type: {types}")
    t = types.pop()
    return (
        t.concat(items, ordered=ordered)
        if hasattr(t, "concat")
        else pdarrayconcatenate(items, ordered=ordered)
    )


def report_mem(pre=""):
    cfg = get_config()
    used = get_mem_used() / (cfg["numLocales"] * cfg["physicalMemory"])
    print(f"{pre} mem use: {get_mem_used()/(1024**4): .2f} TB ({used:.1%})")


def register(a, name):
    """
    Register an arkouda object with a user-specified name. Backwards compatible
    with earlier arkouda versions.
    """
    cb = identity
    if type(a) in {Datetime, Timedelta, BitVector, IPv4}:
        # These classes wrap a pdarray, so two names must be updated
        a = a.register(name)
        # Get the registered name
        n = a.name
        # Re-convert object if necessary
        reg = cb(a)
        # Assign registered name to wrapper object
        reg.name = n
        # Assign same name to underlying pdarray
        reg.values.name = n
    else:
        a = a.register(name)
        reg = cb(a)
    return reg


def register_all(data, prefix, overwrite=True):
    from arkouda.dataframe import DataFrame

    def sanitize(k):
        return str(k).replace(" ", "_")

    if overwrite:
        att = attach_all(prefix)
        for k in data:
            ksan = sanitize(k)
            if ksan in att:
                att[ksan].unregister()
    if isinstance(data, dict):
        return {k: register(v, f"{prefix}{sanitize(k)}") for k, v in data.items()}
    elif isinstance(data, DataFrame):
        return DataFrame({k: register(v, f"{prefix}{sanitize(k)}") for k, v in data.items()})
    elif isinstance(data, list):
        return [register(v, f"{prefix}{i}") for i, v in enumerate(data)]
    elif isinstance(data, tuple):
        return tuple([register(v, f"{prefix}{i}") for i, v in enumerate(data)])
    else:
        try:
            return data.register(prefix)
        except Exception:
            raise RegistrationError(f"Failed to register object of type '{type(data)}'")


def attach_all(prefix):
    pat = re.compile(f"(?:\\d+_)?{prefix}[\\w.]+")
    res = pat.findall(str(list_symbol_table()))
    return {k[len(prefix) :]: attach(k) for k in res}


def unregister_all(prefix):
    att = attach_all(prefix)
    for v in att.values():
        v.unregister()


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


def arkouda_to_numpy(A: pdarray, tmp_dir: str = "") -> np.ndarray:
    """
    Convert from arkouda to numpy using disk rather than sockets.
    """
    warn(
        "This function is deprecated and will be removed in a later version of Arkouda."
        " Use arkouda.pdarray.to_ndarray instead.",
        DeprecationWarning,
    )

    rng = np.random.randint(2**64, dtype=np.uint64)
    tmp_dir = os.getcwd() if not tmp_dir else tmp_dir
    A.save(f"{tmp_dir}/{rng}")
    files = sorted(f"{tmp_dir}/{f}" for f in os.listdir(tmp_dir) if f.startswith(str(rng)))

    B = np.zeros(A.size, dtype=np.int64)
    i = 0
    for file in files:
        with h5py.File(file) as hf:
            a = hf["array"]
            B[i : i + a.size] = a[:]
            i += a.size
        os.remove(file)

    return B


def numpy_to_arkouda(
    A: np.ndarray, tmp_dir: str = ""
) -> Union[pdarray, Strings, Mapping[str, Union[pdarray, Strings]]]:
    """
    Convert from numpy to arkouda using disk rather than sockets.
    """
    warn(
        "This function is deprecated and will be removed in a later version of Arkouda."
        " Use arkouda.array(x) instead.",
        DeprecationWarning,
    )

    rng = np.random.randint(2**64, dtype=np.uint64)
    tmp_dir = os.getcwd() if not tmp_dir else tmp_dir
    with h5py.File(f"{tmp_dir}/{rng}.hdf5", "w") as f:
        arr = f.create_dataset("arr", (A.shape[0],), dtype="int64")
        arr[:] = A[:]

    B = read(f"{tmp_dir}/{rng}.hdf5", "arr")
    os.remove(f"{tmp_dir}/{rng}.hdf5")

    return B


def convert_if_categorical(values):
    """
    Convert a Categorical array to Strings for display
    """

    if isinstance(values, Categorical):
        values = values.categories[values.codes]
    return values


def attach(name: str, dtype: str = "infer"):
    """
    Attaches to a known element name. If a type is passed, the server will use that type
    to pull the corresponding parts, otherwise the server will try to infer the type
    """
    repMsg = cast(str, generic_msg(cmd="genericAttach", args=f"{dtype}+{name}"))

    if repMsg.split("+")[0] == "categorical":
        return Categorical.from_return_msg(repMsg)
    elif repMsg.split("+")[0] == "segarray":
        return SegArray.from_return_msg(repMsg)
    elif repMsg.split("+")[0] == "series":
        from arkouda.series import Series

        return Series.from_return_msg(repMsg)
    else:
        dtype = repMsg.split()[2]

        if dtype == "str":
            return Strings.from_return_msg(repMsg)
        else:
            return create_pdarray(repMsg)
