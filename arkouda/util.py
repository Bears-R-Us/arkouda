import pandas as pd  # type: ignore
import re
import numpy as np  # type: ignore
import h5py #type: ignore
import os

from arkouda import __version__
from arkouda.client_dtypes import BitVector, BitVectorizer, IPv4
from arkouda.timeclass import Datetime, Timedelta
from arkouda.pdarrayclass import attach_pdarray, pdarray
from arkouda.pdarraysetops import concatenate as pdarrayconcatenate
from arkouda.pdarraycreation import arange
from arkouda.pdarraysetops import unique
from arkouda.pdarrayIO import read_hdf
from arkouda.client import get_config, get_mem_used
from arkouda.groupbyclass import GroupBy, broadcast, coargsort
from arkouda.infoclass import information, AllSymbols
from arkouda.categorical import Categorical

identity = lambda x: x


def get_callback(x):
    if type(x) in {Datetime, Timedelta, IPv4}:
        return type(x)
    elif hasattr(x, '_cast'):
        return x._cast
    elif type(x) == BitVector:
        return BitVectorizer(width=x.width, reverse=x.reverse)
    else:
        return identity


# TODO - moving this into arkouda, function name should probably be changed.
def concatenate(items, ordered=True):
    if len(items) > 0:
        types = set([type(x) for x in items])
        if len(types) != 1:
            raise TypeError("Items must all have same type: {}".format(types))
        t = types.pop()
        if t == BitVector:
            widths = set([x.width for x in items])
            revs = set([x.reverse for x in items])
            if len(widths) != 1 or len(revs) != 1:
                raise TypeError("BitVectors must all have same width and direction")
        callback = get_callback(list(items)[0])
        if hasattr(t, 'concat'):
            concat = t.concat
        else:
            concat = pdarrayconcatenate
    else:
        callback = identity
        concat = pdarrayconcatenate
    return callback(concat(items, ordered=ordered))


def report_mem(pre=''):
    cfg = get_config()
    used = get_mem_used() / (cfg['numLocales']*cfg['physicalMemory'])
    print(f"{pre} mem use: {get_mem_used()/(1024**4): .2f} TB ({used:.1%})")


def register(a, name):
    """
    Register an arkouda object with a user-specified name. Backwards compatible
    with earlier arkouda versions.
    """
    if pd.to_datetime(__version__.lstrip('v')) >= pd.to_datetime('2021.04.14'):
        # New versions register in-place and don't need callback
        cb = identity
    else:
        # Older versions return a new object that loses higher-level dtypes and must be re-converted
        cb = get_callback(a)
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
        return str(k).replace(' ', '_')
    if overwrite:
        att = attach_all(prefix)
        for k in data:
            ksan = sanitize(k)
            if ksan in att:
                att[ksan].unregister()
    if isinstance(data, dict):
        return {k:register(v, f'{prefix}{sanitize(k)}') for k, v in data.items()}
    elif isinstance(data, DataFrame):
        return DataFrame({k:register(v, f'{prefix}{sanitize(k)}') for k, v in data.items()})
    elif isinstance(data, list):
        return [register(v, f'{prefix}{i}') for i, v in enumerate(data)]
    elif isinstance(data, tuple):
        return tuple([register(v, f'{prefix}{i}') for i, v in enumerate(data)])
    elif isinstance(data, GroupBy):
        data.permutation = register(data.permutation, f'{prefix}permutation')
        data.segments = register(data.segments, f'{prefix}segments')
        data.unique_keys = register_all(data.unique_keys, f'{prefix}unique_keys_')
        return data
    else:
        raise TypeError(f"Cannot register objects of type {type(data)}")


def attach_all(prefix):
    pat = re.compile(prefix+'\\w+')
    res = pat.findall(information(AllSymbols))
    return {k[len(prefix):]:attach_pdarray(k) for k in res}


def unregister_all(prefix):
    att = attach_all(prefix)
    for v in att.values():
        v.unregister()


def enrich_inplace(data, keynames, aggregations, **kwargs):
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
        if reduction == 'count':
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
    #TODO - look into the comment below
    # I think this suffers from overflow errors on large arrays.
    #if perm.sum() != (perm.size * (perm.size -1)) / 2:
    #    raise ValueError("The indicated permutation is invalid.")
    if unique(perm).size != perm.size:
        raise ValueError("The array is not a permutation.")
    return coargsort([perm, arange(0, perm.size)])


def most_common(g, values):
    """
    Find the most common value for each key in a GroupBy object.

    Parameters
    ----------
    g : ak.GroupBy
        Grouping of keys
    values : array-like
        Values in which to find most common

    Returns
    -------
    unique_keys : (list of) arrays
        Unique key of each group
    most_common_values : array-like
        The most common value for each key
    """
    # Give each key an integer index
    keyidx = g.broadcast(arange(g.unique_keys[0].size), permute=True)
    # Annex values and group by (key, val)
    bykeyval = GroupBy([keyidx, values])
    # Count number of records for each (key, val)
    (ki, uval), count = bykeyval.count()
    # Group out value
    bykey = GroupBy(ki, assume_sorted=True)
    # Find the index of the most frequent value for each key
    _, topidx = bykey.argmax(count)
    # Gather the most frequent values
    return uval[topidx]


def arkouda_to_numpy(A: pdarray, tmp_dir: str='') -> np.ndarray:
    """
    Convert from arkouda to numpy using disk rather than sockets.
    """
    rng = np.random.randint(2**64, dtype=np.uint64)
    tmp_dir = os.getcwd() if not tmp_dir else tmp_dir
    A.save(f"{tmp_dir}/{rng}")
    files = sorted(
        f"{tmp_dir}/{f}"
        for f in os.listdir(tmp_dir)
        if f.startswith(str(rng))
    )

    B = np.zeros(A.size, dtype=np.int64)
    i = 0
    for file in files:
        with h5py.File(file) as hf:
            a = hf['array']
            B[i:i+a.size] = a[:]
            i += a.size
        os.remove(file)

    return B


def numpy_to_arkouda(A: np.ndarray, tmp_dir: str = '') -> pdarray:
    """
    Convert from numpy to arkouda using disk rather than sockets.
    """
    rng = np.random.randint(2 ** 64, dtype=np.uint64)
    tmp_dir = os.getcwd() if not tmp_dir else tmp_dir
    with h5py.File(f'{tmp_dir}/{rng}.hdf5', 'w') as f:
        arr = f.create_dataset('arr', (A.shape[0],), dtype='int64')
        arr[:] = A[:]

    B = read_hdf('arr', f'{tmp_dir}/{rng}.hdf5')
    os.remove(f'{tmp_dir}/{rng}.hdf5')

    return B


def convert_if_categorical(values):
    """
    Convert a cetegorical array to strings for display
    """

    if isinstance(values, Categorical):
        values = values.categories[values.codes]
    return values
