import itertools
from typing import Any, Iterable, List, Optional, Tuple, Union, cast, overload

import numpy as np
import pandas as pd
from typeguard import typechecked

from arkouda.client import generic_msg, get_array_ranks
from arkouda.numpy.dtypes import (
    NUMBER_FORMAT_STRINGS,
    DTypes,
    NumericDTypes,
    SeriesDTypes,
    bigint,
)
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import float64, get_byteorder, get_server_byteorder
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import (
    int_scalars,
    isSupportedInt,
    isSupportedNumber,
    numeric_scalars,
    resolve_scalar_dtype,
    str_,
)
from arkouda.numpy.dtypes import uint64 as akuint64
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.strings import Strings

__all__ = [
    "array",
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "full_like",
    "arange",
    "linspace",
    "randint",
    "uniform",
    "standard_normal",
    "random_strings_uniform",
    "random_strings_lognormal",
    "from_series",
    "bigint_from_uint_arrays",
    "promote_to_common_dtype",
    "scalar_array",
]


@typechecked
def from_series(series: pd.Series, dtype: Optional[Union[type, str]] = None) -> Union[pdarray, Strings]:
    """
    Converts a Pandas Series to an Arkouda pdarray or Strings object. If
    dtype is None, the dtype is inferred from the Pandas Series. Otherwise,
    the dtype parameter is set if the dtype of the Pandas Series is to be
    overridden or is  unknown (for example, in situations where the Series
    dtype is object).

    Parameters
    ----------
    series : Pandas Series
        The Pandas Series with a dtype of bool, float64, int64, or string
    dtype : Optional[type]
        The valid dtype types are np.bool, np.float64, np.int64, and np.str

    Returns
    -------
    Union[pdarray,Strings]

    Raises
    ------
    TypeError
        Raised if series is not a Pandas Series object
    ValueError
        Raised if the Series dtype is not bool, float64, int64, string, datetime, or timedelta

    Examples
    --------
    >>> import arkouda as ak
    >>> np.random.seed(1701)
    >>> ak.from_series(pd.Series(np.random.randint(0,10,5)))
    array([4 3 3 5 0])

    >>> ak.from_series(pd.Series(['1', '2', '3', '4', '5']),dtype=np.int64)
    array([1 2 3 4 5])

    >>> np.random.seed(1701)
    >>> ak.from_series(pd.Series(np.random.uniform(low=0.0,high=1.0,size=3)))
    array([0.089433234324597599 0.1153776854774361 0.51874393620990389])

    >>> ak.from_series(
    ...     pd.Series([
    ...         '0.57600036956445599',
    ...         '0.41619265571741659',
    ...         '0.6615356693784662',
    ...     ]),
    ...     dtype=np.float64,
    ... )
    array([0.57600036956445599 0.41619265571741659 0.6615356693784662])

    >>> np.random.seed(1864)
    >>> ak.from_series(pd.Series(np.random.choice([True, False],size=5)))
    array([True True True False False])

    >>> ak.from_series(pd.Series(['True', 'False', 'False', 'True', 'True']), dtype=bool)
    array([True True True True True])

    >>> ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e'], dtype="string"))
    array(['a', 'b', 'c', 'd', 'e'])

    >>> ak.from_series(pd.Series(pd.to_datetime(['1/1/2018', np.datetime64('2018-01-01')])))
    array([1514764800000000000 1514764800000000000])

    Notes
    -----
    The supported datatypes are bool, float64, int64, string, and datetime64[ns]. The
    data type is either inferred from the the Series or is set via the dtype parameter.

    Series of datetime or timedelta are converted to Arkouda arrays of dtype int64 (nanoseconds)

    A Pandas Series containing strings has a dtype of object. Arkouda assumes the Series
    contains strings and sets the dtype to str
    """
    if not dtype:
        dt = series.dtype.name
    else:
        dt = str(dtype)
    try:
        """
        If the Series has a object dtype, set dtype to string to comply with method
        signature that does not require a dtype; this is required because Pandas can infer
        non-str dtypes from the input np or Python array.
        """
        if dt == "object":
            dt = "string"

        n_array = series.to_numpy(dtype=SeriesDTypes[dt])  # type: ignore
    except KeyError:
        raise ValueError(
            f"dtype {dt} is unsupported. Supported dtypes are bool, float64, int64, string, "
            f"datetime64[ns], and timedelta64[ns]"
        )
    return array(n_array)


def array(
    a: Union[pdarray, np.ndarray, Iterable],
    dtype: Union[np.dtype, type, str, None] = None,
    max_bits: int = -1,
) -> Union[pdarray, Strings]:
    """
    Convert a Python or Numpy Iterable to a pdarray or Strings object, sending
    the corresponding data to the arkouda server.

    Parameters
    ----------
    a: Union[pdarray, np.ndarray]
        array of a supported dtype
    dtype: np.dtype, type, or str
        The target dtype to cast values to
    max_bits: int
        Specifies the maximum number of bits; only used for bigint pdarrays

    Returns
    -------
    pdarray or Strings
        A pdarray instance stored on arkouda server or Strings instance, which
        is composed of two pdarrays stored on arkouda server

    Raises
    ------
    TypeError
        Raised if a is not a pdarray, np.ndarray, or Python Iterable such as a
        list, array, tuple, or deque
    RuntimeError
        Raised if nbytes > maxTransferBytes, a.dtype is not supported (not in DTypes),
        or if the product of a size and a.itemsize > maxTransferBytes
    ValueError
        Raised if a has rank is not in get_array_ranks(), or if the returned message is malformed or does
        not contain the fields required to generate the array.

    See Also
    --------
    pdarray.to_ndarray

    Notes
    -----
    The number of bytes in the input array cannot exceed `ak.client.maxTransferBytes`,
    otherwise a RuntimeError will be raised. This is to protect the user
    from overwhelming the connection between the Python client and the arkouda
    server, under the assumption that it is a low-bandwidth connection. The user
    may override this limit by setting ak.client.maxTransferBytes to a larger value,
    but should proceed with caution.

    If the pdrray or ndarray is of type U, this method is called twice recursively
    to create the Strings object and the two corresponding pdarrays for string
    bytes and offsets, respectively.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.array(np.arange(1,10))
    array([1 2 3 4 5 6 7 8 9])

    >>> ak.array(range(1,10))
    array([1 2 3 4 5 6 7 8 9])

    >>> strings = ak.array([f'string {i}' for i in range(0,5)])
    >>> type(strings)
    <class 'arkouda.numpy.strings.Strings'>
    """
    from arkouda.numpy import cast as akcast

    # If a is already a pdarray, do nothing
    if isinstance(a, pdarray):
        casted = a if dtype is None else akcast(a, dtype)
        if dtype == bigint and max_bits != -1:
            casted.max_bits = max_bits
        return casted
    from arkouda.client import maxTransferBytes

    # If a is not already a numpy.ndarray, convert it
    if not isinstance(a, np.ndarray):
        try:
            if dtype is not None and dtype != bigint:
                # if the user specified dtype, use that dtype
                a = np.array(a, dtype=dtype)
            elif all(isSupportedInt(i) for i in a) and any(2**64 > i > 2**63 for i in a):
                # all supportedInt values but some >2**63, default to uint (see #1297)
                # iterating shouldn't be too expensive since
                # this is after the `isinstance(a, pdarray)` check
                a = np.array(a, dtype=np.uint)
            else:
                # let numpy decide the type
                a = np.array(a)
        except (RuntimeError, TypeError, ValueError):
            raise TypeError("a must be a pdarray, np.ndarray, or convertible to a numpy array")

    #   Special case, to get around error when putting negative numbers in a bigint

    if dtype == bigint:
        if a.dtype == "int64" and (a < 0).any():
            return akcast(array(a), bigint)

    if a.dtype == bigint or a.dtype.name not in DTypes or dtype == bigint:
        # We need this array whether the number of dimensions is 1 or greater.
        uint_arrays: List[Union[pdarray, Strings]] = []

    if (a.dtype == bigint or dtype == bigint) and a.ndim in get_array_ranks() and a.ndim > 1:
        sh = a.shape
        try:
            # attempt to break bigint into multiple uint64 arrays
            # early out if we would have more uint arrays than can fit in max_bits
            early_out = (max_bits // 64) + (max_bits % 64 != 0) if max_bits != -1 else float("inf")
            while (a != 0).any() and len(uint_arrays) < early_out:
                low, a = a % 2**64, a // 2**64
                uint_arrays.append(array(np.array(low, dtype=np.uint), dtype=akuint64))
            # If uint_arrays is empty, this will create an empty ak array and reshape it.
            if not uint_arrays:
                return zeros(size=sh, dtype=bigint, max_bits=max_bits)
            else:
                return bigint_from_uint_arrays(uint_arrays[::-1], max_bits=max_bits)
        except TypeError:
            raise RuntimeError(f"Unhandled dtype {a.dtype}")

    if a.ndim != 1 and a.dtype.name not in NumericDTypes:
        raise TypeError("Must be an iterable or have a numeric DType")

    # Return multi-dimensional pdarray if a.ndim in get_array_ranks()
    # otherwise raise an error

    if a.ndim not in get_array_ranks():
        raise ValueError(f"array rank {a.ndim} not in compiled ranks {get_array_ranks()}")

    # Check if array of strings
    # if a.dtype == numpy.object_ need to check first element
    if "U" in a.dtype.kind or (a.dtype == np.object_ and a.size > 0 and isinstance(a[0], str)):
        # encode each string and add a null byte terminator
        encoded = [i for i in itertools.chain.from_iterable(map(lambda x: x.encode() + b"\x00", a))]
        nbytes = len(encoded)
        if nbytes > maxTransferBytes:
            raise RuntimeError(
                f"Creating pdarray would require transferring {nbytes} bytes, which exceeds "
                f"allowed transfer size. Increase ak.client.maxTransferBytes to force."
            )
        encoded_np = np.array(encoded, dtype=np.uint8)
        rep_msg = generic_msg(
            cmd=f"arraySegString<{encoded_np.dtype.name}>",
            args={"size": encoded_np.size},
            payload=_array_memview(encoded_np),
            send_binary=True,
        )
        parts = cast(str, rep_msg).split("+", maxsplit=3)
        return (
            Strings.from_parts(parts[0], parts[1])
            if dtype is None
            else akcast(Strings.from_parts(parts[0], parts[1]), dtype)
        )

    # If not strings, then check that dtype is supported in arkouda
    if dtype == bigint or a.dtype.name not in DTypes:
        # 2 situations result in attempting to call `bigint_from_uint_arrays`
        # 1. user specified i.e. dtype=ak.bigint
        # 2. too big to fit into other numpy types (dtype = object)
        try:
            # attempt to break bigint into multiple uint64 arrays
            # early out if we would have more uint arrays than can fit in max_bits
            early_out = (max_bits // 64) + (max_bits % 64 != 0) if max_bits != -1 else float("inf")
            while any(a != 0) and len(uint_arrays) < early_out:
                if isinstance(a, np.ndarray):
                    low, a = a.astype("O") % 2**64, a.astype("O") // 2**64
                else:
                    low, a = a % 2**64, a // 2**64
                uint_arrays.append(array(np.array(low, dtype=np.uint), dtype=akuint64))
            return bigint_from_uint_arrays(uint_arrays[::-1], max_bits=max_bits)
        except TypeError:
            raise RuntimeError(f"Unhandled dtype {a.dtype}")
    else:
        from arkouda.numpy.util import _infer_shape_from_size

        shape, ndim, full_size = _infer_shape_from_size(a.shape)

        # Do not allow arrays that are too large
        if (full_size * a.itemsize) > maxTransferBytes:
            raise RuntimeError(
                "Array exceeds allowed transfer size. Increase ak.client.maxTransferBytes to allow"
            )
        if a.ndim > 1 and a.flags["F_CONTIGUOUS"] and not a.flags["OWNDATA"]:
            # Make a copy if the array was shallow-transposed (to avoid error #3757)
            a_ = a.copy()
        else:
            a_ = a
        # Pack binary array data into a bytes object with a command header
        # including the dtype and size. If the server has a different byteorder
        # than our numpy array we need to swap to match since the server expects
        # native endian bytes
        aview = _array_memview(a_)

        server_byte_order = get_server_byteorder()
        if (server_byte_order == "big" and a.dtype.byteorder == "<") or (
            server_byte_order == "little" and a.dtype.byteorder == ">"
        ):
            a = a.view(a.dtype.newbyteorder("S")).byteswap()

        rep_msg = generic_msg(
            cmd=f"array<{a_.dtype.name},{ndim}>",
            args={"dtype": a_.dtype.name, "shape": tuple(a_.shape), "seg_string": False},
            payload=aview,
            send_binary=True,
        )
        return create_pdarray(rep_msg) if dtype is None else akcast(create_pdarray(rep_msg), dtype)


@typechecked
def promote_to_common_dtype(arrays: List[pdarray]) -> Tuple[Any, List[pdarray]]:
    """
    Promote a list of pdarrays to a common dtype.

    Parameters
    ----------
    arrays : List[pdarray]
        List of pdarrays to promote

    Returns
    -------
    dtype, List[pdarray]
        The common dtype of the pdarrays and the list of pdarrays promoted to that dtype

    Raises
    ------
    TypeError
        Raised if any pdarray is a non-numeric type

    See Also
    --------
    pdarray.promote_dtype

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.arange(5)
    >>> b = ak.ones(5, dtype=ak.float64)
    >>> dtype, promoted = ak.promote_to_common_dtype([a, b])
    >>> dtype
    dtype('float64')
    >>> all(isinstance(p, ak.pdarray) and p.dtype == dtype for p in promoted)
    True
    """
    # find the common dtype of the input arrays
    dt = np.common_type(*[np.empty(0, dtype=a.dtype) for a in arrays])

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dt) if a.dtype != dt else a for a in arrays]

    return (akdtype(dt), arrays)


def _array_memview(a) -> memoryview:
    if (get_byteorder(a.dtype) == "<" and get_server_byteorder() == "big") or (
        get_byteorder(a.dtype) == ">" and get_server_byteorder() == "little"
    ):
        return memoryview(a.byteswap())
    else:
        return memoryview(a)


def bigint_from_uint_arrays(arrays, max_bits=-1):
    """
    Create a bigint pdarray from an iterable of uint pdarrays.
    The first item in arrays will be the highest 64 bits and
    the last item will be the lowest 64 bits.

    Parameters
    ----------
    arrays : Sequence[pdarray]
        An iterable of uint pdarrays used to construct the bigint pdarray.
        The first item in arrays will be the highest 64 bits and
        the last item will be the lowest 64 bits.
    max_bits : int
        Specifies the maximum number of bits; only used for bigint pdarrays

    Returns
    -------
    pdarray
        bigint pdarray constructed from uint arrays

    Raises
    ------
    TypeError
        Raised if any pdarray in arrays has a dtype other than uint or
        if the pdarrays are not the same size.
    RuntimeError
        Raised if there is a server-side error thrown

    See Also
    --------
    pdarray.bigint_to_uint_arrays

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.bigint_from_uint_arrays([ak.ones(5, dtype=ak.uint64), ak.arange(5, dtype=ak.uint64)])
    >>> a
    array([18446744073709551616 18446744073709551617 18446744073709551618
    18446744073709551619 18446744073709551620])

    >>> a.dtype
    dtype(bigint)

    >>> all(a[i] == 2**64 + i for i in range(5))
    True
    """
    if not arrays:
        return create_pdarray(
            generic_msg(
                cmd="big_int_creation<bigint,1>",
                args={
                    "arrays": arrays,
                    "num_arrays": len(arrays),
                    "shape": (0,),
                    "max_bits": max_bits,
                },
            )
        )
    if not all(isinstance(a, pdarray) and a.dtype == akuint64 for a in arrays):
        raise TypeError("Sequence must contain only uint pdarrays")
    if len({a.size for a in arrays}) != 1:
        raise TypeError("All pdarrays must be same size")
    if not isinstance(arrays, list):
        arrays = list(arrays)

    if max_bits != -1:
        # truncate if we have more uint arrays than can fit in max_bits
        max_num_arrays = (max_bits // 64) + (max_bits % 64 != 0)
        if len(arrays) > max_num_arrays:
            # only want max_num_arrays from the right (because those are the lowest bits)
            arrays = arrays[-max_num_arrays:]

    return create_pdarray(
        generic_msg(
            cmd=f"big_int_creation<bigint,{arrays[0].ndim}>",
            args={
                "arrays": arrays,
                "num_arrays": len(arrays),
                "shape": arrays[0].shape,
                "max_bits": max_bits,
            },
        )
    )


@typechecked
def zeros(
    size: Union[int_scalars, Tuple[int_scalars, ...], str],
    dtype: Union[np.dtype, type, str, bigint] = float64,
    max_bits: Optional[int] = None,
) -> pdarray:
    """
    Create a pdarray filled with zeros.

    Parameters
    ----------
    size : int_scalars or tuple of int_scalars
        Size or shape of the array
    dtype : all_scalars
        Type of resulting array, default ak.float64
    max_bits: int
        Specifies the maximum number of bits; only used for bigint pdarrays
        Included for consistency, as zeros are represented as all zeros, regardless
        of the value of max_bits

    Returns
    -------
    pdarray
        Zeros of the requested size or shape and dtype

    Raises
    ------
    TypeError
        Raised if the supplied dtype is not supported

    RuntimeError
        Raised if the size parameter is neither an int nor a str that is parseable to an int.

    ValueError
        Raised if the rank of the given shape is not in get_array_ranks() or is empty
        Raised if max_bits is not NONE and ndim does not equal 1

    See Also
    --------
    ones, zeros_like

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.zeros(5, dtype=ak.int64)
    array([0 0 0 0 0])

    >>> ak.zeros(5, dtype=ak.float64)
    array([0.00000000000000000 0.00000000000000000 0.00000000000000000
           0.00000000000000000 0.00000000000000000])

    >>> ak.zeros(5, dtype=ak.bool_)
    array([False False False False False])

    """
    dtype = akdtype(dtype)  # normalize dtype
    dtype_name = dtype.name if isinstance(dtype, bigint) else cast(np.dtype, dtype).name
    # check dtype for error
    if dtype_name not in NumericDTypes:
        raise TypeError(f"unsupported dtype {dtype}")

    from arkouda.numpy.util import (
        _infer_shape_from_size,  # placed here to avoid circ import
    )

    shape, ndim, full_size = _infer_shape_from_size(size)

    if ndim not in get_array_ranks():
        raise ValueError(f"array rank {ndim} is not in {get_array_ranks()}")

    if isinstance(shape, tuple) and len(shape) == 0:
        raise ValueError("size () not currently supported in ak.zeros.")

    repMsg = generic_msg(cmd=f"create<{dtype_name},{ndim}>", args={"shape": shape})

    return create_pdarray(repMsg, max_bits=max_bits)


@typechecked
def ones(
    size: Union[int_scalars, Tuple[int_scalars, ...], str],
    dtype: Union[np.dtype, type, str, bigint] = float64,
    max_bits: Optional[int] = None,
) -> pdarray:
    """
    Create a pdarray filled with ones.

    Parameters
    ----------
    size : int_scalars or tuple of int_scalars
        Size or shape of the array
    dtype : Union[float64, int64, bool]
        Resulting array type, default ak.float64
    max_bits: int
        Specifies the maximum number of bits; only used for bigint pdarrays
        Included for consistency, as ones are all zeros ending on a one, regardless
        of max_bits

    Returns
    -------
    pdarray
        Ones of the requested size or shape and dtype

    Raises
    ------
    TypeError
        Raised if the supplied dtype is not supported

    RuntimeError
        Raised if the size parameter is neither an int nor a str that is parseable to an int.

    ValueError
        Raised if the rank of the given shape is not in get_array_ranks() or is empty

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.ones(5, dtype=ak.int64)
    array([1 1 1 1 1])

    >>> ak.ones(5, dtype=ak.float64)
    array([1.00000000000000000 1.00000000000000000 1.00000000000000000
           1.00000000000000000 1.00000000000000000])

    >>> ak.ones(5, dtype=ak.bool_)
    array([True True True True True])

    Notes
    -----
    Logic for generating the pdarray is delegated to the ak.full method.
    """
    return full(size=size, fill_value=1, dtype=dtype, max_bits=max_bits)


@typechecked
def full(
    size: Union[int_scalars, Tuple[int_scalars, ...], str],
    fill_value: Union[numeric_scalars, str],
    dtype: Optional[Union[np.dtype, type, str, bigint]] = None,
    max_bits: Optional[int] = None,
) -> Union[pdarray, Strings]:
    """
    Create a pdarray filled with fill_value.

    Parameters
    ----------
    size: int_scalars or tuple of int_scalars
        Size or shape of the array
    fill_value: int_scalars or str
        Value with which the array will be filled
    dtype: all_scalars
        Resulting array type, default float64
    max_bits: int
        Specifies the maximum number of bits; only used for bigint pdarrays

    Returns
    -------
    pdarray or Strings
        array of the requested size and dtype filled with fill_value

    Raises
    ------
    TypeError
        Raised if the supplied dtype is not supported

    RuntimeError
        Raised if the size parameter is neither an int nor a str that is parseable to an int.

    ValueError
        Raised if the rank of the given shape is not in get_array_ranks() or is empty
        Raised if max_bits is not NONE and ndim does not equal 1

    See Also
    --------
    zeros, ones

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.full(5, 7, dtype=ak.int64)
    array([7 7 7 7 7])

    >>> ak.full(5, 9, dtype=ak.float64)
    array([9.00000000000000000 9.00000000000000000 9.00000000000000000
           9.00000000000000000 9.00000000000000000])

    >>> ak.full(5, 5, dtype=ak.bool_)
    array([True True True True True])
    """
    from arkouda.numpy.dtypes import dtype as ak_dtype

    if isinstance(fill_value, str):
        return _full_string(size, fill_value)
    elif ak_dtype(dtype) == str_ or dtype == Strings:
        return _full_string(size, str_(fill_value))

    dtype = dtype if dtype is not None else resolve_scalar_dtype(fill_value)

    dtype = akdtype(dtype)  # normalize dtype
    dtype_name = dtype.name if isinstance(dtype, bigint) else cast(np.dtype, dtype).name
    # check dtype for error
    if dtype_name not in NumericDTypes:
        raise TypeError(f"unsupported dtype {dtype}")
    from arkouda.numpy.util import (
        _infer_shape_from_size,  # placed here to avoid circ import
    )

    shape, ndim, full_size = _infer_shape_from_size(size)

    if ndim not in get_array_ranks():
        raise ValueError(f"array rank {ndim} not in compiled ranks {get_array_ranks()}")

    if isinstance(shape, tuple) and len(shape) == 0:
        raise ValueError("size () not currently supported in ak.full.")

    repMsg = generic_msg(cmd=f"create<{dtype_name},{ndim}>", args={"shape": shape})

    a = create_pdarray(repMsg)
    a.fill(fill_value)

    if max_bits:
        a.max_bits = max_bits
    return a


@typechecked
def scalar_array(
    value: numeric_scalars, dtype: Optional[Union[np.dtype, type, str, bigint]] = None
) -> pdarray:
    """
    Create a pdarray from a single scalar value.

    Parameters
    ----------
    value: numeric_scalars
        Value to create pdarray from

    Returns
    -------
    pdarray
        pdarray with a single element

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.scalar_array(5)
    array([5])

    >>> ak.scalar_array(7.0)
    array([7.00000000000000000])

    Raises
    ------
    RuntimeError
        Raised if value cannot be cast as dtype
    """

    if dtype is not None:
        _dtype = akdtype(dtype)
    else:
        _dtype = resolve_scalar_dtype(value)

    return create_pdarray(
        generic_msg(
            cmd=f"createScalarArray<{_dtype}>",
            args={"value": value},
        )
    )


@typechecked
def _full_string(
    size: Union[int_scalars, str],
    fill_value: str,
) -> Strings:
    """
    Create a Strings object filled with fill_value.

    Parameters
    ----------
    size: int_scalars
        Size of the array (only rank-1 arrays supported)
    fill_value: str
        Value with which the array will be filled

    Returns
    -------
    Strings
        array of the requested size and dtype filled with fill_value
    """
    repMsg = generic_msg(cmd="segmentedFull", args={"size": size, "fill_value": fill_value})
    return Strings.from_return_msg(cast(str, repMsg))


@typechecked
def zeros_like(pda: pdarray) -> pdarray:
    """
    Create a zero-filled pdarray of the same size and dtype as an existing
    pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.zeros(pda.size, pda.dtype)

    Raises
    ------
    TypeError
        Raised if the pda parameter is not a pdarray.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.zeros_like(ak.ones(5,dtype=ak.int64))
    array([0 0 0 0 0])

    >>> ak.zeros_like(ak.ones(5,dtype=ak.float64))
    array([0.00000000000000000 0.00000000000000000 0.00000000000000000
           0.00000000000000000 0.00000000000000000])

    >>> ak.zeros_like(ak.ones(5,dtype=ak.bool_))
    array([False False False False False])
    """
    return zeros(tuple(pda.shape), pda.dtype, pda.max_bits)


@typechecked
def ones_like(pda: pdarray) -> pdarray:
    """
    Create a one-filled pdarray of the same size and dtype as an existing
    pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.ones(pda.size, pda.dtype)

    Raises
    ------
    TypeError
        Raised if the pda parameter is not a pdarray.

    See Also
    --------
    ones, zeros_like

    Notes
    -----
    Logic for generating the pdarray is delegated to the ak.ones method.
    Accordingly, the supported dtypes match are defined by the ak.ones method.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.ones_like(ak.zeros(5,dtype=ak.int64))
    array([1 1 1 1 1])

    >>> ak.ones_like(ak.zeros(5,dtype=ak.float64))
    array([1.00000000000000000 1.00000000000000000 1.00000000000000000
           1.00000000000000000 1.00000000000000000])

    >>> ak.ones_like(ak.zeros(5,dtype=ak.bool_))
    array([True True True True True])
    """
    return ones(tuple(pda.shape), pda.dtype, pda.max_bits)


@typechecked
def full_like(pda: pdarray, fill_value: numeric_scalars) -> Union[pdarray, Strings]:
    """
    Create a pdarray filled with fill_value of the same size and dtype as an existing
    pdarray.

    Parameters
    ----------
    pda: pdarray
        Array to use for size and dtype
    fill_value: int_scalars
        Value with which the array will be filled

    Returns
    -------
    pdarray
        Equivalent to ak.full(pda.size, fill_value, pda.dtype)

    Raises
    ------
    TypeError
        Raised if the pda parameter is not a pdarray.

    See Also
    --------
    ones_like, zeros_like

    Notes
    -----
    Logic for generating the pdarray is delegated to the ak.full method.
    Accordingly, the supported dtypes match are defined by the ak.full method.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.full_like(ak.full(5,7,dtype=ak.int64),6)
    array([6 6 6 6 6])

    >>> ak.full_like(ak.full(7,9,dtype=ak.float64),10)
    array([10.00000000000000000 10.00000000000000000 10.00000000000000000
           10.00000000000000000 10.00000000000000000 10.00000000000000000 10.00000000000000000])

    >>> ak.full_like(ak.full(5,True,dtype=ak.bool_),False)
    array([False False False False False])
    """
    return full(tuple(pda.shape), fill_value, pda.dtype, pda.max_bits)


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def arange(
    __arg1: int_scalars,
    *,
    dtype: Optional[Union[np.dtype, type, bigint]] = None,
    max_bits: Optional[int] = None,
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def arange(
    __arg1: int_scalars,
    __arg2: int_scalars,
    *,
    dtype: Optional[Union[np.dtype, type, bigint]] = None,
    max_bits: Optional[int] = None,
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def arange(
    __arg1: int_scalars,
    __arg2: int_scalars,
    __arg3: int_scalars,
    *,
    dtype: Optional[Union[np.dtype, type, bigint]] = None,
    max_bits: Optional[int] = None,
) -> pdarray: ...


def arange(
    *args: int_scalars,
    dtype: Optional[Union[np.dtype, type, bigint]] = None,
    max_bits: Optional[int] = None,
) -> pdarray:
    """# noqa: DAR102
    arange([start,] stop[, step,] dtype=int64)

    Create a pdarray of consecutive integers within the interval [start, stop).
    If only one arg is given then arg is the stop parameter. If two args are
    given, then the first arg is start and second is stop. If three args are
    given, then the first arg is start, second is stop, third is step.

    The return value is cast to type dtype

    Parameters
    ----------
    start : int_scalars, optional
    stop  : int_scalars, optional
    step  : int_scalars, optional
        if one of these three is supplied, it's used as stop, and start = 0, step = 1
        if two of them are supplied, start = start, stop = stop, step = 1
        if all three are supplied, start = start, stop = stop, step = step
    dtype: np.dtype, type, or str
        The target dtype to cast values to
    max_bits: int
        Specifies the maximum number of bits; only used for bigint pdarrays

    Returns
    -------
    pdarray
        Integers from start (inclusive) to stop (exclusive) by step

    Raises
    ------
    ValueError
        Raised if none of start, stop, step was supplied
    TypeError
        Raised if start, stop, or step is not an int object
    ZeroDivisionError
        Raised if step == 0

    See Also
    --------
    linspace, zeros, ones, randint

    Notes
    -----
    Negative steps result in decreasing values. Currently, only int64
    pdarrays can be created with this method. For float64 arrays, use
    the linspace method.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.arange(0, 5, 1)
    array([0 1 2 3 4])

    >>> ak.arange(5, 0, -1)
    array([5 4 3 2 1])

    >>> ak.arange(0, 10, 2)
    array([0 2 4 6 8])

    >>> ak.arange(-5, -10, -1)
    array([-5 -6 -7 -8 -9])
    """

    from arkouda.numpy import cast as akcast

    start: int_scalars
    stop: int_scalars
    step: int_scalars

    if len(args) == 0:
        raise ValueError("A stopping value must be supplied to arange.")
    elif len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop, step = args[0], args[1], 1
    elif len(args) == 3:
        start, stop, step = args[0], args[1], args[2]
    else:
        raise RuntimeError("arange does not accept more than arguments.")

    if step == 0:
        raise ZeroDivisionError("division by zero")

    aktype = akdtype(akint64 if dtype is None else dtype)

    # check the conditions that cause numpy to return an empty array, and
    # return one also.  This includes a fix needed for empty bigint arrays.

    # The fix: ak.array calls ak.cast to handle the dtype parameter.  This
    # caused an error on empty arrays of bigint type (the empty array defaulted
    # to float64, and cast fails when  converting empty float64 arrays to bigint).
    # So empty arrays are initially created as akint64, and then cast to
    # the requested dtype.
    # This matters for several tests in tests/series_test.py

    if (start == stop) | ((np.sign(stop - start) * np.sign(step)) <= 0):
        return akcast(array([], dtype=akint64), dt=aktype)

    if isSupportedInt(start) and isSupportedInt(stop) and isSupportedInt(step):
        arg_dtypes = [resolve_scalar_dtype(arg) for arg in (start, stop, step)]
        akmax_bits = -1 if max_bits is None else max_bits
        arg_dtype = "int64"
        if dtype in ["bigint", bigint] or "bigint" in arg_dtypes or akmax_bits != -1:
            arg_dtype = "bigint"
        elif "uint64" in arg_dtypes:
            arg_dtype = "uint64"

        if step < 0:
            stop = stop + 2

        repMsg = generic_msg(
            cmd=f"arange<{arg_dtype},1>", args={"start": start, "stop": stop, "step": step}
        )
        arr = create_pdarray(repMsg, max_bits=max_bits)
        return arr if aktype == akint64 else akcast(arr, dt=aktype)

    raise TypeError(f"start, stop, step must be ints; got {args!r}")


@typechecked
def linspace(start: numeric_scalars, stop: numeric_scalars, length: int_scalars) -> pdarray:
    """
    Create a pdarray of linearly-spaced floats in a closed interval.

    Parameters
    ----------
    start : numeric_scalars
        Start of interval (inclusive)
    stop : numeric_scalars
        End of interval (inclusive)
    length : int_scalars
        Number of points

    Returns
    -------
    pdarray
        Array of evenly spaced float values along the interval

    Raises
    ------
    TypeError
        Raised if start or stop is not a float or int or if length is not an int

    See Also
    --------
    arange

    Notes
    -----
    If that start is greater than stop, the pdarray values are generated
    in descending order.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.linspace(0, 1, 5)
    array([0.00000000000000000 0.25 0.5 0.75 1.00000000000000000])

    >>> ak.linspace(start=1, stop=0, length=5)
    array([1.00000000000000000 0.75 0.5 0.25 0.00000000000000000])

    >>> ak.linspace(start=-5, stop=0, length=5)
    array([-5.00000000000000000 -3.75 -2.5 -1.25 0.00000000000000000])
    """
    if not isSupportedNumber(start) or not isSupportedNumber(stop):
        raise TypeError("both start and stop must be an int, np.int64, float, or np.float64")
    if not isSupportedNumber(length):
        raise TypeError("length must be an int or int64")
    repMsg = generic_msg(cmd="linspace", args={"start": start, "stop": stop, "len": length})
    return create_pdarray(repMsg)


@typechecked
def randint(
    low: numeric_scalars,
    high: numeric_scalars,
    size: Union[int_scalars, Tuple[int_scalars, ...]] = 1,
    dtype=akint64,
    seed: Optional[int_scalars] = None,
) -> pdarray:
    """
    Generate a pdarray of randomized int, float, or bool values in a
    specified range bounded by the low and high parameters.

    Parameters
    ----------
    low : numeric_scalars
        The low value (inclusive) of the range
    high : numeric_scalars
        The high value (exclusive for int, inclusive for float) of the range
    size : int_scalars or tuple of int_scalars
        The size or shape of the returned array
    dtype : Union[int64, float64, bool]
        The dtype of the array
    seed : int_scalars, optional
        Index for where to pull the first returned value


    Returns
    -------
    pdarray
        Values drawn uniformly from the specified range having the desired dtype

    Raises
    ------
    TypeError
        Raised if dtype.name not in DTypes, size is not an int, low or high is
        not an int or float, or seed is not an int
    ValueError
        Raised if size < 0 or if high < low

    Notes
    -----
    Calling randint with dtype=float64 will result in uniform non-integral
    floating point values.

    Ranges >= 2**64 in size is undefined behavior because
    it exceeds the maximum value that can be stored on the server (uint64)

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.randint(0, 10, 5, seed=1701)
    array([6 5 1 6 3])

    >>> ak.randint(0, 1, 3, seed=1701, dtype=ak.float64)
    array([0.011410423448327005 0.73618171558685619 0.12367222192448891])

    >>> ak.randint(0, 1, 5, seed=1701, dtype=ak.bool_)
    array([False True False True False])
    """
    from arkouda.numpy.random import randint

    return randint(low=low, high=high, size=size, dtype=dtype, seed=seed)


@typechecked
def uniform(
    size: int_scalars,
    low: numeric_scalars = float(0.0),
    high: numeric_scalars = 1.0,
    seed: Union[None, int_scalars] = None,
) -> pdarray:
    """
    Generate a pdarray with uniformly distributed random float values
    in a specified range.

    Parameters
    ----------
    low : float_scalars
        The low value (inclusive) of the range, defaults to 0.0
    high : float_scalars
        The high value (inclusive) of the range, defaults to 1.0
    size : int_scalars
        The length of the returned array
    seed : int_scalars, optional
        Value used to initialize the random number generator

    Returns
    -------
    pdarray
        Values drawn uniformly from the specified range

    Raises
    ------
    TypeError
        Raised if dtype.name not in DTypes, size is not an int, or if
        either low or high is not an int or float
    ValueError
        Raised if size < 0 or if high < low

    Notes
    -----
    The logic for uniform is delegated to the ak.randint method which
    is invoked with a dtype of float64

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.uniform(3,seed=1701)
    array([0.011410423448327005 0.73618171558685619 0.12367222192448891])

    >>> ak.uniform(size=3,low=0,high=5,seed=0)
    array([0.30013431967121934 0.47383036230759112 1.0441791878997098])
    """
    return randint(low=low, high=high, size=size, dtype="float64", seed=seed)


@typechecked
def standard_normal(size: int_scalars, seed: Union[None, int_scalars] = None) -> pdarray:
    """
    Draw real numbers from the standard normal distribution.

    Parameters
    ----------
    size : int_scalars
        The number of samples to draw (size of the returned array)
    seed : int_scalars
        Value used to initialize the random number generator

    Returns
    -------
    pdarray
        The array of random numbers

    Raises
    ------
    TypeError
        Raised if size is not an int
    ValueError
        Raised if size < 0

    See Also
    --------
    randint

    Notes
    -----
    For random samples from :math:`N(\\mu, \\sigma^2)`, use:

    ``(sigma * standard_normal(size)) + mu``

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.standard_normal(3,1)
    array([-0.68586185091150265 1.1723810583573377 0.567584107142031])
    """
    from arkouda.numpy.random import standard_normal

    return standard_normal(size=size, seed=seed)


@typechecked
def random_strings_uniform(
    minlen: int_scalars,
    maxlen: int_scalars,
    size: int_scalars,
    characters: str = "uppercase",
    seed: Union[None, int_scalars] = None,
) -> Strings:
    """
    Generate random strings with lengths uniformly distributed between
    minlen and maxlen, and with characters drawn from a specified set.

    Parameters
    ----------
    minlen : int_scalars
        The minimum allowed length of string
    maxlen : int_scalars
        The maximum allowed length of string
    size : int_scalars
        The number of strings to generate
    characters : (uppercase, lowercase, numeric, printable, binary)
        The set of characters to draw from
    seed :  Union[None, int_scalars], optional
        Value used to initialize the random number generator

    Returns
    -------
    Strings
        The array of random strings

    Raises
    ------
    ValueError
        Raised if minlen < 0, maxlen < minlen, or size < 0

    See Also
    --------
    random_strings_lognormal, randint

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.random_strings_uniform(minlen=1, maxlen=5, seed=8675309, size=5)
    array(['ECWO', 'WSS', 'TZG', 'RW', 'C'])

    >>> ak.random_strings_uniform(minlen=1, maxlen=5, seed=8675309, size=5,
    ... characters='printable')
    array(['2 .z', 'aom', '2d|', 'o(', 'M'])
    """
    if minlen < 0 or maxlen <= minlen or size < 0:
        raise ValueError("Incompatible arguments: minlen < 0, maxlen " + "<= minlen, or size < 0")

    repMsg = generic_msg(
        cmd="randomStrings",
        args={
            "size": NUMBER_FORMAT_STRINGS["int64"].format(size),
            "dist": "uniform",
            "chars": characters,
            "arg1": NUMBER_FORMAT_STRINGS["int64"].format(minlen),
            "arg2": NUMBER_FORMAT_STRINGS["int64"].format(maxlen),
            "seed": seed,
        },
    )
    return Strings.from_return_msg(cast(str, repMsg))


@typechecked
def random_strings_lognormal(
    logmean: numeric_scalars,
    logstd: numeric_scalars,
    size: int_scalars,
    characters: str = "uppercase",
    seed: Optional[int_scalars] = None,
) -> Strings:
    """
    Generate random strings with log-normally distributed lengths and
    with characters drawn from a specified set.

    Parameters
    ----------
    logmean : numeric_scalars
        The log-mean of the length distribution
    logstd :  numeric_scalars
        The log-standard-deviation of the length distribution
    size : int_scalars
        The number of strings to generate
    characters : (uppercase, lowercase, numeric, printable, binary)
        The set of characters to draw from
    seed : int_scalars, optional
        Value used to initialize the random number generator

    Returns
    -------
    Strings
        The Strings object encapsulating a pdarray of random strings

    Raises
    ------
    TypeError
        Raised if logmean is neither a float nor a int, logstd is not a float,
        seed is not an int, size is not an int, or if characters is not a str
    ValueError
        Raised if logstd <= 0 or size < 0

    See Also
    --------
    random_strings_lognormal, randint

    Notes
    -----
    The lengths of the generated strings are distributed $Lognormal(\\mu, \\sigma^2)$,
    with :math:`\\mu = logmean` and :math:`\\sigma = logstd`. Thus, the strings will
    have an average length of :math:`exp(\\mu + 0.5*\\sigma^2)`, a minimum length of
    zero, and a heavy tail towards longer strings.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.random_strings_lognormal(2, 0.25, 5, seed=1)
    array(['VWHJEX', 'BEBBXJHGM', 'RWOVKBUR', 'LNJCSDXD', 'NKEDQC'])

    >>> ak.random_strings_lognormal(2, 0.25, 5, seed=1, characters='printable')
    array(['eL96<O', ')o-GOe lR', ')PV yHf(', '._b3Yc&K', ',7Wjef'])
    """
    if not isSupportedNumber(logmean) or not isSupportedNumber(logstd):
        raise TypeError("both logmean and logstd must be an int, np.int64, float, or np.float64")
    if logstd <= 0 or size < 0:
        raise ValueError("Incompatible arguments: logstd <= 0 or size < 0")

    repMsg = generic_msg(
        cmd="randomStrings",
        args={
            "size": NUMBER_FORMAT_STRINGS["int64"].format(size),
            "dist": "lognormal",
            "chars": characters,
            "arg1": NUMBER_FORMAT_STRINGS["float64"].format(logmean),
            "arg2": NUMBER_FORMAT_STRINGS["float64"].format(logstd),
            "seed": seed,
        },
    )
    return Strings.from_return_msg(cast(str, repMsg))
