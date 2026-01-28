import itertools

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing import cast as type_cast

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typeguard import typechecked

from arkouda.numpy._typing._typing import _NumericLikeDType, _StringDType
from arkouda.numpy.dtypes import (
    ARKOUDA_SUPPORTED_NUMBERS,
    NUMBER_FORMAT_STRINGS,
    NumericDTypes,
    bigint,
    bool_,
    bool_scalars,
    float64,
    get_byteorder,
    get_server_byteorder,
    int_scalars,
    is_supported_int,
    is_supported_number,
    numeric_scalars,
    resolve_scalar_dtype,
    str_,
)
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import uint64 as akuint64
from arkouda.numpy.strings import Strings


if TYPE_CHECKING:
    from arkouda.numpy.pdarrayclass import pdarray
else:
    pdarray = TypeVar("pdarray")

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
    "logspace",
    "randint",
    "uniform",
    "standard_normal",
    "random_strings_uniform",
    "random_strings_lognormal",
    "bigint_from_uint_arrays",
    "promote_to_common_dtype",
    "scalar_array",
]


def _deepcopy(a: pdarray) -> pdarray:
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarrayclass import create_pdarray

    rep_msg = generic_msg(
        cmd=f"deepcopy<{a.dtype.name},{a.ndim}>",
        args={"x": a},
    )
    return create_pdarray(rep_msg)


# ======================
# Overloads for ak.array
# ======================


@overload
def array(
    a: Union[pdarray, Strings, Iterable[Any], NDArray[Any]],
    dtype: _StringDType = ...,
    *,
    copy: bool = ...,
    max_bits: int = ...,
) -> Strings: ...


@overload
def array(
    a: Union[pdarray, Strings, Iterable[Any], NDArray[Any]],
    dtype: _NumericLikeDType = ...,  # object covers np.dtype, type, int/float/bool dtypes, etc.
    *,
    copy: bool = ...,
    max_bits: int = ...,
) -> pdarray: ...


@overload
def array(
    a: Union[pdarray, Strings, Iterable[Any], NDArray[Any]],
    dtype=None,
    *,
    copy: bool = ...,
    max_bits: int = ...,
) -> Union[pdarray, Strings]: ...


@overload
def array(
    a: Union[pdarray, List[_NumericLikeDType]],
    dtype: Union[_StringDType, _NumericLikeDType, None] = None,
    copy: bool = False,
    max_bits: int = -1,
) -> pdarray: ...


@overload
def array(
    a: Union[Strings, Iterable[_StringDType]],
    dtype: Union[_StringDType, _NumericLikeDType, None] = None,
    copy: bool = False,
    max_bits: int = -1,
) -> Strings: ...


@overload
def array(
    a: Union[pdarray, Strings, Iterable[Any], NDArray[Any]],
    dtype: Union[_StringDType, _NumericLikeDType, None] = None,
    copy: bool = False,
    max_bits: int = -1,
) -> Union[pdarray, Strings]: ...


def array(
    a: Union[pdarray, np.ndarray, Iterable, Strings],
    dtype: Union[np.dtype, type, str, None] = None,
    copy: bool = False,
    max_bits: int = -1,
) -> Union[pdarray, Strings]:
    """
    Convert a Python, NumPy, or Arkouda array-like into a `pdarray` or `Strings` object,
    transferring data to the Arkouda server.

    Parameters
    ----------
    a : Union[pdarray, np.ndarray, Iterable, Strings]
        The array-like input to convert. Supported types include Arkouda `Strings`, `pdarray`,
        NumPy `ndarray`, or Python iterables such as list, tuple, range, or deque.

    dtype : Union[np.dtype, type, str], optional
        The target dtype to cast values to. This may be a NumPy dtype object,
        a NumPy scalar type (e.g. `np.int64`), or a string (e.g. `'int64'`, `'str'`).

    copy : bool, default=False
        If True, a deep copy of the array is made. If False, no copy is made if the input
        is already a `pdarray`. **Note**: Arkouda does not currently support views or shallow copies.
        This differs from NumPy. Also, the default (`False`) is chosen to reduce performance overhead.

    max_bits : int, optional
        The maximum number of bits for bigint arrays. Ignored for other dtypes.

    Returns
    -------
    Union[pdarray, Strings]
        A `pdarray` stored on the Arkouda server, or a `Strings` object.

    Raises
    ------
    TypeError
        - If `a` is not a `pdarray`, `np.ndarray`, or Python iterable.
        - If `a` is of string type and `dtype` is not `ak.str_`.

    RuntimeError
        - If input size exceeds `ak.client.maxTransferBytes`.
        - If `a.dtype` is unsupported or incompatible with Arkouda.
        - If `a.size * a.itemsize > maxTransferBytes`.

    ValueError
        - If `a`'s rank is not supported (see `get_array_ranks()`).
        - If the server response is malformed or missing required fields.

    See Also
    --------
    pdarray.to_ndarray
        Convert back from Arkouda to NumPy.

    Notes
    -----
    - Arkouda does not currently support shallow copies or views; all copies are deep.
    - The number of bytes transferred to the server is limited by `ak.client.maxTransferBytes`.
      This prevents saturating the network during large transfers. To increase this limit,
      set `ak.client.maxTransferBytes` to a larger value manually.
    - If the input is a Unicode string array (`dtype.kind == 'U'` or `dtype='str'`),
      this function recursively creates a `Strings` object from two internal `pdarray`s
      (one for offsets and one for concatenated string bytes).

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.array(np.arange(1, 10))
    array([1 2 3 4 5 6 7 8 9])

    >>> ak.array(range(1, 10))
    array([1 2 3 4 5 6 7 8 9])

    >>> strings = ak.array([f'string {i}' for i in range(5)])
    >>> type(strings)
    <class 'arkouda.numpy.strings.Strings'>
    """
    from arkouda.client import generic_msg, get_array_ranks
    from arkouda.numpy.numeric import cast as akcast
    from arkouda.numpy.pdarrayclass import create_pdarray, pdarray

    if isinstance(a, str):
        raise TypeError(
            "Cannot convert a scalar string to an Arkouda array; pass a sequence of strings instead."
        )

    if isinstance(a, pdarray) and (a.dtype == dtype or dtype is None):
        return _deepcopy(a) if copy else a

    if isinstance(a, Strings):
        if dtype and dtype != "str_":
            raise TypeError(f"Cannot cast Strings to dtype {dtype} in ak.array")
        return (
            Strings(cast(pdarray, array([], dtype="int64")), 0) if a.size == 0 else a[:] if copy else a
        )

    # If a is already a pdarray, do nothing
    if isinstance(a, pdarray):
        dtype = dtype if dtype else a.dtype
        casted = type_cast(pdarray, akcast(a, dtype))
        if dtype == bigint and max_bits != -1:
            casted.max_bits = max_bits
        return type_cast(Union[pdarray, Strings], casted)

    from arkouda.client import maxTransferBytes

    if isinstance(a, Iterable) and not isinstance(a, (list, tuple, np.ndarray, pd.Series, set, dict)):
        a = list(a)

    # If a is not already a numpy.ndarray, convert it

    if not isinstance(a, np.ndarray):
        try:
            if dtype is not None and dtype != bigint:
                # if the user specified dtype, use that dtype
                a = np.array(a, dtype=dtype)
            elif (
                all(is_supported_int(i) for i in a)
                and all(i >= 0 for i in a)
                and any(2**64 > i >= 2**63 for i in a)
            ):
                # all supportedInt values but some >=2**63, default to uint (see #1297)
                # iterating shouldn't be too expensive since
                # this is after the `isinstance(a, pdarray)` check
                a = np.array(a, dtype=np.uint)
            else:
                # let numpy decide the type
                a = np.array(a)
        except (RuntimeError, TypeError, ValueError):
            raise TypeError("a must be a pdarray, np.ndarray, or convertible to a numpy array")
    if dtype is not None and dtype not in [bigint, "bigint"]:
        a = a.astype(dtype)
    if a.dtype == object and dtype is None and all(isinstance(x, (int, np.integer)) for x in a.flat):
        dtype = bigint

    if a.ndim == 0 and a.dtype.name not in NumericDTypes:
        raise TypeError("Must be an iterable or have a numeric DType")

    # Return multi-dimensional pdarray if a.ndim in get_array_ranks()
    # otherwise raise an error

    if a.ndim not in get_array_ranks():
        raise ValueError(f"array rank {a.ndim} not in compiled ranks {get_array_ranks()}")
    if a.dtype.kind in ("i", "u", "f", "O") and (dtype == bigint or dtype == "bigint"):
        return _bigint_from_numpy(a, max_bits)
    elif not (
        np.issubdtype(a.dtype, np.str_)
        or (a.dtype == np.object_ and a.size > 0 and isinstance(a[0], str))
    ) and (dtype == bigint or dtype == "bigint"):
        raise TypeError("Cannot convert type to bigint")

    # Check if array of strings
    # if a.dtype == numpy.object_ need to check first element
    if np.issubdtype(a.dtype, np.str_) or (
        a.dtype == np.object_ and a.size > 0 and isinstance(a[0], str)
    ):
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
        strings = Strings.from_return_msg(cast(str, rep_msg))
        return strings if dtype is None else type_cast(Union[pdarray, Strings], akcast(strings, dtype))

    # If not strings, then check that dtype is supported in arkouda

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
        args={
            "dtype": a_.dtype.name,
            "shape": tuple(a_.shape),
            "seg_string": False,
        },
        payload=aview,
        send_binary=True,
    )
    return (
        create_pdarray(rep_msg)
        if dtype is None
        else type_cast(Union[pdarray, Strings], akcast(create_pdarray(rep_msg), dtype))
    )


def _bigint_from_numpy(np_a: np.ndarray, max_bits: int) -> pdarray:
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarrayclass import create_pdarray, pdarray

    a = np.asarray(np_a)
    out_shape = a.shape
    flat = a.ravel()  # ndarray (view when possible)

    # Empty → empty bigint (shape-preserving)
    if a.size == 0:
        return zeros(size=a.shape, dtype=bigint, max_bits=max_bits)

    # Only ints/object should reach here (strings handled upstream)
    if a.dtype.kind not in ("i", "u", "f", "O"):
        raise TypeError(f"bigint requires numeric input, got dtype={a.dtype}")

    if a.dtype.kind == "O":
        # Only allow Python ints and floats
        if not all(isinstance(x, (int, float)) for x in a.flat):
            raise TypeError("bigint requires numeric input, got non-numeric object")
        if any(isinstance(x, float) for x in a.flat):
            a = a.astype(np.float64, copy=False)

    # Fast all-zero path or single limb path
    if a.dtype.kind in ("i", "u", "f"):
        if not np.any(a):
            return zeros(size=a.shape, dtype=bigint, max_bits=max_bits)

        if a.dtype.kind == "i":
            ak_a = array(a.astype(np.int64, copy=False))
        elif a.dtype.kind == "u":
            ak_a = array(a.astype(np.uint64, copy=False))
        else:
            ak_a = array(a.astype(np.float64, copy=False))

        # Send a single limb array
        return create_pdarray(
            generic_msg(
                cmd=f"big_int_creation_one_limb<{ak_a.dtype},{ak_a.ndim}>",
                args={
                    "array": ak_a,
                    "shape": ak_a.shape,
                    "max_bits": max_bits,
                },
            )
        )
    else:
        if all(int(x) == 0 for x in a.flat):
            return zeros(size=a.shape, dtype=bigint, max_bits=max_bits)
    any_neg = np.any(flat < 0)
    req_bits: int
    if any_neg:
        req_bits = max(flat.max().bit_length(), (-flat.min()).bit_length()) + 1
    else:
        req_bits = flat.max().bit_length()
    req_limbs = (req_bits + 63) // 64
    mask = (1 << 64) - 1
    uint_arrays: List[Union[pdarray, Strings]] = []
    # attempt to break bigint into multiple uint64 arrays
    for _ in range(req_limbs):
        low = flat & mask
        flat = flat >> 64  # type: ignore
        # low, flat = flat & mask, flat >> 64
        uint_arrays.append(array(np.array(low, dtype=np.uint), dtype=akuint64))
    return create_pdarray(
        generic_msg(
            cmd=f"big_int_creation_multi_limb<{uint_arrays[0].dtype},1>",
            args={
                "arrays": uint_arrays,
                "num_arrays": len(uint_arrays),
                "signed": any_neg,
                "shape": flat.shape,
                "max_bits": max_bits,
            },
        )
    ).reshape(out_shape)


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
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarrayclass import create_pdarray, pdarray

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
    from arkouda.client import generic_msg, get_array_ranks
    from arkouda.numpy.pdarrayclass import create_pdarray

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

    rep_msg = generic_msg(cmd=f"create<{dtype_name},{ndim}>", args={"shape": shape})

    return create_pdarray(rep_msg, max_bits=max_bits)


@typechecked
def ones(
    size: Union[int_scalars, Tuple[int_scalars, ...], str],
    dtype: Union[np.dtype, type, str, bigint] = float64,
    max_bits: Optional[int] = None,
) -> Union[pdarray, Strings]:
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
    pdarray or Strings
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
    fill_value: Union[numeric_scalars, np.bool, str],
    dtype: Union[np.dtype, type, str, bigint] = float64,
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
    from arkouda.client import generic_msg, get_array_ranks
    from arkouda.numpy.dtypes import dtype as ak_dtype
    from arkouda.numpy.pdarrayclass import create_pdarray
    from arkouda.numpy.util import _infer_shape_from_size

    shape, ndim, full_size = _infer_shape_from_size(size)

    if isinstance(fill_value, str):
        return _full_string(full_size, fill_value)
    elif ak_dtype(dtype) == str_ or dtype == Strings:
        return _full_string(full_size, str_(fill_value))

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

    rep_msg = generic_msg(cmd=f"create<{dtype_name},{ndim}>", args={"shape": shape})

    a = create_pdarray(rep_msg)
    a.fill(fill_value)

    if max_bits:
        a.max_bits = max_bits
    return a


@typechecked
def scalar_array(
    value: Union[numeric_scalars, bool_scalars],
    dtype: Optional[Union[np.dtype, type, str, bigint]] = None,
) -> pdarray:
    """
    Create a pdarray from a single scalar value.

    Parameters
    ----------
    value: numeric_scalars
        Value to create pdarray from
    dtype: np.dtype, type, str, bigint, or None
        The data type of the created array.

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
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarrayclass import create_pdarray

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
    from arkouda.client import generic_msg

    rep_msg = generic_msg(cmd="segmentedFull", args={"size": size, "fill_value": fill_value})
    return Strings.from_return_msg(cast(str, rep_msg))


@typechecked
def zeros_like(pda: pdarray) -> pdarray:
    """
    Create a zero-filled pdarray of the same size and dtype as an existing
    pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for shape and dtype

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
def ones_like(pda: pdarray) -> Union[pdarray, Strings]:
    """
    Create a one-filled pdarray of the same size and dtype as an existing
    pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for shape and dtype

    Returns
    -------
    pdarray or Strings
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
def full_like(pda: pdarray, fill_value: Union[numeric_scalars, bool_, str_]) -> Union[pdarray, Strings]:
    """
    Create a pdarray filled with fill_value of the same size and dtype as an existing
    pdarray.

    Parameters
    ----------
    pda: pdarray
        Array to use for shape and dtype
    fill_value: int_scalars
        Value with which the array will be filled

    Returns
    -------
    pdarray
        Equivalent to ak.full(pda.shape, fill_value, pda.dtype)

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
    """
    Create a pdarray of consecutive integers within the interval [start, stop).

    Called as: arange([start,] stop[, step,] dtype=int64).

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
    from arkouda.client import generic_msg
    from arkouda.numpy import cast as akcast
    from arkouda.numpy.pdarrayclass import create_pdarray, pdarray

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
        return type_cast(pdarray, akcast(array([], dtype=akint64), dt=aktype))

    if is_supported_int(start) and is_supported_int(stop) and is_supported_int(step):
        arg_dtypes = [resolve_scalar_dtype(arg) for arg in (start, stop, step)]
        akmax_bits = -1 if max_bits is None else max_bits
        arg_dtype = "int64"
        if dtype in ["bigint", bigint] or "bigint" in arg_dtypes or akmax_bits != -1:
            arg_dtype = "bigint"
        elif "uint64" in arg_dtypes:
            arg_dtype = "uint64"

        if step < 0:
            stop = stop + 2

        rep_msg = generic_msg(
            cmd=f"arange<{arg_dtype},1>",
            args={"start": start, "stop": stop, "step": step},
        )

        arr = create_pdarray(rep_msg, max_bits=max_bits)
        return arr if aktype == akint64 else akcast(arr, dt=aktype)

    raise TypeError(f"start, stop, step must be ints; got {args!r}")


@typechecked
def logspace(
    start: Union[numeric_scalars, pdarray],
    stop: Union[numeric_scalars, pdarray],
    num: int_scalars = 50,
    base: numeric_scalars = 10.0,
    endpoint: Union[None, bool] = True,
    dtype: Optional[type] = float64,
    axis: int_scalars = 0,
) -> pdarray:
    """
    Create a pdarray of numbers evenly spaced on a log scale.

    Parameters
    ----------
    start : Union[numeric_scalars, pdarray]
        The starting value of the sequence.
    stop : Union[numeric_scalars, pdarray]
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    base : numeric_scalars, optional
        the base of the log space, defaults to 10.0.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : Union[None, float64]
        allowed for compatibility with numpy, but ignored.  Outputs are always float
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    pdarray
        There are `num` equally spaced (logarithmically) samples in the closed interval
        base**``[start, stop]`` or the half-open interval base**``[start, stop)``
        (depending on whether `endpoint` is True or False).

    Raises
    ------
    TypeError
        Raised if start or stop is not a float or a pdarray, or if num
        is not an int, or if endpoint is not a bool, or if dtype is anything
        other than None or float64, or axis is not an integer.
    ValueError
        Raised if axis is not a valid axis for the given data, or if base < 0.

    See Also
    --------
    linspace

    Notes
    -----
    If start is greater than stop, the pdarray values are generated
    in descending order.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.logspace(2,3,3,4)
    array([16.00000000000000000 32.00000000000000000 64.00000000000000000])
    >>> ak.logspace(2,3,3,4,endpoint=False)
    array([16.00000000000000000 25.398416831491197 40.317473596635935])
    >>> ak.logspace(0,1,3,4)
    array([1.00000000000000000 2.00000000000000000 4.00000000000000000])
    >>> ak.logspace(1,0,3,4)
    array([4.00000000000000000 2.00000000000000000 1.00000000000000000])
    >>> ak.logspace(0,1,3,endpoint=False)
    array([1.00000000000000000 2.1544346900318838 4.6415888336127784])
    >>> ak.logspace(0,ak.array([2,3]),3,base=2)
    array([array([1.00000000000000000 1.00000000000000000])
        array([2.00000000000000000 2.8284271247461903])
        array([4.00000000000000000 8.00000000000000000])])
    >>> ak.logspace(ak.array([0,1]),3,3,base=3)
    array([array([1.00000000000000000 3.00000000000000000])
        array([5.196152422706632 9.00000000000000000])
        array([27.00000000000000000 27.00000000000000000])])
    >>> ak.logspace(ak.array([0,1]),ak.array([2,3]),3,base=4)
    array([array([1.00000000000000000 4.00000000000000000])
        array([4.00000000000000000 16.00000000000000000])
        array([16.00000000000000000 64.00000000000000000])])
    """
    if dtype not in (None, float64):
        raise TypeError("dtype must be None or float64")
    if base <= 0:
        raise ValueError("base must be positive")
    if endpoint is None:
        endpoint = True

    return base ** linspace(start, stop, num, endpoint=endpoint, dtype=float64, axis=axis)


@typechecked
def linspace(
    start: Union[numeric_scalars, pdarray],
    stop: Union[numeric_scalars, pdarray],
    num: int_scalars = 50,
    endpoint: Union[None, bool] = True,
    dtype: Optional[type] = float64,
    axis: int_scalars = 0,
) -> pdarray:
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : Union[numeric_scalars, pdarray]
        The starting value of the sequence.
    stop : Union[numeric_scalars, pdarray]
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype, optional
        Allowed for compatibility with numpy linspace, but anything entered
        is ignored.  The output is always ak.float64.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    pdarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).

    Raises
    ------
    TypeError
        Raised if start or stop is not a float or a pdarray, or if num
        is not an int, or if endpoint is not a bool, or if dtype is anything
        other than None or float64, or axis is not an integer.
    ValueError
        Raised if axis is not a valid axis for the given data.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.linspace(0,1,3)
    array([0.00000000000000000 0.5 1.00000000000000000])
    >>> ak.linspace(1,0,3)
    array([1.00000000000000000 0.5 0.00000000000000000])
    >>> ak.linspace(0,1,3,endpoint=False)
    array([0.00000000000000000 0.33333333333333331 0.66666666666666663])
    >>> ak.linspace(0,ak.array([2,3]),3)
    array([array([0.00000000000000000 0.00000000000000000])
        array([1.00000000000000000 1.5]) array([2.00000000000000000 3.00000000000000000])])
    >>> ak.linspace(ak.array([0,1]),3,3)
    array([array([0.00000000000000000 1.00000000000000000])
        array([1.5 2.00000000000000000]) array([3.00000000000000000 3.00000000000000000])])
    >>> ak.linspace(ak.array([0,1]),ak.array([2,3]),3)
    array([array([0.00000000000000000 1.00000000000000000])
        array([1.00000000000000000 2.00000000000000000])
        array([2.00000000000000000 3.00000000000000000])])
    """
    from arkouda import newaxis
    from arkouda.numeric import transpose
    from arkouda.numpy.manipulation_functions import tile
    from arkouda.numpy.pdarrayclass import pdarray
    from arkouda.numpy.util import (
        _integer_axis_validation,
        broadcast_shapes,
        broadcast_to,
    )

    if dtype not in (None, float64):
        raise TypeError("dtype must be None or float64")
    if endpoint is None:
        endpoint = True

    start_: Union[numeric_scalars, pdarray] = start
    stop_: Union[numeric_scalars, pdarray] = stop

    #   First make sure everything's a float.

    if isinstance(start_, pdarray):
        start_ = start_.astype(float64)
    elif isinstance(start_, int):
        start_ = float(start_)

    if isinstance(stop_, pdarray):
        stop_ = stop_.astype(float64)
    elif isinstance(stop_, int):
        stop_ = float(stop_)

    #   Determine whether this is all scalars, or if vectors are involved.

    if isinstance(start_, pdarray) and isinstance(stop_, pdarray):
        #  they must be broadcast to a matching shape
        if start_.shape != stop_.shape:
            newshape = broadcast_shapes(start_.shape, stop_.shape)
            start_ = broadcast_to(start_, newshape)
            stop_ = broadcast_to(stop_, newshape)

    #   If one is a scalar and other a vector, we use full_like to "promote" the scalar one.

    else:
        if isinstance(start_, pdarray) and isinstance(stop_, (ARKOUDA_SUPPORTED_NUMBERS, str_, bool_)):
            full_pda = full_like(start_, stop_)
            assert isinstance(full_pda, pdarray)
            stop_ = full_pda

        elif isinstance(stop_, pdarray) and isinstance(start_, (ARKOUDA_SUPPORTED_NUMBERS, str_, bool_)):
            full_pda = full_like(stop_, start_)
            assert isinstance(full_pda, pdarray)
            start_ = full_pda

    divisor = num - 1 if endpoint else num

    #   In the vector case, by the time we reach here, start_ and stop_ are the same
    #   shape.  They are tiled by num (the size of the linspace), the delta is
    #   computed, and a solution is calculated involving start_ plus arange(num)
    #   multipled by delta.

    if isinstance(start_, pdarray) and isinstance(stop_, pdarray):
        pad: Tuple[int, int] = (int(num), int(1))
        start_ = tile(start_, pad).reshape((num,) + start_.shape)
        stop_ = tile(stop_, pad).reshape((num,) + stop_.shape)
        delta_: pdarray = (stop_ - start_) / divisor
        result = start_ + arange(num)[(...,) + (newaxis,) * (delta_.ndim - 1)] * delta_

        # Handle the axis parameter if needed

        if axis != 0:
            valid, axis_ = _integer_axis_validation(axis, result.ndim)
            if not valid:
                raise IndexError(f"{axis} is not a valid axis for the result of linspace.")
            axes = list(range(result.ndim))
            axes[axis_] = 0
            axes[0] = axis_
            result = transpose(result, tuple(axes))

    #   Scalar case is pretty straightforward.

    elif isinstance(start_, ARKOUDA_SUPPORTED_NUMBERS):
        if axis == 0:
            delta = (stop_ - start_) / divisor
            result = full(num, start_) + arange(num).astype(float64) * delta
        else:
            raise ValueError("axis should not be supplied when start and stop are scalars.")

    return result


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

    >>> ak.randint(0, 2, 5, seed=1701, dtype=ak.bool_)
    array([False True False True False])
    """
    from arkouda.numpy.random import randint

    return randint(low=low, high=high, size=size, dtype=dtype, seed=seed)


@typechecked
def uniform(
    size: Union[int_scalars, Tuple[int_scalars, ...]],
    low: numeric_scalars = float(0.0),
    high: numeric_scalars = 1.0,
    seed: Union[None, int_scalars] = None,
) -> pdarray:
    """
    Generate a pdarray with uniformly distributed random float values
    in a specified range.

    Parameters
    ----------
    size : Union[int_scalars, Tuple[int_scalars]
        The length or shape of the returned array
    low : float_scalars
        The low value (inclusive) of the range, defaults to 0.0
    high : float_scalars
        The high value (inclusive) of the range, defaults to 1.0
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
    from arkouda.numpy.util import _infer_shape_from_size

    shape, ndim, full_size = _infer_shape_from_size(size)
    if full_size < 0:
        raise ValueError("The size parameter must be >= 0")

    return (
        randint(low=low, high=high, size=size, dtype="float64", seed=seed)
        if ndim == 1
        else randint(low=low, high=high, size=full_size, dtype="float64", seed=seed).reshape(shape)
    )


@typechecked
def standard_normal(size: int_scalars, seed: Union[None, int_scalars] = None) -> pdarray:
    r"""
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
    array([-0.68586185091150265 1.1723810583573377 0.5675841071420...])
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
    from arkouda.client import generic_msg

    if minlen < 0 or maxlen <= minlen or size < 0:
        raise ValueError("Incompatible arguments: minlen < 0, maxlen " + "<= minlen, or size < 0")

    rep_msg = generic_msg(
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
    return Strings.from_return_msg(cast(str, rep_msg))


@typechecked
def random_strings_lognormal(
    logmean: numeric_scalars,
    logstd: numeric_scalars,
    size: int_scalars,
    characters: str = "uppercase",
    seed: Optional[int_scalars] = None,
) -> Strings:
    r"""
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
    from arkouda.client import generic_msg

    if not is_supported_number(logmean) or not is_supported_number(logstd):
        raise TypeError("both logmean and logstd must be an int, np.int64, float, or np.float64")
    if logstd <= 0 or size < 0:
        raise ValueError("Incompatible arguments: logstd <= 0 or size < 0")

    rep_msg = generic_msg(
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
    return Strings.from_return_msg(cast(str, rep_msg))
