import numpy as np
import struct

from arkouda.client import generic_msg, maxTransferBytes
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS
from arkouda.dtypes import dtype as akdtype
from arkouda.pdarrayclass import pdarray, create_pdarray

__all__ = ["array", "zeros", "ones", "zeros_like", "ones_like", "arange",
           "linspace", "randint"]

def array(a):
    """
    Convert an iterable to a pdarray, sending data to the arkouda server.

    Parameters
    ----------
    a : array_like
        Rank-1 array of a supported dtype

    Returns
    -------
    pdarray
        Instance of pdarray stored on arkouda server

    See Also
    --------
    pdarray.to_ndarray

    Notes
    -----
    The number of bytes in the input array cannot exceed `arkouda.maxTransferBytes`,
    otherwise a RuntimeError will be raised. This is to protect the user
    from overwhelming the connection between the Python client and the arkouda
    server, under the assumption that it is a low-bandwidth connection. The user
    may override this limit by setting ak.maxTransferBytes to a larger value, 
    but should proceed with caution.

    Examples
    --------
    >>> a = [3, 5, 7]
    >>> b = ak.array(a)
    >>> b
    array([3, 5, 7])
   
    >>> type(b)
    arkouda.pdarray    
    """
    # If a is already a pdarray, do nothing
    if isinstance(a, pdarray):
        return a
    # If a is not already a numpy.ndarray, convert it
    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except:
            raise TypeError("Argument must be array-like")
    # Only rank 1 arrays currently supported
    if a.ndim != 1:
        raise RuntimeError("Only rank-1 arrays supported")
    # Check that dtype is supported in arkouda
    if a.dtype.name not in DTypes:
        raise RuntimeError("Unhandled dtype {}".format(a.dtype))
    # Do not allow arrays that are too large
    size = a.size
    if (size * a.itemsize) > maxTransferBytes:
        raise RuntimeError("Array exceeds allowed transfer size. Increase ak.maxTransferBytes to allow")
    # Pack binary array data into a bytes object with a command header
    # including the dtype and size
    fmt = ">{:n}{}".format(size, structDtypeCodes[a.dtype.name])
    req_msg = "array {} {:n} ".format(a.dtype.name, size).encode() + struct.pack(fmt, *a)
    rep_msg = generic_msg(req_msg, send_bytes=True)
    return create_pdarray(rep_msg)

def zeros(size, dtype=np.float64):
    """
    Create a pdarray filled with zeros.

    Parameters
    ----------
    size : int
        Size of the array (only rank-1 arrays supported)
    dtype : {float64, int64, bool}
        Type of resulting array, default float64

    Returns
    -------
    pdarray
        Zeros of the requested size and dtype

    See Also
    --------
    ones, zeros_like

    Examples
    --------
    >>> ak.zeros(5, dtype=ak.int64)
    array([0, 0, 0, 0, 0])
    >>> ak.zeros(5, dtype=ak.float64)
    array([0, 0, 0, 0, 0])
    >>> ak.zeros(5, dtype=ak.bool)
    array([False, False, False, False, False])
    """
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    kind, itemsize = translate_np_dtype(dtype)
    repMsg = generic_msg("create {} {}".format(dtype.name, size))
    return create_pdarray(repMsg)

def ones(size, dtype=float64):
    """
    Create a pdarray filled with ones.

    Parameters
    ----------
    size : int
        Size of the array (only rank-1 arrays supported)
    dtype : {float64, int64, bool}
        Resulting array type, default float64

    Returns
    -------
    pdarray
        Ones of the requested size and dtype

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> ak.ones(5, dtype=ak.int64)
    array([1, 1, 1, 1, 1])
    >>> ak.ones(5, dtype=ak.float64)
    array([1, 1, 1, 1, 1])
    >>> ak.ones(5, dtype=ak.bool)
    array([True, True, True, True, True])
    """
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    kind, itemsize = translate_np_dtype(dtype)
    repMsg = generic_msg("create {} {}".format(dtype.name, size))
    a = create_pdarray(repMsg)
    a.fill(1)
    return a

def zeros_like(pda):
    """
    Create a zero-filled pdarray of the same size and dtype as an existing pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.zeros(pda.size, pda.dtype)

    See Also
    --------
    zeros, ones_like
    """
    if isinstance(pda, pdarray):
        return zeros(pda.size, pda.dtype)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def ones_like(pda):
    """
    Create a one-filled pdarray of the same size and dtype as an existing pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.ones(pda.size, pda.dtype)

    See Also
    --------
    ones, zeros_like
    """
    if isinstance(pda, pdarray):
        return ones(pda.size, pda.dtype)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def arange(*args):
    """
    arange([start,] stop[, stride])

    Create a pdarray of consecutive integers within the interval [start, stop).
    If only one arg is given then arg is the stop parameter. If two args are given
    then the first arg is start and second is stop. If three args are given
    then the first arg is start, second is stop, third is stride.

    Parameters
    ----------
    start : int, optional
        Starting value (inclusive), the default starting value is 0
    stop : int
        Stopping value (exclusive)
    stride : int, optional
        The difference between consecutive elements, the default stride is 1,
        if stride is specified then start must also be specified

    Returns
    -------
    pdarray, int64
        Integers from start (inclusive) to stop (exclusive) by stride

    See Also
    --------
    linspace, zeros, ones, randint
    
    Notes
    -----
    Negative strides result in decreasing values. Currently, only int64 pdarrays
    can be created with this function. For float64 arrays, use linspace.

    Examples
    --------
    >>> ak.arange(0, 5, 1)
    array([0, 1, 2, 3, 4])

    >>> ak.arange(5, 0, -1)
    array([5, 4, 3, 2, 1])

    >>> ak.arange(0, 10, 2)
    array([0, 2, 4, 6, 8])
    """
   
    #if one arg is given then arg is stop
    if len(args) == 1:
        start = 0
        stop = args[0]
        stride = 1

    #if two args are given then first arg is start and second is stop
    if len(args) == 2:
        start = args[0]
        stop = args[1]
        stride = 1

    #if three args are given then first arg is start,
    #second is stop, third is stride
    if len(args) == 3:
        start = args[0]
        stop = args[1]
        stride = args[2]

    if stride == 0:
        raise ZeroDivisionError("division by zero")

    if isinstance(start, int) and isinstance(stop, int) and isinstance(stride, int):
        # TO DO: fix bug in server that goes 2 steps too far for negative strides
        if stride < 0:
            stop = stop + 2
        repMsg = generic_msg("arange {} {} {}".format(start, stop, stride))
        return create_pdarray(repMsg)
    else:
        raise TypeError("start,stop,stride must be type int {} {} {}".format(start,stop,stride))

def linspace(start, stop, length):
    """
    Create a pdarray of linearly spaced points in a closed interval.

    Parameters
    ----------
    start : scalar
        Start of interval (inclusive)
    stop : scalar
        End of interval (inclusive)
    length : int
        Number of points

    Returns
    -------
    pdarray, float64
        Array of evenly spaced points along the interval

    See Also
    --------
    arange

    Examples
    --------
    >>> ak.linspace(0, 1, 5)
    array([0, 0.25, 0.5, 0.75, 1])
    """
    starttype = resolve_scalar_dtype(start)
    startstr = NUMBER_FORMAT_STRINGS[starttype].format(start)
    stoptype = resolve_scalar_dtype(stop)
    stopstr = NUMBER_FORMAT_STRINGS[stoptype].format(stop)
    lentype = resolve_scalar_dtype(length)
    if lentype != 'int64':
        raise TypeError("Length must be int64")
    lenstr = NUMBER_FORMAT_STRINGS[lentype].format(length)
    repMsg = generic_msg("linspace {} {} {}".format(startstr, stopstr, lenstr))
    return create_pdarray(repMsg)


def randint(low, high, size, dtype=int64):
    """
    Generate a pdarray with random values in a specified range.

    Parameters
    ----------
    low : int
        The low value (inclusive) of the range
    high : int
        The high value (exclusive for int, inclusive for float) of the range
    size : int
        The length of the returned array
    dtype : {int64, float64, bool}
        The dtype of the array

    Returns
    -------
    pdarray
        Values drawn uniformly from the specified range having the desired dtype

    Notes
    -----
    Calling randint with dtype=float64 will result in uniform non-integral
    floating point values.

    Examples
    --------
    >>> ak.randint(0, 10, 5)
    array([5, 7, 4, 8, 3])

    >>> ak.randint(0, 1, 3, dtype=ak.float64)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])

    >>> ak.randint(0, 1, 5, dtype=ak.bool)
    array([True, False, True, True, True])
    """
    # TO DO: separate out into int and float versions
    # TO DO: float version should accept non-integer low and high
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    if isinstance(low, int) and isinstance(high, int) and isinstance(size, int):
        kind, itemsize = translate_np_dtype(dtype)
        repMsg = generic_msg("randint {} {} {} {}".format(low,high,size,dtype.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("min,max,size must be int {} {} {}".format(low,high,size));
