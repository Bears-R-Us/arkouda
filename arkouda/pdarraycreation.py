import numpy as np
import struct
from typing import Iterable, Union
from typeguard import typechecked
from arkouda.client import generic_msg
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS
from arkouda.dtypes import dtype as akdtype
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.strings import Strings

__all__ = ["array", "zeros", "ones", "zeros_like", "ones_like", "arange",
           "linspace", "randint", "uniform", "standard_normal",
           "random_strings_uniform", "random_strings_lognormal"]

numericDTypes = frozenset(["bool", "int64", "float64"]) 

RANDINT_TYPES = {'int64','float64'}

def array(a : Union[pdarray,np.ndarray, Iterable]) -> Union[pdarray, Strings]:
    """
    Convert an iterable to a pdarray or Strings object, sending the corresponding
    data to the arkouda server. 

    Parameters
    ----------
    a : Union[pdarray, np.ndarray]
        Rank-1 array of a supported dtype

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
        If a is not one-dimensional, nbytes > maxTransferBytes, a.dtype is
        not supported (not in DTypes), or if the product of a size and
        a.itemsize > maxTransferBytes

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
    
    If the pdrray or ndarray is of type U, this method is called twice recursively 
    to create the Strings object and the two corresponding pdarrays for string 
    bytes and offsets, respectively.

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
    from arkouda.client import maxTransferBytes
    # If a is not already a numpy.ndarray, convert it
    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except:
            raise TypeError(('a must be a pdarray, np.ndarray, or convertible to' +
                            ' a numpy array'))
    # Only rank 1 arrays currently supported
    if a.ndim != 1:
        raise RuntimeError("Only rank-1 pdarrays or ndarrays supported")
    # Check if array of strings
    if a.dtype.kind == 'U':
        encoded = np.array([elem.encode() for elem in a])
        # Length of each string, plus null byte terminator
        lengths = np.array([len(elem) for elem in encoded]) + 1
        # Compute zero-up segment offsets
        offsets = np.cumsum(lengths) - lengths
        # Allocate and fill bytes array with string segments
        nbytes = offsets[-1] + lengths[-1]
        if nbytes > maxTransferBytes:
            raise RuntimeError(("Creating pdarray would require transferring {} bytes," +
                                " which exceeds allowed transfer size. Increase " +
                                "ak.maxTransferBytes to force.").format(nbytes))
        values = np.zeros(nbytes, dtype=np.uint8)
        for s, o in zip(encoded, offsets):
            for i, b in enumerate(s):
                values[o+i] = b
        # Recurse to create pdarrays for offsets and values, then return Strings object
        return Strings(array(offsets), array(values))
    # If not strings, then check that dtype is supported in arkouda
    if a.dtype.name not in DTypes:
        raise RuntimeError("Unhandled dtype {}".format(a.dtype))
    # Do not allow arrays that are too large
    size = a.size
    if (size * a.itemsize) > maxTransferBytes:
        raise RuntimeError(("Array exceeds allowed transfer size. Increase " +
                            "ak.maxTransferBytes to allow"))
    # Pack binary array data into a bytes object with a command header
    # including the dtype and size
    fmt = ">{:n}{}".format(size, structDtypeCodes[a.dtype.name])
    req_msg = "array {} {:n} ".\
                       format(a.dtype.name, size).encode() + struct.pack(fmt, *a)
    rep_msg = generic_msg(req_msg, send_bytes=True)
    return create_pdarray(rep_msg)

def zeros(size : int, dtype : type=np.float64) -> pdarray:
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
        
    Raises
    ------
    TypeError
        Raised if the supplied dtype is not supported or if the size
         parameter is neither an int nor a str that is parseable to an int.

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
    if not np.isscalar(size):
        raise TypeError("size must be a scalar, not {}".\
                                     format(size.__class__.__name__))
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in numericDTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    kind, itemsize = translate_np_dtype(dtype)
    repMsg = generic_msg("create {} {}".format(dtype.name, size))
    return create_pdarray(repMsg)

def ones(size : int, dtype : type=float64) -> pdarray:
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
        
    Raises
    ------
    TypeError
        Raised if the supplied dtype is not supported or if the size
         parameter is neither an int nor a str that is parseable to an int.

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
    if not np.isscalar(size):
        raise TypeError("size must be a scalar, not {}".\
                                            format(size.__class__.__name__))
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in numericDTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    kind, itemsize = translate_np_dtype(dtype)
    repMsg = generic_msg("create {} {}".format(dtype.name, size))
    a = create_pdarray(repMsg)
    a.fill(1)
    return a

@typechecked
def zeros_like(pda : pdarray) -> pdarray:
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
        
    Raises
    ------
    TypeError
        Raised if pda is not a pdarray

    See Also
    --------
    zeros, ones_like
    """
    return zeros(pda.size, pda.dtype)

@typechecked
def ones_like(pda : pdarray) -> pdarray:
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
        
    Raises
    ------
        Raised if pda is not a pdarray

    See Also
    --------
    ones, zeros_like
    """
    return ones(pda.size, pda.dtype)

def arange(*args) -> pdarray:
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
        
    Raises
    ------
    TypeError
        Raised if start, stop, or stride is not an int object
    ZeroDivisionError
        Raised if stride == 0

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

    if not all((np.isscalar(start), np.isscalar(stop), np.isscalar(stride))):
        raise TypeError("all arguments must be scalars")

    if stride == 0:
        raise ZeroDivisionError("division by zero")

    if isinstance(start, int) and isinstance(stop, int) and isinstance(stride, int):
        # TO DO: fix bug in server that goes 2 steps too far for negative strides
        if stride < 0:
            stop = stop + 2
        repMsg = generic_msg("arange {} {} {}".format(start, stop, stride))
        return create_pdarray(repMsg)
    else:
        raise TypeError("start,stop,stride must be type int {} {} {}".\
                                    format(start,stop,stride))

def linspace(start : int, stop : int, length : int) -> pdarray:
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
        
    Raises
    ------
    TypeError
        Raised if start or stop is not a scalar or if length is not int

    See Also
    --------
    arange

    Examples
    --------
    >>> ak.linspace(0, 1, 5)
    array([0, 0.25, 0.5, 0.75, 1])
    """
    if not all((np.isscalar(start), np.isscalar(stop), np.isscalar(length))):
        raise TypeError("all arguments must be scalars")

    starttype = resolve_scalar_dtype(start)

    try: 
        startstr = NUMBER_FORMAT_STRINGS[starttype].format(start)
    except KeyError as ke:
        raise TypeError(('The start parameter must be an int or a scalar that'  +
                        ' can be parsed to an int, but is a {}'.format(ke)))
    stoptype = resolve_scalar_dtype(stop)

    try: 
        stopstr = NUMBER_FORMAT_STRINGS[stoptype].format(stop)
    except KeyError as ke:
        raise TypeError(('The stop parameter must be an int or a scalar that'  +
                        ' can be parsed to an int, but is a {}'.format(ke)))

    lentype = resolve_scalar_dtype(length)
    if lentype != 'int64':
        raise TypeError("The length parameter must be an int64")

    try: 
        lenstr = NUMBER_FORMAT_STRINGS[lentype].format(length)
    except KeyError as ke:
        raise TypeError(('The length parameter must be an int or a scalar that'  +
                        ' can be parsed to an int, but is a {}'.format(ke)))

    repMsg = generic_msg("linspace {} {} {}".format(startstr, stopstr, lenstr))
    return create_pdarray(repMsg)

def randint(low : Union[int,float], high : Union[int,float], size : int, dtype=int64) -> pdarray:
    """
    Generate a pdarray with random values in a specified range.

    Parameters
    ----------
    low : Union[int,float]
        The low value (inclusive) of the range
    high : Union[int,float]
        The high value (exclusive for int, inclusive for float) of the range
    size : int
        The length of the returned array
    dtype : {int64, float64, bool}
        The dtype of the array

    Returns
    -------
    pdarray
        Values drawn uniformly from the specified range having the desired dtype
        
    Raises
    ------
    TypeError
        Raised if dtype.name not in DTypes, size is not an int, low or if 
        not a scalar
    ValueError
        Raised if size < 0 or if high < low

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
    if not all((np.isscalar(low), np.isscalar(high), np.isscalar(size))):
        raise TypeError("all arguments must be scalars")
    if resolve_scalar_dtype(size) != 'int64':
        raise TypeError("The size parameter must be an integer")
    if resolve_scalar_dtype(low) not in RANDINT_TYPES:
        raise TypeError("The low parameter must be an integer or float")
    if resolve_scalar_dtype(high) not in RANDINT_TYPES:
        raise TypeError("The high parameter must be an integer or float")
    if size < 0 or high < low:
        raise ValueError("size must be > 0 and high > low")
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    lowstr = NUMBER_FORMAT_STRINGS[dtype.name].format(low)
    highstr = NUMBER_FORMAT_STRINGS[dtype.name].format(high)
    sizestr = NUMBER_FORMAT_STRINGS['int64'].format(size)
    repMsg = generic_msg("randint {} {} {} {}".\
                         format(sizestr, dtype.name, lowstr, highstr))
    return create_pdarray(repMsg)

@typechecked
def uniform(size : int, low : float=0.0, high : float=1.0) -> pdarray:
    """
    Generate a pdarray with uniformly distributed random values 
    in a specified range.

    Parameters
    ----------
    low : float
        The low value (inclusive) of the range
    high : float
        The high value (inclusive) of the range
    size : int
        The length of the returned array

    Returns
    -------
    pdarray, float64
        Values drawn uniformly from the specified range

    Raises
    ------
    TypeError
        Raised if dtype.name not in DTypes, size is not an int, or if
        either low or high is not an int or float
    ValueError
        Raised if size < 0 or if high < low

    Examples
    --------
    >>> ak.uniform(3)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])
    """
    return randint(low=low, high=high, size=size, dtype='float64')

    
@typechecked
def standard_normal(size : int) -> pdarray:
    """
    Draw real numbers from the standard normal distribution.

    Parameters
    ----------
    size : int
        The number of samples to draw (size of the returned array)
    
    Returns
    -------
    pdarray, float64
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
    For random samples from :math:`N(\mu, \sigma^2)`, use:

    ``(sigma * standard_normal(size)) + mu``
    """
    if size < 0:
        raise ValueError("The size parameter must be > 0")
    msg = "randomNormal {}".format(NUMBER_FORMAT_STRINGS['int64'].format(size))
    repMsg = generic_msg(msg)
    return create_pdarray(repMsg)

@typechecked
def random_strings_uniform(minlen : int, maxlen : int, size : int, 
                           characters : str='uppercase') -> Strings:
    """
    Generate random strings with lengths uniformly distributed between 
    minlen and maxlen, and with characters drawn from a specified set.

    Parameters
    ----------
    minlen : int
        The minimum allowed length of string
    maxlen : int
        The maximum allowed length of string
    size : int
        The number of strings to generate
    characters : (uppercase, lowercase, numeric, printable, binary)
        The set of characters to draw from

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
    """
    if minlen < 0 or maxlen < minlen or size < 0:
        raise ValueError(("Incompatible arguments: minlen < 0, maxlen < minlen, " +
                          "or size < 0"))
    msg = "randomStrings {} {} {} {} {}".\
                            format(NUMBER_FORMAT_STRINGS['int64'].format(size),
                            "uniform", characters,
                            NUMBER_FORMAT_STRINGS['int64'].format(minlen),
                            NUMBER_FORMAT_STRINGS['int64'].format(maxlen))
    repMsg = generic_msg(msg)
    return Strings(*(repMsg.split('+')))

@typechecked
def random_strings_lognormal(logmean : Union[float, int], logstd : float, 
                             size : int, characters : str='uppercase') -> Strings:
    """
    Generate random strings with log-normally distributed lengths and 
    with characters drawn from a specified set.

    Parameters
    ----------
    logmean : Union[float, int]
        The log-mean of the length distribution
    logstd : float
        The log-standard-deviation of the length distribution
    size : int
        The number of strings to generate
    characters : (uppercase, lowercase, numeric, printable, binary)
        The set of characters to draw from

    Returns
    -------
    Strings
        The array of random strings
    
    Raises
    ------
    TypeError
        Raised if logmean is not a float or int, logstd is not a float, 
        size is not an int, or if characters is not a str
    ValueError
        Raised if logstd <= 0 or size < 0

    See Also
    --------
    random_strings_lognormal, randint

    Notes
    -----
    The lengths of the generated strings are distributed $Lognormal(\mu, \sigma^2)$,
    with :math:`\mu = logmean` and :math:`\sigma = logstd`. Thus, the strings will
    have an average length of :math:`exp(\mu + 0.5*\sigma^2)`, a minimum length of 
    zero, and a heavy tail towards longer strings.
    """
    if not isinstance(logmean, float) and not isinstance(logmean, int):
        raise TypeError("The logmean must be a float or int")
    if not isinstance(logstd, float):
        raise TypeError("The logstd must be a float")
    if not isinstance(size, int):
        raise TypeError("The size must be an integer")
    if not isinstance(characters, str):
        raise TypeError("characters must be a str")
    if logstd <= 0 or size < 0:
        raise ValueError("Incompatible arguments: logstd <= 0 or size < 0")
    msg = "randomStrings {} {} {} {} {}".\
                             format(NUMBER_FORMAT_STRINGS['int64'].format(size),
                             "lognormal", characters,
                             NUMBER_FORMAT_STRINGS['float64'].format(logmean),
                             NUMBER_FORMAT_STRINGS['float64'].format(logstd))
    repMsg = generic_msg(msg)
    return Strings(*(repMsg.split('+')))
