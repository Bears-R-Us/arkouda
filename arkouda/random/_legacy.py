from typing import Optional, Tuple, Union, cast

from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.numpy.dtypes import NUMBER_FORMAT_STRINGS, DTypes
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import int_scalars, numeric_scalars
from arkouda.pdarrayclass import create_pdarray, pdarray


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
    size : int_scalars
        The length of the returned array
    dtype : Union[int64, float64, bool]
        The dtype of the array
    seed : int_scalars, optional
        Seed to allow for reproducible random number generation


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
    >>> ak.randint(0, 10, 5)
    array([5, 7, 4, 8, 3])

    >>> ak.randint(0, 1, 3, dtype=ak.float64)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])

    >>> ak.randint(0, 1, 5, dtype=ak.bool_)
    array([True, False, True, True, True])

    >>> ak.randint(1, 5, 10, seed=2)
    array([4, 3, 1, 3, 4, 4, 2, 4, 3, 2])

    >>> ak.randint(1, 5, 3, dtype=ak.float64, seed=2)
    array([2.9160772326374946, 4.353429832157099, 4.5392023718621486])

    >>> ak.randint(1, 5, 10, dtype=ak.bool, seed=2)
    array([False, True, True, True, True, False, True, True, True, True])
    """
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

    if full_size < 0 or ndim < 1 or high < low:
        raise ValueError("size must be >= 0, ndim >= 1, and high >= low")
    dtype = akdtype(dtype)  # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError(f"unsupported dtype {dtype}")

    repMsg = generic_msg(
        cmd=f"randint<{dtype.name},{ndim}>",
        args={
            "shape": shape,
            "low": NUMBER_FORMAT_STRINGS[dtype.name].format(low),
            "high": NUMBER_FORMAT_STRINGS[dtype.name].format(high),
            "seed": seed if seed is not None else -1,
        },
    )
    return create_pdarray(repMsg)


@typechecked
def standard_normal(
    size: Union[int_scalars, Tuple[int_scalars, ...]], seed: Union[None, int_scalars] = None
) -> pdarray:
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
    For random samples from :math:`N(\\mu, \\sigma^2)`, use:

    ``(sigma * standard_normal(size)) + mu``

    Examples
    --------
    >>> ak.standard_normal(3,1)
    array([-0.68586185091150265, 1.1723810583573375, 0.567584107142031])
    """
    shape: Union[int_scalars, Tuple[int_scalars, ...]] = 1
    if isinstance(size, tuple):
        shape = cast(Tuple, size)
        full_size = 1
        for s in cast(Tuple, shape):
            full_size *= s
        ndim = len(shape)
    else:
        full_size = cast(int, size)
        if full_size < 0:
            raise ValueError("The size parameter must be > 0")
        shape = full_size
        ndim = 1
    return create_pdarray(
        generic_msg(
            cmd=f"randomNormal<{ndim}>",
            args={"shape": shape, "seed": seed},
        )
    )


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
    pdarray, float64
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
    >>> ak.uniform(3)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])

    >>> ak.uniform(size=3,low=0,high=5,seed=0)
    array([0.30013431967121934, 0.47383036230759112, 1.0441791878997098])
    """
    return randint(low=low, high=high, size=size, dtype="float64", seed=seed)
