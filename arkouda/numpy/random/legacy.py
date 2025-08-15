from typing import Optional, Tuple, Union, cast

from typeguard import typechecked

from arkouda.numpy.dtypes import NUMBER_FORMAT_STRINGS, DTypes, int_scalars, numeric_scalars
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.random.generator import default_rng

__all__ = [
    "choice",
    "exponential",
    "integers",
    "logistic",
    "lognormal",
    "normal",
    "permutation",
    "poisson",
    "random",
    "randint",
    "seed",
    "shuffle",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "uniform",
]


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
    >>> import arkouda as ak
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
    from arkouda.client import generic_msg

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

    from arkouda.numpy.dtypes import isSupportedFloat

    if dtype == akint64:
        if isSupportedFloat(low):
            low = int(low)
        if isSupportedFloat(high):
            high = int(high)

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
    size: Union[int_scalars, Tuple[int_scalars, ...]],
    seed: Union[None, int_scalars] = None,
) -> pdarray:
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
    array([-0.68586185091150265, 1.1723810583573375, 0.567584107142031])
    """
    from arkouda.client import generic_msg

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
    >>> ak.uniform(3)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])

    >>> ak.uniform(size=3,low=0,high=5,seed=0)
    array([0.30013431967121934, 0.47383036230759112, 1.0441791878997098])
    """
    return randint(low=low, high=high, size=size, dtype="float64", seed=seed)


#   In the experimental stuff below, there is a global object called theGenerator
#   This is what I use in order to implement a global seed.

#   These functions will be called as:
#      ak.random.seed(the_seed_value), and
#      ak.random.integers(lower_limit, upper_limit, how_many_to_make)
#      etc.


def defaultGeneratorExists():  # used in all of the fns below to determine
    global theGenerator  # noqa: F824
    try:
        theGenerator  # this will succeed if the object exists, and fail if not
    except NameError:
        return False
    else:
        return True


def seed(seed=None):
    # reseeding always causes the destruction of an existing generator, because
    # there is no way to reseed a chapel randomStream.  So if there is no existing
    # global generator, we create one with the seed, otherwise we destroy it and
    # make a new one with the seed.
    global theGenerator
    if defaultGeneratorExists():
        del theGenerator  # not strictly necessary, but I have plans for this
    theGenerator = default_rng(seed)


#   All of the functions below are called as ak.random.function_name.  They
#   pass their arguments to the appropriate function method in theGenerator.


def integers(low=0, high=10, size=5):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.integers(low, high, size)


def choice(a, size=None, replace=True, p=None):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.choice(a, size, replace, p)


def exponential(scale=1.0, size=None, method="zig"):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.exponential(scale, size, method)


def standard_exponential(size=None, method="zig"):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.standard_exponential(size, method)


def logistic(loc=0.0, scale=0.0, size=None):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.logistic(loc, scale, size)


def lognormal(mean=0.0, sigma=1.0, size=None, method="zig"):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.lognormal(mean, sigma, size, method)


def normal(loc=0.0, scale=1.0, size=None, method="zig"):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.normal(loc, scale, size, method)


def random(size=None):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.random(size)


def standard_gamma(shape, size=None):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.standard_gamma(shape, size)


def shuffle(x, method="FisherYates"):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.shuffle(x, method)


def permutation(x, method="Argsort"):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.permutation(x, method)


def poisson(lam=1.0, size=None):
    if not defaultGeneratorExists():
        seed()
    return theGenerator.poisson(lam, size)

#   I implemented the two below, before realize they're already defined above.
#   So for now, these are commented out.  I think that long-term, the ones with
#   the global seed are the ones we'll keep.  But for now, for going through drafts,
#   I'm sticking to functions that don't already exist in this file.

# def standard_normal(shape, size=None, method="zig"):
#    if not defaultGeneratorExists():
#        seed()
#    return theGenerator.standard_normal(size, method)


# def uniform(low=0.0, high=1.0, size=None):
#    if not defaultGeneratorExists():
#        seed()
#    return theGenerator.uniform(low, high, size)
