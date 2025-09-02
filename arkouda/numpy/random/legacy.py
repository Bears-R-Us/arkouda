from typing import Optional, Tuple, Union, cast

from typeguard import typechecked

from arkouda.numpy.dtypes import NUMBER_FORMAT_STRINGS, DTypes
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import float64 as akfloat64
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import int_scalars, numeric_scalars
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.random.generator import default_rng, Generator

__all__ = [
    "choice",
    "exponential",
    "integers",
    "logistic",
    "lognormal",
    "normal",
    "permutation",
    "poisson",
    "rand",
    "random",
    "randint",
    "seed",
    "shuffle",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "uniform",
]

theGenerator : Optional[Generator] = None  # used below to check if generator exists


@typechecked
def rand(*size: int_scalars, seed: Union[None, int_scalars] = None) -> Union[pdarray, akfloat64]:
    """
    Generate a pdarray of float values in the range (0,1).

    Parameters
    ----------
    size : int
        Dimensions of the returned array. Multiple arguments define a shape tuple.

    seed : int_scalars, optional
        The seed for the random number generator

    Returns
    -------
    pdarray, float_scalar
        Values drawn uniformly from the range (0,1).
        Returned as pdarray if size is provided, else as scalar.

    Raises
    ------
    TypeError
        Raised if size is not an int or a sequence of ints, or if seed is not an int

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.rand(3,seed=1701)
    array([0.011410423448327005 0.73618171558685619 0.12367222192448891])
    """
    from arkouda.numpy.util import _infer_shape_from_size

    if not size:  # meaning the tuple is empty, i.e. we are returning a scalar
        return uniform(1, seed=seed)[0]
    else:
        shape, ndim, full_size = _infer_shape_from_size(size)
        if ndim == 1:
            return uniform(full_size, seed=seed)
        else:
            return uniform(full_size, seed=seed).reshape(shape)


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
    dtype : Union[int64, float_scalar, bool]
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


def globalGeneratorExists():
    """
    Used to determine is a generator has already been created.

    Returns
    -------
    boolean
        True if theGenerator is not None
        False if theGenerator is None
    """
    return theGenerator is not None

def getGlobalGenerator() -> Generator:
    """
    Used to simplify the boilerplate code for each function.
    """
    seed() if not globalGeneratorExists() else None

    if theGenerator:
        return theGenerator
    else:
        raise RuntimeError("Default RNG failed to initialize")

def seed(seed=None):
    """
    Implements global seed by seeding theGenerator.

    Parameters
    ----------
    seed : int, None
        the seed for the global generator.  Can be left out.

    Notes
    -----
    Reseeding always causes the destruction of an existing generator, because
    there is no way to reseed a chapel randomStream.
    The existing generator is deleted, though python doesn't require that.
    Python-side, it suffices to create a new generator with the new seed.
    """
    global theGenerator

    if globalGeneratorExists():
        del theGenerator

    theGenerator = default_rng(seed)


#   All of the functions below are called as ak.random.function_name.  They
#   pass their arguments to the appropriate function method in theGenerator.


def integers(low, high=None, size=None, dtype=akint64, endpoint=False):
    """
    Return random integers from range (low,high) if endpoint = True, else (low,high].

    Return random integers from the “discrete uniform” distribution of the specified dtype.
    If high is None (the default), then results are from 0 to low.

    Parameters
    ----------
    low: numeric_scalars
        If low and high are both defined, range is low to high.  But if high is none,
        the range is 0 to low.

    high: numeric_scalars
        If provided, and endpoint is True, the highest value to be included in the range.
        If provided, and endpoint is False, 1 more than the highest value to be included.
        See above for behavior if high=None.

    size: numeric_scalars
        If scalar, size = output size.  If tuple, size = output shape. Default is None,
        in which case a scalar is returned.

    dtype: type
        The type of the output.

    endpoint: boolean
        if True, high is included in the range.  If False, the range ends at high-1.

    Returns
    -------
    pdarray, numeric_scalar
        Values drawn uniformly from the specified range having the desired dtype.
        Returned as pdarray if size is provided, else as scalar.

    Examples
    --------
    >>> ak.random.seed(1)
    >>> ak.random.integers(5, 20, 10)
    array([7 19 5 16 19 11 18 15 10 5])
    >>> ak.random.integers(5, size=10)
    array([5 9 7 7 9 9 7 7 8 6])

    """
    return getGlobalGenerator().integers(low, high, size, dtype, endpoint)


def choice(a, size=None, replace=True, p=None):
    """
    Generate a random sample from a.

    Parameters
    ----------
    a: int or pdarray
        If a is an integer, randomly sample from ak.arange(a).
        If a is a pdarray, randomly sample from a.

    size: int, optional
        Number of elements to be sampled

    replace: bool, optional
        If True, sample with replacement. Otherwise sample without replacement.
        Defaults to True

    p: pdarray, optional
        p is the probabilities or weights associated with each element of a.

    Returns
    -------
    pdarray, numeric_scalar
        Sample or samples from the input a.
        Returned as pdarray if size is provided, else as scalar.

    Examples
    --------
    >>> ak.random.seed(1701)
    >>> ak.random.choice(ak.arange(10),size=5,replace=True)
    array([6 5 1 6 3])
    """
    return getGlobalGenerator().choice(a, size, replace, p)


def exponential(scale=1.0, size=None, method="zig"):
    r"""
    Draw samples from an exponential distribution.

    Its probability density function is

    .. math::
        f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta}),

    for ``x > 0`` and 0 elsewhere. :math:`\beta` is the scale parameter,
    which is the inverse of the rate parameter :math:`\lambda = 1/\beta`.
    The rate parameter is an alternative, widely used parameterization
    of the exponential distribution.

    Parameters
    ----------
    scale: float or pdarray
        The scale parameter, :math:`\beta = 1/\lambda`. Must be
        non-negative. An array must have the same size as the size argument.
    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.
    method : str, optional
        Either 'inv' or 'zig'. 'inv' uses the default inverse CDF method.
        'zig' uses the Ziggurat method.

    Returns
    -------
    pdarray, float_scalar
        Drawn samples from the parameterized exponential distribution.
        Returned as pdarray if size is provided, else as scalar.

    Examples
    --------
    >>> ak.random.seed(1701)
    >>> ak.random.exponential(scale=1.0,size=3,method='zig')
    array([0.35023958744297734 1.3308542074773211 1.819197246298274])

    """
    return getGlobalGenerator().exponential(scale, size, method)


def standard_exponential(size=None, method="zig"):
    """
    Draw samples from the standard exponential distribution.

    `standard_exponential` is identical to the exponential distribution
    with a scale parameter of 1.

    Parameters
    ----------
    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.
    method : str, optional
        Either 'inv' or 'zig'. 'inv' uses the default inverse CDF method.
        'zig' uses the Ziggurat method.

    Returns
    -------
    pdarray, float_scalar
        Drawn samples from the standard exponential distribution.
        Returned as pdarray if size is provided, else as scalar.

    Examples
    --------
    >>> ak.random.seed(5551212)
    >>> ak.random.standard_exponential(size=3,method="zig")
    array([0.0036288331189547511 0.12747464978660919 2.4564938704378503])
    """
    return getGlobalGenerator().standard_exponential(size, method)


def logistic(loc=0.0, scale=0.0, size=None):
    r"""
    Draw samples from a logistic distribution.

    Samples are drawn from a logistic distribution with specified parameters,
    loc (location or mean, also median), and scale (>0).

    Parameters
    ----------
    loc: float or pdarray of floats, optional
        Parameter of the distribution. Default of 0.

    scale: float or pdarray of floats, optional
        Parameter of the distribution. Must be non-negative. Default is 1.

    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.

    Notes
    -----
    The probability density for the Logistic distribution is

    .. math::
       P(x) = \frac{e^{-(x - \mu)/s}}{s( 1 + e^{-(x - \mu)/s})^2}

    where :math:`\mu` is the location and :math:`s` is the scale.

    The Logistic distribution is used in Extreme Value problems where it can act
    as a mixture of Gumbel distributions, in Epidemiology, and by the World Chess Federation (FIDE)
    where it is used in the Elo ranking system, assuming the performance of each player
    is a logistically distributed random variable.

    Returns
    -------
    pdarray, float_scalar
        Samples drawn from a logistic distribution.
        Returned as pdarray if size is provided, else as scalar.

    See Also
    --------
    normal

    Examples
    --------
    >>> ak.random.seed(17)
    >>> ak.random.logistic(3, 2.5, 3)
    array([1.1319566682702642 -7.1665150633720014 7.7208667145173608])
    """
    return getGlobalGenerator().logistic(loc, scale, size)


def lognormal(mean=0.0, sigma=1.0, size=None, method="zig"):
    r"""
    Draw samples from a log-normal distribution with specified mean,
    standard deviation, and array shape.

    Note that the mean and standard deviation are not the values for the distribution itself,
    but of the underlying normal distribution it is derived from.

    Parameters
    ----------
    mean: float or pdarray of floats, optional
        Mean of the distribution. Default of 0.

    sigma: float or pdarray of floats, optional
        Standard deviation of the distribution. Must be non-negative. Default of 1.

    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.

    method : str, optional
        Either 'box' or 'zig'. 'box' uses the Box–Muller transform
        'zig' uses the Ziggurat method.

    Notes
    -----
    A variable `x` has a log-normal distribution if `log(x)` is normally distributed.
    The probability density for the log-normal distribution is:

    .. math::
       p(x) = \frac{1}{\sigma x \sqrt{2\pi}} e^{-\frac{(\ln(x)-\mu)^2}{2\sigma^2}}

    where :math:`\mu` is the mean and :math:`\sigma` the standard deviation of the normally
    distributed logarithm of the variable.
    A log-normal distribution results if a random variable is the product of a
    large number of independent, identically-distributed variables in the same
    way that a normal distribution results if the variable is
    the sum of a large number of independent, identically-distributed variables.

    Returns
    -------
    pdarray, float_scalar
        Samples drawn from a lognormal distribution.
        Returned as pdarray if size is provided, else as scalar.

    See Also
    --------
    normal

    Examples
    --------
    >>> ak.random.seed(17)
    >>> ak.random.lognormal(3, 2.5, 3)
    array([75.587346973566639 9.4194790331678568 1.0996120079897966])

    """
    return getGlobalGenerator().lognormal(mean, sigma, size, method)


def normal(loc=0.0, scale=1.0, size=None, method="zig"):
    r"""
    Draw samples from a normal (Gaussian) distribution.

    Parameters
    ----------
    loc: float or pdarray of floats, optional
        Mean of the distribution. Default of 0.

    scale: float or pdarray of floats, optional
        Standard deviation of the distribution. Must be non-negative. Default of 1.

    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.

    method : str, optional
        Either 'box' or 'zig'. 'box' uses the Box–Muller transform
        'zig' uses the Ziggurat method.

    Notes
    -----
    The probability density for the Gaussian distribution is:

    .. math::
       p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    where :math:`\mu` is the mean and :math:`\sigma` the standard deviation.
    The square of the standard deviation, :math:`\sigma^2`, is called the variance.

    Returns
    -------
    pdarray, float_scalar
        Samples drawn from a normal distribution.
        Returned as pdarray if size is provided, else as scalar.

    See Also
    --------
    standard_normal
    uniform

    Examples
    --------
    >>> ak.random.seed(17)
    >>> ak.random.normal(3, 2.5, 3)
    array([4.3252889011033728 2.2427797827243081 0.09495739757471533])

    """
    return getGlobalGenerator().normal(loc, scale, size, method)


def random(size=None):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    Results are from the uniform distribution over the stated interval.

    Parameters
    ----------
    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.

    Returns
    -------
    pdarray, float_scalar
        Samples drawn from a uniform distribution.
        Returned as pdarray if size is provided, else as scalar.

    Notes
    -----
    To sample over `[a,b)`, use uniform or multiply the output of random by `(b - a)` and add `a`:

     ``(b - a) * random() + a``

    See Also
    --------
    uniform

    Examples
    --------
    >>> ak.random.seed(42)
    >>> ak.random.random()
    0.7739560485559633
    >>> ak.random.random(3)
    array([0.30447083571882388 0.89653821715718895 0.34737575437149532])

    """
    return getGlobalGenerator().random(size)


def standard_gamma(shape, size=None):
    r"""
    Draw samples from a standard gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    shape (sometimes designated “k”) and scale (sometimes designated “theta”),
    where both parameters are > 0.

    Parameters
    ----------
    shape: numeric_scalars
        specified parameter (sometimes designated “k”)
    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.

    Returns
    -------
    pdarray, float_scalar
        Samples drawn from a standard gamma distribution.
        Returned as pdarray if size is provided, else as scalar.

    Notes
    -----
    The probability density function for the Gamma distribution is

    .. math::
        p(x) = x^{k-1}\frac{e^{\frac{-x}{\theta}}}{\theta^k\Gamma(k)}

    Examples
    --------
    >>> ak.random.seed(16309)
    >>> ak.random.standard_gamma(1)
    0.22445153117925773
    >>> ak.random.standard_gamma(1, size=3)
    array([0.85277675774402018 3.1253116338237561 0.95808096440750634])

    """  # noqa: W605
    return getGlobalGenerator().standard_gamma(shape, size)


def shuffle(
    x,
    method: str = "FisherYates",
    *,
    feistel_rounds: int = 16,
    feistel_key: int | None = None,
) :
    """
    Randomly shuffle the elements of a `pdarray` in place.

    This method performs a reproducible in-place shuffle of the array `x`
    using the specified strategy. Three methods are available:

    Parameters
    ----------
    x : pdarray
        The array to be shuffled in place. Must be a one-dimensional Arkouda array.

    method : {"FisherYates","MergeShuffle","Feistel"}, optional
        - "FisherYates": A **serial, global** Fisher–Yates shuffle implemented in Chapel.
          Simple and fast for small/medium arrays, but **not distributed** — the entire
          array must fit on one locale.
        - "MergeShuffle": A **fully distributed** shuffle that combines local randomization
          and cross-locale probabilistic merging. Scales to large datasets and maintains
          good statistical uniformity across locales.
        - "Feistel": A **keyed permutation** of indices via a Feistel PRP over [0, N),
          then applies that permutation to `x`. Works for any `N` (uses cycle-walking
          when N is not a power of two). **Distributed-friendly** and reproducible.
          Not intended for cryptographic security.

        Default is "FisherYates".

    feistel_rounds : int, optional (keyword-only)
        Number of Feistel rounds (default 16). Higher may cost more time.

    feistel_key : int or None, optional (keyword-only)
        64-bit key for the Feistel permutation. If None, the backend should derive
        a key from the RNG stream so results remain deterministic given the client RNG state.

    Raises
    ------
    TypeError
        If `x` is not a `pdarray`.

    ValueError
        If an unsupported shuffle method is specified, or if `feistel_key` is not a 64-bit integer.

    Notes
    -----
    - The shuffle modifies `x` in place.
    - The result is deterministic given the client RNG state.
    - For `"MergeShuffle"`, reproducibility is guaranteed **only if the number of locales
      remains fixed** between runs. Changing locale count will yield different permutations.
    - Use `"FisherYates"` only for small arrays or testing.
    - Use `"MergeShuffle"` for production-scale distributed shuffling.
    - Use `"Feistel"` when you need a keyed, reproducible permutation of indices that
      also scales in distributed settings.

    Examples
    --------
    >>> ak.random.seed(18)
    >>> pda = ak.arange(10)
    >>> pda
    array([0 1 2 3 4 5 6 7 8 9])
    >>> ak.random.shuffle(pda, method="FisherYates")
    >>> pda
    array([0 8 2 7 9 4 6 3 5 1])
    >>> ak.random.shuffle(pda, method="MergeShuffle")
    >>> pda
    array([5 6 9 3 8 2 7 0 4 1])
    >>> ak.random.shuffle(pda, method="Feistel", feistel_rounds=18)
    >>> pda
    array([4 2 1 6 9 3 0 5 8 7])
    >>> ak.random.shuffle(pda, method="Feistel", feistel_key=0x1234_5678_9ABC_DEF0, feistel_rounds=18)
    >>> pda
    array([2 3 6 9 8 5 1 4 7 0])
    """

    getGlobalGenerator().shuffle(
        x, method=method, feistel_rounds=feistel_rounds, feistel_key=feistel_key
    )


def permutation(x, method="Argsort"):
    """
    Randomly permute a sequence, or return a permuted range.

    Parameters
    ----------
    x: int or pdarray
        If x is an integer, randomly permute ak.arange(x). If x is an array,
        make a copy and shuffle the elements randomly.
    method: str = 'Argsort'
        The method for generating the permutation.
        Allowed values: 'FisherYates', 'Argsort'

        If 'Argsort' is selected, the permutation will be generated by
        an argsort performed on randomly generated floats.

    Returns
    -------
    pdarray
        pdarray of permuted elements

    Raises
    ------
    ValueError
        Raised if method is not an allowed value.
    TypeError
        Raised if x is not of type int or pdarray.

    Examples
    --------
    >>> ak.random.seed(1984)
    >>> ak.random.permutation(ak.arange(10))
    array([4 7 0 2 5 3 6 1 8 9])
    """
    return getGlobalGenerator().permutation(x, method)


def poisson(lam=1.0, size=None):
    r"""
    Draw samples from a Poisson distribution.

    The Poisson distribution is the limit of the binomial distribution for large N.

    Parameters
    ----------
    lam: float or pdarray
        Expected number of events occurring in a fixed-time interval, must be >= 0.
        An array must have the same size as the size argument.
    size: numeric_scalars, optional
        Output shape. Default is None, in which case a single value is returned.

    Notes
    -----
    The probability mass function for the Poisson distribution is:

    .. math::
       f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

    For events with an expected separation :math:`\lambda`, the Poisson distribution
    :math:`f(k; \lambda)` describes the probability of :math:`k` events occurring
    within the observed interval :math:`\lambda`

    Returns
    -------
    pdarray, int_scalar
        Samples drawn from a Poisson distribution.
        Returned as pdarray if size is provided, else as scalar.

    Examples
    --------
    >>> ak.random.seed(2525)
    >>> ak.random.poisson(lam=3, size=5)
    array([3 4 3 3 5])

    """
    return getGlobalGenerator().poisson(lam, size)
