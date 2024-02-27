import numpy.random as np_random

from arkouda.dtypes import dtype as to_numpy_dtype
from arkouda.dtypes import int64 as akint64


class Generator:
    """
    ``Generator`` exposes a number of methods for generating random
    numbers drawn from a variety of probability distributions. In addition to
    the distribution-specific arguments, each method takes a keyword argument
    `size` that defaults to ``None``. If `size` is ``None``, then a single
    value is generated and returned. If `size` is an integer, then a 1-D
    array filled with generated values is returned.

    Parameters
    ----------
    seed : int
        Seed to allow for reproducible random number generation.

    See Also
    --------
    default_rng : Recommended constructor for `Generator`.
    """

    def __init__(self, seed=None):
        self._seed = seed
        self._np_generator = np_random.default_rng(seed)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        _str = self.__class__.__name__
        # be sure to update if we add support for non-pcg generators
        _str += "(PCG64)"
        return _str

    def integers(self, low, high=None, size=None, dtype=akint64, endpoint=False):
        """
        Return random integers from low (inclusive) to high (exclusive),
        or if endpoint=True, low (inclusive) to high (inclusive).

        Return random integers from the “discrete uniform” distribution of the specified dtype.
        If high is None (the default), then results are from 0 to low.

        Parameters
        ----------
        low: numeric_scalars
            Lowest (signed) integers to be drawn from the distribution (unless high=None,
            in which case this parameter is 0 and this value is used for high).

        high: numeric_scalars
            If provided, one above the largest (signed) integer to be drawn from the distribution
            (see above for behavior if high=None)

        size: numeric_scalars
            Output shape. Default is None, in which case a single value is returned.

        dtype: dtype, optional
            Desired dtype of the result. The default value is ak.int64.

        endpoint: bool, optional
            If true, sample from the interval [low, high] instead of the default [low, high).
            Defaults to False

        Returns
        -------
        pdarray, numeric_scalar
            Values drawn uniformly from the specified range having the desired dtype,
            or a single such random int if size not provided.

        Examples
        --------
        >>> rng = ak.random.default_rng()
        >>> rng.integers(5, 20, 10)
        array([15, 13, 10, 8, 5, 18, 16, 14, 7, 13])  # random
        >>> rng.integers(5, size=10)
        array([2, 4, 0, 0, 0, 3, 1, 5, 5, 3])  # random
        """
        from arkouda.random._legacy import randint

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.integers(
                low=low, high=high, dtype=to_numpy_dtype(dtype), endpoint=endpoint
            )
        if high is None:
            high = low + 1
            low = 0
        elif endpoint:
            high = high + 1
        return randint(low=low, high=high, size=size, dtype=dtype, seed=self._seed)

    def random(self, size=None):
        """
        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the uniform distribution over the stated interval.

        Parameters
        ----------
        size: numeric_scalars, optional
            Output shape. Default is None, in which case a single value is returned.

        Returns
        -------
        pdarray
            Pdarray of random floats (unless size=None, in which case a single float is returned).

        Notes
        -----
        To sample over `[a,b)`, use uniform or multiply the output of random by `(b - a)` and add `a`:

         ``(b - a) * random() + a``

        See Also
        --------
        uniform

        Examples
        --------
        >>> rng = ak.random.default_rng()
        >>> rng.random()
        0.47108547995356098 # random
        >>> rng.random(3)
        array([0.055256829926011691, 0.62511314008006458, 0.16400145561571539]) # random
        """
        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.random()
        return self.uniform(low=0.0, high=1.0, size=size)

    def standard_normal(self, size=None):
        """
        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size: numeric_scalars, optional
            Output shape. Default is None, in which case a single value is returned.

        Returns
        -------
        pdarray
            Pdarray of floats (unless size=None, in which case a single float is returned).

        Notes
        -----
        For random samples from :math:`N(\\mu, \\sigma^2)`, use:

        ``(sigma * standard_normal(size)) + mu``


        Examples
        --------
        >>> rng = ak.random.default_rng()
        >>> rng.standard_normal()
        2.1923875335537315 # random
        >>> rng.standard_normal(3)
        array([0.8797352989638163, -0.7085325853376141, 0.021728052940979934])  # random
        """
        from arkouda.random._legacy import standard_normal

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.standard_normal()
        return standard_normal(size=size, seed=self._seed)

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval [low, high).
        In other words, any value within the given interval is equally likely to be drawn by uniform.

        Parameters
        ----------
        low: float, optional
            Lower boundary of the output interval. All values generated will be greater than or
            equal to low. The default value is 0.

        high: float, optional
            Upper boundary of the output interval. All values generated will be less than high.
            high must be greater than or equal to low. The default value is 1.0.

        size: numeric_scalars, optional
            Output shape. Default is None, in which case a single value is returned.

        Returns
        -------
        pdarray
            Pdarray of floats (unless size=None, in which case a single float is returned).

        See Also
        --------
        integers
        random

        Examples
        --------
        >>> rng = ak.random.default_rng()
        >>> rng.uniform(-1, 1, 3)
        array([0.030785499755523249, 0.08505865366367038, -0.38552048588998722])  # random
        """
        from arkouda.random._legacy import uniform

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.uniform(low=low, high=high)
        return uniform(low=low, high=high, size=size, seed=self._seed)


def default_rng(seed=None):
    """
    Construct a new Generator.

    Right now we only support PCG64, since this is what is available in chapel.

    Parameters
    ----------
    seed: {None, int, Generator}, optional
        A seed to initialize the `Generator`. If None, then the seed will
        be generated by chapel in an implementation specific manner based on the current time.
        This behavior is currently unstable and may change in the future. If an int,
        then the value must be non-negative. If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    Generator
        The initialized generator object.
    """
    if isinstance(seed, Generator):
        # Pass through a Generator.
        return seed
    return Generator(seed)
