import numpy as np
import numpy.random as np_random

from arkouda.numpy.dtypes import _val_isinstance_of_union
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import dtype as to_numpy_dtype
from arkouda.numpy.dtypes import float64 as akfloat64
from arkouda.numpy.dtypes import float_scalars
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import int_scalars, numeric_scalars
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray

__all__ = [
    "Generator",
    "default_rng",
    "float_array_or_scalar_helper",
]


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

    name_dict: dict
        Dictionary mapping the server side names associated with
        the generators for each dtype.

    state: int
        The current state we are in the random number generation stream.
        This information makes it so calls to any dtype generator
        function affects the stream of random numbers for the other generators.
        This mimics the behavior we see in numpy

    See Also
    --------
    default_rng : Recommended constructor for `Generator`.
    """

    def __init__(self, name_dict=None, seed=None, state=1):
        self._seed = seed
        self._np_generator = np_random.default_rng(seed)
        self._name_dict = name_dict
        self._state = state

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        _str = self.__class__.__name__
        # be sure to update if we add support for non-pcg generators
        _str += "(PCG64)"
        return _str

    def choice(self, a, size=None, replace=True, p=None):
        """
        Generate a randomly sample from a.

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
            p is the probabilities or weights associated with each element of a

        Returns
        -------
        pdarray, numeric_scalar
            A pdarray containing the sampled values or a single random value if size not provided.
        """
        from arkouda.client import generic_msg

        if size is None:
            ret_scalar = True
            size = 1
        else:
            ret_scalar = False

        from arkouda.numpy import cast as akcast

        if _val_isinstance_of_union(a, int_scalars):
            is_domain = True
            dtype = to_numpy_dtype(akint64)
            pop_size = a
        elif isinstance(a, pdarray):
            is_domain = False
            dtype = to_numpy_dtype(a.dtype)
            pop_size = a.size
        else:
            raise TypeError("choice only accepts a pdarray or int scalar.")

        if not replace and size > pop_size:
            raise ValueError("Cannot take a larger sample than population when replace is False")

        has_weights = p is not None
        if has_weights:
            if not isinstance(p, pdarray):
                raise TypeError("weights must be a pdarray")
            if p.dtype != akfloat64:
                p = akcast(p, akfloat64)
        else:
            p = ""

        # weighted sample requires float and non-weighted uses int
        name = self._name_dict[to_numpy_dtype(akfloat64 if has_weights else akint64)]

        rep_msg = generic_msg(
            cmd=f"choice<{dtype.name}>",
            args={
                "gName": name,
                "aName": a,
                "wName": p,
                "numSamples": size,
                "replace": replace,
                "hasWeights": has_weights,
                "isDom": is_domain,
                "popSize": pop_size,
                "state": self._state,
            },
        )
        # for the non-weighted domain case we pull pop_size numbers from the generator.
        # for other cases we may be more than the numbers drawn, but that's okay. The important
        # thing is not repeating any positions in the state.
        self._state += pop_size

        pda = create_pdarray(rep_msg)
        return pda if not ret_scalar else pda[0]

    def exponential(self, scale=1.0, size=None, method="zig"):
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
        pdarray
            Drawn samples from the parameterized exponential distribution.
        """
        _, scale = float_array_or_scalar_helper("exponential", "scale", scale, size)
        if (scale < 0).any() if isinstance(scale, pdarray) else scale < 0:
            raise TypeError("scale must be non-negative.")
        return scale * self.standard_exponential(size, method=method)

    def standard_exponential(self, size=None, method="zig"):
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
        pdarray
            Drawn samples from the standard exponential distribution.
        """
        from arkouda.client import generic_msg
        from arkouda.numpy.util import _infer_shape_from_size

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.standard_exponential(method=method)

        shape, ndim, full_size = _infer_shape_from_size(size)
        if full_size < 0:
            raise ValueError("The size parameter must be > 0")

        rep_msg = generic_msg(
            cmd=f"standardExponential<{ndim}>",
            args={
                "name": self._name_dict[akdtype("float64")],
                "size": shape,
                "method": method.upper(),
                "has_seed": self._seed is not None,
                "state": self._state,
            },
        )
        self._state += full_size if method.upper() == "INV" else 1
        return create_pdarray(rep_msg)

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
        >>> import arkouda as ak
        >>> rng = ak.random.default_rng()
        >>> rng.integers(5, 20, 10)
        array([15, 13, 10, 8, 5, 18, 16, 14, 7, 13])  # random
        >>> rng.integers(5, size=10)
        array([2, 4, 0, 0, 0, 3, 1, 5, 5, 3])  # random
        """
        from arkouda.client import generic_msg
        from arkouda.numpy.util import _infer_shape_from_size

        # normalize dtype so things like "int" will work
        dtype = to_numpy_dtype(dtype)

        if dtype is akfloat64:
            raise TypeError("Unsupported dtype dtype('float64') for integers")

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.integers(low=low, high=high, dtype=dtype, endpoint=endpoint)

        if high is None:
            high = low
            low = 0
        elif not endpoint:
            high = high - 1

        shape, ndim, full_size = _infer_shape_from_size(size)
        if full_size < 0:
            raise ValueError("The size parameter must be > 0")

        rep_msg = generic_msg(
            cmd=f"uniformGenerator<{dtype.name},{ndim}>",
            args={
                "name": self._name_dict[dtype],
                "low": low,
                "high": high,
                "shape": shape,
                "state": self._state,
            },
        )
        self._state += full_size
        return create_pdarray(rep_msg)

    def logistic(self, loc=0.0, scale=1.0, size=None):
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
        pdarray
            Pdarray of floats (unless size=None, in which case a single float is returned).

        See Also
        --------
        normal

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.random.default_rng(17).logistic(3, 2.5, 3)
        array([1.1319566682702642 -7.1665150633720014 7.7208667145173608])
        """
        from arkouda.client import generic_msg

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.logistic(loc=loc, scale=scale, size=size)

        is_single_mu, mu = float_array_or_scalar_helper("logistic", "loc", loc, size)
        is_single_scale, scale = float_array_or_scalar_helper("logistic", "scale", scale, size)
        if (scale < 0).any() if isinstance(scale, pdarray) else scale < 0:
            raise TypeError("scale must be non-negative.")

        rep_msg = generic_msg(
            cmd="logisticGenerator",
            args={
                "name": self._name_dict[akdtype("float64")],
                "mu": mu,
                "is_single_mu": is_single_mu,
                "scale": scale,
                "is_single_scale": is_single_scale,
                "size": size,
                "has_seed": self._seed is not None,
                "state": self._state,
            },
        )
        # we only generate one val using the generator in the symbol table
        self._state += 1
        return create_pdarray(rep_msg)

    def lognormal(self, mean=0.0, sigma=1.0, size=None, method="zig"):
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
        pdarray
            Pdarray of floats (unless size=None, in which case a single float is returned).

        See Also
        --------
        normal

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.random.default_rng(17).lognormal(3, 2.5, 3)
        array([7.3866978126031091 106.20159494048757 4.5424399190667666])
        """
        from arkouda.numpy import exp

        norm_arr = self.normal(loc=mean, scale=sigma, size=size, method=method)
        return exp(norm_arr) if size is not None else np.exp(norm_arr)

    def normal(self, loc=0.0, scale=1.0, size=None, method="zig"):
        r"""
        Draw samples from a normal (Gaussian) distribution

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
        pdarray
            Pdarray of floats (unless size=None, in which case a single float is returned).

        See Also
        --------
        standard_normal
        uniform

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.random.default_rng(17).normal(3, 2.5, 3)
        array([2.3673425816523577 4.0532529435624589 2.0598322696795694])
        """
        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.standard_normal(loc, scale)

        if (scale < 0).any() if isinstance(scale, pdarray) else scale < 0:
            raise TypeError("scale must be non-negative.")

        return loc + scale * self.standard_normal(size=size, method=method)

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
        >>> import arkouda as ak
        >>> rng = ak.random.default_rng()
        >>> rng.random()
        0.47108547995356098 # random
        >>> rng.random(3)
        array([0.055256829926011691, 0.62511314008006458, 0.16400145561571539]) # random
        """
        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.random()
        return self.uniform(size=size)

    def standard_gamma(self, shape, size=None):
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
        pdarray
            Pdarray of floats (unless size=None, in which case a single float is returned).

        Notes
        -----
        The probability density function for the Gamma distribution is

        .. math::
            p(x) = x^{k-1}\frac{e^{\frac{-x}{\theta}}}{\theta^k\Gamma(k)}

        Examples
        --------
        >>> import arkouda as ak
        >>> rng = ak.random.default_rng()
        >>> rng.standard_gamma(1)
        0.8729704388729135 # random
        >>> rng.standard_gamma(1, size=3)
        array([0.4879818539586227 0.6534654349920751 0.40990997253631162]) # random
        """  # noqa: W605
        from arkouda.client import generic_msg
        from arkouda.numpy.util import _infer_shape_from_size

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.standard_gamma(shape=shape)

        # rename shape to avoid conflict, it's also referred to as k
        # in the numpy doc string
        is_single_k, k_arg = float_array_or_scalar_helper("gamma", "shape", shape, size)

        shape, ndim, full_size = _infer_shape_from_size(size)
        if full_size < 0:
            raise ValueError("The size parameter must be > 0")

        rep_msg = generic_msg(
            cmd=f"standardGamma<{ndim}>",
            args={
                "name": self._name_dict[akdtype("float64")],
                "size": shape,
                "is_single_k": is_single_k,
                "k_arg": k_arg,
                "has_seed": self._seed is not None,
                "state": self._state,
            },
        )
        self._state += 1
        return create_pdarray(rep_msg)

    def standard_normal(self, size=None, method="zig"):
        r"""
        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size: numeric_scalars, optional
            Output shape. Default is None, in which case a single value is returned.

        method : str, optional
            Either 'box' or 'zig'. 'box' uses the Box–Muller transform
            'zig' uses the Ziggurat method.

        Returns
        -------
        pdarray
            Pdarray of floats (unless size=None, in which case a single float is returned).

        Notes
        -----
        For random samples from :math:`N(\mu, \sigma^2)`, either call `normal` or do:

        .. math::
            (\sigma \cdot \texttt{standard_normal}(size)) + \mu

        See Also
        --------
        normal

        Examples
        --------
        >>> import arkouda as ak
        >>> rng = ak.random.default_rng()
        >>> rng.standard_normal()
        2.1923875335537315 # random
        >>> rng.standard_normal(3)
        array([0.8797352989638163, -0.7085325853376141, 0.021728052940979934])  # random
        """
        from arkouda.client import generic_msg
        from arkouda.numpy.util import _infer_shape_from_size

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.standard_normal()

        shape, ndim, full_size = _infer_shape_from_size(size)
        if full_size < 0:
            raise ValueError("The size parameter must be > 0")

        rep_msg = generic_msg(
            cmd=f"standardNormalGenerator<{ndim}>",
            args={
                "name": self._name_dict[akdtype("float64")],
                "shape": shape,
                "method": method.upper(),
                "has_seed": self._seed is not None,
                "state": self._state,
            },
        )
        # since we generate 2*size uniform samples for box-muller transform
        self._state += full_size * 2
        return create_pdarray(rep_msg)

    def shuffle(self, x):
        """
        Randomly shuffle a pdarray in place.

        Parameters
        ----------
        x: pdarray
            shuffle the elements of x randomly in place

        """
        from arkouda.client import generic_msg

        if not isinstance(x, pdarray):
            raise TypeError("shuffle only accepts a pdarray.")
        dtype = to_numpy_dtype(x.dtype)
        name = self._name_dict[to_numpy_dtype(akint64)]
        ndim = len(x.shape)
        generic_msg(
            cmd=f"shuffle<{dtype.name},{ndim}>",
            args={
                "name": name,
                "x": x,
                "shape": x.shape,
                "state": self._state,
            },
        )
        self._state += x.size

    def permutation(self, x, method="Argsort"):
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
        """
        from arkouda.client import generic_msg

        if _val_isinstance_of_union(x, int_scalars):
            is_domain_perm = True
            dtype = to_numpy_dtype(akint64)
            shape = x
            size = x
            ndim = 1
        elif isinstance(x, pdarray):
            is_domain_perm = False
            dtype = to_numpy_dtype(x.dtype)
            shape = x.shape
            size = x.size
            ndim = len(shape)
        else:
            raise TypeError("permutation only accepts a pdarray or int scalar.")

        if method.lower() == "fisheryates":
            # we have to use the int version since we permute the domain
            name = self._name_dict[to_numpy_dtype(akint64)]
            rep_msg = generic_msg(
                cmd=f"permutation<{dtype.name},{ndim}>",
                args={
                    "name": name,
                    "x": x,
                    "shape": shape,
                    "size": size,
                    "isDomPerm": is_domain_perm,
                    "state": self._state,
                },
            )
            self._state += size
            return create_pdarray(rep_msg)
        elif method.lower() == "argsort":
            from arkouda.numpy.sorting import argsort

            perm = argsort(self.random(size))
            if is_domain_perm:
                return perm
            else:
                return x[perm]
        else:
            raise ValueError("method did not match allowed values: Serial, Argsort")

    def poisson(self, lam=1.0, size=None):
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
        pdarray
            Pdarray of ints (unless size=None, in which case a single int is returned).

        Examples
        --------
        >>> import arkouda as ak
        >>> rng = ak.random.default_rng()
        >>> rng.poisson(lam=3, size=5)
        array([5 3 2 2 3])  # random
        """
        from arkouda.client import generic_msg

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.poisson(lam, size)

        is_single_lambda, lam = float_array_or_scalar_helper("poisson", "lam", lam, size)
        if (lam < 0).any() if isinstance(lam, pdarray) else lam < 0:
            raise TypeError("lam must be non-negative.")

        rep_msg = generic_msg(
            cmd="poissonGenerator",
            args={
                "name": self._name_dict[akdtype("float64")],
                "lam": lam,
                "is_single_lambda": is_single_lambda,
                "size": size,
                "has_seed": self._seed is not None,
                "state": self._state,
            },
        )
        # we only generate one val using the generator in the symbol table
        self._state += 1
        return create_pdarray(rep_msg)

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
        >>> import arkouda as ak
        >>> rng = ak.random.default_rng()
        >>> rng.uniform(-1, 1, 3)
        array([0.030785499755523249, 0.08505865366367038, -0.38552048588998722])  # random
        """
        from arkouda.client import generic_msg
        from arkouda.numpy.util import _infer_shape_from_size

        if size is None:
            # delegate to numpy when return size is 1
            return self._np_generator.uniform(low=low, high=high)

        shape, ndim, full_size = _infer_shape_from_size(size)
        if full_size < 0:
            raise ValueError("The size parameter must be > 0")

        dt = akdtype("float64")
        rep_msg = generic_msg(
            cmd=f"uniformGenerator<{dt.name},{ndim}>",
            args={
                "name": self._name_dict[dt],
                "low": low,
                "high": high,
                "shape": shape,
                "state": self._state,
            },
        )
        self._state += full_size
        return create_pdarray(rep_msg)


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
    from arkouda.client import generic_msg

    if isinstance(seed, Generator):
        # Pass through the generator
        return seed
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True

    state = 1
    # chpl has to know the type of the generator, in order to avoid having to declare
    # the type of the generator beforehand (which is not what numpy does)
    # we declare a generator for each type and fast-forward the state

    name_dict = dict()
    for dt in akdtype("int64"), akdtype("uint64"), akdtype("float64"), akdtype("bool"):
        name_dict[dt] = generic_msg(
            cmd=f"createGenerator<{dt.name}>",
            args={"has_seed": has_seed, "seed": seed, "state": state},
        ).split()[1]

    return Generator(name_dict, seed if has_seed else None, state=state)


def float_array_or_scalar_helper(func_name, var_name, var, size):
    if _val_isinstance_of_union(var, numeric_scalars):
        is_scalar = True
        if not _val_isinstance_of_union(var, float_scalars):
            var = float(var)
    elif isinstance(var, pdarray):
        is_scalar = False
        if size != var.size:
            raise TypeError(f"array of {var_name} must have same size as return size")
        if var.dtype != akfloat64:
            from arkouda.numpy import cast as akcast

            var = akcast(var, akfloat64)
    else:
        raise TypeError(f"{func_name} only accepts a pdarray or float scalar for {var_name}")
    return is_scalar, var
