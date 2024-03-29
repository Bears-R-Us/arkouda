# flake8: noqa
# mypy: ignore-errors
from _typeshed import Incomplete


from scipy.stats._distn_infrastructure import rv_continuous


class chi2(rv_continuous):
    r'''
    A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



    '''

    def _argcheck(self, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _argcheck_rvs(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _attach_argparser_methods(self, ):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _attach_methods(self, ):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _cdf(self, x, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _cdf_single(self, x, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _cdfvec(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _construct_argparser(self, meths_to_inspect, locscale_in, locscale_out):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _construct_default_doc(self, longname=None, docdict=None, discrete='continuous'):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _construct_doc(self, docdict, shapes_vals=None):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _ctor_param(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _entropy(self, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _fit_loc_scale_support(self, data, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _fitstart(self, data, args=None):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _get_support(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _isf(self, p, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _logcdf(self, x, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _logpdf(self, x, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _logpxf(self, x, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _logsf(self, x, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _mom0_sc(self, m, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _mom1_sc(self, m, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _mom_integ0(self, x, m, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _mom_integ1(self, q, m, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _moment_error(self, theta, x, data_moments):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _munp(self, n, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _nlff_and_penalty(self, x, args, log_fitfun):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _nnlf(self, x, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _nnlf_and_penalty(self, x, args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _open_support_mask(self, x, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _param_info(self, ):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _parse_arg_template(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _parse_args(self, df, loc=0, scale=1):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _parse_args_rvs(self, df, loc=0, scale=1, size=None):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _parse_args_stats(self, df, loc=0, scale=1, moments='mv'):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _pdf(self, x, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _penalized_nlpsf(self, theta, x):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _penalized_nnlf(self, theta, x):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _ppf(self, p, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _ppf_single(self, q, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _ppf_to_solve(self, x, q, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _ppfvec(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _random_state(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _reduce_func(self, args, kwds, data=None):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _rvs(self, df, size=None, random_state=None):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _sf(self, x, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _shape_info(self, ):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _stats(self, df):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _stats_has_moments(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _support_mask(self, x, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _unpack_loc_scale(self, theta):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def _updated_ctor_param(self, ):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def a(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def b(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def badvalue(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def cdf(self, x, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def entropy(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def expect(self, func=None, args=(self, ), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def fit(self, data, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def fit_loc_scale(self, data, *args):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def freeze(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def generic_moment(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def interval(self, confidence, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def isf(self, q, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def logcdf(self, x, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def logpdf(self, x, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def logsf(self, x, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def mean(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def median(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def moment(self, order, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def moment_type(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def name(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def nnlf(self, theta, x):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def numargs(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def pdf(self, x, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def ppf(self, q, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def random_state(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def rvs(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def sf(self, x, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def shapes(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def stats(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def std(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def support(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def var(self, *args, **kwds):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def vecentropy(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...

    def xtol(self, *args, **kwargs):
        r'''
        A chi-squared continuous random variable.

        For the noncentral chi-square distribution, see `ncx2`.

        As an instance of the `rv_continuous` class, `chi2` object inherits from it
        a collection of generic methods (see below for the full list),
        and completes them with details specific for this particular distribution.

        Methods
        -------
        rvs(df, loc=0, scale=1, size=1, random_state=None)
            Random variates.
        pdf(x, df, loc=0, scale=1)
            Probability density function.
        logpdf(x, df, loc=0, scale=1)
            Log of the probability density function.
        cdf(x, df, loc=0, scale=1)
            Cumulative distribution function.
        logcdf(x, df, loc=0, scale=1)
            Log of the cumulative distribution function.
        sf(x, df, loc=0, scale=1)
            Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
        logsf(x, df, loc=0, scale=1)
            Log of the survival function.
        ppf(q, df, loc=0, scale=1)
            Percent point function (inverse of ``cdf`` --- percentiles).
        isf(q, df, loc=0, scale=1)
            Inverse survival function (inverse of ``sf``).
        moment(order, df, loc=0, scale=1)
            Non-central moment of the specified order.
        stats(df, loc=0, scale=1, moments='mv')
            Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
        entropy(df, loc=0, scale=1)
            (Differential) entropy of the RV.
        fit(data)
            Parameter estimates for generic data.
            See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
            keyword arguments.
        expect(func, args=(df,), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
            Expected value of a function (of one argument) with respect to the distribution.
        median(df, loc=0, scale=1)
            Median of the distribution.
        mean(df, loc=0, scale=1)
            Mean of the distribution.
        var(df, loc=0, scale=1)
            Variance of the distribution.
        std(df, loc=0, scale=1)
            Standard deviation of the distribution.
        interval(confidence, df, loc=0, scale=1)
            Confidence interval with equal areas around the median.

        See Also
        --------
        ncx2

        Notes
        -----
        The probability density function for `chi2` is:

        .. math::

            f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                       x^{k/2-1} \exp \left( -x/2 \right)

        for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
        in the implementation).

        `chi2` takes ``df`` as a shape parameter.

        The chi-squared distribution is a special case of the gamma
        distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
        ``scale = 2``.

        The probability density above is defined in the "standardized" form. To shift
        and/or scale the distribution use the ``loc`` and ``scale`` parameters.
        Specifically, ``chi2.pdf(x, df, loc, scale)`` is identically
        equivalent to ``chi2.pdf(y, df) / scale`` with
        ``y = (x - loc) / scale``. Note that shifting the location of a distribution
        does not make it a "noncentral" distribution; noncentral generalizations of
        some distributions are available in separate classes.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import chi2
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)

        Calculate the first four moments:

        >>> df = 55
        >>> mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

        Display the probability density function (``pdf``):

        >>> x = np.linspace(chi2.ppf(0.01, df),
        ...                 chi2.ppf(0.99, df), 100)
        >>> ax.plot(x, chi2.pdf(x, df),
        ...        'r-', lw=5, alpha=0.6, label='chi2 pdf')

        Alternatively, the distribution object can be called (as a function)
        to fix the shape, location and scale parameters. This returns a "frozen"
        RV object holding the given parameters fixed.

        Freeze the distribution and display the frozen ``pdf``:

        >>> rv = chi2(df)
        >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

        Check accuracy of ``cdf`` and ``ppf``:

        >>> vals = chi2.ppf([0.001, 0.5, 0.999], df)
        >>> np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))
        True

        Generate random numbers:

        >>> r = chi2.rvs(df, size=1000)

        And compare the histogram:

        >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        >>> ax.set_xlim([x[0], x[-1]])
        >>> ax.legend(loc='best', frameon=False)
        >>> plt.show()



        '''
        ...
