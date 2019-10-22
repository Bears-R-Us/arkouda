import numpy as np

from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, resolve_scalar_dtype, create_pdarray
from arkouda.pdarraysetops import unique

__all__ = ["abs", "log", "exp", "cumsum", "cumprod", "sin", "cos",
           "any", "all", "is_sorted", "sum", "prod", "min", "max",
           "argmin", "argmax", "mean", "var", "std", "where",
           "histogram", "value_counts"]

def parse_single_value(msg):
    """
    Attempt to convert a scalar return value from the arkouda server to a numpy
    scalar in Python. The user should not call this function directly.
    """
    dtname, value = msg.split()
    dtype = np.dtype(dtname)
    if dtype == np.bool:
        if value == "True":
            return np.bool(True)
        elif value == "False":
            return np.bool(False)
        else:
            raise ValueError("unsupported value from server {} {}".format(dtype.name, value))
    try:
        return dtype.type(value)
    except:
        raise ValueError("unsupported value from server {} {}".format(dtype.name, value))

def abs(pda):
    """
    Return the element-wise absolute value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("abs", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def log(pda):
    """
    Return the element-wise natural log of the array. 

    Notes
    -----
    Logarithms with other bases can be computed as follows:

    >>> A = ak.array([1, 10, 100])
    # Natural log
    >>> ak.log(A)
    array([0, 2.3025850929940459, 4.6051701859880918])
    # Log base 10
    >>> ak.log(A) / np.log(10)
    array([0, 1, 2])
    # Log base 2
    >>> ak.log(A) / np.log(2)
    array([0, 3.3219280948873626, 6.6438561897747253])
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("log", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def exp(pda):
    """
    Return the element-wise exponential of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("exp", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cumsum(pda):
    """
    Return the cumulative sum over the array. 

    The sum is inclusive, such that the ``i`` th element of the 
    result is the sum of elements up to and including ``i``.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("cumsum", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cumprod(pda):
    """
    Return the cumulative product over the array. 

    The product is inclusive, such that the ``i`` th element of the 
    result is the product of elements up to and including ``i``.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("cumprod", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sin(pda):
    """
    Return the element-wise sine of the array.
    """
    if isinstance(pda,pdarray):
        repMsg = generic_msg("efunc {} {}".format("sin",pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cos(pda):
    """
    Return the element-wise cosine of the array.
    """
    if isinstance(pda,pdarray):
        repMsg = generic_msg("efunc {} {}".format("cos",pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def any(pda):
    """
    Return True iff any element of the array evaluates to True.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("any", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def all(pda):
    """
    Return True iff all elements of the array evaluate to True.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("all", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def is_sorted(pda):
    """
    Return True iff the array is monotonically non-decreasing.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("is_sorted", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sum(pda):
    """
    Return the sum of all elements in the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("sum", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def prod(pda):
    """
    Return the product of all elements in the array. Return value is
    always a float.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("prod", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def min(pda):
    """
    Return the minimum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("min", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def max(pda):
    """
    Return the maximum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("max", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def argmin(pda):
    """
    Return the index of the first minimum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmin", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmax(pda):
    """
    Return the index of the first maximum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmax", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def mean(pda):
    """
    Return the mean of the array.
    """
    return pda.sum() / pda.size

def var(pda, ddof=0):
    """
    Return the variance of values in the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to find the variance
    ddof : int
        "Delta Degrees of Freedom" used in calculating mean

    Returns
    -------
    float
        The scalar variance of the array

    See Also
    --------
    mean, std

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean((x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.
    """
    if not isinstance(pda, pdarray):
        raise TypeError("must be pdarray {}".format(pda))
    if ddof >= pda.size:
        raise ValueError("var: ddof must be less than number of values")
    m = mean(pda)
    return ((pda - m)**2).sum() / (pda.size - ddof)

def std(pda, ddof=0):
    """
    Return the standard deviation of values in the array. The standard
    deviation is implemented as the square root of the variance.

    Parameters
    ----------
    pda : pdarray
        values for which to find the variance
    ddof : int
        "Delta Degrees of Freedom" used in calculating mean

    Returns
    -------
    float
        The scalar standard deviation of the array

    See Also
    --------
    mean, var

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e., ``std = sqrt(mean((x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
    the divisor ``N - ddof`` is used instead. In standard statistical
    practice, ``ddof=1`` provides an unbiased estimator of the variance
    of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The
    standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.
    """
    return np.sqrt(var(pda, ddof=ddof))
    
def where(condition, A, B):
    """
    Return an array with elements chosen from A and B based on a conditioning array.
    
    Parameters
    ----------
    condition : pdarray
        Used to choose values from A or B
    A : scalar or pdarray
        Value(s) used when condition is True
    B : scalar or pdarray
        Value(s) used when condition is False

    Returns
    -------
    pdarray
        Values chosen from A and B according to condition

    Notes
    -----
    A and B must have the same dtype.
    """
    if not isinstance(condition, pdarray):
        raise TypeError("must be pdarray {}".format(condition))
    if isinstance(A, pdarray) and isinstance(B, pdarray):
        repMsg = generic_msg("efunc3vv {} {} {} {}".format("where",
                                                           condition.name,
                                                           A.name,
                                                           B.name))
    # For scalars, try to convert it to the array's dtype
    elif isinstance(A, pdarray) and np.isscalar(B):
        repMsg = generic_msg("efunc3vs {} {} {} {} {}".format("where",
                                                              condition.name,
                                                              A.name,
                                                              A.dtype.name,
                                                              A.format_other(B)))
    elif isinstance(B, pdarray) and np.isscalar(A):
        repMsg = generic_msg("efunc3sv {} {} {} {} {}".format("where",
                                                              condition.name,
                                                              B.dtype.name,
                                                              B.format_other(A),
                                                              B.name))
    elif np.isscalar(A) and np.isscalar(B):
        # Scalars must share a common dtype (or be cast)
        dtA = resolve_scalar_dtype(A)
        dtB = resolve_scalar_dtype(B)
        # Make sure at least one of the dtypes is supported
        if not (dtA in DTypes or dtB in DTypes):
            raise TypeError("Not implemented for scalar types {} and {}".format(dtA, dtB))
        # If the dtypes are the same, do not cast
        if dtA == dtB:
            dt = dtA
        # If the dtypes are different, try casting one direction then the other
        elif dtB in DTypes and np.can_cast(A, dtB):
            A = np.dtype(dtB).type(A)
            dt = dtB
        elif dtA in DTypes and np.can_cast(B, dtA):
            B = np.dtype(dtA).type(B)
            dt = dtA
        # Cannot safely cast
        else:
            raise TypeError("Cannot cast between scalars {} and {} to supported dtype".format(A, B))
        repMsg = generic_msg("efunc3ss {} {} {} {} {} {}".format("where",
                                                                 condition.name,
                                                                 dt,
                                                                 A,
                                                                 dt,
                                                                 B))
    return create_pdarray(repMsg)


def histogram(pda, bins=10):
    """
    Compute a histogram of evenly spaced bins over the range of an array.
    
    Parameters
    ----------
    pda : pdarray
        The values to histogram

    bins : int
        The number of equal-size bins to use (default: 10)

    Returns
    -------
    pdarray
        The number of values present in each bin

    See Also
    --------
    value_counts

    Notes
    -----
    The bins are evenly spaced in the interval [pda.min(), pda.max()]. Currently,
    the user must re-compute the bin edges, e.g. with np.linspace (see below) 
    in order to plot the histogram.

    Examples
    --------
    >>> A = ak.arange(0, 10, 1)
    >>> nbins = 3
    >>> h = ak.histogram(A, bins=nbins)
    >>> h
    array([3, 3, 4])
    # Recreate the bin edges in NumPy
    >>> binEdges = np.linspace(A.min(), A.max(), nbins+1)
    >>> binEdges
    array([0., 3., 6., 9.])
    # To plot, use only the left edges, and export the histogram to NumPy
    >>> plt.plot(binEdges[:-1], h.to_ndarray())
    """
    if isinstance(pda, pdarray) and isinstance(bins, int):
        repMsg = generic_msg("histogram {} {}".format(pda.name, bins))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} and bins must be an int {}".format(pda,bins))


def value_counts(pda):
    """
    Count the occurrences of the unique values of an array.

    Parameters
    ----------
    pda : pdarray, int64
        The array of values to count

    Returns
    -------
    unique_values : pdarray, int64
        The unique values, sorted in ascending order

    counts : pdarray, int64
        The number of times the corresponding unique value occurs

    See Also
    --------
    unique, histogram

    Notes
    -----
    This function differs from ``histogram()`` in that it only returns counts 
    for values that are present, leaving out empty "bins".

    Examples
    --------
    >>> A = ak.array([2, 0, 2, 4, 0, 0])
    >>> ak.value_counts(A)
    (array([0, 2, 4]), array([3, 2, 1]))
    """
    return unique(pda, return_counts=True)
