import numpy.polynomial.polyutils as nputils

__all__ = [
    'mapdomain',
]


def mapdomain(x, old, new):
    """Similar to numpy.polynomial.polyutils.mapdomain, with two
       changes: no forced conversion to an ndarray, and optimizations
       for common special cases (esp. the default parameters.
    """

    off, scl = nputils.mapparms(old, new)
    if off == 0 and scl == 1.0:
        return x

    if off == 0:
        return scl*x

    if scl == 1.0:
        return off + x

    return off + scl*x

