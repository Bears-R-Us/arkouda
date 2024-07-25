import arkouda as ak
import numpy as np
import numpy.polynomial as nppoly
from . import polyutils as pu

__all__ = [
    'polyval',
    'Polynomial'
]


def polyval(x, c, tensor=True):
    if not isinstance(x, ak.pdarray):
        return nppoly.polynomial.polyval(x, c, tensor)

    # TODO: once Polynomial support in Numba is out of the experimental stage,
    # it should be possible to send a nppoly.Polynomial(c) directly instead

    # calling `nppoly.polynomial.polyval(x, c, tensor)` directly in `poly` below
    # works fine, but it means executing the following one-off code inside the
    # loop, so for efficiency reasons, the code from numpy's `polyval` is copied:
    c = np.array(c, ndmin=1, copy=False)  # uses copy=None in more recent numpy
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1:
        return ak.ones_like(x)*c[0]
    else:
        # generic
        def poly(x):
            c0 = c[-1] #+ x*0
            for i in range(2, len(c) + 1):
                c0 = c[-i] + c0*x
            return c0

    return ak.for_each(x, poly)


class Polynomial(nppoly.Polynomial):
    _val = staticmethod(polyval)

    def __call__(self, arg):
        arg = pu.mapdomain(arg, self.domain, self.window)
        return self._val(arg, self.coef)

