"""
Support arkouda arrays in numpy.polynomial:
 https://numpy.org/doc/stable/reference/routines.polynomials.html
"""

from numpy.polynomial import set_default_printstyle
from .polynomial import Polynomial

__all__ = [
    "set_default_printstyle",
    "polynomial", "Polynomial",
]
