# flake8: noqa

import warnings

warnings.warn(
    "arkouda.sorting is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.sorting instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.sorting import SortingAlgorithm, argsort, coargsort, searchsorted, sort
