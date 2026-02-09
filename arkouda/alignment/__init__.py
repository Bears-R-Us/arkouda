# flake8: noqa
import warnings

warnings.warn(
    "arkouda.alignment is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.alignment instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.alignment import (
    NonUniqueError,
    align,
    find,
    in1d_intervals,
    interval_lookup,
    is_cosorted,
    left_align,
    lookup,
    right_align,
    search_intervals,
    unsqueeze,
    zero_up,
)
