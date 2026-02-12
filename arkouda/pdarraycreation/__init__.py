# flake8: noqa

import warnings

warnings.warn(
    "arkouda.pdarraycreation is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.pdarraycreation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.pdarraycreation import (
    arange,
    array,
    bigint_from_uint_arrays,
    full,
    full_like,
    linspace,
    ones,
    ones_like,
    promote_to_common_dtype,
    randint,
    random_strings_lognormal,
    random_strings_uniform,
    scalar_array,
    standard_normal,
    uniform,
    zeros,
    zeros_like,
)
