# flake8: noqa

import warnings

warnings.warn(
    "arkouda.join is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.join instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.join import compute_join_size, gen_ranges, join_on_eq_with_dt
