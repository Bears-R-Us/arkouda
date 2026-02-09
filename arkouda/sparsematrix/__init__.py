# flake8: noqa

import warnings

warnings.warn(
    "arkouda.sparsematrix is deprecated and will be removed in a future release. "
    "Please use arkouda.scipy.sparsematrix instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.scipy.sparsematrix import (
    create_sparse_matrix,
    random_sparse_matrix,
    sparse_matrix_matrix_mult,
)
