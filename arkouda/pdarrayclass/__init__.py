# flake8: noqa

import warnings

warnings.warn(
    "arkouda.pdarrayclass is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.pdarrayclass instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.pdarrayclass import (
    RegistrationError,
    all,
    allclose,
    any,
    argmax,
    argmaxk,
    argmin,
    argmink,
    clear,
    clz,
    corr,
    cov,
    ctz,
    divmod,
    dot,
    fmod,
    is_sorted,
    logical_not,
    max,
    maxk,
    mean,
    min,
    mink,
    mod,
    parity,
    pdarray,
    popcount,
    power,
    prod,
    rotl,
    rotr,
    sqrt,
    std,
    sum,
    var,
)
