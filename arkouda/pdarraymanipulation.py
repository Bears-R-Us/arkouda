from typing import Tuple, List, Literal, Union, Optional
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray

import numpy as np

__all__ = ["vstack"]


@typechecked
def vstack(
    tup: Union[Tuple[pdarray], List[pdarray]],
    *,
    dtype: Optional[Union[type, str]] = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> pdarray:
    """
    Stack a sequence of arrays vertically (row-wise).

    This is equivalent to concatenation along the first axis after 1-D arrays of
    shape `(N,)` have been reshaped to `(1,N)`.

    Parameters
    ----------
    tup : Tuple[pdarray]
        The arrays to be stacked
    dtype : Optional[Union[type, str]], optional
        The data-type of the output array. If not provided, the output
        array will be determined using `np.common_type` on the
        input arrays Defaults to None
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"], optional
        Controls what kind of data casting may occur - currently unused

    Returns
    -------

    pdarray
        The stacked array
    """

    if casting != "same_kind":
        # TODO: wasn't clear from the docs what each of the casting options does
        raise NotImplementedError(f"casting={casting} is not yet supported")

    # ensure all arrays have the same number of dimensions
    ndim = tup[0].ndim
    for a in tup:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions")

    # establish the dtype of the output array
    if dtype is None:
        dtype_ = np.common_type([a.dtype for a in tup])
    else:
        dtype_ = dtype

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dtype_) if a.dtype != dtype_ else a for a in tup]

    # stack the arrays along the first axis
    return create_pdarray(
        generic_msg(
            cmd=f"stack{ndim}D",
            args={
                "names": list(arrays),
                "n": len(arrays),
                "axis": 0,
            },
        )
    )
