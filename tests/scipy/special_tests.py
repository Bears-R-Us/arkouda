import numpy as np
import pytest
from scipy.special import xlogy as scipy_xlogy

import arkouda as ak
from arkouda.scipy.special import xlogy
from arkouda.client import get_max_array_rank, get_array_ranks


#  bumpup is useful for multi-dimensional testing.
#  Given an array of shape(m1,m2,...m), it will broadcast it to shape (2,m1,m2,...,m).

def bumpup (a) :
    if a.ndim == 1 :
        blob = (2,a.size)
    else :
        blob = list(a.shape)
        blob.insert(0,2)
        blob = tuple(blob)
    return np.broadcast_to(a,blob)

class TestStats:
    @pytest.mark.parametrize("np_x", [3, 5, np.float64(6), np.array([1.0, 2.0, 4.5])])
    @pytest.mark.parametrize(
        "np_y", [np.array([1, 2, 3]), np.array([10, 100, 100]), np.array([-1, 0, np.nan])]
    )
    def test_xlogy(self, np_x, np_y):
        x = ak.array(np_x) if isinstance(np_x, np.ndarray) else np_x
        y = ak.array(np_y)

        ak_result = xlogy(x, y)
        scipy_result = scipy_xlogy(np_x, np_y)

        assert np.allclose(ak_result.to_ndarray(), scipy_result, equal_nan=True)

    # for the potential multi-dimensional case, create higher-dim versions of the above
    # pairs.
        if get_max_array_rank() > 1 :
            for n in range(2,get_max_array_rank()) :
                np_x = np.ascontiguousarray(bumpup(np_x)) # contiguous is needed for the
                np_y = np.ascontiguousarray(bumpup(np_y)) # conversion to pdarrays below
                x = ak.array(np_x)
                y = ak.array(np_y) 
        # Note the the "bumpup" is done whether or not this rank is in get_array_ranks
        # so that the rank at each iteration will be correct.
        # But the test is only applied for ranks that are in get_array_ranks
                if n in get_array_ranks() :
                    ak_result = xlogy(x, y, ddof=ddof, lambda_=lambda_)
                    scipy_result = scipy_xlogy(np_x, np_y, ddof=ddof, axis=0, lambda_=lambda_)

                    assert np.allclose(ak_result, scipy_result, equal_nan=True)

