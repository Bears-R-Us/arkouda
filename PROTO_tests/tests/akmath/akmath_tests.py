import math

import numpy as np

import arkouda as ak
from arkouda.math import xlogy
from arkouda.pdarrayclass import pdarray


class TestStats:
    def test_xlogy(self):
        from scipy.special import xlogy as scipy_xlogy

        ys = [ak.array([1, 2, 3]), ak.array([10, 100, 100]), ak.array([-1, 0, np.nan])]
        xs = [3, 5, np.float64(6), ak.array([1.0, 2.0, 4.5])]

        for y in ys:
            for x in xs:
                ak_result = xlogy(x, y)

                np_y = y.to_ndarray()
                np_x = x
                if isinstance(np_x, pdarray):
                    np_x = np_x.to_ndarray()

                scipy_result = scipy_xlogy(np_x, np_y)

                for i in range(len(ak_result)):
                    if math.isnan(ak_result[i]):
                        assert math.isnan(scipy_result[i])
                    elif math.isinf(ak_result[i]):
                        assert math.isinf(scipy_result[i])
                    else:
                        assert abs(ak_result[i] - scipy_result[i]) < 0.1 / 10**6
