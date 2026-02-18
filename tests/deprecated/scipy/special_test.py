import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.scipy.special import xlogy
from arkouda.numpy.pdarrayclass import pdarray


class StatsTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)

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

                assert np.allclose(ak_result.to_ndarray(), scipy_result, equal_nan=True)
