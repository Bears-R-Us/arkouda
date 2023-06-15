import arkouda as ak
import numpy as np
import pytest


class TestExample2:
    def test_one(self):
        a = ak.arange(10)
        b = np.arange(10)
        assert a.to_list() == b.tolist()

    def test_two(self):
        a = [
            ak.array([0, 1, 2, 3]),
            ak.array([5, 6]),
            ak.array([7]),
            ak.array([8, 9, 10, 12])
        ]
        sa = ak.SegArray.from_multi_array(a)
        assert sa.to_list() == [x.to_list() for x in a]