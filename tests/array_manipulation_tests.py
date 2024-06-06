import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak
import pytest


class ArrayManipulationTests(ArkoudaTest):
    @pytest.mark.skipif(ArkoudaTest.ndim < 2, reason="vstack requires server with 'max_array_dims' >= 2")
    def test_vstack(self):
        a = [ak.random.randint(0, 10, 25) for _ in range(4)]
        n = [x.to_ndarray() for x in a]

        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        self.assertListEqual(n_vstack.tolist(), a_vstack.to_list())
