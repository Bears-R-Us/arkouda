import arkouda as ak
import pytest
import numpy as np

NUMERIC_TYPES = [ak.int64, ak.float64, ak.bool, ak.uint64]

class NumericTest:
    def _get_seed(self):
        return

    @pytest.mark.parametrize("numeric_type", NUMERIC_TYPES)
    def test_seeded_rng(self, numeric_type):
        seed = pytest.seed if pytest.seed is not None else 8675309

        # Make sure unseeded runs differ
        a = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=numeric_type)
        b = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=numeric_type)
        assert not (a == b).all()
        # self.assertFalse((a == b).all())
        # Make sure seeded results are same
        a = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=numeric_type, seed=seed)
        b = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=numeric_type, seed=seed)
        assert a.to_list() == b.to_list()
        # self.assertListEqual(a.to_list(), b.to_list())