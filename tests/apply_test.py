import pytest

import arkouda as ak

"""
Encapsulates a variety of arkouda apply test cases.
"""


class TestApply:

    @classmethod
    def setup_class(cls):
        cls.size = 1000

    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_inc(self, dtype):
        a = ak.arange(TestApply.size, dtype=dtype)
        b = ak.apply(a, "lambda x,: x+1")
        assert ak.all(b == a + 1)
        assert b.dtype == dtype

    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_cube(self, dtype):
        a = ak.arange(TestApply.size, dtype=dtype)
        b = ak.apply(a, "lambda x,: x*x*x")
        assert ak.all(b == a * a * a)
        assert b.dtype == dtype

