import pytest

import arkouda as ak

"""
Encapsulates a variety of arkouda apply test cases.
"""



def supports_apply():
    try:
        arr = ak.arange(1)
        ak.apply(arr, lambda x: x)
        return True
    except RuntimeError as e:
        if "Python module not supported with this version of Chapel" in str(e):
            return False
        else:
            raise e

class TestApply:

    @classmethod
    def setup_class(cls):
        if not supports_apply():
            pytest.skip("apply not supported")
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

    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_lambda(self, dtype):
        a = ak.arange(TestApply.size, dtype=dtype)
        b = ak.apply(a, lambda x: x*x*x)
        assert ak.all(b == a * a * a)
        assert b.dtype == dtype

    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_times_pi(self, dtype):
        import math
        def times_pi(x):
            return x*math.pi

        a = ak.arange(TestApply.size, dtype=dtype)
        b = ak.apply(a, times_pi, "float64")

        res = ak.cast(a, "float64") * math.pi

        assert ak.all(b == res)
        assert b.dtype == "float64"

