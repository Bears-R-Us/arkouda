import sys

import pytest

import arkouda as ak

"""
Encapsulates a variety of arkouda apply test cases.
"""


def supports_apply():
    return ak.numpy.pdarrayclass.parse_single_value(ak.client.generic_msg("isPythonModuleSupported"))


class TestApply:
    def test_apply_docstrings(self):
        import doctest

        apply_module = sys.modules["arkouda.apply"]
        result = doctest.testmod(
            apply_module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @classmethod
    def setup_class(cls):
        if not supports_apply():
            pytest.skip("apply not supported")

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_inc(self, prob_size, dtype):
        a = ak.arange(prob_size, dtype=dtype)
        b = ak.apply(a, "lambda x,: x+1")
        assert ak.all(b == a + 1)
        assert b.dtype == dtype

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_cube(self, prob_size, dtype):
        a = ak.arange(prob_size, dtype=dtype)
        b = ak.apply(a, "lambda x,: x*x*x")
        assert ak.all(b == a * a * a)
        assert b.dtype == dtype

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_lambda(self, prob_size, dtype):
        a = ak.arange(prob_size, dtype=dtype)
        b = ak.apply(a, lambda x: x * x * x)
        assert ak.all(b == a * a * a)
        assert b.dtype == dtype

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_times_pi(self, prob_size, dtype):
        import math

        def times_pi(x):
            return x * math.pi

        a = ak.arange(prob_size, dtype=dtype)
        b = ak.apply(a, times_pi, "float64")

        res = ak.cast(a, "float64") * math.pi

        assert ak.all(b == res)
        assert b.dtype == "float64"

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_apply_shapes(self, prob_size, dtype):
        for r in ak.client.get_array_ranks():
            size = int(prob_size ** (1 / r))
            shape = (size,) * r
            a = ak.randint(1, 100, shape, dtype)
            b = ak.apply(a, lambda x: x**4 - 1)
            assert ak.all(b == a**4 - 1)
            assert b.dtype == dtype
