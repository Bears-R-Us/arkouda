import operator

import numpy as np
import pandas as pd
import pytest

from arkouda.pandas.extension import ArkoudaArray


# ---- helpers ---------------------------------------------------------------


def _ea(values, dtype="ak_int64"):
    # construct the ExtensionArray via pandas dtype resolution
    return pd.array(values, dtype=dtype)


def _np(values, dtype=np.int64):
    return np.array(values, dtype=dtype)


class TestArkoudaArrayExtensionArithmeticOps:
    # ---- core: dunder dispatch exists ------------------------------------------

    def test_dunder_add_exists_on_ea_type(self):
        x = _ea([1, 2, 3], dtype="ak_int64")
        assert hasattr(type(x), "__add__")
        assert hasattr(type(x), "__radd__")

    @pytest.mark.parametrize(
        "op, np_op",
        [
            (operator.add, operator.add),
            (operator.sub, operator.sub),
            (operator.mul, operator.mul),
            (operator.truediv, operator.truediv),
            (operator.floordiv, operator.floordiv),
            (operator.mod, operator.mod),
            (operator.pow, operator.pow),
        ],
    )
    def test_binary_ops_ea_ea_dispatch_and_values_int64(self, op, np_op):
        x = _ea([1, 2, 3], dtype="ak_int64")
        y = _ea([10, 20, 30], dtype="ak_int64")

        # IMPORTANT: exercise Python operator dispatch (dunder), not _arith_method directly
        out = op(x, y)

        assert isinstance(out, ArkoudaArray)

        expected = np_op(_np([1, 2, 3]), _np([10, 20, 30]))
        # pandas may produce float for true division; your EA currently converts via to_numpy
        np.testing.assert_allclose(out.to_numpy(), expected)

    @pytest.mark.parametrize(
        "op, np_op",
        [
            (operator.add, operator.add),
            (operator.sub, operator.sub),
            (operator.mul, operator.mul),
            (operator.truediv, operator.truediv),
            (operator.floordiv, operator.floordiv),
            (operator.mod, operator.mod),
            (operator.pow, operator.pow),
        ],
    )
    def test_binary_ops_ea_scalar_dispatch_and_values_int64(self, op, np_op):
        x = _ea([1, 2, 3], dtype="ak_int64")
        s = 5

        out = op(x, s)
        assert isinstance(out, ArkoudaArray)

        expected = np_op(_np([1, 2, 3]), s)
        np.testing.assert_allclose(out.to_numpy(), expected)

    @pytest.mark.parametrize(
        "op, np_op",
        [
            (operator.add, operator.add),
            (operator.sub, operator.sub),
            (operator.mul, operator.mul),
            (operator.truediv, operator.truediv),
            (operator.floordiv, operator.floordiv),
            (operator.mod, operator.mod),
            (operator.pow, operator.pow),
        ],
    )
    def test_reflected_ops_scalar_ea_dispatch_and_values_int64(self, op, np_op):
        x = _ea([1, 2, 3], dtype="ak_int64")
        s = 5

        # reflected: scalar op EA
        out = op(s, x)
        assert isinstance(out, ArkoudaArray)

        expected = np_op(s, _np([1, 2, 3]))
        np.testing.assert_allclose(out.to_numpy(), expected)

    # ---- dtype coverage: float64 ------------------------------------------------

    def test_add_float64(self):
        x = _ea([1.5, 2.5, 3.0], dtype="ak_float64")
        y = _ea([10.0, 20.0, 30.0], dtype="ak_float64")

        out = x + y
        assert isinstance(out, ArkoudaArray)
        np.testing.assert_allclose(out.to_numpy(), _np([11.5, 22.5, 33.0], dtype=np.float64))

    def test_truediv_int64_produces_expected_values(self):
        # This test is intentionally about values; dtype may be float depending on backend behavior.
        x = _ea([1, 2, 3], dtype="ak_int64")
        y = _ea([2, 2, 2], dtype="ak_int64")

        out = x / y
        np.testing.assert_allclose(out.to_numpy(), np.array([0.5, 1.0, 1.5]))

    # ---- error paths ------------------------------------------------------------

    def test_add_incompatible_type_raises_typeerror(self):
        x = _ea([1, 2, 3], dtype="ak_int64")

        # choose something your _arith_method rejects (non-scalar, non-EA)
        class Weird:
            pass

        with pytest.raises(TypeError):
            _ = x + Weird()

    def test_add_length_mismatch_raises(self):
        x = _ea([1, 2, 3], dtype="ak_int64")
        y = _ea([10, 20], dtype="ak_int64")

        # depending on arkouda/pandas behavior, this could be ValueError or something else;
        # but it should definitely not silently succeed.
        with pytest.raises(ValueError):
            _ = x + y
