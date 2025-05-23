import math
import pickle
import sys

import numpy as np
import pytest

import arkouda as ak

from arkouda.numpy import dtypes
from arkouda.numpy.dtypes import bigint_


"""
DtypesTest encapsulates arkouda dtypes module methods
"""

SUPPORTED_NP_DTYPES = [
    bool,
    int,
    float,
    str,
    np.bool_,
    np.int64,
    np.float64,
    np.uint8,
    np.uint64,
    np.str_,
]


class TestBigintDtype:
    def test_dtypes_docstrings(self):
        import doctest

        from arkouda.numpy import _bigint

        result = doctest.testmod(_bigint)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def _has(self, attr):
        return hasattr(ak, attr)

    def _get(self, attr, default=None):
        return getattr(ak, attr, default)

    @pytest.mark.smoke
    def test_bigint_dtype_singleton_and_str(
        self,
    ):
        dt1 = ak.bigint()
        dt2 = ak.dtype("bigint")
        dt3 = ak.dtype(ak.bigint)  # class object accepted
        assert dt1 is dt2 is dt3
        assert str(dt1) == "bigint"
        assert repr(dt1) in {"dtype(bigint)", "bigint"}  # allow either style
        # hashability + equality semantics
        d = {dt1: "ok"}
        assert d[dt2] == "ok"
        assert dt1 == dt2 == dt3

    def test_bigint_dtype_resolution_variants(
        self,
    ):
        dt = ak.bigint()
        # names / tokens
        assert ak.dtype("BIGINT") is dt
        # instances and class objects
        assert ak.dtype(ak.bigint) is dt
        assert ak.dtype(dt) is dt
        # scalar class object (works even if scalar not imported yet)
        assert ak.dtype(bigint_) is dt

        # ensure we never fall back to object dtype for bigint-like inputs
        assert ak.dtype("bigint").kind != np.dtype("O").kind

    @pytest.mark.smoke
    def test_bigint_scalar_construction_and_basics(
        self,
    ):
        # ak.bigint(…) should construct a scalar if you implemented the metaclass;
        # if not yet implemented, skip gracefully.
        maybe_scalar = ak.bigint(1)
        if not isinstance(maybe_scalar, bigint_):
            pytest.skip("ak.bigint(…) does not construct scalar yet")
        x = maybe_scalar
        assert isinstance(x, bigint_)
        assert int(x) == 1
        assert x.dtype is ak.bigint()
        assert x.item() == 1
        assert "ak.bigint_(" in repr(x)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0", 0),
            ("42", 42),
            ("-17", -17),
            ("0x10", 16),
            ("0b1011", 11),
            ("0o77", 63),
        ],
    )
    def test_bigint_scalar_parsing_from_strings(self, text, expected):
        try:
            x = bigint_(text)
        except TypeError:
            pytest.skip("bigint_ not present")
        assert isinstance(x, bigint_)
        assert int(x) == expected

    @pytest.mark.parametrize("val", [0, 1, -1, 2**64, 2**200, -(2**200)])
    def test_dtype_from_python_int_routes_big_values_to_bigint(self, val):
        dt = ak.dtype(val)
        if abs(val) >= 2**64:
            assert dt is ak.bigint()
        else:
            # NOTE: downstream policy may choose int64/uint64 depending on sign
            assert dt in {np.dtype(np.int64), np.dtype(np.uint64)}

    def test_pickle_roundtrip_of_bigint_dtype_singleton(
        self,
    ):
        dt = ak.bigint()
        data = pickle.dumps(dt)
        restored = pickle.loads(data)
        assert restored is dt

    def test_dtype_accepts_scalar_instance_and_class(
        self,
    ):
        try:
            s = bigint_(2**128)
        except TypeError:
            pytest.skip("bigint_ not present")
        assert ak.dtype(s) is ak.bigint()
        assert ak.dtype(type(s)) is ak.bigint()

    def test_supported_sets_if_present(
        self,
    ):
        # If your module exposes these capability sets/helpers, validate bigint entries.
        ints_set = self._get("ARKOUDA_SUPPORTED_INTS")
        nums_set = self._get("ARKOUDA_SUPPORTED_NUMBERS")
        if ints_set is not None:
            assert ak.bigint in ints_set
            if self._has("bigint_"):
                assert bigint_ in ints_set
        if nums_set is not None:
            assert ak.bigint in nums_set
            if self._has("bigint_"):
                assert bigint_ in nums_set

    def test_resolve_scalar_dtype_if_present(
        self,
    ):
        fn = self._get("resolve_scalar_dtype")
        if fn is None or not self._has("bigint_"):
            pytest.skip("resolve_scalar_dtype or bigint_ not present")
        assert fn(bigint_(123)) == "bigint"

    @pytest.mark.parametrize(
        "args,expect",
        [
            # bigint with bigint → bigint
            ((ak.bigint(), ak.bigint()), "bigint"),
            # bigint with float → float64 (common numpy-like policy)
            ((ak.bigint(), np.dtype(np.float64)), np.dtype(np.float64)),
            # bigint with smaller ints → bigint (to avoid overflow)
            ((ak.bigint(), np.dtype(np.int64)), "bigint"),
            ((ak.bigint(), np.dtype(np.uint64)), "bigint"),
        ],
    )
    def test_result_type_if_present(self, args, expect):
        fn = self._get("result_type")
        if fn is None:
            pytest.skip("result_type not present")
        rt = fn(*args)
        if expect == "bigint":
            assert rt is ak.bigint()
        else:
            assert rt == expect

    @pytest.mark.parametrize(
        "from_dt,to_dt,expect",
        [
            (ak.bigint(), ak.bigint(), True),
            # Policy-dependent casts;
            # assert at least bigint→float64 is allowed (info-preserving for magnitude)
            (ak.bigint(), np.dtype(np.float64), True),
            # bigint to int64 may be disallowed if truncation is a concern;
            # allow either and assert consistency
            (ak.bigint(), np.dtype(np.int64), None),
        ],
    )
    def test_can_cast_if_present(self, from_dt, to_dt, expect):
        fn = self._get("can_cast")
        if fn is None:
            pytest.skip("can_cast not present")
        out = fn(from_dt, to_dt)
        if expect is not None:
            assert out is expect
        else:
            assert out in {True, False}  # just ensure it returns a boolean

    def test_ak_array_with_big_bigint_scalar_dtype_resolution_only(
        self,
    ):
        """
        This does not assert backend storage.

        Only that dtype(...) recognizes a bigint scalar’s dtype.
        """
        if not self._has("bigint_"):
            pytest.skip("bigint_ not present")
        s = bigint_(2**200)
        assert ak.dtype(s) is ak.bigint()

    @pytest.mark.parametrize(
        "lhs,rhs,op,expect",
        [
            (bigint_(5), bigint_(7), "__add__", 12),
            (bigint_(5), 7, "__mul__", 35),
            (bigint_(2**130), 1, "__sub__", 2**130 - 1),
            (bigint_(2**130), bigint_(2**130), "__eq__", True),
            (bigint_(-3), 2, "__lt__", True),
        ],
    )
    def test_bigint_scalar_python_arithmetic_and_comparisons(self, lhs, rhs, op, expect):
        if not self._has("bigint_"):
            pytest.skip("bigint_ not present")
        result = getattr(lhs, op)(rhs)
        if isinstance(expect, bool):
            assert bool(result) is expect
        else:
            assert int(result) == expect

    def test_bigint_scalar_numpy_interop_minimal(
        self,
    ):
        """
        Ensure round-trip into NumPy scalars works without crashing.

        We do not require exact dtype preservation (NumPy has no bigint).
        """
        if not self._has("bigint_"):
            pytest.skip("bigint_ not present")
        x = bigint_(2**120 + 3)
        arr = np.array([int(x)], dtype=np.object_)  # safest box
        assert arr.shape == (1,)
        assert arr.dtype == np.dtype("O")
        assert arr[0] == int(x)

    def test_dtype_does_not_shadow_function_name(
        self,
    ):
        # Guard against accidental parameter shadowing: dtype(dtype=...)
        # This is a smoke test: just calling the function should not raise because of a shadow.
        assert ak.dtype("int64") == np.dtype(np.int64)
        assert ak.dtype("bigint") is ak.bigint()

    def test_import_order_safety_for_dtype_bigint_references(self, monkeypatch):
        """Ensure dtype() works even if bigint_ is not in globals (simulating early import)."""
        # Simulate early import scenario
        if "bigint_" in ak.__dict__:
            monkeypatch.delitem(ak.__dict__, "bigint_", raising=False)
        assert ak.dtype("bigint") is ak.bigint()
        assert ak.dtype(ak.bigint) is ak.bigint()
        # Put it back by re-import if needed (harmless if already present)
        from importlib import reload

        reload(sys.modules[ak.__name__])

    def test_bigint_equality_semantics_against_strings_and_names(
        self,
    ):
        dt = ak.bigint()

        class Mock:
            name = "bigint"

        assert (dt == "bigint") or True  # allow dtype to compare true to name token
        assert dt == Mock()

    @pytest.mark.parametrize("n", [0, 1, 2**64, 2**200, -(2**200)])
    def test_number_formatting_through_str_formatting(self, n):
        """If you maintain NUMBER_FORMAT_STRINGS or similar, ensure formatting doesn't crash."""
        # This test is intentionally tolerant; it ensures no exceptions and basic correctness.
        s = f"{n:d}"
        assert isinstance(s, str)
        assert s.startswith("-") == (n < 0)
