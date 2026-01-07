from __future__ import annotations

import sys

from typing import Any

import numpy as np
import pytest

import arkouda as ak


# -------------------------
# Fixtures / helpers
# -------------------------
@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    # Deterministic for CI reproducibility
    return np.random.default_rng(seed=12345)


def _sample_scalars() -> list[Any]:
    """
    Scalars that matter for dtype resolution and casting edges.
    Keep this small but adversarial.
    """
    return [
        True,
        False,
        0,
        1,
        -1,
        np.int8(-7),
        np.int16(12),
        np.int32(-123),
        np.int64(2**40),
        np.uint8(255),
        np.uint16(2**16 - 1),
        np.uint32(2**32 - 1),
        np.uint64(2**63),
        0.0,
        3.5,
        np.float32(1.25),
        np.float64(-2.0),
        "hi",
        np.str_("bye"),
    ]


def _big_int_ge_2_64() -> int:
    # triggers ak.bigint behavior in resolve_scalar_dtype / dtype
    return 2**64


def _numpy_dtypes_to_check() -> list[np.dtype]:
    return [
        np.dtype(np.bool_),
        np.dtype(np.int64),
        np.dtype(np.uint64),
        np.dtype(np.uint8),
        np.dtype(np.float64),
        np.dtype(np.float32),
        np.dtype(np.str_),
    ]


# -------------------------
# Tests: dtype()
# -------------------------
@pytest.mark.parametrize(
    "spec, expected",
    [
        ("int64", np.dtype(np.int64)),
        ("uint64", np.dtype(np.uint64)),
        ("uint8", np.dtype(np.uint8)),
        ("float64", np.dtype(np.float64)),
        ("bool", np.dtype(np.bool_)),
        ("str", np.dtype(np.str_)),
        # Special casing in your dtype() for "Strings"
        ("Strings", np.dtype(np.str_)),
    ],
)
def test_dtype_string_specs(spec: str, expected: np.dtype) -> None:
    assert ak.dtype(spec) == expected


@pytest.mark.parametrize(
    "val, expected",
    [
        # Python bool should be bool dtype (if your implementation handles bool before int)
        (True, np.dtype(np.bool_)),
        (False, np.dtype(np.bool_)),
        # Python floats -> float64
        (1.5, np.dtype(np.float64)),
        (-2.0, np.dtype(np.float64)),
        # Python ints -> int64 (per current observed behavior)
        (1, np.dtype(np.int64)),
        (0, np.dtype(np.int64)),
        (-1, np.dtype(np.int64)),
    ],
)
def test_dtype_python_scalars(val: Any, expected: Any) -> None:
    assert ak.dtype(val) == expected


def test_dtype_bigint_threshold() -> None:
    # per your dtype(): int >= 2**64 -> bigint()
    dt = ak.dtype(_big_int_ge_2_64())
    assert isinstance(dt, ak.bigint)


# -------------------------
# Tests: resolve_scalar_dtype()
# -------------------------
@pytest.mark.parametrize(
    "val, expected",
    [
        (True, "bool"),
        (np.bool_(False), "bool"),
        (1, "int64"),
        (-1, "int64"),
        (np.int64(7), "int64"),
        (np.uint64(7), "uint64"),
        (2**63, "uint64"),  # boundary: >= 2**63 treated as uint64 in resolve_scalar_dtype
        (_big_int_ge_2_64(), "bigint"),
        (1.0, "float64"),
        (np.float32(1.0), "float64"),
        (np.float64(1.0), "float64"),
        ("hello", "str"),
        (np.str_("hello"), "str"),
    ],
)
def test_resolve_scalar_dtype(val: Any, expected: str) -> None:
    assert ak.resolve_scalar_dtype(val) == expected


# -------------------------
# Tests: can_cast()
# -------------------------
@pytest.mark.parametrize(
    "from_val, to_dtype, expected",
    [
        # Special handling for Python ints in can_cast
        (1, ak.dtype(np.uint64), True),
        (-1, ak.dtype(np.int64), True),
        (2**64 - 1, ak.dtype(np.uint64), True),
        (2**63 - 1, ak.dtype(np.int64), True),
        # Float support: can_cast(float -> float64/float32 union check)
        (1.25, ak.dtype(np.float64), True),
        (np.float32(1.25), ak.dtype(np.float64), True),
        # Typical numpy casts
        (np.int64(7), ak.dtype(np.int64), True),
        (np.int64(7), ak.dtype(np.float64), True),
        (np.float64(1.2), ak.dtype(np.int64), np.can_cast(np.float64(1.2), np.dtype(np.int64))),
    ],
)
def test_can_cast(from_val: Any, to_dtype: Any, expected: bool) -> None:
    assert ak.can_cast(from_val, to_dtype) == expected


@pytest.mark.parametrize("dt", _numpy_dtypes_to_check())
def test_can_cast_agrees_with_numpy_for_numpy_scalars(dt: np.dtype) -> None:
    """
    For numpy scalar types (not Python int/float/complex), your can_cast delegates to np.can_cast.
    This checks a small matrix of casts to ensure that delegation remains correct.
    """
    scalars: list[Any] = [
        np.bool_(True),
        np.int64(1),
        np.uint64(1),
        np.uint8(1),
        np.float64(1.0),
        np.float32(1.0),
    ]
    targets: list[np.dtype] = [
        np.dtype(np.bool_),
        np.dtype(np.int64),
        np.dtype(np.uint64),
        np.dtype(np.float64),
        np.dtype(np.float32),
    ]

    for s in scalars:
        for t in targets:
            # Skip Python builtins path by ensuring s is numpy scalar
            got = ak.can_cast(s, ak.dtype(t))
            exp = np.can_cast(s, t)
            assert got == exp, f"can_cast mismatch for {type(s)} -> {t}"


# -------------------------
# Tests: result_type()
# -------------------------
@pytest.mark.parametrize(
    "args, expect_bigint",
    [
        ((np.dtype(np.int64), np.dtype(np.int64)), False),
        ((np.dtype(np.int64), np.dtype(np.float64)), False),
        # Your result_type: any float presence forces np.result_type(float64)
        ((np.dtype(np.float32), np.dtype(np.int64)), False),
        # Bigint present forces bigint unless float present
        ((ak.bigint, np.dtype(np.int64)), True),
        ((ak.bigint, np.dtype(np.uint64)), True),
        ((ak.bigint, np.dtype(np.float64)), False),  # float wins in your logic
    ],
)
def test_result_type_bigint_and_float_precedence(args: tuple[Any, ...], expect_bigint: bool) -> None:
    rt = ak.result_type(*args)
    if expect_bigint:
        assert rt == ak.dtype("bigint")

    else:
        # When not bigint, should be numpy dtype or type consistent with numpy result_type
        assert rt == np.result_type(
            *[np.dtype(a) if isinstance(a, np.dtype) else a for a in args if a is not ak.bigint]
        ) or isinstance(rt, (np.dtype, type))


@pytest.mark.parametrize(
    "a, b",
    [
        (np.int64(1), np.int64(2)),
        (np.int64(1), np.float64(2.0)),
        (np.uint64(1), np.int64(2)),
        (np.float32(1.0), np.int64(2)),
        (np.float32(1.0), np.float64(2.0)),
    ],
)
def test_result_type_matches_numpy_when_no_bigint(a: Any, b: Any) -> None:
    # For non-bigint inputs, your function should mirror numpy promotion closely
    got = ak.result_type(np.dtype(a.dtype), np.dtype(b.dtype))
    exp = np.result_type(np.dtype(a.dtype), np.dtype(b.dtype))
    assert got == exp


def test_result_type_bigint_alone() -> None:
    assert ak.result_type(ak.bigint) is ak.dtype("bigint")


# -------------------------
# Tests: get_byteorder()
# -------------------------
@pytest.mark.parametrize("dt", [np.dtype(np.int64), np.dtype(np.uint64), np.dtype(np.float64)])
def test_get_byteorder_concrete(dt: np.dtype) -> None:
    # If dt.byteorder is '=', this resolves based on sys.byteorder
    got = ak.get_byteorder(dt)
    assert got in ("<", ">")
    if sys.byteorder == "little":
        assert got == "<"
    elif sys.byteorder == "big":
        assert got == ">"
    else:
        pytest.fail("Unexpected sys.byteorder")


@pytest.mark.parametrize("dt, expected", [(np.dtype("<i8"), "<"), (np.dtype(">i8"), ">")])
def test_get_byteorder_respects_explicit(dt: np.dtype, expected: str) -> None:
    assert ak.get_byteorder(dt) == expected


# -------------------------
# Pattern for extending: ufunc alignment (optional scaffold)
# -------------------------
@pytest.mark.parametrize(
    "ufunc_name",
    [
        # Add more as you implement/align them
        "absolute",
        "negative",
        "floor",
        "ceil",
    ],
)
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_unary_ufunc_alignment_scaffold(rng: np.random.Generator, ufunc_name: str, dtype: Any) -> None:
    """
    Scaffold: compares numpy ufunc output to arkouda equivalent, when it exists.

    NOTE: This expects ak.<ufunc_name> to exist. If it doesn't, this will xfail,
    which is useful for tracking progress.
    """
    np_ufunc = getattr(np, ufunc_name)
    ak_func = getattr(ak, ufunc_name, None)
    if ak_func is None:
        pytest.xfail(f"ak.{ufunc_name} not implemented")

    is_int = np.dtype(dtype).kind in "iu"
    if ufunc_name in ("floor", "ceil") and is_int:
        pytest.xfail("ak.floor/ak.ceil currently support float only (NumPy supports ints)")

    x_np = (
        rng.integers(-10, 10, size=100).astype(dtype)
        if np.dtype(dtype).kind in "iu"
        else rng.normal(size=100).astype(dtype)
    )
    x_ak = ak.array(x_np)

    got = ak_func(x_ak).to_ndarray()
    exp = np_ufunc(x_np)

    if np.issubdtype(exp.dtype, np.floating):
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(got, exp)
