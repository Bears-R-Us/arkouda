import re

import numpy as np
import pytest

import arkouda as ak


############################
# Shared helpers
############################


def gen_strings():
    """
    Full test set including Unicode characters.
    Used for tests that intentionally track divergence from NumPy.
    """
    return np.array(
        ["", "a", "AbC", "123", "hello world", "Σ", "ß", "3.14", "  "],
        dtype=object,
    )


def gen_ascii_strings():
    """ASCII-only subset where Arkouda is expected to match NumPy exactly."""
    return np.array(
        ["", "a", "AbC", "123", "hello world", "3.14", "  "],
        dtype=object,
    )


def assert_ak_np_equal(ak_obj, np_obj):
    ak_np = ak_obj.to_ndarray() if hasattr(ak_obj, "to_ndarray") else ak_obj
    np.testing.assert_array_equal(ak_np, np_obj)


def assert_bool_equal(ak_arr, np_arr):
    np.testing.assert_array_equal(ak_arr.to_ndarray(), np_arr)


############################
# Case operations (ASCII)
############################


@pytest.mark.parametrize("op", ["lower", "upper", "title", "capitalize"])
def test_strings_case_ops_ascii(op):
    data = gen_ascii_strings()

    ak_arr = ak.array(data)
    np_arr = data.astype(str)

    ak_res = getattr(ak_arr, op)()
    np_res = np.char.__dict__[op](np_arr)

    assert_ak_np_equal(ak_res, np_res)


############################
# Case operations (Unicode xfail)
############################


@pytest.mark.xfail(
    reason="Arkouda Strings case operations are not fully Unicode-aware "
    "(e.g., Greek Sigma Σ does not lowercase to \\u03C3). Issue #5288.",
    strict=True,
)
@pytest.mark.parametrize("op", ["lower", "upper", "title", "capitalize"])
def test_strings_case_ops_unicode_expected_mismatch(op):
    data = gen_strings()

    ak_arr = ak.array(data)
    np_arr = data.astype(str)

    ak_res = getattr(ak_arr, op)()
    np_res = np.char.__dict__[op](np_arr)

    assert_ak_np_equal(ak_res, np_res)


############################
# Predicate operations (strict)
############################


@pytest.mark.parametrize(
    "op",
    [
        "isdigit",
        "isalpha",
        "isalnum",
        "islower",
        "isupper",
        # istitle is handled separately as xfail (see below)
        "isspace",
        "isdecimal",
        # isnumeric is handled separately as xfail (see below)
    ],
)
def test_strings_predicates(op):
    data = gen_strings()

    ak_arr = ak.array(data)
    np_arr = np.array([str(x) for x in data], dtype=object)

    ak_res = getattr(ak_arr, op)()
    np_res = np.vectorize(lambda x: getattr(x, op)())(np_arr)

    assert_bool_equal(ak_res, np_res)


@pytest.mark.xfail(
    reason="Arkouda Strings.istitle does not match Python/NumPy semantics "
    "(e.g., returns True for digit-only and whitespace-only strings). "
    "Issue #5289",
    strict=True,
)
def test_strings_predicate_istitle_unicode_expected_mismatch():
    data = gen_strings()

    ak_arr = ak.array(data)
    np_arr = np.array([str(x) for x in data], dtype=object)

    ak_res = ak_arr.istitle()
    np_res = np.vectorize(lambda x: x.istitle())(np_arr)

    assert_bool_equal(ak_res, np_res)


@pytest.mark.xfail(
    reason="Arkouda Strings.isnumeric does not match Python/NumPy semantics "
    "(e.g., returns True for the empty string). Issue #5290.",
    strict=True,
)
def test_strings_predicate_isnumeric_expected_mismatch():
    data = gen_strings()

    ak_arr = ak.array(data)
    np_arr = np.array([str(x) for x in data], dtype=object)

    ak_res = ak_arr.isnumeric()
    np_res = np.vectorize(lambda x: x.isnumeric())(np_arr)

    assert_bool_equal(ak_res, np_res)


############################
# Binary operations
############################


def test_strings_eq_ne():
    data = gen_strings()

    ak_arr = ak.array(data)
    np_arr = data.astype(str)

    assert_bool_equal(ak_arr == ak_arr, np_arr == np_arr)
    assert_bool_equal(ak_arr != ak_arr, np_arr != np_arr)


def test_strings_eq_scalar():
    data = gen_strings()
    scalar = "a"

    ak_arr = ak.array(data)
    ak_res = ak_arr == scalar

    np_res = np.array([x == scalar for x in data])

    assert_bool_equal(ak_res, np_res)


############################
# Indexing and slicing
############################


def test_strings_indexing():
    data = gen_strings()
    ak_arr = ak.array(data)

    for i in range(-len(data), len(data)):
        assert ak_arr[i] == data[i]


@pytest.mark.parametrize("slc", [slice(None), slice(1, 5), slice(None, None, 2)])
def test_strings_slicing(slc):
    data = gen_strings()
    ak_arr = ak.array(data)

    assert_ak_np_equal(ak_arr[slc], data[slc])


############################
# Regex alignment
############################


def test_strings_contains_regex():
    data = gen_strings()
    ak_arr = ak.array(data)
    pattern = r"\d+"

    ak_res = ak_arr.contains(pattern, regex=True)
    py_res = np.array([bool(re.search(pattern, s)) for s in data])

    assert_bool_equal(ak_res, py_res)


############################
# Error alignment
############################


def test_strings_size_mismatch_eq():
    a = ak.array(["a", "b"])
    b = ak.array(["a"])

    with pytest.raises(ValueError):
        _ = a == b


def test_strings_invalid_regex():
    ak_arr = ak.array(["abc"])

    # Python's regex compiler raises re.error (PatternError on py3.13+) for invalid patterns.
    with pytest.raises(re.error):
        ak_arr.contains("[", regex=True)
