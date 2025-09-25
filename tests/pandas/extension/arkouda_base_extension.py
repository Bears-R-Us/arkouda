import numpy as np
from numpy.testing import assert_array_equal
import pytest

import arkouda as ak
from arkouda.pandas.extension import (
    ArkoudaArray,
    ArkoudaCategoricalArray,
    ArkoudaStringArray,
)


class TestArkoudaBaseExtension:
    def test_base_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_base_array

        result = doctest.testmod(
            _arkouda_base_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("sort", [False, True])
    @pytest.mark.parametrize("use_na_sentinel", [True, False])
    def test_factorize_int_basic(self, sort, use_na_sentinel):
        """
        Int array has no NAs; first-appearance order vs sorted uniques;
        NA sentinel only affects behavior if there are NAs (there aren't here).
        """
        a = ArkoudaArray(ak.array([1, 2, 1, 3]))
        codes, uniques = a.factorize(sort=sort, use_na_sentinel=use_na_sentinel)

        if not sort:
            # First appearance: uniques [1, 2, 3]
            assert_array_equal(uniques, np.array([1, 2, 3]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))
        else:
            # Sorted: uniques [1, 2, 3] (same here, but codes recomputed from sorted order)
            assert_array_equal(uniques, np.array([1, 2, 3]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))

    @pytest.mark.parametrize("sort", [False, True])
    def test_factorize_float_with_nan_default_sentinel(self, sort):
        """
        Float array treats NaN as missing -> -1 sentinel by default.
        """
        a = ArkoudaArray(ak.array([1.0, np.nan, 1.0, 2.0]))
        codes, uniques = a.factorize(sort=sort)

        if not sort:
            # First appearance uniques: [1.0, 2.0]
            assert_array_equal(uniques, np.array([1.0, 2.0]))
            assert_array_equal(codes, np.array([0, -1, 0, 1]))
        else:
            # Sorted uniques: [1.0, 2.0] (same set)
            assert_array_equal(uniques, np.array([1.0, 2.0]))
            assert_array_equal(codes, np.array([0, -1, 0, 1]))

    def test_factorize_float_with_nan_no_sentinel(self):
        """
        With use_na_sentinel=False, NaNs get a valid code == len(uniques).
        """
        a = ArkoudaArray(ak.array([1.0, np.nan, 1.0, 2.0]))
        codes, uniques = a.factorize(sort=False, use_na_sentinel=False)
        # uniques from first appearance: [1.0, 2.0]; NaN code == 2
        assert_array_equal(uniques, np.array([1.0, 2.0]))
        assert_array_equal(codes, np.array([0, 2, 0, 1]))

    def test_factorize_float_all_nan(self):
        """
        Edge case: all values are NaN -> codes all sentinel, uniques empty.
        """
        a = ArkoudaArray(ak.array([np.nan, np.nan]))
        codes, uniques = a.factorize()
        assert_array_equal(uniques, np.array([], dtype=float))
        assert_array_equal(codes, np.array([-1, -1], dtype=np.int64))

    @pytest.mark.parametrize("sort", [False, True])
    def test_factorize_strings_basic(self, sort):
        """
        Strings: no NA handling; empty strings are treated as normal values.
        """
        s = ak.array(["a", "b", "a", "c"])
        a = ArkoudaStringArray(s)
        codes, uniques = a.factorize(sort=sort)

        if not sort:
            assert_array_equal(uniques, np.array(["a", "b", "c"]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))
        else:
            # Sorted: ["a", "b", "c"] -> same result for this set
            assert_array_equal(uniques, np.array(["a", "b", "c"]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))

    def test_factorize_strings_with_empty_string(self):
        """
        Explicitly ensure "" is treated as a normal value (not missing).
        """
        s = ak.array(["", "x", "", "y"])
        a = ArkoudaStringArray(s)
        codes, uniques = a.factorize(sort=False)
        assert_array_equal(uniques, np.array(["", "x", "y"]))
        assert_array_equal(codes, np.array([0, 1, 0, 2]))

    @pytest.mark.parametrize("sort", [False, True])
    def test_factorize_categorical_basic(self, sort):
        """
        Categorical: factorization operates over observed values (not categories table),
        honoring first-appearance vs sorted order semantics of the observed data.
        """
        s = ak.array(["red", "blue", "red", "green"])
        cat = ak.Categorical(s)  # construct from Strings
        a = ArkoudaCategoricalArray(cat)
        codes, uniques = a.factorize(sort=sort)

        if not sort:
            # first appearance uniques: ["red", "blue", "green"]
            assert_array_equal(uniques, np.array(["red", "blue", "green"]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))
        else:
            # sorted uniques: ["blue", "green", "red"]
            assert_array_equal(uniques, np.array(["blue", "green", "red"]))
            # remapped codes according to sorted order:
            # red->2, blue->0, green->1
            assert_array_equal(codes, np.array([2, 0, 2, 1]))

    def test_factorize_stability_first_appearance_vs_sorted(self):
        """
        Sanity check that switching sort changes code assignments consistently.
        """
        x = ak.array([2, 1, 3, 2])
        a = ArkoudaArray(x)

        codes_unsorted, uniques_unsorted = a.factorize(sort=False)
        codes_sorted, uniques_sorted = a.factorize(sort=True)

        # First appearance uniques: [2, 1, 3]
        assert_array_equal(uniques_unsorted, np.array([2, 1, 3]))
        assert_array_equal(codes_unsorted, np.array([0, 1, 2, 0]))

        # Sorted uniques: [1, 2, 3]
        assert_array_equal(uniques_sorted, np.array([1, 2, 3]))
        # mapping old->new: 2->1, 1->0, 3->2  => [1,0,2,1]
        assert_array_equal(codes_sorted, np.array([1, 0, 2, 1]))
