import random
import string

import numpy as np
import pytest

import arkouda as ak
import arkouda.numpy.char as ak_char


def _np_isnumeric(py_list: list[str]) -> np.ndarray:
    """
    NumPy alignment target:
    np.char.isnumeric applies Python's str.isnumeric elementwise.
    """
    a = np.array(py_list, dtype=str)
    return np.char.isnumeric(a)


class TestArkoudaNumpyCharAlignment:
    def test_public_api_reexports_match_numpy(self):
        # These should be exactly the NumPy objects re-exported
        assert ak_char.bool_ is np.bool_
        assert ak_char.int_ is np.int_
        assert ak_char.integer is np.integer
        assert ak_char.object_ is np.object_
        assert ak_char.str_ is np.str_
        assert ak_char.character is np.character

        # Ensure isnumeric exists and is callable
        assert callable(ak_char.isnumeric)

    def test_isnumeric_rejects_non_strings(self):
        with pytest.raises(TypeError, match=r"input to isnumeric must be Strings"):
            ak_char.isnumeric(ak.arange(3))

        with pytest.raises(TypeError, match=r"input to isnumeric must be Strings"):
            ak_char.isnumeric(["1", "2", "3"])

    @pytest.mark.xfail(
        reason="Known mismatch: empty string treated as numeric in Arkouda "
        "(should be False like NumPy). Issue #5243"
    )
    @pytest.mark.parametrize(
        "py_list",
        [
            ["Strings 0", "Strings 1", "Strings 2", "120", "121", "122"],
            ["", "0", "00", "  ", "3.14", "-1", "+2", "１２３", "٣", "二"],  # noqa: RUF001
            ["1e3", "⅕", "²", "₇", "2³₇", "2³x₇", "٣٤٥", "१२३"],
        ],
    )
    def test_isnumeric_matches_numpy_char(self, py_list):
        s = ak.array(py_list)

        got = ak_char.isnumeric(s).to_ndarray()
        exp = _np_isnumeric(py_list)

        assert got.dtype == np.bool_
        np.testing.assert_array_equal(got, exp)

    def test_isnumeric_unicode_special_examples_from_docstring(self):
        py_list = ["3.14", "\u0030", "\u00b2", "2³₇", "2³x₇"]
        s = ak.array(py_list)

        got = ak_char.isnumeric(s).to_ndarray()
        exp = _np_isnumeric(py_list)

        np.testing.assert_array_equal(got, exp)

    @pytest.mark.xfail(
        reason="Known mismatch: empty string treated as numeric in Arkouda "
        "(should be False like NumPy). Issue #5243"
    )
    def test_isnumeric_randomized_matches_python_and_numpy(self):
        # Keep this deterministic and cheap
        rng = random.Random(0)

        # Build strings from digits, letters, punctuation, whitespace,
        # plus a few unicode numeric characters.
        unicode_numerics = ["²", "³", "₇", "⅕", "٣", "१२३", "１２３"]  # noqa: RUF001
        alphabet = string.digits + string.ascii_letters + string.punctuation + " \t"

        py_list: list[str] = []
        for _ in range(200):
            if rng.random() < 0.15:
                py_list.append(rng.choice(unicode_numerics))
                continue
            n = rng.randint(0, 12)
            py_list.append("".join(rng.choice(alphabet) for _ in range(n)))

        s = ak.array(py_list)
        got = ak_char.isnumeric(s).to_ndarray()

        # Primary truth: Python's str.isnumeric elementwise
        py_truth = np.array([x.isnumeric() for x in py_list], dtype=np.bool_)
        np.testing.assert_array_equal(got, py_truth)

        # Secondary cross-check: NumPy's char implementation
        np.testing.assert_array_equal(got, _np_isnumeric(py_list))
