import numpy as np
import pytest

import arkouda as ak
from arkouda.numpy import char
from arkouda.testing import assert_arkouda_array_equivalent


class TestChar:
    def test_char_docstrings(self):
        import doctest

        result = doctest.testmod(char, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    #   The isnumeric test generates random strings, some with numerics only, some with
    #   alphanumerics, and runs them through both numpy and arkouda, checking for
    #   equivalence.

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_isnumeric(self, prob_size):
        import random
        import string

        seed = pytest.seed if pytest.seed is not None else 8675309
        numerics = [chr(i) for i in range(0x110000) if np.char.isnumeric(chr(i))]  # unicode numerics
        alphanumerics = string.digits + string.ascii_letters
        random.seed(seed)
        prob_set = list()
        for i in range(prob_size):
            heads_or_tails = random.choice([True, False])
            if heads_or_tails:
                prob_set.append("".join(random.choice(numerics) for j in range(4)))
            else:
                prob_set.append("".join(random.choice(alphanumerics) for j in range(4)))
        nda = np.array(prob_set)
        pda = ak.array(nda)
        for_comparison = np.char.isnumeric(nda)
        assert_arkouda_array_equivalent(for_comparison, pda.isnumeric())
        assert_arkouda_array_equivalent(for_comparison, ak.isnumeric(pda))
