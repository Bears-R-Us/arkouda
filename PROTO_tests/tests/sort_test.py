import arkouda as ak
import pytest

from arkouda.sorting import SortingAlgorithm


class TestSort:
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    def test_sort_dtype(self, size, dtype):
        pda = ak.randint(0, 100, size, dtype)
        for algo in SortingAlgorithm:
            spda = ak.sort(pda, algo)
            assert ak.is_sorted(spda)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_sort_bigint(self, size):
        pda = ak.randint(0, 100, size, dtype=ak.uint64)
        shift_up = pda + 2**200
        for algo in SortingAlgorithm:
            sorted_pda = ak.sort(pda, algo)
            sorted_bi = ak.sort(shift_up, algo)
            assert (sorted_bi - 2**200).to_list() == sorted_pda.to_list()

    def test_bit_boundary_hardcode(self):
        # test hardcoded 16/17-bit boundaries with and without negative values
        a = ak.array([1, -1, 32767])  # 16 bit
        b = ak.array([1, 0, 32768])  # 16 bit
        c = ak.array([1, -1, 32768])  # 17 bit
        for algo in SortingAlgorithm:
            for arr in a, b, c:
                assert ak.is_sorted(ak.sort(arr, algo))

        # test hardcoded 64-bit boundaries with and without negative values
        d = ak.array([1, -1, 2**63 - 1])
        e = ak.array([1, 0, 2**63 - 1])
        f = ak.array([1, -(2**63), 2**63 - 1])
        for algo in SortingAlgorithm:
            for arr in d, e, f:
                assert ak.is_sorted(ak.sort(arr, algo))

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_bit_boundary(self, size):
        # test 17-bit sort
        lower = -(2**15)
        upper = 2**16
        a = ak.randint(lower, upper, size)
        for algo in SortingAlgorithm:
            assert ak.is_sorted(ak.sort(a, algo))

    def test_error_handling(self):
        # Test RuntimeError from bool NotImplementedError
        ak_bools = ak.randint(0, 1, 1000, dtype=ak.bool)
        bools = ak.randint(0, 1, 1000, dtype=bool)

        for algo in SortingAlgorithm:
            with pytest.raises(ValueError):
                ak.sort(ak_bools, algo)

            with pytest.raises(ValueError):
                ak.sort(bools, algo)

            # Test TypeError from sort attempt on non-pdarray
            with pytest.raises(TypeError):
                ak.sort(list(range(0, 10)), algo)

            # Test attempt to sort Strings object, which is unsupported
            with pytest.raises(TypeError):
                ak.sort(ak.array(["String {}".format(i) for i in range(0, 10)]), algo)
