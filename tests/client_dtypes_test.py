import ipaddress
import random

import pytest

import arkouda as ak
from arkouda import client_dtypes


INT_TYPES = [ak.int64, ak.uint64]


class TestClientDTypes:
    """
    Note: BitVector operations are not tested here because the class is
    only a wrapper on a pdarray to display as such.
    The class does not actually store values as bit-values,
    it only converts to a bit representation for display.
    Thus, pdarray testing covers these operations.
    """

    def test_client_dtypes_docstrings(self):
        import doctest

        result = doctest.testmod(
            client_dtypes, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("dtype", INT_TYPES)
    def test_bit_vector_creation(self, dtype):
        bv_answer = ["...", "..|", ".|.", ".||"]
        bv_rev_answer = ["...", "|..", ".|.", "||."]

        arr = ak.arange(4, dtype=dtype)
        bv = ak.BitVector(arr, width=3)
        assert isinstance(bv, ak.BitVector)
        assert bv.tolist() == bv_answer
        assert bv.dtype == ak.bitType

        # Test reversed
        arr = ak.arange(4, dtype=dtype)
        bv = ak.BitVector(arr, width=3, reverse=True)
        assert isinstance(bv, ak.BitVector)
        assert bv.tolist() == bv_rev_answer
        assert bv.dtype == ak.bitType

        # test use of vectorizer function
        arr = ak.arange(4, dtype=dtype)
        bvectorizer = ak.BitVectorizer(3)
        bv = bvectorizer(arr)
        assert isinstance(bv, ak.BitVector)
        assert bv.tolist() == bv_answer
        assert bv.dtype == ak.bitType

    def test_bit_vector_error_handling(self):
        # fail on argument types
        with pytest.raises(TypeError):
            ak.BitVector(17, width=4)
        arr = ak.array([1.1, 8.3])
        with pytest.raises(TypeError):
            ak.BitVector(arr)

    @pytest.mark.parametrize("dtype", INT_TYPES)
    def test_bit_vector_upper_bound(self, dtype):
        bv_answer = [
            "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||..",
            "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||.|",
            "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||.",
        ]

        arr = ak.arange(2**64 - 4, 2**64 - 1, dtype=dtype)
        bv = ak.BitVector(arr, width=64)
        assert isinstance(bv, ak.BitVector)
        assert bv.tolist() == bv_answer
        assert bv.dtype == ak.bitType

    def test_field_creation(self):
        values = ak.arange(4)
        names = ["8", "4", "2", "1"]
        f = ak.Fields(values, names)
        assert isinstance(f, ak.Fields)
        assert f.tolist() == ["---- (0)", "---1 (1)", "--2- (2)", "--21 (3)"]
        assert f.dtype == ak.bitType

        # Named fields with reversed bit order
        values = ak.array([0, 1, 5, 8, 12])
        names = ["Bit1", "Bit2", "Bit3", "Bit4"]
        f = ak.Fields(values, names, MSB_left=False, separator="//")
        expected = [
            "----//----//----//----// (0)",
            "Bit1//----//----//----// (1)",
            "Bit1//----//Bit3//----// (5)",
            "----//----//----//Bit4// (8)",
            "----//----//Bit3//Bit4// (12)",
        ]
        assert f.tolist() == expected

        values = ak.arange(8, dtype=ak.uint64)
        names = [f"Bit{x}" for x in range(65)]
        with pytest.raises(ValueError):
            f = ak.Fields(values, names)

        names = ["t", "t"]
        with pytest.raises(ValueError):
            f = ak.Fields(values, names)

        names = ["t", ""]
        with pytest.raises(ValueError):
            f = ak.Fields(values, names)

        names = ["abc", "123"]
        with pytest.raises(ValueError):
            f = ak.Fields(values, names, separator="abc")

        with pytest.raises(ValueError):
            f = ak.Fields(values, names)

        with pytest.raises(ValueError):
            f = ak.Fields(values, names, pad="|!~", separator="//")

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INT_TYPES)
    def test_ipv4_creation(self, prob_size, dtype):
        ips = ak.randint(1, 2**32, prob_size)
        ip_list = ak.array(ips, dtype=dtype)
        ipv4 = ak.IPv4(ip_list)
        py_ips = [ipaddress.IPv4Address(ip).compressed for ip in ips.tolist()]

        assert isinstance(ipv4, ak.IPv4)
        assert ipv4.tolist() == py_ips
        assert ipv4.dtype == ak.bitType

        with pytest.raises(TypeError):
            ipv4 = ak.IPv4(f"{ips[0]}")
        with pytest.raises(TypeError):
            ipv4 = ak.IPv4(ak.array([ips[0] + 0.177]))

        # Test handling of python dotted-quad input
        ipv4 = ak.ip_address(py_ips)
        assert isinstance(ipv4, ak.IPv4)
        assert ipv4.tolist() == py_ips
        assert ipv4.dtype == ak.bitType

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_ipv4_normalization(self, prob_size):
        ips = ak.randint(1, 2**32, prob_size)
        ip_list = ak.array(ips)
        ipv4 = ak.IPv4(ip_list)
        ip_as_dot = [ipaddress.IPv4Address(ip) for ip in ips.tolist()]
        ip_as_int = [ipv4.normalize(ipd) for ipd in ip_as_dot]
        assert ips.tolist() == ip_as_int

    def test_is_ipv4(self):
        prob_size = 100
        x = [random.getrandbits(32) for _ in range(prob_size)]

        ans = ak.is_ipv4(ak.array(x, dtype=ak.uint64))
        assert ans.tolist() == [True] * prob_size

        ipv4 = ak.IPv4(ak.array(x))
        assert ak.is_ipv4(ipv4).tolist() == [True] * prob_size

        x = [
            random.getrandbits(64) if i < prob_size / 2 else random.getrandbits(32)
            for i in range(prob_size)
        ]
        ans = ak.is_ipv4(ak.array(x, ak.uint64))
        assert ans.tolist() == [i >= prob_size / 2 for i in range(prob_size)]

        with pytest.raises(TypeError):
            ak.is_ipv4(ak.array(x, ak.float64))

        with pytest.raises(RuntimeError):
            ak.is_ipv4(ak.array(x, dtype=ak.uint64), ak.arange(2, dtype=ak.uint64))

    def test_is_ipv6(self):
        prob_size = 100
        x = [random.getrandbits(128) for _ in range(prob_size)]
        low = ak.array([i & (2**64 - 1) for i in x], dtype=ak.uint64)
        high = ak.array([i >> 64 for i in x], dtype=ak.uint64)

        assert ak.is_ipv6(high, low).tolist() == [True] * prob_size

        x = [
            random.getrandbits(64) if i < prob_size / 2 else random.getrandbits(32)
            for i in range(prob_size)
        ]
        ans = ak.is_ipv6(ak.array(x, ak.uint64))
        assert ans.tolist() == [i < prob_size / 2 for i in range(prob_size)]

        with pytest.raises(TypeError):
            ak.is_ipv6(ak.array(x, ak.float64))

        with pytest.raises(RuntimeError):
            ak.is_ipv6(ak.cast(ak.array(x), ak.int64), ak.cast(ak.arange(2), ak.int64))
