import random

import pytest

import arkouda as ak
from arkouda import client_dtypes


class TestClientDTypeTests:
    """
    Note: BitVector operations are not tested here because the class is
    only a wrapper on a pdarray to display as such.
    The class does not actually store values as bit-values,
    it only converts to a bit representation for display.
    Thus, pdarray testing covers these operations.
    """

    def test_BitVector_creation(self):
        arr = ak.arange(4)
        bv = ak.BitVector(arr, width=3)
        assert isinstance(bv, client_dtypes.BitVector)
        assert bv.to_list() == ["...", "..|", ".|.", ".||"]
        assert bv.dtype == ak.bitType

        # Test reversed
        arr = ak.arange(4, dtype=ak.uint64)  # Also test with uint64 input
        bv = ak.BitVector(arr, width=3, reverse=True)
        assert isinstance(bv, client_dtypes.BitVector)
        assert bv.to_list() == ["...", "|..", ".|.", "||."]
        assert bv.dtype == ak.bitType

        # test use of vectorizer function
        arr = ak.arange(4)
        bvectorizer = ak.BitVectorizer(3)
        bv = bvectorizer(arr)
        assert isinstance(bv, client_dtypes.BitVector)
        assert bv.to_list() == ["...", "..|", ".|.", ".||"]
        assert bv.dtype == ak.bitType

        # fail on argument types
        with pytest.raises(TypeError):
            bv = ak.BitVector(17, width=4)
        arr = ak.array([1.1, 8.3])
        with pytest.raises(TypeError):
            bv - ak.BitVector(arr)

    def test_Field_creation(self):
        values = ak.arange(4)
        names = ["8", "4", "2", "1"]
        f = ak.Fields(values, names)
        assert isinstance(f, ak.Fields)
        assert f.to_list() == ["---- (0)", "---1 (1)", "--2- (2)", "--21 (3)"]
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
        assert f.to_list() == expected

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

    def test_ipv4_creation(self):
        # Test handling of int64 input
        ip_list = ak.array([3232235777], dtype=ak.int64)
        ipv4 = ak.IPv4(ip_list)

        assert isinstance(ipv4, ak.IPv4)
        assert ipv4.to_list() == ["192.168.1.1"]
        assert ipv4.dtype == ak.bitType

        # Test handling of uint64 input
        ip_list = ak.array([3232235777], dtype=ak.uint64)
        ipv4 = ak.IPv4(ip_list)

        assert isinstance(ipv4, ak.IPv4)
        assert ipv4.to_list() == ["192.168.1.1"]
        assert ipv4.dtype == ak.bitType

        with pytest.raises(TypeError):
            ipv4 = ak.IPv4("3232235777")
        with pytest.raises(TypeError):
            ipv4 = ak.IPv4(ak.array([3232235777.177]))

        # Test handling of python dotted-quad input
        ipv4 = ak.ip_address(["192.168.1.1"])
        assert isinstance(ipv4, ak.IPv4)
        assert ipv4.to_list() == ["192.168.1.1"]
        assert ipv4.dtype == ak.bitType

    def test_ipv4_normalization(self):
        ip_list = ak.array([3232235777])
        ipv4 = ak.IPv4(ip_list)
        ip_as_int = ipv4.normalize("192.168.1.1")
        assert 3232235777 == ip_as_int

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_is_ipv4(self, size):
        x = [random.getrandbits(32) for i in range(size)]

        ans = ak.is_ipv4(ak.array(x, dtype=ak.uint64))
        assert ans.to_list() == [True] * size

        ipv4 = ak.IPv4(ak.array(x))
        assert ak.is_ipv4(ipv4).to_list() == [True] * size

        x = [random.getrandbits(64) if i < size/2 else random.getrandbits(32) for i in range(size)]
        ans = ak.is_ipv4(ak.array(x, ak.uint64))
        assert ans.to_list() == [i >= size/2 for i in range(size)]

        with pytest.raises(TypeError):
            ak.is_ipv4(ak.array(x, ak.float64))

        with pytest.raises(RuntimeError):
            ak.is_ipv4(ak.array(x, dtype=ak.uint64), ak.arange(2, dtype=ak.uint64))

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_is_ipv6(self, size):
        x = [random.getrandbits(128) for i in range(size)]
        low = ak.array([i & (2**64 - 1) for i in x], dtype=ak.uint64)
        high = ak.array([i >> 64 for i in x], dtype=ak.uint64)

        assert ak.is_ipv6(high, low).to_list() == [True] * size

        x = [random.getrandbits(64) if i < 5 else random.getrandbits(32) for i in range(10)]
        ans = ak.is_ipv6(ak.array(x, ak.uint64))
        assert ans.to_list() == [i < 5 for i in range(10)]

        with pytest.raises(TypeError):
            ak.is_ipv6(ak.array(x, ak.float64))

        with pytest.raises(RuntimeError):
            ak.is_ipv6(ak.cast(ak.array(x), ak.int64), ak.cast(ak.arange(2), ak.int64))