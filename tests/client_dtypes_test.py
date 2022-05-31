from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import client_dtypes


class ClientDTypeTests(ArkoudaTest):
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
        self.assertIsInstance(bv, client_dtypes.BitVector)
        self.assertListEqual(bv.to_ndarray().tolist(), ["...", "..|", ".|.", ".||"])
        self.assertTrue(bv.dtype == ak.bitType)

        # Test reversed
        arr = ak.arange(4, dtype=ak.uint64)  # Also test with uint64 input
        bv = ak.BitVector(arr, width=3, reverse=True)
        self.assertIsInstance(bv, client_dtypes.BitVector)
        self.assertListEqual(bv.to_ndarray().tolist(), ["...", "|..", ".|.", "||."])
        self.assertTrue(bv.dtype == ak.bitType)

        # test use of vectorizer function
        arr = ak.arange(4)
        bvectorizer = ak.BitVectorizer(3)
        bv = bvectorizer(arr)
        self.assertIsInstance(bv, client_dtypes.BitVector)
        self.assertListEqual(bv.to_ndarray().tolist(), ["...", "..|", ".|.", ".||"])
        self.assertTrue(bv.dtype == ak.bitType)

        # fail on argument types
        with self.assertRaises(TypeError):
            bv = ak.BitVector(17, width=4)
        arr = ak.array([1.1, 8.3])
        with self.assertRaises(TypeError):
            bv - ak.BitVector(arr)

    def test_Field_creation(self):
        values = ak.arange(4)
        names = ["8", "4", "2", "1"]
        f = ak.Fields(values, names)
        self.assertIsInstance(f, ak.Fields)
        self.assertListEqual(f.to_ndarray().tolist(), ["---- (0)", "---1 (1)", "--2- (2)", "--21 (3)"])
        self.assertTrue(f.dtype == ak.bitType)

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
        self.assertListEqual(f.to_ndarray().tolist(), expected)

        values = ak.arange(8, dtype=ak.uint64)
        names = [f"Bit{x}" for x in range(65)]
        with self.assertRaises(ValueError):
            f = ak.Fields(values, names)

        names = ["t", "t"]
        with self.assertRaises(ValueError):
            f = ak.Fields(values, names)

        names = ["t", ""]
        with self.assertRaises(ValueError):
            f = ak.Fields(values, names)

        names = ["abc", "123"]
        with self.assertRaises(ValueError):
            f = ak.Fields(values, names, separator="abc")

        with self.assertRaises(ValueError):
            f = ak.Fields(values, names)

        with self.assertRaises(ValueError):
            f = ak.Fields(values, names, pad="|!~", separator="//")

    def test_ipv4_creation(self):
        # Test handling of int64 input
        ip_list = ak.array([3232235777], dtype=ak.int64)
        ipv4 = ak.IPv4(ip_list)

        self.assertIsInstance(ipv4, ak.IPv4)
        self.assertListEqual(ipv4.to_ndarray().tolist(), ["192.168.1.1"])
        self.assertTrue(ipv4.dtype == ak.bitType)

        # Test handling of uint64 input
        ip_list = ak.array([3232235777], dtype=ak.uint64)
        ipv4 = ak.IPv4(ip_list)

        self.assertIsInstance(ipv4, ak.IPv4)
        self.assertListEqual(ipv4.to_ndarray().tolist(), ["192.168.1.1"])
        self.assertTrue(ipv4.dtype == ak.bitType)

        with self.assertRaises(TypeError):
            ipv4 = ak.IPv4("3232235777")
        with self.assertRaises(TypeError):
            ipv4 = ak.IPv4(ak.array([3232235777.177]))

        # Test handling of python dotted-quad input
        ipv4 = ak.ip_address(["192.168.1.1"])
        self.assertIsInstance(ipv4, ak.IPv4)
        self.assertListEqual(ipv4.to_ndarray().tolist(), ["192.168.1.1"])
        self.assertTrue(ipv4.dtype == ak.bitType)

    def test_ipv4_normalization(self):
        ip_list = ak.array([3232235777])
        ipv4 = ak.IPv4(ip_list)
        ip_as_int = ipv4.normalize("192.168.1.1")
        self.assertEqual(3232235777, ip_as_int)
