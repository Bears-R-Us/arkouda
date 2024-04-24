import numpy as np
import pytest
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.scipy import chisquare as akchisquare


class RandomTest(ArkoudaTest):
    def test_integers(self):
        # verify same seed gives different but reproducible arrays
        rng = ak.random.default_rng(18)
        first = rng.integers(-(2**32), 2**32, 10)
        second = rng.integers(-(2**32), 2**32, 10)
        self.assertNotEqual(first.to_list(), second.to_list())

        rng = ak.random.default_rng(18)
        same_seed_first = rng.integers(-(2**32), 2**32, 10)
        same_seed_second = rng.integers(-(2**32), 2**32, 10)
        self.assertEqual(first.to_list(), same_seed_first.to_list())
        self.assertEqual(second.to_list(), same_seed_second.to_list())

        # test endpoint
        rng = ak.random.default_rng()
        all_zero = rng.integers(0, 1, 20)
        self.assertTrue(all(all_zero.to_ndarray() == 0))

        not_all_zero = rng.integers(0, 1, 20, endpoint=True)
        self.assertTrue(any(not_all_zero.to_ndarray() != 0))

        # verify that switching dtype and function from seed is still reproducible
        rng = ak.random.default_rng(74)
        uint_arr = rng.integers(0, 2**32, size=10, dtype="uint")
        float_arr = rng.uniform(-1.0, 1.0, size=5)
        bool_arr = rng.integers(0, 1, size=20, dtype="bool")
        int_arr = rng.integers(-(2**32), 2**32, size=10, dtype="int")

        rng = ak.random.default_rng(74)
        same_seed_uint_arr = rng.integers(0, 2**32, size=10, dtype="uint")
        same_seed_float_arr = rng.uniform(-1.0, 1.0, size=5)
        same_seed_bool_arr = rng.integers(0, 1, size=20, dtype="bool")
        same_seed_int_arr = rng.integers(-(2**32), 2**32, size=10, dtype="int")

        self.assertEqual(uint_arr.to_list(), same_seed_uint_arr.to_list())
        self.assertEqual(float_arr.to_list(), same_seed_float_arr.to_list())
        self.assertEqual(bool_arr.to_list(), same_seed_bool_arr.to_list())
        self.assertEqual(int_arr.to_list(), same_seed_int_arr.to_list())

        # verify within bounds (lower inclusive and upper exclusive)
        rng = ak.random.default_rng()
        bounded_arr = rng.integers(-5, 5, 1000)
        self.assertTrue(all(bounded_arr.to_ndarray() >= -5))
        self.assertTrue(all(bounded_arr.to_ndarray() < 5))

    def test_shuffle(self):
        # verify same seed gives reproducible arrays
        rng = ak.random.default_rng(18)

        int_pda = rng.integers(-(2**32), 2**32, 10)
        pda_copy = int_pda[:]
        # shuffle int_pda in place
        rng.shuffle(int_pda)
        # verify all the same elements are in permutation as the original
        self.assertEqual(ak.sort(int_pda).to_list(), ak.sort(pda_copy).to_list())

        float_pda = rng.uniform(-(2**32), 2**32, 10)
        pda_copy = float_pda[:]
        rng.shuffle(float_pda)
        # verify all the same elements are in permutation as the original
        self.assertEqual(ak.sort(float_pda).to_list(), ak.sort(pda_copy).to_list())

        rng = ak.random.default_rng(18)

        pda = rng.integers(-(2**32), 2**32, 10)
        rng.shuffle(pda)
        self.assertEqual(pda.to_list(), int_pda.to_list())

        pda = rng.uniform(-(2**32), 2**32, 10)
        rng.shuffle(pda)
        self.assertTrue(np.allclose(pda.to_list(), float_pda.to_list()))

    def test_permutation(self):
        # verify same seed gives reproducible arrays
        rng = ak.random.default_rng(18)
        # providing just a number permutes the range(num)
        range_permute = rng.permutation(20)
        self.assertEqual(ak.arange(20).to_list(), ak.sort(range_permute).to_list())

        pda = rng.integers(-(2**32), 2**32, 10)
        array_permute = rng.permutation(pda)
        # verify all the same elements are in permutation as the original
        self.assertEqual(ak.sort(pda).to_list(), ak.sort(array_permute).to_list())

        pda = rng.uniform(-(2**32), 2**32, 10)
        float_array_permute = rng.permutation(pda)
        # verify all the same elements are in permutation as the original
        self.assertTrue(np.allclose(ak.sort(pda).to_list(), ak.sort(float_array_permute).to_list()))

        rng = ak.random.default_rng(18)
        same_seed_range_permute = rng.permutation(20)
        self.assertEqual(range_permute.to_list(), same_seed_range_permute.to_list())

        pda = rng.integers(-(2**32), 2**32, 10)
        same_seed_array_permute = rng.permutation(pda)
        self.assertEqual(array_permute.to_list(), same_seed_array_permute.to_list())

        pda = rng.uniform(-(2**32), 2**32, 10)
        same_seed_float_array_permute = rng.permutation(pda)
        # verify all the same elements are in permutation as the original
        self.assertTrue(
            np.allclose(float_array_permute.to_list(), same_seed_float_array_permute.to_list())
        )

    def test_uniform(self):
        # verify same seed gives different but reproducible arrays
        rng = ak.random.default_rng(18)
        first = rng.uniform(-(2**32), 2**32, 10)
        second = rng.uniform(-(2**32), 2**32, 10)
        self.assertNotEqual(first.to_list(), second.to_list())

        rng = ak.random.default_rng(18)
        same_seed_first = rng.uniform(-(2**32), 2**32, 10)
        same_seed_second = rng.uniform(-(2**32), 2**32, 10)
        self.assertEqual(first.to_list(), same_seed_first.to_list())
        self.assertEqual(second.to_list(), same_seed_second.to_list())

        # verify within bounds (lower inclusive and upper exclusive)
        rng = ak.random.default_rng()
        bounded_arr = rng.uniform(-5, 5, 1000)
        self.assertTrue(all(bounded_arr.to_ndarray() >= -5))
        self.assertTrue(all(bounded_arr.to_ndarray() < 5))

    def test_choice_hypothesis_testing(self):
        # perform a weighted sample and use chisquare to test
        # if the observed frequency matches the expected frequency

        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05
        rng = ak.random.default_rng(43)
        num_samples = 10**4

        weights = ak.array([0.25, 0.15, 0.20, 0.10, 0.30])
        weighted_sample = rng.choice(ak.arange(5), size=num_samples, p=weights)

        # count how many of each category we saw
        uk, f_obs = ak.GroupBy(weighted_sample).count()

        # I think the keys should always be sorted but just in case
        if not ak.is_sorted(uk):
            f_obs = f_obs[ak.argsort(uk)]

        f_exp = weights * num_samples
        _, pval = akchisquare(f_obs=f_obs, f_exp=f_exp)

        # if pval <= 0.05, the difference from the expected distribution is significant
        self.assertTrue(pval > 0.05)

    def test_choice(self):
        # verify without replacement works
        rng = ak.random.default_rng()
        # test domains and selecting all
        domain_choice = rng.choice(20, 20, replace=False)
        # since our populations and sample size is the same without replacement,
        # we should see all values
        self.assertEqual(ak.sort(domain_choice).to_list(), ak.arange(20).to_list())

        # test arrays and not selecting all
        perm = rng.permutation(100)
        array_choice = rng.choice(perm, 95, replace=False)
        # verify all unique
        _, count = ak.GroupBy(array_choice).count()
        self.assertTrue((count == 1).all())

        # test single value
        scalar = rng.choice(5)
        self.assertIs(type(scalar), np.int64)
        self.assertIn(scalar, [0, 1, 2, 3, 4])

    @pytest.mark.skip(reason="skip until issue #3118 is resolved")
    def test_choice_flags(self):
        # use numpy to randomly generate a set seed
        seed = np.random.default_rng().choice(2**63)

        rng = ak.random.default_rng(seed)
        weights = rng.uniform(size=10)
        a_vals = [
            10,
            rng.integers(0, 2**32, size=10, dtype="uint"),
            rng.uniform(-1.0, 1.0, size=10),
            rng.integers(0, 1, size=10, dtype="bool"),
            rng.integers(-(2**32), 2**32, size=10, dtype="int"),
        ]

        rng = ak.random.default_rng(seed)
        choice_arrays = []
        for a in a_vals:
            for size in 5, 10:
                for replace in True, False:
                    for p in [None, weights]:
                        choice_arrays.append(rng.choice(a, size, replace, p))

        # reset generator to ensure we get the same arrays
        rng = ak.random.default_rng(seed)
        for a in a_vals:
            for size in 5, 10:
                for replace in True, False:
                    for p in [None, weights]:
                        previous = choice_arrays.pop(0)
                        current = rng.choice(a, size, replace, p)
                        self.assertTrue(np.allclose(previous.to_list(), current.to_list()))

    def test_legacy_randint(self):
        testArray = ak.random.randint(0, 10, 5)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)

        testArray = ak.random.randint(np.int64(0), np.int64(10), np.int64(5))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)

        testArray = ak.random.randint(np.float64(0), np.float64(10), np.int64(5))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)

        test_ndarray = testArray.to_ndarray()

        for value in test_ndarray:
            self.assertTrue(0 <= value <= 10)

        test_array = ak.random.randint(0, 1, 3, dtype=ak.float64)
        self.assertEqual(ak.float64, test_array.dtype)

        test_array = ak.random.randint(0, 1, 5, dtype=ak.bool)
        self.assertEqual(ak.bool, test_array.dtype)

        test_ndarray = test_array.to_ndarray()

        # test resolution of modulus overflow - issue #1174
        test_array = ak.random.randint(-(2**63), 2**63 - 1, 10)
        to_validate = np.full(10, -(2**63))
        self.assertFalse((test_array.to_ndarray() == to_validate).all())

        for value in test_ndarray:
            self.assertTrue(value in [True, False])

        with self.assertRaises(TypeError):
            ak.random.randint(low=5)

        with self.assertRaises(TypeError):
            ak.random.randint(high=5)

        with self.assertRaises(TypeError):
            ak.random.randint()

        with self.assertRaises(ValueError):
            ak.random.randint(low=0, high=1, size=-1, dtype=ak.float64)

        with self.assertRaises(ValueError):
            ak.random.randint(low=1, high=0, size=1, dtype=ak.float64)

        with self.assertRaises(TypeError):
            ak.random.randint(0, 1, "1000")

        with self.assertRaises(TypeError):
            ak.random.randint("0", 1, 1000)

        with self.assertRaises(TypeError):
            ak.random.randint(0, "1", 1000)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.randint(low=np.uint8(1), high=np.uint16(100), size=np.uint32(100))

    def test_legacy_randint_with_seed(self):
        values = ak.random.randint(1, 5, 10, seed=2)

        self.assertListEqual([4, 3, 1, 3, 2, 4, 4, 2, 3, 4], values.to_list())

        values = ak.random.randint(1, 5, 10, dtype=ak.float64, seed=2)
        self.assertListEqual(
            [
                2.9160772326374946,
                4.353429832157099,
                4.5392023718621486,
                4.4019932101126606,
                3.3745324569952304,
                1.1642002901528308,
                4.4714086874555292,
                3.7098921109084522,
                4.5939589352472314,
                4.0337935981006172,
            ],
            values.to_list(),
        )

        values = ak.random.randint(1, 5, 10, dtype=ak.bool, seed=2)
        self.assertListEqual(
            [False, True, True, True, True, False, True, True, True, True],
            values.to_list(),
        )

        values = ak.random.randint(1, 5, 10, dtype=bool, seed=2)
        self.assertListEqual(
            [False, True, True, True, True, False, True, True, True, True],
            values.to_list(),
        )

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.randint(np.uint8(1), np.uint32(5), np.uint16(10), seed=np.uint8(2))

    def test_legacy_uniform(self):
        testArray = ak.random.uniform(3)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(ak.float64, testArray.dtype)
        self.assertEqual([3], testArray.shape)

        testArray = ak.random.uniform(np.int64(3))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(ak.float64, testArray.dtype)
        self.assertEqual([3], testArray.shape)

        uArray = ak.random.uniform(size=3, low=0, high=5, seed=0)
        self.assertListEqual(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098],
            uArray.to_list(),
        )

        uArray = ak.random.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        self.assertListEqual(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098],
            uArray.to_list(),
        )

        with self.assertRaises(TypeError):
            ak.random.uniform(low="0", high=5, size=100)

        with self.assertRaises(TypeError):
            ak.random.uniform(low=0, high="5", size=100)

        with self.assertRaises(TypeError):
            ak.random.uniform(low=0, high=5, size="100")

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.uniform(low=np.uint8(0), high=5, size=np.uint32(100))

    def test_legacy_standard_normal(self):
        pda = ak.random.standard_normal(100)
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)

        pda = ak.random.standard_normal(np.int64(100))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)

        pda = ak.random.standard_normal(np.int64(100), np.int64(1))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)

        npda = pda.to_ndarray()
        pda = ak.random.standard_normal(np.int64(100), np.int64(1))

        self.assertListEqual(npda.tolist(), pda.to_list())

        with self.assertRaises(TypeError):
            ak.random.standard_normal("100")

        with self.assertRaises(TypeError):
            ak.random.standard_normal(100.0)

        with self.assertRaises(ValueError):
            ak.random.standard_normal(-1)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.standard_normal(np.uint8(100))
        ak.random.standard_normal(np.uint16(100))
        ak.random.standard_normal(np.uint32(100))
