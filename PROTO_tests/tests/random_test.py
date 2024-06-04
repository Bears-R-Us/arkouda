from collections import Counter

import numpy as np
import pytest
from scipy import stats as sp_stats

import arkouda as ak
from arkouda.scipy import chisquare as akchisquare


class TestRandom:
    def test_integers(self):
        # verify same seed gives different but reproducible arrays
        rng = ak.random.default_rng(18)
        first = rng.integers(-(2**32), 2**32, 10)
        second = rng.integers(-(2**32), 2**32, 10)
        assert first.to_list() != second.to_list()

        rng = ak.random.default_rng(18)
        same_seed_first = rng.integers(-(2**32), 2**32, 10)
        same_seed_second = rng.integers(-(2**32), 2**32, 10)
        assert first.to_list() == same_seed_first.to_list()
        second.to_list() == same_seed_second.to_list()

        # test endpoint
        rng = ak.random.default_rng()
        all_zero = rng.integers(0, 1, 20)
        assert all(all_zero.to_ndarray() == 0)

        not_all_zero = rng.integers(0, 1, 20, endpoint=True)
        assert any(not_all_zero.to_ndarray() != 0)

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

        assert uint_arr.to_list() == same_seed_uint_arr.to_list()
        assert float_arr.to_list() == same_seed_float_arr.to_list()
        assert bool_arr.to_list() == same_seed_bool_arr.to_list()
        assert int_arr.to_list() == same_seed_int_arr.to_list()

        # verify within bounds (lower inclusive and upper exclusive)
        rng = ak.random.default_rng()
        bounded_arr = rng.integers(-5, 5, 1000)
        assert all(bounded_arr.to_ndarray() >= -5)
        assert all(bounded_arr.to_ndarray() < 5)

    def test_shuffle(self):
        # verify same seed gives reproducible arrays
        rng = ak.random.default_rng(18)

        int_pda = rng.integers(-(2**32), 2**32, 10)
        pda_copy = int_pda[:]
        # shuffle int_pda in place
        rng.shuffle(int_pda)
        # verify all the same elements are in permutation as the original
        assert (ak.sort(int_pda) == ak.sort(pda_copy)).all()

        float_pda = rng.uniform(-(2**32), 2**32, 10)
        pda_copy = float_pda[:]
        rng.shuffle(float_pda)
        # verify all the same elements are in permutation as the original
        assert (ak.sort(float_pda) == ak.sort(pda_copy)).all()

        rng = ak.random.default_rng(18)

        pda = rng.integers(-(2**32), 2**32, 10)
        rng.shuffle(pda)
        assert (pda == int_pda).all()

        pda = rng.uniform(-(2**32), 2**32, 10)
        rng.shuffle(pda)
        assert np.allclose(pda.to_list(), float_pda.to_list())

    def test_permutation(self):
        # verify same seed gives reproducible arrays
        rng = ak.random.default_rng(18)
        # providing just a number permutes the range(num)
        range_permute = rng.permutation(20)
        assert (ak.arange(20) == ak.sort(range_permute)).all()

        pda = rng.integers(-(2**32), 2**32, 10)
        array_permute = rng.permutation(pda)
        # verify all the same elements are in permutation as the original
        assert (ak.sort(pda) == ak.sort(array_permute)).all()

        pda = rng.uniform(-(2**32), 2**32, 10)
        float_array_permute = rng.permutation(pda)
        # verify all the same elements are in permutation as the original
        assert np.allclose(ak.sort(pda).to_list(), ak.sort(float_array_permute).to_list())

        rng = ak.random.default_rng(18)
        same_seed_range_permute = rng.permutation(20)
        assert (range_permute == same_seed_range_permute).all()

        pda = rng.integers(-(2**32), 2**32, 10)
        same_seed_array_permute = rng.permutation(pda)
        assert (array_permute == same_seed_array_permute).all()

        pda = rng.uniform(-(2**32), 2**32, 10)
        same_seed_float_array_permute = rng.permutation(pda)
        # verify all the same elements are in permutation as the original
        assert np.allclose(float_array_permute.to_list(), same_seed_float_array_permute.to_list())

    def test_uniform(self):
        # verify same seed gives different but reproducible arrays
        rng = ak.random.default_rng(18)
        first = rng.uniform(-(2**32), 2**32, 10)
        second = rng.uniform(-(2**32), 2**32, 10)
        assert first.to_list() != second.to_list()

        rng = ak.random.default_rng(18)
        same_seed_first = rng.uniform(-(2**32), 2**32, 10)
        same_seed_second = rng.uniform(-(2**32), 2**32, 10)
        assert np.allclose(first.to_list(), same_seed_first.to_list())
        assert np.allclose(second.to_list(), same_seed_second.to_list())

        # verify within bounds (lower inclusive and upper exclusive)
        rng = ak.random.default_rng()
        bounded_arr = rng.uniform(-5, 5, 1000)
        assert all(bounded_arr.to_ndarray() >= -5)
        assert all(bounded_arr.to_ndarray() < 5)

    def test_choice(self):
        # verify without replacement works
        rng = ak.random.default_rng()
        # test domains and selecting all
        domain_choice = rng.choice(20, 20, replace=False)
        # since our populations and sample size is the same without replacement,
        # we should see all values
        assert (ak.sort(domain_choice) == ak.arange(20)).all()

        # test arrays and not selecting all
        perm = rng.permutation(100)
        array_choice = rng.choice(perm, 95, replace=False)
        # verify all unique
        _, count = ak.GroupBy(array_choice).size()
        assert (count == 1).all()

        # test single value
        scalar = rng.choice(5)
        assert type(scalar) is np.int64
        assert scalar in [0, 1, 2, 3, 4]

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
                        assert np.allclose(previous.to_list(), current.to_list())

    def test_normal(self):
        rng = ak.random.default_rng(17)
        both_scalar = rng.normal(loc=10, scale=2, size=10).to_list()
        scale_scalar = rng.normal(loc=ak.array([0, 10, 20]), scale=1, size=3).to_list()
        loc_scalar = rng.normal(loc=10, scale=ak.array([1, 2, 3]), size=3).to_list()
        both_array = rng.normal(loc=ak.array([0, 10, 20]), scale=ak.array([1, 2, 3]), size=3).to_list()

        # redeclare rng with same seed to test reproducibility
        rng = ak.random.default_rng(17)
        assert rng.normal(loc=10, scale=2, size=10).to_list() == both_scalar
        assert rng.normal(loc=ak.array([0, 10, 20]), scale=1, size=3).to_list() == scale_scalar
        assert rng.normal(loc=10, scale=ak.array([1, 2, 3]), size=3).to_list() == loc_scalar
        assert (
            rng.normal(loc=ak.array([0, 10, 20]), scale=ak.array([1, 2, 3]), size=3).to_list()
            == both_array
        )

    def test_poissson(self):
        rng = ak.random.default_rng(17)
        num_samples = 5
        # scalar lambda
        scal_lam = 2
        scal_sample = rng.poisson(lam=scal_lam, size=num_samples).to_list()

        # array lambda
        arr_lam = ak.arange(5)
        arr_sample = rng.poisson(lam=arr_lam, size=num_samples).to_list()

        # reset rng with same seed and ensure we get same results
        rng = ak.random.default_rng(17)
        assert rng.poisson(lam=scal_lam, size=num_samples).to_list() == scal_sample
        assert rng.poisson(lam=arr_lam, size=num_samples).to_list() == arr_sample

    def test_choice_hypothesis_testing(self):
        # perform a weighted sample and use chisquare to test
        # if the observed frequency matches the expected frequency

        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.
        rng = ak.random.default_rng(43)
        num_samples = 10**4

        weights = ak.array([0.25, 0.15, 0.20, 0.10, 0.30])
        weighted_sample = rng.choice(ak.arange(5), size=num_samples, p=weights)

        # count how many of each category we saw
        uk, f_obs = ak.GroupBy(weighted_sample).size()

        # I think the keys should always be sorted but just in case
        if not ak.is_sorted(uk):
            f_obs = f_obs[ak.argsort(uk)]

        f_exp = weights * num_samples
        _, pval = akchisquare(f_obs=f_obs, f_exp=f_exp)

        # if pval <= 0.05, the difference from the expected distribution is significant
        assert pval > 0.05

    def test_normal_hypothesis_testing(self):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.
        rng = ak.random.default_rng(43)
        num_samples = 10**4

        mean = rng.uniform(-10, 10)
        deviation = rng.uniform(0, 10)
        sample = rng.normal(loc=mean, scale=deviation, size=num_samples)
        sample_list = sample.to_list()

        # first test if samples are normal at all
        _, pval = sp_stats.normaltest(sample_list)

        # if pval <= 0.05, the difference from the expected distribution is significant
        assert pval > 0.05

        # second goodness of fit test against the distribution with proper mean and std
        good_fit_res = sp_stats.goodness_of_fit(
            sp_stats.norm, sample_list, known_params={"loc": mean, "scale": deviation}
        )
        assert good_fit_res.pvalue > 0.05

    def test_poisson_hypothesis_testing(self):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.
        rng = ak.random.default_rng(43)
        num_samples = 10**4
        lam = rng.uniform(0, 10)

        sample = rng.poisson(lam=lam, size=num_samples)
        count_dict = Counter(sample.to_list())

        # the sum of exp freq and obs freq must be within 1e-08, so we use
        # the isf (inverse survival function where survival function is 1-cdf) to
        # find out how many elements we need to ensure we're within that tolerance
        num_elems = int(sp_stats.poisson.isf(1e-09, mu=lam))

        obs_counts = np.array([0] * num_elems)
        for k, v in count_dict.items():
            obs_counts[k] = v

        # use the probability mass function to get the probability of seeing each value
        # and multiply by num_samples to get the expected counts
        exp_counts = sp_stats.poisson.pmf(range(num_elems), mu=lam) * num_samples
        _, pval = sp_stats.chisquare(f_obs=obs_counts, f_exp=exp_counts)
        assert pval > 0.05

    def test_legacy_randint(self):
        testArray = ak.random.randint(0, 10, 5)
        assert isinstance(testArray, ak.pdarray)
        assert 5 == len(testArray)
        assert ak.int64 == testArray.dtype

        testArray = ak.random.randint(np.int64(0), np.int64(10), np.int64(5))
        assert isinstance(testArray, ak.pdarray)
        assert 5 == len(testArray)
        assert ak.int64 == testArray.dtype

        testArray = ak.random.randint(np.float64(0), np.float64(10), np.int64(5))
        assert isinstance(testArray, ak.pdarray)
        assert 5 == len(testArray)
        assert ak.int64 == testArray.dtype

        test_ndarray = testArray.to_ndarray()

        for value in test_ndarray:
            assert 0 <= value <= 10

        test_array = ak.random.randint(0, 1, 3, dtype=ak.float64)
        assert ak.float64 == test_array.dtype

        test_array = ak.random.randint(0, 1, 5, dtype=ak.bool)
        assert ak.bool == test_array.dtype

        test_ndarray = test_array.to_ndarray()

        # test resolution of modulus overflow - issue #1174
        test_array = ak.random.randint(-(2**63), 2**63 - 1, 10)
        to_validate = np.full(10, -(2**63))
        assert not (test_array.to_ndarray() == to_validate).all()

        for value in test_ndarray:
            assert value in [True, False]

        with pytest.raises(TypeError):
            ak.random.randint(low=5)

        with pytest.raises(TypeError):
            ak.random.randint(high=5)

        with pytest.raises(TypeError):
            ak.random.randint()

        with pytest.raises(ValueError):
            ak.random.randint(low=0, high=1, size=-1, dtype=ak.float64)

        with pytest.raises(ValueError):
            ak.random.randint(low=1, high=0, size=1, dtype=ak.float64)

        with pytest.raises(TypeError):
            ak.random.randint(0, 1, "1000")

        with pytest.raises(TypeError):
            ak.random.randint("0", 1, 1000)

        with pytest.raises(TypeError):
            ak.random.randint(0, "1", 1000)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.randint(low=np.uint8(1), high=np.uint16(100), size=np.uint32(100))

    def test_legacy_randint_with_seed(self):
        values = ak.random.randint(1, 5, 10, seed=2)

        assert [4, 3, 1, 3, 2, 4, 4, 2, 3, 4] == values.to_list()

        values = ak.random.randint(1, 5, 10, dtype=ak.float64, seed=2)

        assert [
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
        ] == values.to_list()

        values = ak.random.randint(1, 5, 10, dtype=ak.bool, seed=2)
        assert [False, True, True, True, True, False, True, True, True, True] == values.to_list()

        values = ak.random.randint(1, 5, 10, dtype=bool, seed=2)
        assert [False, True, True, True, True, False, True, True, True, True] == values.to_list()

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.randint(np.uint8(1), np.uint32(5), np.uint16(10), seed=np.uint8(2))

    def test_legacy_uniform(self):
        testArray = ak.random.uniform(3)
        assert isinstance(testArray, ak.pdarray)
        assert 3 == len(testArray)
        assert ak.float64 == testArray.dtype

        testArray = ak.random.uniform(np.int64(3))
        assert isinstance(testArray, ak.pdarray)
        assert 3 == len(testArray)
        assert ak.float64 == testArray.dtype

        uArray = ak.random.uniform(size=3, low=0, high=5, seed=0)
        assert np.allclose(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098], uArray.to_list()
        )

        uArray = ak.random.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        assert np.allclose(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098], uArray.to_list()
        )

        with pytest.raises(TypeError):
            ak.random.uniform(low="0", high=5, size=100)

        with pytest.raises(TypeError):
            ak.random.uniform(low=0, high="5", size=100)

        with pytest.raises(TypeError):
            ak.random.uniform(low=0, high=5, size="100")

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.uniform(low=np.uint8(0), high=5, size=np.uint32(100))

    def test_legacy_standard_normal(self):
        pda = ak.random.standard_normal(100)
        assert isinstance(pda, ak.pdarray)
        assert 100 == len(pda)
        assert ak.float64 == pda.dtype

        pda = ak.random.standard_normal(np.int64(100))
        assert isinstance(pda, ak.pdarray)
        assert 100 == len(pda)
        assert ak.float64 == pda.dtype

        pda = ak.random.standard_normal(np.int64(100), np.int64(1))
        assert isinstance(pda, ak.pdarray)
        assert 100 == len(pda)
        assert ak.float64 == pda.dtype

        npda = pda.to_ndarray()
        pda = ak.random.standard_normal(np.int64(100), np.int64(1))

        assert np.allclose(npda.tolist(), pda.to_list())

        with pytest.raises(TypeError):
            ak.random.standard_normal("100")

        with pytest.raises(TypeError):
            ak.random.standard_normal(100.0)

        with pytest.raises(ValueError):
            ak.random.standard_normal(-1)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.standard_normal(np.uint8(100))
        ak.random.standard_normal(np.uint16(100))
        ak.random.standard_normal(np.uint32(100))
