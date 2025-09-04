from collections import Counter
from itertools import product
import math
import os

import numpy as np
import pytest
from scipy import stats as sp_stats

import arkouda as ak
from arkouda.numpy import random
from arkouda.scipy import chisquare as akchisquare
from arkouda.testing import assert_almost_equivalent

INT_FLOAT = [ak.int64, ak.float64]


class TestRandom:
    def test_random_docstrings(self):
        import doctest

        result = doctest.testmod(random, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_nl_neq(1)
    def test_random_generator_docstrings(self):
        import doctest

        from arkouda.numpy.random import generator

        result = doctest.testmod(generator, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_integers(self):
        # verify same seed gives different but reproducible arrays
        rng = ak.random.default_rng(pytest.seed)
        first = rng.integers(-(2**32), 2**32, 10)
        second = rng.integers(-(2**32), 2**32, 10)
        assert first.tolist() != second.tolist()

        rng = ak.random.default_rng(pytest.seed)
        same_seed_first = rng.integers(-(2**32), 2**32, 10)
        same_seed_second = rng.integers(-(2**32), 2**32, 10)
        assert first.tolist() == same_seed_first.tolist()
        assert second.tolist() == same_seed_second.tolist()

        # test endpoint
        rng = ak.random.default_rng()
        all_zero = rng.integers(0, 1, 20)
        assert all(all_zero.to_ndarray() == 0)

        not_all_zero = rng.integers(0, 1, 20, endpoint=True)
        assert any(not_all_zero.to_ndarray() != 0)

        # verify that switching dtype and function from seed is still reproducible
        rng = ak.random.default_rng(pytest.seed + 1)
        uint_arr = rng.integers(0, 2**32, size=10, dtype="uint")
        float_arr = rng.uniform(-1.0, 1.0, size=5)
        bool_arr = rng.integers(0, 1, size=20, dtype="bool")
        int_arr = rng.integers(-(2**32), 2**32, size=10, dtype="int")

        rng = ak.random.default_rng(pytest.seed + 1)
        same_seed_uint_arr = rng.integers(0, 2**32, size=10, dtype="uint")
        same_seed_float_arr = rng.uniform(-1.0, 1.0, size=5)
        same_seed_bool_arr = rng.integers(0, 1, size=20, dtype="bool")
        same_seed_int_arr = rng.integers(-(2**32), 2**32, size=10, dtype="int")

        assert uint_arr.tolist() == same_seed_uint_arr.tolist()
        assert float_arr.tolist() == same_seed_float_arr.tolist()
        assert bool_arr.tolist() == same_seed_bool_arr.tolist()
        assert int_arr.tolist() == same_seed_int_arr.tolist()

        # verify within bounds (lower inclusive and upper exclusive)
        rng = ak.random.default_rng()
        bounded_arr = rng.integers(-5, 5, 1000)
        assert all(bounded_arr.to_ndarray() >= -5)
        assert all(bounded_arr.to_ndarray() < 5)

    @pytest.mark.parametrize("data_type", INT_FLOAT)
    @pytest.mark.parametrize("method", ["FisherYates", "MergeShuffle", "Feistel"])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_shuffle(self, data_type, method, size):
        # ints are checked for equality; floats are checked for closeness

        def check(a, b, t):
            return (a == b).all() if t is ak.int64 else np.allclose(a.tolist(), b.tolist())

        # verify all the same elements are in the shuffle as in the original

        rng = ak.random.default_rng(pytest.seed)
        rnfunc = rng.integers if data_type is ak.int64 else rng.uniform
        pda = rnfunc(-(2**32), 2**32, size)
        pda_copy = pda[:]
        rng.shuffle(pda, method=method)

        assert not ak.all(pda == pda_copy)
        assert check(ak.sort(pda), ak.sort(pda_copy), data_type)

        # verify same seed gives reproducible arrays

        rng = ak.random.default_rng(pytest.seed)
        rnfunc = rng.integers if data_type is ak.int64 else rng.uniform
        pda_prime = rnfunc(-(2**32), 2**32, size)
        rng.shuffle(pda_prime, method=method)

        assert check(pda, pda_prime, data_type)

    @pytest.mark.parametrize("data_type", INT_FLOAT)
    @pytest.mark.parametrize("method", ["FisherYates", "Argsort"])
    def test_permutation(self, data_type, method):
        # ints are checked for equality; floats are checked for closeness

        def check(a, b, t):
            return (a == b).all() if t is ak.int64 else np.allclose(a.tolist(), b.tolist())

        # verify all the same elements are in the permutation as in the original

        rng = ak.random.default_rng(pytest.seed)
        range_permute = rng.permutation(20, method=method)
        assert (ak.arange(20) == ak.sort(range_permute)).all()  # range is always int

        # verify same seed gives reproducible arrays

        rng = ak.random.default_rng(pytest.seed)
        rnfunc = rng.integers if data_type is ak.int64 else rng.uniform
        pda = rnfunc(-(2**32), 2**32, 10)
        permuted = rng.permutation(pda, method=method)
        assert check(ak.sort(pda), ak.sort(permuted), data_type)

        # verify same seed gives reproducible permutations

        rng = ak.random.default_rng(pytest.seed)
        same_seed_range_permute = rng.permutation(20, method=method)
        assert check(range_permute, same_seed_range_permute, data_type)

        # verify all the same elements are in permutation as in the original

        rng = ak.random.default_rng(pytest.seed)
        rnfunc = rng.integers if data_type is ak.int64 else rng.uniform
        pda_p = rnfunc(-(2**32), 2**32, 10)
        permuted_p = rng.permutation(pda_p, method=method)
        assert check(ak.sort(pda_p), ak.sort(permuted_p), data_type)

    def test_uniform(self):
        # verify same seed gives different but reproducible arrays
        rng = ak.random.default_rng(pytest.seed)
        first = rng.uniform(-(2**32), 2**32, 10)
        second = rng.uniform(-(2**32), 2**32, 10)
        assert first.tolist() != second.tolist()

        rng = ak.random.default_rng(pytest.seed)
        same_seed_first = rng.uniform(-(2**32), 2**32, 10)
        same_seed_second = rng.uniform(-(2**32), 2**32, 10)
        assert np.allclose(first.tolist(), same_seed_first.tolist())
        assert np.allclose(second.tolist(), same_seed_second.tolist())

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

        rng = ak.random.default_rng(pytest.seed)
        weights = rng.uniform(size=10)
        a_vals = [
            10,
            rng.integers(0, 2**32, size=10, dtype="uint"),
            rng.uniform(-1.0, 1.0, size=10),
            rng.integers(0, 1, size=10, dtype="bool"),
            rng.integers(-(2**32), 2**32, size=10, dtype="int"),
        ]

        rng = ak.random.default_rng(pytest.seed)
        choice_arrays = []
        for a in a_vals:
            for size in 5, 10:
                for replace in True, False:
                    for p in [None, weights]:
                        choice_arrays.append(rng.choice(a, size, replace, p))

        # reset generator to ensure we get the same arrays
        rng = ak.random.default_rng(pytest.seed)
        for a in a_vals:
            for size in 5, 10:
                for replace in True, False:
                    for p in [None, weights]:
                        previous = choice_arrays.pop(0)
                        current = rng.choice(a, size, replace, p)
                        assert np.allclose(previous.tolist(), current.tolist())

    def test_logistic(self):
        scal = 2
        arr = ak.arange(5)

        for loc, scale in product([scal, arr], [scal, arr]):
            rng = ak.random.default_rng(pytest.seed)
            num_samples = 5
            log_sample = rng.logistic(loc=loc, scale=scale, size=num_samples).tolist()

            rng = ak.random.default_rng(pytest.seed)
            assert rng.logistic(loc=loc, scale=scale, size=num_samples).tolist() == log_sample

    def test_lognormal(self):
        scal = 2
        arr = ak.arange(5)

        for mean, sigma in product([scal, arr], [scal, arr]):
            rng = ak.random.default_rng(pytest.seed)
            num_samples = 5
            log_sample = rng.lognormal(mean=mean, sigma=sigma, size=num_samples).tolist()

            rng = ak.random.default_rng(pytest.seed)
            assert rng.lognormal(mean=mean, sigma=sigma, size=num_samples).tolist() == log_sample

    def test_normal(self):
        rng = ak.random.default_rng(pytest.seed)
        both_scalar = rng.normal(loc=10, scale=2, size=10).tolist()
        scale_scalar = rng.normal(loc=ak.array([0, 10, 20]), scale=1, size=3).tolist()
        loc_scalar = rng.normal(loc=10, scale=ak.array([1, 2, 3]), size=3).tolist()
        both_array = rng.normal(loc=ak.array([0, 10, 20]), scale=ak.array([1, 2, 3]), size=3).tolist()

        # redeclare rng with same seed to test reproducibility
        rng = ak.random.default_rng(pytest.seed)
        assert rng.normal(loc=10, scale=2, size=10).tolist() == both_scalar
        assert rng.normal(loc=ak.array([0, 10, 20]), scale=1, size=3).tolist() == scale_scalar
        assert rng.normal(loc=10, scale=ak.array([1, 2, 3]), size=3).tolist() == loc_scalar
        assert (
            rng.normal(loc=ak.array([0, 10, 20]), scale=ak.array([1, 2, 3]), size=3).tolist()
            == both_array
        )

    def test_standard_gamma(self):
        rng = ak.random.default_rng(pytest.seed)
        num_samples = 5
        # scalar shape
        scal_sample = rng.standard_gamma(2, size=num_samples).tolist()

        # array shape
        arr_sample = rng.standard_gamma(ak.arange(5), size=num_samples).tolist()

        # reset rng with same seed and ensure we get same results
        rng = ak.random.default_rng(pytest.seed)
        assert rng.standard_gamma(2, size=num_samples).tolist() == scal_sample
        assert rng.standard_gamma(ak.arange(num_samples), size=num_samples).tolist() == arr_sample

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_standard_gamma_no_seed(self, size):
        num_tests = 10

        for i in range(num_tests):
            rng = ak.random.default_rng()

            arr_sample = rng.standard_gamma(rng.uniform(0, 10, size), size=size)

            assert ak.sum(arr_sample > 0) > size / 2, "Majority of values should be > 0."

    def test_standard_gamma_hypothesis_testing(self):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.

        rng = ak.random.default_rng(pytest.seed)
        num_samples = 10**3

        k = rng.uniform(0, 10)
        sample = rng.standard_gamma(k, size=num_samples)
        sample_list = sample.tolist()

        # second goodness of fit test against the distribution with proper mean and std
        good_fit_res = sp_stats.goodness_of_fit(sp_stats.gamma, sample_list, known_params={"a": k})
        assert good_fit_res.pvalue > 0.05

    def test_standard_gamma_kolmogorov_smirnov_testing(self):
        from scipy.stats import gamma, kstest

        num_samples = 10**3

        rng = ak.random.default_rng(pytest.seed)
        k = rng.uniform(0, 10)
        sample = rng.standard_gamma(k, size=num_samples).to_ndarray()
        ks = kstest(sample, gamma.cdf, args=(k, 0, 1))
        assert ks.pvalue > 0.05

        rng = ak.random.default_rng(pytest.seed)
        k = 0.5
        sample = rng.standard_gamma(k, size=num_samples).to_ndarray()
        ks = kstest(sample, gamma.cdf, args=(k, 0, 1))
        assert ks.pvalue > 0.05

        rng = ak.random.default_rng(pytest.seed)
        k = 5
        sample = rng.standard_gamma(k, size=num_samples).to_ndarray()
        ks = kstest(sample, gamma.cdf, args=(k, 0, 1))
        assert ks.pvalue > 0.05

    def test_poisson(self):
        rng = ak.random.default_rng(pytest.seed)
        num_samples = 5
        # scalar lambda
        scal_lam = 2
        scal_sample = rng.poisson(lam=scal_lam, size=num_samples).tolist()

        # array lambda
        arr_lam = ak.arange(5)
        arr_sample = rng.poisson(lam=arr_lam, size=num_samples).tolist()

        # reset rng with same seed and ensure we get same results
        rng = ak.random.default_rng(pytest.seed)
        assert rng.poisson(lam=scal_lam, size=num_samples).tolist() == scal_sample
        assert rng.poisson(lam=arr_lam, size=num_samples).tolist() == arr_sample

    def test_poisson_seed_reproducibility(self):
        # test resolution of issue #3322, same seed gives same result across machines / num locales
        iseed = 11  # retains non pytest.seed because it asserts specific values
        # test with many orders of magnitude to ensure we test different remainders and case where
        # all elements are pulled to first locale i.e. total_size < (minimum elemsPerStream = 256)
        saved_seeded_file_patterns = ["second_order*", "third_order*", "fourth_order*"]

        # directory of this file
        file_dir = os.path.dirname(os.path.realpath(__file__))
        for i, f_name in zip(range(2, 5), saved_seeded_file_patterns):
            generated = ak.random.default_rng(seed=iseed).poisson(size=10**i)
            saved = ak.read_parquet(f"{file_dir}/saved_seeded_random/{f_name}")["array"]
            assert (generated == saved).all()

    def test_exponential(self):
        rng = ak.random.default_rng(pytest.seed)
        num_samples = 5
        # scalar scale
        scal_scale = 2
        scal_sample = rng.exponential(scale=scal_scale, size=num_samples).tolist()

        # array scale
        arr_scale = ak.arange(5)
        arr_sample = rng.exponential(scale=arr_scale, size=num_samples).tolist()

        # reset rng with same seed and ensure we get same results
        rng = ak.random.default_rng(pytest.seed)
        assert rng.exponential(scale=scal_scale, size=num_samples).tolist() == scal_sample
        assert rng.exponential(scale=arr_scale, size=num_samples).tolist() == arr_sample

    def test_choice_hypothesis_testing(self):
        # perform a weighted sample and use chisquare to test
        # if the observed frequency matches the expected frequency

        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.

        rng = ak.random.default_rng(pytest.seed)
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

    @pytest.mark.parametrize("method", ["zig", "inv"])
    def test_exponential_hypothesis_testing(self, method):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.

        rng = ak.random.default_rng(pytest.seed)
        num_samples = 10**4

        scale = rng.uniform(0, 10)
        sample = rng.exponential(scale=scale, size=num_samples, method=method)
        sample_list = sample.tolist()

        # do the Kolmogorov-Smirnov test for goodness of fit
        ks_res = sp_stats.kstest(
            rvs=sample_list,
            cdf=sp_stats.expon.cdf,
            args=(0, scale),
        )
        assert ks_res.pvalue > 0.05

    @pytest.mark.parametrize("method", ["zig", "box"])
    def test_normal_hypothesis_testing(self, method):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.

        rng = ak.random.default_rng(pytest.seed)
        num_samples = 10**4

        mean = rng.uniform(-10, 10)
        deviation = rng.uniform(0, 10)
        sample = rng.normal(loc=mean, scale=deviation, size=num_samples, method=method)
        sample_list = sample.tolist()

        # first test if samples are normal at all
        _, pval = sp_stats.normaltest(sample_list)

        # if pval <= 0.05, the difference from the expected distribution is significant
        assert pval > 0.05

        # second goodness of fit test against the distribution with proper mean and std
        good_fit_res = sp_stats.goodness_of_fit(
            sp_stats.norm, sample_list, known_params={"loc": mean, "scale": deviation}
        )
        assert good_fit_res.pvalue > 0.05

    @pytest.mark.parametrize("method", ["zig", "box"])
    def test_lognormal_hypothesis_testing(self, method):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.

        rng = ak.random.default_rng(pytest.seed)
        num_samples = 10**4

        mean = rng.uniform(-10, 10)
        deviation = rng.uniform(0, 10)
        sample = rng.lognormal(mean=mean, sigma=deviation, size=num_samples, method=method)

        log_sample_list = np.log(sample.to_ndarray()).tolist()

        # first test if samples are normal at all
        _, pval = sp_stats.normaltest(log_sample_list)

        # if pval <= 0.05, the difference from the expected distribution is significant
        assert pval > 0.05

        # second goodness of fit test against the distribution with proper mean and std
        good_fit_res = sp_stats.goodness_of_fit(
            sp_stats.norm,
            log_sample_list,
            known_params={"loc": mean, "scale": deviation},
        )
        assert good_fit_res.pvalue > 0.05

    def test_poisson_hypothesis_testing(self):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.

        rng = ak.random.default_rng(pytest.seed)
        num_samples = 10**4
        lam = rng.uniform(0, 10)

        sample = rng.poisson(lam=lam, size=num_samples)
        count_dict = Counter(sample.tolist())

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

    def test_logistic_hypothesis_testing(self):
        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05.

        rng = ak.random.default_rng(pytest.seed)
        num_samples = 10**4
        mu = rng.uniform(0, 10)
        scale = rng.uniform(0, 10)

        sample = rng.logistic(loc=mu, scale=scale, size=num_samples)
        sample_list = sample.tolist()

        # second goodness of fit test against the distribution with proper mean and std
        good_fit_res = sp_stats.goodness_of_fit(
            sp_stats.logistic, sample_list, known_params={"loc": mu, "scale": scale}
        )
        assert good_fit_res.pvalue > 0.05

    def test_legacy_rand_scalar(self):
        assert math.isclose(ak.random.rand(seed=1701), np.float64(0.011410423448327005))

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_legacy_rand(self, size):
        known = ak.array([0.011410423448327005, 0.73618171558685619])
        given = ak.random.rand(2, seed=1701)
        assert_almost_equivalent(known, given)

    @pytest.mark.skip_if_rank_not_compiled(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_legacy_rand2D(self, size):
        known = ak.array(
            [[0.011410423448327005, 0.73618171558685619], [0.12367222192448891, 0.95616789699591898]]
        )
        given = ak.random.rand(2, 2, seed=1701)
        assert_almost_equivalent(known, given)

    @pytest.mark.skip_if_rank_not_compiled(3)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_legacy_rand3D(self, size):
        known = ak.array(
            [
                [
                    [0.011410423448327005, 0.73618171558685619],
                    [0.12367222192448891, 0.95616789699591898],
                ],
                [
                    [0.36427886480971333, 0.71482330432026153],
                    [0.66334928222218692, 0.071647713917930858],
                ],
            ]
        )
        given = ak.random.rand(2, 2, 2, seed=1701)
        assert_almost_equivalent(known, given)

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

        test_array = ak.random.randint(0, 1, 5, dtype=ak.bool_)
        assert ak.bool_ == test_array.dtype

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

        with pytest.raises(ValueError, match="size must be >= 0, ndim >= 1, and high >= low"):
            ak.random.randint(low=0, high=1, size=-1, dtype=ak.float64)

        with pytest.raises(ValueError, match="size must be >= 0, ndim >= 1, and high >= low"):
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
        #  This test retains the non pytest.seed, because it asserts specific values.
        values = ak.random.randint(1, 5, 10, seed=2)

        assert [4, 3, 1, 3, 2, 4, 4, 2, 3, 4] == values.tolist()

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
        ] == values.tolist()

        values = ak.random.randint(1, 5, 10, dtype=ak.bool_, seed=2)
        assert [
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
        ] == values.tolist()

        values = ak.random.randint(1, 5, 10, dtype=bool, seed=2)
        assert [
            False,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
        ] == values.tolist()

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

        #  The next two tests also retain the non pytest.seed, because they assert specific values.

        uArray = ak.random.uniform(size=3, low=0, high=5, seed=0)
        assert np.allclose(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098],
            uArray.tolist(),
        )

        uArray = ak.random.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        assert np.allclose(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098],
            uArray.tolist(),
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

        assert np.allclose(npda.tolist(), pda.tolist())

        with pytest.raises(TypeError):
            ak.random.standard_normal("100")

        with pytest.raises(TypeError):
            ak.random.standard_normal(100.0)

        with pytest.raises(ValueError, match="The size parameter must be > 0"):
            ak.random.standard_normal(-1)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random.standard_normal(np.uint8(100))
        ak.random.standard_normal(np.uint16(100))
        ak.random.standard_normal(np.uint32(100))
