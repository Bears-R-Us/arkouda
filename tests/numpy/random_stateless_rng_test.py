import math

import numpy as np
import pytest

from scipy import stats as sp_stats

import arkouda as ak

from arkouda.numpy.dtypes import float64, uint64
from arkouda.numpy.pdarraycreation import zeros as ak_zeros


@pytest.mark.requires_chapel_module("SplitMix64RNG")
class TestStatelessRNG:
    def test_stateless_u64_same_params_reproducible(self):
        rng = ak.random.default_rng()
        n = 10_000
        seed = 123
        stream = 0
        start_idx = 0

        a = rng._stateless_u64_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        b = rng._stateless_u64_from_index(n, seed=seed, stream=stream, start_idx=start_idx)

        assert a.dtype == uint64
        assert b.dtype == uint64
        assert ak.all(a == b)

    def test_stateless_u64_stream_changes_sequence(self):
        rng = ak.random.default_rng()
        n = 10_000
        seed = 123
        start_idx = 0

        a = rng._stateless_u64_from_index(n, seed=seed, stream=0, start_idx=start_idx)
        b = rng._stateless_u64_from_index(n, seed=seed, stream=1, start_idx=start_idx)

        # They should not be identical elementwise
        assert ak.any(a != b)

    def test_stateless_u64_start_idx_matches_slice(self):
        rng = ak.random.default_rng()
        n_total = 20_000
        seed = 999
        stream = 7

        full = rng._stateless_u64_from_index(n_total, seed=seed, stream=stream, start_idx=0)

        start_idx = 5_000
        n = 3_000
        sub = rng._stateless_u64_from_index(n, seed=seed, stream=stream, start_idx=start_idx)

        assert ak.all(sub == full[start_idx : start_idx + n])

    def test_fill_with_stateless_u64_inplace_matches_allocate(self):
        rng = ak.random.default_rng()
        n = 10_000
        seed = 42
        stream = 3
        start_idx = 1234

        x = ak_zeros(n, dtype=uint64)
        rng._fill_with_stateless_u64_from_index(x, seed=seed, stream=stream, start_idx=start_idx)

        y = rng._stateless_u64_from_index(n, seed=seed, stream=stream, start_idx=start_idx)

        assert x.dtype == uint64
        assert y.dtype == uint64
        assert ak.all(x == y)

    def test_stateless_uniform01_dtype_and_range(self):
        rng = ak.random.default_rng()
        n = 50_000
        seed = 1
        stream = 0
        start_idx = 0

        u = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)

        assert u.dtype == float64
        assert ak.all(u >= 0.0)
        assert ak.all(u < 1.0)

    def test_stateless_uniform01_same_params_reproducible(self):
        rng = ak.random.default_rng()
        n = 10_000
        seed = 321
        stream = 2
        start_idx = 99

        a = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        b = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)

        assert a.dtype == float64
        assert ak.all(a == b)

    def test_stateless_uniform01_start_idx_matches_slice(self):
        rng = ak.random.default_rng()
        n_total = 30_000
        seed = 2026
        stream = 11

        full = rng._stateless_uniform_01_from_index(n_total, seed=seed, stream=stream, start_idx=0)

        start_idx = 7_000
        n = 4_000
        sub = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)

        assert ak.all(sub == full[start_idx : start_idx + n])

    def test_fill_uniform01_inplace_matches_allocate(self):
        rng = ak.random.default_rng()
        n = 10_000
        seed = 8
        stream = 9
        start_idx = 10

        x = ak_zeros(n, dtype=float64)
        rng._fill_stateless_uniform_01_from_index(x, seed=seed, stream=stream, start_idx=start_idx)

        y = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)

        assert x.dtype == float64
        assert ak.all(x == y)

    def test_stateless_uniform01_stream_changes_sequence(self):
        rng = ak.random.default_rng()
        n = 10_000
        seed = 1234
        start_idx = 0

        a = rng._stateless_uniform_01_from_index(n, seed=seed, stream=0, start_idx=start_idx)
        b = rng._stateless_uniform_01_from_index(n, seed=seed, stream=1, start_idx=start_idx)

        assert ak.any(a != b)

    # -----------------------------
    # Helpers (non-flaky thresholds)
    # -----------------------------
    @staticmethod
    def _six_sigma_half(n: int) -> float:
        """6-sigma bound for deviation from 0.5 for a Bernoulli(0.5) mean."""
        return 6.0 * math.sqrt(0.25 / n)

    @staticmethod
    def _corr_bound(n: int) -> float:
        """Loose ~6/sqrt(n) correlation bound."""
        return 6.0 / math.sqrt(n)

    # -----------------------------
    # Uniform[0,1) statistical tests
    # -----------------------------
    def test_stateless_uniform01_ks_test(self):
        rng = ak.random.default_rng()
        n = 80_000
        seed, stream, start_idx = 123, 0, 0

        u = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        u_np = u.to_ndarray()

        # KS test against continuous U[0, 1)
        stat = sp_stats.kstest(u_np, "uniform")
        # Keep alpha conservative to avoid flakes; fixed seed makes this stable.
        assert stat.pvalue > 1e-4, stat

    def test_stateless_uniform01_chisquare_binned(self):
        rng = ak.random.default_rng()
        n = 200_000
        k = 256
        seed, stream, start_idx = 456, 2, 10

        u = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        u_np = u.to_ndarray()

        # Bin into k equal-width buckets on [0,1)
        bins = np.floor(u_np * k).astype(np.int64)
        # Safety: u in [0,1), so bins in [0,k-1]
        assert bins.min() >= 0
        assert bins.max() < k

        counts = np.bincount(bins, minlength=k)
        chisq = sp_stats.chisquare(counts)

        assert chisq.pvalue > 1e-4, chisq

    def test_stateless_uniform01_no_pathological_endpoints(self):
        rng = ak.random.default_rng()
        n = 200_000
        seed, stream, start_idx = 999, 7, 0

        u = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        u_np = u.to_ndarray()

        # Should never hit 1.0 due to top-53-bit mapping
        assert not np.any(u_np == 1.0)
        # Exact 0.0 is possible but astronomically rare; with this n it should be 0.
        assert np.sum(u_np == 0.0) == 0

    # -----------------------------
    # uint64 bit-level sanity tests
    # -----------------------------
    def test_stateless_u64_monobit_selected_bits(self):
        rng = ak.random.default_rng()
        n = 250_000
        seed, stream, start_idx = 42, 0, 0

        r = rng._stateless_u64_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        r_np = r.to_ndarray().astype(np.uint64, copy=False)

        # Check a few low and high bits
        bits_to_check = list(range(0, 8)) + list(range(56, 64))
        tol = self._six_sigma_half(n)

        for b in bits_to_check:
            ones = ((r_np >> np.uint64(b)) & np.uint64(1)).mean()
            assert abs(ones - 0.5) < tol, (b, ones, tol)

    def test_stateless_u64_low_byte_histogram_chisquare(self):
        rng = ak.random.default_rng()
        n = 300_000
        seed, stream, start_idx = 123, 3, 1000

        r = rng._stateless_u64_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        r_np = r.to_ndarray().astype(np.uint64, copy=False)

        low_byte = (r_np & np.uint64(0xFF)).astype(np.int64, copy=False)
        counts = np.bincount(low_byte, minlength=256)

        chisq = sp_stats.chisquare(counts)
        assert chisq.pvalue > 1e-4, chisq

    # -----------------------------
    # Independence / correlation
    # -----------------------------
    def test_stateless_uniform01_lag1_autocorrelation_small(self):
        rng = ak.random.default_rng()
        n = 120_000
        seed, stream, start_idx = 2026, 0, 0

        u = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=start_idx)
        u_np = u.to_ndarray()

        # Lag-1 correlation
        x = u_np[:-1]
        y = u_np[1:]
        corr = np.corrcoef(x, y)[0, 1]

        assert abs(corr) < self._corr_bound(len(x)), corr

    def test_stateless_uniform01_cross_stream_correlation_small(self):
        rng = ak.random.default_rng()
        n = 120_000
        seed, start_idx = 777, 0

        u0 = rng._stateless_uniform_01_from_index(
            n, seed=seed, stream=0, start_idx=start_idx
        ).to_ndarray()
        u1 = rng._stateless_uniform_01_from_index(
            n, seed=seed, stream=1, start_idx=start_idx
        ).to_ndarray()

        corr = np.corrcoef(u0, u1)[0, 1]
        assert abs(corr) < self._corr_bound(n), corr

    def test_stateless_uniform01_startidx_two_sample_ks(self):
        rng = ak.random.default_rng()
        n = 80_000
        seed, stream = 31415, 9

        u0 = rng._stateless_uniform_01_from_index(n, seed=seed, stream=stream, start_idx=0).to_ndarray()
        u_l = rng._stateless_uniform_01_from_index(
            n, seed=seed, stream=stream, start_idx=10_000_000
        ).to_ndarray()

        # Same distribution despite different index offsets
        stat = sp_stats.ks_2samp(u0, u_l)
        assert stat.pvalue > 1e-4, stat

    # -----------------------------
    # Practical "shuffle key" sanity
    # -----------------------------
    def test_stateless_u64_shuffle_key_not_index_biased(self):
        rng = ak.random.default_rng()
        n = 200_000
        seed, stream, start_idx = 9999, 0, 0

        keys = rng._stateless_u64_from_index(
            n, seed=seed, stream=stream, start_idx=start_idx
        ).to_ndarray()
        # Shuffle-by-keys => take indices of smallest fraction of keys
        frac = 0.01
        m = int(n * frac)

        idx = np.argpartition(keys, m)[:m]

        # If keys are “random enough”, selected indices should be ~uniform over 10 deciles
        dec = (idx * 10) // n
        counts = np.bincount(dec, minlength=10)

        chisq = sp_stats.chisquare(counts)
        assert chisq.pvalue > 1e-4, (chisq, counts)
