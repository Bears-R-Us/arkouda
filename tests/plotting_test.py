# tests/test_plotting.py
# Uses non-interactive backend and closes figures to avoid resource leaks.
import math
import os

import matplotlib
import numpy as np
import pytest

import arkouda as ak


# Import targets under test
from arkouda.plotting import plot_dist


matplotlib.use("Agg")  # must be set before importing pyplot


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    from matplotlib import pyplot as plt

    plt.close("all")


def _connected_to_arkouda():
    try:
        import arkouda as ak

        # Ak will raise if not connected; get_config hits server.
        ak.get_config()
        return True
    except Exception:
        return False


arkouda = pytest.importorskip("arkouda", reason="arkouda not installed")
pdarrayclass = pytest.importorskip("arkouda.pdarrayclass", reason="arkouda not installed")


class TestPlotting:
    def test_plotting_docstrings(self, tmp_path):
        import doctest

        from arkouda import plotting

        matplotlib.use("Agg")  # Use non-GUI backend to suppress plots
        # Run doctests inside a temp directory so any files (e.g., savefig) go here
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = doctest.testmod(plotting)
        finally:
            os.chdir(cwd)

        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_plot_dist_numpy_edges_counts_returns_fig_axes_and_titles(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=1_000)
        counts, edges = np.histogram(data, bins=20)

        fig, axes = plot_dist(edges, counts, xlabel="Value", show=False)

        assert fig is not None
        assert np.asarray(axes).shape == (2,)
        assert axes[0].get_title().lower() == "distribution"
        assert axes[1].get_title().lower() == "cumulative distribution"
        assert axes[0].get_xlabel() == "Value"
        # CDF should end at ~1
        y = axes[1].lines[0].get_ydata()
        assert math.isclose(float(y[-1]), 1.0, rel_tol=1e-12, abs_tol=1e-12)

    def test_plot_dist_accepts_centers_plus_counts(self):
        # centers (N) with counts (N)
        centers = np.linspace(-1, 1, 11)
        counts = np.arange(11, dtype=float)
        fig, axes = plot_dist(centers, counts, show=False)
        assert fig is not None
        assert np.asarray(axes).shape == (2,)

    def test_plot_dist_length_mismatch_raises(self):
        b = np.array([0.0, 1.0, 2.0])  # 3
        h = np.array(
            [5.0, 1.0, 2.0, 3.0, 4.0]
        )  # 5  -> mismatch (neither N nor N+1 relative to edges/centers)
        with pytest.raises(ValueError):
            plot_dist(b, h)

    def test_plot_dist_log_toggle(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        counts = np.array([1.0, 0.0, 10.0])  # has positives -> log can be applied safely
        fig, axes = plot_dist(edges, counts, log=True)
        assert axes[0].get_yscale() == "log"

        fig2, axes2 = plot_dist(edges, counts, log=False)
        assert axes2[0].get_yscale() == "linear"

    def test_plot_dist_newfig_false_draws_into_current_figure(self):
        # Make a current figure first; plot_dist(newfig=False) should reuse it
        from matplotlib import pyplot as plt

        fig0 = plt.figure()
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        counts = np.array([1.0, 2.0, 3.0])
        fig, axes = plot_dist(edges, counts, newfig=False)
        assert fig is fig0
        assert np.asarray(axes).shape == (2,)

    def test_plot_dist_accepts_arkouda_edges_and_counts_with_extra_slot(self):
        import arkouda as ak

        # Arkouda histogram returns edges (N+1) and counts (N+1); plot_dist should drop the final count.
        edges, counts = ak.histogram(ak.arange(10), 3)
        fig, axes = plot_dist(edges, counts, xlabel="Value")
        assert fig is not None
        assert np.asarray(axes).shape == (2,)
        assert axes[0].get_xlabel() == "Value"
        # CDF ends at 1
        y = axes[1].lines[0].get_ydata()
        assert math.isclose(float(y[-1]), 1.0, rel_tol=1e-12, abs_tol=1e-12)

    def test_plot_dist_respects_max_transfer_bytes(self, monkeypatch):
        import arkouda as ak

        # Create tiny ak arrays (valid input),
        # then force an artificially small transfer cap to trigger ValueError.
        edges, counts = ak.histogram(ak.arange(10), 3)

        # Make sure arrays would normally be well under the limit:
        nbytes_edges = edges.size * edges.to_ndarray().dtype.itemsize
        nbytes_counts = counts.size * counts.to_ndarray().dtype.itemsize
        assert nbytes_edges < ak.client.maxTransferBytes
        assert nbytes_counts < ak.client.maxTransferBytes

        # Patch maxTransferBytes to 1 to guarantee failure
        monkeypatch.setattr(ak.client, "maxTransferBytes", 1, raising=False)

        with pytest.raises(ValueError):
            plot_dist(edges, counts)

    def test_hist_all_returns_fig_axes_numpy_dataframe(self, monkeypatch):
        # Minimal numeric data; if hist_all expects ak types, adapt or guard this test accordingly.
        df = ak.DataFrame(
            {
                "a": ak.random.rand(100),
                "b": ak.random.rand(100),
                "c": ak.random.rand(100),
                "d": ak.random.rand(100),
            }
        )

        fig, axes = ak.hist_all(df)
        assert fig is not None
        # Allow either a flat array or list; just ensure we got multiple axes
        assert len(np.ravel(axes)) >= 2
