class TestPlotting:
    def test_plotting_docstrings(self):
        import doctest

        import matplotlib

        from arkouda import plotting

        matplotlib.use("Agg")  # Use non-GUI backend to suppress plots

        result = doctest.testmod(plotting)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
