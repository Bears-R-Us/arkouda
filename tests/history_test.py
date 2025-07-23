class TestHistory:
    def test_history_docstrings(self):
        import doctest

        from arkouda import history

        result = doctest.testmod(history, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
