class TestAccessor:
    def test_alignment_docstrings(self):
        import doctest

        from arkouda import accessor

        result = doctest.testmod(accessor, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
