class TestRow:
    def test_row_docstrings(self):
        import doctest

        from arkouda.pandas import row

        result = doctest.testmod(row, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
