class TestArkoudaCategoricalExtension:
    def test_base_categorical_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_categorical_array

        result = doctest.testmod(
            _arkouda_categorical_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
