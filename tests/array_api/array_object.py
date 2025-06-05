import pytest

import arkouda.array_api as xp


class TestArrayObject:
    def test_array_object_docstrings(self):
        import doctest

        from arkouda.array_api import array_object

        result = doctest.testmod(
            array_object, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skipif(pytest.nl > 1, reason="Multi-local will produce different chunk_info")
    def test_chunk_info(self):
        a = xp.zeros(5)
        chunks = a.chunk_info()
        assert chunks == [[0]]

    @pytest.mark.skipif(pytest.nl > 1, reason="Multi-local will produce different chunk_info")
    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_chunk_info_2dim(self):
        a = xp.zeros((2, 2))
        chunks = a.chunk_info()
        assert chunks == [[0], [0]]

    @pytest.mark.skipif(pytest.nl <= 1, reason="Multi-local will produce different chunk_info")
    def test_chunk_info_2dim_nl1(self):
        a = xp.zeros(10)
        chunks = a.chunk_info()
        assert len(chunks) > 0
        assert chunks[0][0] == 0
        assert chunks[0][1] > 0
