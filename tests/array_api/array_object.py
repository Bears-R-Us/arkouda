import pytest

import arkouda.array_api as xp


class TestArrayCreation:

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
    def test_chunk_info_2dim(self):

        a = xp.zeros(10)
        chunks = a.chunk_info()
        assert len(chunks) > 0
        assert chunks[0][0] == 0
        assert chunks[0][1] > 0
