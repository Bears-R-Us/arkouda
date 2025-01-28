import pytest
import arkouda as ak
from arkouda.comm_diagnostics import *


diagnostic_stats_functions = [
    "get_comm_diagnostics_put",
    "get_comm_diagnostics_get",
    "get_comm_diagnostics_put_nb",
    "get_comm_diagnostics_get_nb",
    "get_comm_diagnostics_try_nb",
    "get_comm_diagnostics_amo",
    "get_comm_diagnostics_execute_on",
    "get_comm_diagnostics_execute_on_fast",
    "get_comm_diagnostics_execute_on_nb",
    "get_comm_diagnostics_cache_get_hits",
    "get_comm_diagnostics_cache_get_misses",
    "get_comm_diagnostics_cache_put_hits",
    "get_comm_diagnostics_cache_put_misses",
    "get_comm_diagnostics_cache_num_prefetches",
    "get_comm_diagnostics_cache_num_page_readaheads",
    "get_comm_diagnostics_cache_prefetch_unused",
    "get_comm_diagnostics_cache_prefetch_waited",
    "get_comm_diagnostics_cache_readahead_unused",
    "get_comm_diagnostics_cache_readahead_waited",
]


class TestCommDiagnostics:

    @pytest.mark.parametrize("op", diagnostic_stats_functions)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_comm_diagnostics_single_locale(self, op, size):
        start_comm_diagnostics()
        start_verbose_comm()

        ak.sort(ak.arange(size) * -1.0)
        comm_diagnostic_function = getattr(ak.comm_diagnostics, op)
        result = comm_diagnostic_function()
        assert isinstance(result, ak.pdarray)
        assert result.size == pytest.nl
        if pytest.nl == 1:
            assert result.sum() == 0

        stop_verbose_comm()
        print_comm_diagnostics_table()
        reset_comm_diagnostics()
        stop_comm_diagnostics()

    @pytest.mark.skip_if_nl_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_comm_diagnostics_multi_locale(self, size):
        start_comm_diagnostics()

        ak.sort(ak.arange(size) * -1.0)
        result = get_comm_diagnostics_get() + get_comm_diagnostics_put()
        assert isinstance(result, ak.pdarray)
        assert result.size == pytest.nl
        assert ak.all(result > (size ** (1 / 2)))

        print_comm_diagnostics_table(print_empty_columns=True)
        print_comm_diagnostics_table(print_empty_columns=False)

        stop_comm_diagnostics()

    def test_get_comm_diagnostics(self):
        start_comm_diagnostics()

        df = get_comm_diagnostics()
        assert isinstance(df, ak.DataFrame)
        assert len(df) == pytest.nl

        stop_comm_diagnostics()
