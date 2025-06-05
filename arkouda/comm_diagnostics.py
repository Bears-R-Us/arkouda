from typing import cast as typecast

from arkouda.client import generic_msg
from arkouda.dataframe import DataFrame
from arkouda.numpy.pdarrayclass import create_pdarray

__all__ = [
    "start_comm_diagnostics",
    "stop_comm_diagnostics",
    "reset_comm_diagnostics",
    "print_comm_diagnostics_table",
    "start_verbose_comm",
    "stop_verbose_comm",
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
    "get_comm_diagnostics_wait_nb",
    "get_comm_diagnostics",
]


def start_comm_diagnostics():
    """
    Start counting communication operations across the whole program.

    Returns
    -------
    str
        Completion message.

    """
    rep_msg = generic_msg(
        cmd="startCommDiagnostics",
        args={},
    )
    return typecast(str, rep_msg)


def stop_comm_diagnostics():
    """
    Stop counting communication operations across the whole program.

    Returns
    -------
    str
        Completion message.

    """
    rep_msg = generic_msg(
        cmd="stopCommDiagnostics",
        args={},
    )
    return typecast(str, rep_msg)


def reset_comm_diagnostics():
    """
    Reset aggregate communication counts across the whole program.

    Returns
    -------
    str
        Completion message.

    """
    rep_msg = generic_msg(
        cmd="resetCommDiagnostics",
        args={},
    )
    return typecast(str, rep_msg)


def print_comm_diagnostics_table(print_empty_columns=False):
    """
    Print the current communication counts in a markdown table.

    Uses a row per locale and a column per operation.
    By default, operations for which all locales have a count of zero are not displayed in the table,
    though an argument can be used to reverse that behavior.

    Parameters
    ----------
    print_empty_columns : bool=False

    Note
    ----
    The table will only be printed to the chapel logs.

    Returns
    -------
    str
        Completion message.

    """
    from arkouda import sum as ak_sum

    rep_msg = generic_msg(
        cmd="printCommDiagnosticsTable",
        args={"printEmptyCols": print_empty_columns},
    )

    df = get_comm_diagnostics()
    if print_empty_columns:
        print(df.to_markdown())
    else:
        print(df[[col for col in df.columns if ak_sum(df[col]) != 0]].to_markdown())

    return typecast(str, rep_msg)


def start_verbose_comm():
    """
    Start on-the-fly reporting of communication initiated on any locale.

    Note
    ----
    Reporting will only be printed to the chapel logs.

    Returns
    -------
    str
        Completion message.

    """
    rep_msg = generic_msg(
        cmd="startVerboseComm",
        args={},
    )
    return typecast(str, rep_msg)


def stop_verbose_comm():
    """
    Stop on-the-fly reporting of communication initiated on any locale.

    Returns
    -------
    str
        Completion message.

    """
    rep_msg = generic_msg(
        cmd="stopVerboseComm",
        args={},
    )
    return typecast(str, rep_msg)


def get_comm_diagnostics_put():
    """
    Return blocking puts, in which initiator waits for completion.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsPut",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_get():
    """
    Return blocking gets, in which initiator waits for completion.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsGet",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_put_nb():
    """
    Return non-blocking puts.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsPutNb",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_get_nb():
    """
    Return non-blocking gets.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsGetNb",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_test_nb():
    """
    Return test statistic for non-blocking get/put completions.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsTestNb",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_wait_nb():
    """
    Return blocking waits for non-blocking get/put completions.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsWaitNb",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_try_nb():
    """
    Return test statistics for non-blocking get/put completions.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsTryNb",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_amo():
    """
    Return atomic memory operations statistic.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsAmo",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_execute_on():
    """
    Return blocking remote executions, in which initiator waits for completion.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsExecuteOn",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_execute_on_fast():
    """
    Return blocking remote executions performed by the target locale’s Active Message handler.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsExecuteOnFast",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_execute_on_nb():
    """
    Return non-blocking remote executions.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsExecuteOnNb",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_get_hits():
    """
    Return number of gets that were handled by the cache.

    Gets counted here did not require the cache to communicate in order to return the result.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCacheGetHits",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_get_misses():
    """
    Return number of gets that were not handled by the cache.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCacheGetMisses",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_put_hits():
    """
    Return number of puts that were stored in cache pages that already existed.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCachePutHits",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_put_misses():
    """
    Return number of puts that required the cache to create a new page to store them.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCachePutMisses",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_num_prefetches():
    """
    Return number of prefetches issued to the remote cache at the granularity of cache pages.

    This counter is specifically triggered via calls to chpl_comm_remote_prefetch.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCacheNumPrefetches",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_num_page_readaheads():
    """
    Return number of readaheads issued to the remote cache at the granularity of cache pages.

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCacheNumPageReadaheads",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_prefetch_unused():
    """
    Return number of cache pages that were prefetched but unused.

    Return number of cache pages that were prefetched but evicted from the cache before being accessed
    (i.e., the prefetches were too early).

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCachePrefetchUnused",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_prefetch_waited():
    """
    Return number of cache pages that were prefetched but waited.

    Number of cache pages that were prefetched but did not arrive in the cache before being accessed
    (i.e., the prefetches were too late).

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCachePrefetchWaited",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_readahead_unused():
    """
    Return number of cache pages that were read ahead but unused.

    The number of cache pages that were read ahead but evicted from the cache before being accessed
    (i.e., the readaheads were too early).

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCacheReadaheadUnused",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics_cache_readahead_waited():
    """
    Return number of cache pages that were read ahead but waited.

    Return number of cache pages that were read ahead
    but did not arrive in the cache before being accessed
    (i.e., the readaheads were too late).

    Returns
    -------
    pdarray
        A pdarray, where the size is the number of locales,
        populated with the statistic value from each locale.

    """
    rep_msg = generic_msg(
        cmd="getCommDiagnosticsCacheReadaheadWaited",
        args={},
    )
    return create_pdarray(typecast(str, rep_msg))


def get_comm_diagnostics() -> DataFrame:
    """
    Return a DataFrame with the communication diagnostics statistics.

    Returns
    -------
    DataFrame

    """
    return DataFrame(
        {
            "put": get_comm_diagnostics_put(),
            "get": get_comm_diagnostics_get(),
            "put_nb": get_comm_diagnostics_put_nb(),
            "get_nb": get_comm_diagnostics_get_nb(),
            "try_nb": get_comm_diagnostics_try_nb(),
            "amo": get_comm_diagnostics_amo(),
            "execute_on": get_comm_diagnostics_execute_on(),
            "execute_on_fast": get_comm_diagnostics_execute_on_fast(),
            "execute_on_nb": get_comm_diagnostics_execute_on_nb(),
            "cache_get_hits": get_comm_diagnostics_cache_get_hits(),
            "cache_get_misses": get_comm_diagnostics_cache_get_misses(),
            "cache_put_hits": get_comm_diagnostics_cache_put_hits(),
            "cache_put_misses": get_comm_diagnostics_cache_put_misses(),
            "cache_num_prefetches": get_comm_diagnostics_cache_num_prefetches(),
            "cache_num_page_readaheads": get_comm_diagnostics_cache_num_page_readaheads(),
            "cache_prefetch_unused": get_comm_diagnostics_cache_prefetch_unused(),
            "cache_prefetch_waited": get_comm_diagnostics_cache_prefetch_waited(),
            "cache_readahead_unused": get_comm_diagnostics_cache_readahead_unused(),
            "cache_readahead_waited": get_comm_diagnostics_cache_readahead_waited(),
            "wait_nb": get_comm_diagnostics_wait_nb(),
        }
    )
