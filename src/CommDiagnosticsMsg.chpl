module CommDiagnosticsMsg
{
    use CommDiagnostics, Message, CommandMap, MultiTypeSymbolTable, Reflection, MultiTypeSymEntry;

    proc ak_startCommDiagnostics(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        startCommDiagnostics();
        return new MsgTuple("commDiagnostics started.", MsgType.NORMAL);
    }

    proc ak_stopCommDiagnostics(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        stopCommDiagnostics();
        return new MsgTuple("commDiagnostics stopped.", MsgType.NORMAL);
    }

    proc ak_printCommDiagnosticsTable(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const printEmptyCols = msgArgs['printEmptyCols'].toScalar(bool);
        printCommDiagnosticsTable(printEmptyColumns=printEmptyCols);
        return new MsgTuple("commDiagnostics printed.", MsgType.NORMAL);
    }

    proc ak_resetCommDiagnostics(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        resetCommDiagnostics();
        return new MsgTuple("commDiagnostics reset.", MsgType.NORMAL);
    }

    proc ak_startVerboseComm(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        startVerboseComm();
        return new MsgTuple("commDiagnostics set verbose.", MsgType.NORMAL);
    }

    proc ak_stopVerboseComm(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        stopVerboseComm();
        return new MsgTuple("commDiagnostics unset verbose.", MsgType.NORMAL);
    }

    proc getCommDiagnosticsPut(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const puts = forall i in 0..#numLocales do commD[i].put;
        const e = createSymEntry(puts);
        return st.insert(e);
    }

    proc getCommDiagnosticsGet(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const gets = forall i in 0..#numLocales do commD[i].get;
        const e = createSymEntry(gets);
        return st.insert(e);
    }

    proc getCommDiagnosticsPutNb(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const puts = forall i in 0..#numLocales do commD[i].put_nb;
        const e = createSymEntry(puts);
        return st.insert(e);
    }

    proc getCommDiagnosticsGetNb(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const gets = forall i in 0..#numLocales do commD[i].get_nb;
        const e = createSymEntry(gets);
        return st.insert(e);
    }

    proc getCommDiagnosticsTestNb(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].test_nb;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsWaitNb(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].wait_nb;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsTryNb(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].try_nb;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsAmo(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].amo;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsExecuteOn(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].execute_on;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsExecuteOnFast(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].execute_on_fast;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsExecuteOnNb(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].execute_on_nb;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCacheGetHits(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_get_hits;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCacheGetMisses(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_get_misses;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCachePutHits(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_put_hits;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCachePutMisses(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_put_misses;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCacheNumPrefetches(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_num_prefetches;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCacheNumPageReadaheads(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_num_page_readaheads;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCachePrefetchUnused(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_prefetch_unused;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCachePrefetchWaited(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_prefetch_waited;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCacheReadaheadUnused(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_readahead_unused;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    proc getCommDiagnosticsCacheReadaheadWaited(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const commD = getCommDiagnostics();
        const ret = forall i in 0..#numLocales do commD[i].cache_readahead_waited;
        const e = createSymEntry(ret);
        return st.insert(e);
    }

    registerFunction('startCommDiagnostics', ak_startCommDiagnostics, getModuleName());
    registerFunction('stopCommDiagnostics', ak_stopCommDiagnostics, getModuleName());
    registerFunction('printCommDiagnosticsTable', ak_printCommDiagnosticsTable, getModuleName());
    registerFunction('resetCommDiagnostics', ak_resetCommDiagnostics, getModuleName());
    registerFunction('startVerboseComm', ak_startVerboseComm, getModuleName());
    registerFunction('stopVerboseComm', ak_stopVerboseComm, getModuleName());
    registerFunction('getCommDiagnosticsPut', getCommDiagnosticsPut, getModuleName());
    registerFunction('getCommDiagnosticsGet', getCommDiagnosticsGet, getModuleName());
    registerFunction('getCommDiagnosticsPutNb', getCommDiagnosticsPutNb, getModuleName());
    registerFunction('getCommDiagnosticsGetNb', getCommDiagnosticsGetNb, getModuleName());
    registerFunction('getCommDiagnosticsTestNb', getCommDiagnosticsTestNb, getModuleName());
    registerFunction('getCommDiagnosticsWaitNb', getCommDiagnosticsWaitNb, getModuleName());
    registerFunction('getCommDiagnosticsTryNb', getCommDiagnosticsTryNb, getModuleName());
    registerFunction('getCommDiagnosticsAmo', getCommDiagnosticsAmo, getModuleName());
    registerFunction('getCommDiagnosticsExecuteOn', getCommDiagnosticsExecuteOn, getModuleName());
    registerFunction('getCommDiagnosticsExecuteOnFast', getCommDiagnosticsExecuteOnFast, getModuleName());
    registerFunction('getCommDiagnosticsExecuteOnNb', getCommDiagnosticsExecuteOnNb, getModuleName());
    registerFunction('getCommDiagnosticsCacheGetHits', getCommDiagnosticsCacheGetHits, getModuleName());
    registerFunction('getCommDiagnosticsCacheGetMisses', getCommDiagnosticsCacheGetMisses, getModuleName());
    registerFunction('getCommDiagnosticsCachePutHits', getCommDiagnosticsCachePutHits, getModuleName());
    registerFunction('getCommDiagnosticsCachePutMisses', getCommDiagnosticsCachePutMisses, getModuleName());
    registerFunction('getCommDiagnosticsCacheNumPrefetches', getCommDiagnosticsCacheNumPrefetches, getModuleName());
    registerFunction('getCommDiagnosticsCacheNumPageReadaheads', getCommDiagnosticsCacheNumPageReadaheads, getModuleName());
    registerFunction('getCommDiagnosticsCachePrefetchUnused', getCommDiagnosticsCachePrefetchUnused, getModuleName());
    registerFunction('getCommDiagnosticsCachePrefetchWaited', getCommDiagnosticsCachePrefetchWaited, getModuleName());
    registerFunction('getCommDiagnosticsCacheReadaheadUnused', getCommDiagnosticsCacheReadaheadUnused, getModuleName());
    registerFunction('getCommDiagnosticsCacheReadaheadWaited', getCommDiagnosticsCacheReadaheadWaited, getModuleName());
}