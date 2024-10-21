module SparseMatrixMsg {
    use ServerConfig;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;

    use ArraySetops;
    use Indexing;
    use RadixSortLSD;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    use SparseMatrix;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sparseLogger = new Logger(logLevel, logChannel);

    @arkouda.instantiateAndRegister("random_sparse_matrix")
    proc randomSparseMatrix(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                            type SparseSymEntry_etype, param SparseSymEntry_matLayout: Layout
    ): MsgTuple throws {
        const shape = msgArgs["shape"].toScalarTuple(int, 2), // Hardcode 2D for now
              density = msgArgs["density"].toScalar(real);

        const aV = randSparseMatrix(shape, density, SparseSymEntry_matLayout, SparseSymEntry_etype);
        return st.insert(new shared SparseSymEntry(aV, SparseSymEntry_matLayout));
    }

    @arkouda.instantiateAndRegister("sparse_matrix_matrix_mult")
    proc sparseMatrixMatrixMultMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                                   type SparseSymEntry_etype
    ): MsgTuple throws {
        const e1 = st[msgArgs["arg1"]]: borrowed SparseSymEntry(SparseSymEntry_etype, 2, Layout.CSC),
              e2 = st[msgArgs["arg2"]]: borrowed SparseSymEntry(SparseSymEntry_etype, 2, Layout.CSR);

        const aV = sparseMatMatMult(e1.a, e2.a);
        return st.insert(new shared SparseSymEntry(aV, Layout.CSR));
    }

    @arkouda.instantiateAndRegister("sparse_to_pdarrays")
    proc sparseMatrixtoPdarray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                               type SparseSymEntry_etype, param SparseSymEntry_matLayout: Layout
    ): MsgTuple throws {
        const e = st[msgArgs["matrix"]]: borrowed SparseSymEntry(SparseSymEntry_etype, 2, SparseSymEntry_matLayout);

        const size = e.nnz;
        var rows = makeDistArray(size, int),
            cols = makeDistArray(size, int),
            vals = makeDistArray(size, SparseSymEntry_etype);

        sparseMatToPdarray(e.a, rows, cols, vals, SparseSymEntry_matLayout);

        var responses: [0..2] MsgTuple;
        responses[0] = st.insert(new shared SymEntry(rows));
        responses[1] = st.insert(new shared SymEntry(cols));
        responses[2] = st.insert(new shared SymEntry(vals));
        sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Converted sparse matrix to pdarray");
        return MsgTuple.fromResponses(responses);
    }

    @arkouda.registerCommand("fill_sparse_vals", ignoreWhereClause=true)
    proc fillSparseMatrixMsg(matrix: borrowed SparseSymEntry(?), vals: [?d] ?t /* matrix.etype */) throws
        where t == matrix.etype && d.rank == 1
            do fillSparseMatrix(matrix.a, vals, matrix.matLayout);

    proc fillSparseMatrixMsg(matrix: borrowed SparseSymEntry(?), vals: [?d] ?t) throws
        where t != matrix.etype
            do throw new Error("fillSparseMatrixMsg: type mismatch between matrix and vals");

    proc fillSparseMatrixMsg(matrix: borrowed SparseSymEntry(?), vals: [?d] ?t) throws
        where d.rank != 1 && t == matrix.etype
            do throw new Error("fillSparseMatrixMsg: vals must be rank 1");

    @arkouda.instantiateAndRegister("sparse_matrix_from_pdarrays")
    proc sparseMatrixFromPdarrays(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                                  type SparseSymEntry_etype, param SparseSymEntry_matLayout: Layout
    ): MsgTuple throws {
        const rows = st[msgArgs["rows"]]: borrowed SymEntry(int, 1),
              cols = st[msgArgs["cols"]]: borrowed SymEntry(int, 1),
              vals = st[msgArgs["vals"]]: borrowed SymEntry(SparseSymEntry_etype, 1),
              shape = msgArgs["shape"].toScalarTuple(int, 2); // Hardcode 2D for now

        const aV = sparseMatFromArrays(rows.a, cols.a, vals.a, shape, SparseSymEntry_matLayout, SparseSymEntry_etype);
        return st.insert(new shared SparseSymEntry(aV, SparseSymEntry_matLayout));
    }

}
