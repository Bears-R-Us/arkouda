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
                            type sparse_dtype, param sparse_layout: Layout
    ): MsgTuple throws {
        const shape = msgArgs["shape"].toScalarTuple(int, 2), // Hardcode 2D for now
              density = msgArgs["density"].toScalar(real);

        const aV = randSparseMatrix(shape, density, sparse_layout, sparse_dtype);
        return st.insert(new shared SparseSymEntry(aV, sparse_layout));
    }

    @arkouda.instantiateAndRegister("sparse_matrix_matrix_mult")
    proc sparseMatrixMatrixMultMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                                   type sparse_dtype
    ): MsgTuple throws {
        const e1 = st[msgArgs["arg1"]]: borrowed SparseSymEntry(sparse_dtype, 2, Layout.CSC),
              e2 = st[msgArgs["arg2"]]: borrowed SparseSymEntry(sparse_dtype, 2, Layout.CSR);

        const aV = sparseMatMatMult(e1.a, e2.a);
        return st.insert(new shared SparseSymEntry(aV, Layout.CSR));
    }

    @arkouda.instantiateAndRegister("sparse_to_pdarrays")
    proc sparseMatrixtoPdarray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                               type sparse_dtype, param sparse_layout: Layout
    ): MsgTuple throws {
        const e = st[msgArgs["matrix"]]: borrowed SparseSymEntry(sparse_dtype, 2, sparse_layout);

        const size = e.nnz;
        var rows = makeDistArray(size, int),
            cols = makeDistArray(size, int),
            vals = makeDistArray(size, sparse_dtype);

        sparseMatToPdarray(e.a, rows, cols, vals, sparse_layout);

        var responses: [0..2] MsgTuple;
        responses[0] = st.insert(new shared SymEntry(rows));
        responses[1] = st.insert(new shared SymEntry(cols));
        responses[2] = st.insert(new shared SymEntry(vals));
        sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Converted sparse matrix to pdarray");
        return MsgTuple.fromResponses(responses);
    }

    @arkouda.instantiateAndRegister("fill_sparse_vals")
    proc fillSparseMatrixMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                             type sparse_dtype, param sparse_layout: Layout
    ): MsgTuple throws {
        const e = st[msgArgs["matrix"]]: borrowed SparseSymEntry(sparse_dtype, 2, sparse_layout),
              vals = st[msgArgs["vals"]]: borrowed SymEntry(sparse_dtype, 1);

        fillSparseMatrix(e.a, vals.a, sparse_layout);

        sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Filled sparse Array with values");
        return MsgTuple.success();
    }

}
