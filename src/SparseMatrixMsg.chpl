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

    param distributed = true;

    proc randomSparseMatrixMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message with the details of the new arr

        var vName = st.nextName(); // symbol table key for resulting sparse arr

        var size: int = msgArgs.get("size").getIntValue();
        var density: real = msgArgs.get("density").getRealValue();
        var l: string = msgArgs.getValueOf("layout");


        select l {
            when "CSR" {
                var aV = randSparseMatrix(size, density, layout.CSR, distributed, int); // Hardcode int for now and false for distributed
                st.addEntry(vName, createSparseSymEntry(aV, size, layout.CSR, int));
                repMsg = "created " + st.attrib(vName);
                sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when "CSC" {
                var aV = randSparseMatrix(size, density, layout.CSC, distributed, int); // Hardcode int for now and false for distributed
                st.addEntry(vName, createSparseSymEntry(aV, size, layout.CSC, int));
                repMsg = "created " + st.attrib(vName);
                sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            otherwise {
                var errorMsg = notImplementedError("unsupported layout for sparse matrix",l);
                sparseLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

    }



    proc sparseMatrixMatrixMultMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message with the details of the new arr

        var vName = st.nextName(); // symbol table key for resulting sparse arr

        var gEnt1 = getGenericSparseArrayEntry(msgArgs.getValueOf("arg1"), st);
        var gEnt2 = getGenericSparseArrayEntry(msgArgs.getValueOf("arg2"), st);

        // Hardcode for int right now
        var e1 = gEnt1.toSparseSymEntry(int, dimensions=2, layout.CSC);
        var e2 = gEnt2.toSparseSymEntry(int, dimensions=2, layout.CSR);

        var size = gEnt2.size;

        var aV = sparseMatMatMult(e1.a, e2.a);
        st.addEntry(vName, createSparseSymEntry(aV, size, layout.CSR, int));
        repMsg = "created " + st.attrib(vName);
        sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }



    use CommandMap;
    registerFunction("random_sparse_matrix", randomSparseMatrixMsg, getModuleName());
    registerFunction("sparse_matrix_matrix_mult", sparseMatrixMatrixMultMsg, getModuleName());

}