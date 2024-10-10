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

    proc randomSparseMatrixMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message with the details of the new arr

        var vName = st.nextName(); // symbol table key for resulting sparse arr

        var size: int = msgArgs.get("size").getIntValue();
        var density: real = msgArgs.get("density").getRealValue();
        var l: string = msgArgs.getValueOf("layout");


        select l {
            when "CSR" {
                var aV = randSparseMatrix(size, density, layout.CSR, int);
                st.addEntry(vName, createSparseSymEntry(aV, size, layout.CSR, int));
                repMsg = "created " + st.attrib(vName);
                sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when "CSC" {
                var aV = randSparseMatrix(size, density, layout.CSC, int);
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


    proc sparseMatrixtoPdarray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message with the details of the new arr

        var gEnt = getGenericSparseArrayEntry(msgArgs.getValueOf("matrix"), st);

        var size = gEnt.nnz;
        var rows = makeDistArray(size, int);
        var cols = makeDistArray(size, int);
        var vals = makeDistArray(size, int);

        if gEnt.layoutStr=="CSC" {
            // Hardcode for int right now
            var sparrayEntry = gEnt.toSparseSymEntry(int, dimensions=2, layout.CSC);
            sparseMatToPdarrayCSC(sparrayEntry.a, rows, cols, vals);
        } else if gEnt.layoutStr=="CSR" {
            // Hardcode for int right now
            var sparrayEntry = gEnt.toSparseSymEntry(int, dimensions=2, layout.CSR);
            sparseMatToPdarrayCSR(sparrayEntry.a, rows, cols, vals);
        } else {
            throw getErrorWithContext(
                                    msg="unsupported layout for sparse matrix: %s".format(gEnt.layoutStr),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="NotImplementedError"
                                    );
        }

        var responses: [0..2] MsgTuple;
        responses[0] = st.insert(createSymEntry(rows));
        responses[1] = st.insert(createSymEntry(cols));
        responses[2] = st.insert(createSymEntry(vals));
        sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Converted sparse matrix to pdarray");
        return MsgTuple.fromResponses(responses);
    }


    proc fillSparseMatrixMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message with the details of the new arr

        var gEnt = getGenericSparseArrayEntry(msgArgs.getValueOf("matrix"), st);
        var gEntVals: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("vals"), st);

        //Hardcode int for now
        var vals = toSymEntry(gEntVals,int);
        if gEnt.layoutStr=="CSC" {
            // Hardcode for int right now
            var sparrayEntry = gEnt.toSparseSymEntry(int, dimensions=2, layout.CSC);
            fillSparseMatrix(sparrayEntry.a, vals.a, layout.CSC);
        } else if gEnt.layoutStr=="CSR" {
            // Hardcode for int right now
            var sparrayEntry = gEnt.toSparseSymEntry(int, dimensions=2, layout.CSR);
            fillSparseMatrix(sparrayEntry.a, vals.a, layout.CSR);
        } else {
            throw getErrorWithContext(
                                    msg="unsupported layout for sparse matrix: %s".format(gEnt.layoutStr),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="NotImplementedError"
                                    );
        }
        sparseLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Filled sparse Array with values");
        return MsgTuple.success();
    }



    use CommandMap;
    registerFunction("random_sparse_matrix", randomSparseMatrixMsg, getModuleName());
    registerFunction("sparse_matrix_matrix_mult", sparseMatrixMatrixMultMsg, getModuleName());
    registerFunction("sparse_to_pdarrays", sparseMatrixtoPdarray, getModuleName());
    registerFunction("fill_sparse_vals", fillSparseMatrixMsg, getModuleName());

}
