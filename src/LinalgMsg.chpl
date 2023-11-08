module LinalgMsg {

    use Reflection;
    use ArkoudaTimeCompat;

    use Logging;
    use Message;
    use ServerConfig;
    use CommandMap;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const linalgLogger = new Logger(logLevel, logChannel);

    /*
        Create an identity matrix of a given size with ones along a given diagonal
    */
    proc eyeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const rows = msgArgs.get("rows").getIntValue(), // matrix height
              cols = msgArgs.get("cols").getIntValue(), // matrix width
              diag = msgArgs.get("diag").getIntValue(); // diagonal to set to 1
                                                        // 0  : center diag
                                                        // >0 : upper triangle
                                                        // <0 : lower triangle

        const dtype = str2dtype(msgArgs.getValueOf("dtype")),
              rname = st.nextName();

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s rname: %s aRows: %i: aCols: %i aDiag: %i".doFormat(
            cmd,dtype2str(dtype),rname,rows,cols,diag));

        proc setDiag(ref a: [?d] ?t, k: int, param one: t) where d.rank == 2 {
            // TODO: (this is very inefficient) develop a parallel iterator
            //  for diagonals to ensure computation occurs on the correct locale
            if k == 0 {
                forall ij in 0..<min(rows, cols) do
                    a[ij, ij] = one;
            } else if k > 0 {
                forall i in 0..<min(rows, cols-k) do
                    a[i, i+k] = one;
            } else if k < 0 {
                forall j in 0..<min(rows+k, cols) do
                    a[j-k, j] = one;
            }
        }

        var t = new stopwatch();

        select dtype {
            when DType.Int64 {
                t.start();
                var e = st.addEntry(rname, rows, cols, int);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                setDiag(e.a, diag, 1);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.UInt8 {
                t.start();
                var e = st.addEntry(rname, rows, cols, uint(8));
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                setDiag(e.a, diag, 1:uint(8));
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.UInt64 {
                t.start();
                var e = st.addEntry(rname, rows, cols, uint);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                setDiag(e.a, diag, 1:uint);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.Float64 {
                t.start();
                var e = st.addEntry(rname, rows, cols, real);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                setDiag(e.a, diag, 1.0);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.Bool {
                t.start();
                var e = st.addEntry(rname, rows, cols, bool);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                setDiag(e.a, diag, true);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            otherwise {
                var errorMsg = notImplementedError(getRoutineName(),dtype);
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        const repMsg = "created " + st.attrib(rname);
        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    registerFunction("eye", eyeMsg, getModuleName());

    /*
        Create an array from an existing array with its upper triangle zeroed out
    */
    @arkouda.registerND
    proc trilMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        if nd < 2 {
            const errorMsg = "Array must be at least 2 dimensional for 'tril'";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        return triluHandler(cmd, msgArgs, st, nd, false);
    }

    /*
        Create an array from an existing array with its lower triangle zeroed out
    */
    @arkouda.registerND
    proc triuMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        if nd < 2 {
            const errorMsg = "Array must be at least 2 dimensional for 'triu'";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        return triluHandler(cmd, msgArgs, st, nd, true);
    }

    /*
        Get the lower or upper triangular part of a matrix or a stack of matrices
    */
    proc triluHandler(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                      param nd: int, param upper: bool
    ): MsgTuple throws {
        const name = msgArgs.getValueOf("array"),
              diag = msgArgs.get("diag").getIntValue();

        const rname = st.nextName();

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s rname: %s aName: %s aDiag: %i".doFormat(
            cmd,rname,name,diag));

        var t = new stopwatch();
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        select gEnt.dtype {
            when DType.Int64 {
                var eIn = toSymEntry(gEnt, int, nd);
                t.start();
                var eOut = st.addEntry(rname, (...eIn.tupShape), int);
                eOut.a = eIn.a;
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                zeroTri(eOut.a, diag, 0, upper);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.UInt8 {
                var eIn = toSymEntry(gEnt, uint(8), nd);
                t.start();
                var eOut = st.addEntry(rname, (...eIn.tupShape), uint(8));
                eOut.a = eIn.a;
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                zeroTri(eOut.a, diag, 0:uint(8), upper);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.UInt64 {
                var eIn = toSymEntry(gEnt, uint, nd);
                t.start();
                var eOut = st.addEntry(rname, (...eIn.tupShape), uint);
                eOut.a = eIn.a;
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                zeroTri(eOut.a, diag, 0:uint, upper);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.Float64 {
                var eIn = toSymEntry(gEnt, real, nd);
                t.start();
                var eOut = st.addEntry(rname, (...eIn.tupShape), real);
                eOut.a = eIn.a;
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                zeroTri(eOut.a, diag, 0.0, upper);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            when DType.Bool {
                var eIn = toSymEntry(gEnt, bool, nd);
                t.start();
                var eOut = st.addEntry(rname, (...eIn.tupShape), bool);
                eOut.a = eIn.a;
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "alloc time = %i sec".doFormat(t.elapsed()));

                t.restart();
                zeroTri(eOut.a, diag, false, upper);
                t.stop();
                linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "compute time = %i sec".doFormat(t.elapsed()));
            }
            otherwise {
                var errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        const repMsg = "created " + st.attrib(rname);
        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    private proc zeroTri(ref a: [?d] ?t, diag: int, param zero: t, param upper: bool)
        where d.rank >= 2 && upper == true
    {
        const iThresh = if diag < 0 then abs(diag) else 0,
              jThresh = if diag > 0 then diag else 0;

        forall idx in d {
            const i = idx[d.rank-2],
                  j = idx[d.rank-1];

            if i - iThresh < j - jThresh then a[idx] = zero;
        }
    }

    private proc zeroTri(ref a: [?d] ?t, diag: int, param zero: t, param upper: bool)
        where d.rank >= 2 && upper == false
    {
        const iThresh = if diag < 0 then abs(diag) else 0,
              jThresh = if diag > 0 then diag else 0;

        forall idx in d {
            const i = idx[d.rank-2],
                  j = idx[d.rank-1];

            if i - iThresh > j - jThresh then a[idx] = zero;
        }
    }


}
