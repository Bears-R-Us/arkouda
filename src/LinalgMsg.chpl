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
                const errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
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

    /*
        Multiply two matrices of compatible dimensions, or two stacks of matrices
        of compatible dimensions (the final two dimensions are the matrix dimensions,
        the preceding dimensions are the batch dimensions and must also be compatible)

        Note: the array api specifies that one dimensional arrays can be supported
        in matrix multiplication by adding a degenerate dimension to the array, multiplying,
        and then removing the degenerate method. This procedure expects that such a
        transformation is handled on the server side.
    */
    @arkouda.registerND
    proc matMulMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        if nd < 2 {
            const errorMsg = "Matrix multiplication with arrays of dimension < 2 is not supported";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        const x1Name = msgArgs.getValueOf("x1"),
              x2Name = msgArgs.getValueOf("x2");

        var x1G: borrowed GenSymEntry = getGenericTypedArrayEntry(x1Name, st),
            x2G: borrowed GenSymEntry = getGenericTypedArrayEntry(x2Name, st);

        const rname = st.nextName();

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype1: %s dtype2: %s rname: %s".doFormat(
            cmd,dtype2str(x1G.dtype),dtype2str(x2G.dtype),rname));

        proc doMatMult(type x1Type, type x2Type, type resultType): MsgTuple {
            var x1E = toSymEntry(x1G, x1Type, nd),
                x2E = toSymEntry(x2G, x2Type, nd);

            const (valid, outDims, err) = assertValidDims(x1E, x2E);
            if !valid then return err;

            var eOut = st.addEntry(rname, (...outDims), resultType);
            if nd == 2
                then matMult(x1E.a, x2E.a, eOut.a);
                else batchedMatMult(x1E.a, x2E.a, eOut.a);

            const repMsg = "created " + st.attrib(rname);
            linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select (x1G.dtype, x2G.dtype) {
            when (DType.Int64, DType.Int64) do
                return doMatMult(int, int, int);
            when (DType.Int64, DType.Float64) do
                return doMatMult(int, real, real);
            when (DType.Int64, DType.UInt64) do
                return doMatMult(int, uint, uint);
            when (DType.Float64, DType.Int64) do
                return doMatMult(real, int, real);
            when (DType.Uint64, DType.Int64) do
                return doMatMult(uint, int, uint);
            when (DType.UInt64, DType.UInt64) do
                return doMatMult(uint, uint, uint);
            when (DType.Uint64, DType.Float64) do
                return doMatMult(uint, real, real);
            when (DType.Float64, DType.UInt64) do
                return doMatMult(real, uint, real);
            when (DType.Bool, DType.Int64) do
                return doMatMult(bool, int, int);
            when (DType.Bool, DType.UInt64) do
                return doMatMult(bool, uint, uint);
            when (DType.Bool, DType.Float64) do
                return doMatMult(bool, real, real);
            when (DType.Int64, DType.Bool) do
                return doMatMult(int, bool, int);
            when (DType.UInt64, DType.Bool) do
                return doMatMult(uint, bool, uint);
            when (DType.Float64, DType.Bool) do
                return doMatMult(real, bool, real);
            when (DType.Bool, DType.Bool) do
                return doMatMult(bool, bool, bool);
            when (DType.Float64, DType.Float64) do
                return doMatMult(real, real, real);
            otherwise {
                const errorMsg = notImplementedError(getRoutineName(),x1G.dtype, x2G.dtype);
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    proc assertValidDims(ref x1, ref x2) {
        const (validInputDims, outDims) = matMultDims((...x1.tupShape), (...x2.tupShape));
        if !validInputDims {
            const errorMsg = "Invalid dimensions for matrix multiplication";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return (false, outDims, new MsgTuple(errorMsg, MsgType.ERROR));
        } else {
            return (true, outDims, new MsgTuple("", MsgType.NORMAL));
        }
    }

    proc matMultDims(a: int ...?Na, b: int ...?Nb): (bool, Na*int) {
        var outDims: Na*int;

        // the (batched) matrices must have the same rank
        if Na != Nb then return (false, outDims);

        // the batch dimensions must have the same size
        for param i in 0..<(Na-2) {
            if a[i] != b[i] then return (false, outDims);
            outDims[i] = a[i];
        }

        // the matrix dimensions must have compatible size
        if a[Na-1] != b[Na-2] then return (false, outDims);
        outDims[Na-2] = a[Na-2];
        outDims[Na-1] = b[Na-1];

        return (true, outDims);
    }

    proc batchedMatMult(A: [?D], B, C) {
        // create a domain identical to D, except degenerate along the last two axes
        var degenAxes = D.rank*range;
        for param i in 0..<D.rank {
            if i == D.rank-2 || i == D.rank-1
                then degenAxes[i] = 1..0;
                else degenAxes[i] = D.dim(i);
        }
        // slice the domain along the tuple of ranges (maintains distribution information)
        const DD = D[degenAxes];

        // for each matrix in the batch, perform matrix multiplication
        forall idx in DD {
            var slicer: D.rank*range;
            for param i in 0..<D.rank {
                if i == D.rank-2 then slicer[i] = D.dim(i);
                if i == D.rank-1 then slicer[i] = D.dim(i);
                else slicer[i] = idx[i]..idx[i];
            }

            var a = A[slicer];
            var b = B[slicer];
            var c = C[slicer];

            // TODO: pass 'slicer' and the full matrices to a parallel matMult
            //  instead of creating new slices for each call
            matMult(a, b, c);
        }
    }

    // TODO: not performant at all -- use tiled and parallel matrix multiplication
    //  or maybe use the linear algebra module? (do we want to compile Arkouda with that?)
    proc matMult(A: [?D1] ?t1, B: [?D2] ?t2, C: [?D3] ?T3)
        where D1.rank == 2 && D2.rank == 2 && D3.rank == 2
    {
        const (m      , k      ) = D1.shape,
              (_ /*k*/, n      ) = D2.shape,
              (_ /*m*/, _ /*n*/) = D2.shape;

        for i in 0..<m do
            for j in 0..<n do
                for l in 0..<k do
                    C[i, j] += A[i, l] * B[l, j];
    }

    @arkouda.registerND
    proc transposeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        if nd < 2 {
            const errorMsg = "Matrix transpose with arrays of dimension < 2 is not supported";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        const name = msgArgs.getValueOf("array"),
              rname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s rname: %s".doFormat(
            cmd,dtype2str(gEnt.dtype),rname));

        proc doTranspose(type t) {
            var eIn = toSymEntry(gEnt, t, nd),
                outShape: eIn.tupShape.type;

            outShape[outShape.size-2] <=> outShape.tupShape[outShape.size-1];

            var eOut = st.addEntry(rname, (...outShape), t);
            transpose(eIn.a, eOut.a);

            const repMsg = "created " + st.attrib(rname);
            linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select gEnt.dtype {
            when DType.Int64 do return doTranspose(int);
            when DType.UInt8 do return doTranspose(uint(8));
            when DType.UInt64 do return doTranspose(uint);
            when DType.Float64 do return doTranspose(real);
            when DType.Bool do return doTranspose(bool);
            otherwise {
                const errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    // TODO: performance improvements. Should use tiling to keep data local
    proc transpose(A: [?D], B) {
        forall idx in D {
            var bIdx = idx;
            bIdx[D.rank-2] <=> idx[D.rank-1];
            bIdx[D.rank-1] <=> idx[D.rank-2];
            B[bIdx] = A[idx];
        }
    }
}
