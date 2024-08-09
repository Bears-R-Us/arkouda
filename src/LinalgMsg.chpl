module LinalgMsg {

    use Reflection;
    use Logging;
    use Message;
    use ServerConfig;
    use CommandMap;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use AryUtil;

    import BigInteger;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const linalgLogger = new Logger(logLevel, logChannel);

    /*
        Create an 'identity' matrix of a given size with ones along a given diagonal
        This only creates two dimensional arrays, so the array_nd parameter isn't used.
     	The matrix doesn't have to be square.  The row and col counts are supplied as
     	arguments.
    */
    @arkouda.instantiateAndRegister
    proc eye(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws 
	where array_dtype != BigInteger.bigint
    {

        const rows = msgArgs.get("rows").getIntValue(),
              cols = msgArgs.get("cols").getIntValue(),
              diag = msgArgs.get("diag").getIntValue();

	// rows, cols = dimensions of 2 dimensional matrix.
	// diag = 0 gives ones along center diagonal
	// diag = nonzero moves the diagonal up/right for diag > 0, and down/left for diag < 0
	//        See comment below

        const dtype = type2str(array_dtype),
              rname = st.nextName();

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s rname: %s aRows: %i: aCols: %i aDiag: %i".format(
            cmd,type2str(array_dtype),rname,rows,cols,diag));

        var e = st.addEntry(rname, rows, cols, array_dtype); // all elements initialize to zero

	// Now put the ones where they go, on the main diagonal if diag == 0, otherise
	// up-and-right by 'diag' spaces if diag > 0,
	// or down-and-left by abs(diag) spaces if diag < 0

        if diag == 0 {
            forall ij in 0..<min(rows, cols) do
                e.a[ij, ij] = 1 : array_dtype;
        } else if diag > 0 {
            forall i in 0..<min(rows, cols-diag) do
                e.a[i, i+diag] = 1 : array_dtype;
        } else if diag < 0 {
            forall j in 0..<min(rows+diag, cols) do
                e.a[j-diag, j] = 1 : array_dtype;
        }

        const repMsg = "created " + st.attrib(rname);
        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }


    proc eye(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws 
       where array_dtype == BigInteger.bigint
    {
        return MsgTuple.error("eye does not support the bigint dtype");
    }

    //  tril and triu are identical except for the argument they pass to triluHandler (true for upper, false for lower)
    //  The zeros are written into the upper (or lower) triangle of the array, offset by the value of diag.

    //  Create an array from an existing array with its upper triangle zeroed out

    @arkouda.instantiateAndRegister
    proc tril(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        if array_nd < 2 {
            const errorMsg = "Array must be at least 2 dimensional for 'tril'";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        return triluHandler(cmd, msgArgs, st, array_dtype, array_nd, false);
    }

    //  Create an array from an existing array with its lower triangle zeroed out

    @arkouda.instantiateAndRegister
    proc triu(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        if array_nd < 2 {
            const errorMsg = "Array must be at least 2 dimensional for 'triu'";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        return triluHandler(cmd, msgArgs, st, array_dtype, array_nd, true);
    }

    //  Fetch the arguments, call zeroTri, return result. 

    proc triluHandler(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                      type array_dtype, param array_nd: int, param upper: bool
    ): MsgTuple throws {
        const name = msgArgs.getValueOf("array"),
              diag = msgArgs.get("diag").getIntValue();

        const rname = st.nextName();

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s rname: %s aName: %s aDiag: %i".format(
            cmd,rname,name,diag));

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        var eIn = toSymEntry(gEnt, array_dtype, array_nd),
            eOut = st.addEntry(rname, (...eIn.tupShape), array_dtype);

        eOut.a = eIn.a;
        zeroTri(eOut.a, diag, upper);

        const repMsg = "created " + st.attrib(rname);
        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    // When upper = false, zero out the upper diagonal.

    private proc zeroTri(ref a: [?d] ?t, diag: int, param upper: bool)
        where d.rank >= 2 && upper == false
    {
        const iThresh = if diag < 0 then abs(diag) else 0,
              jThresh = if diag > 0 then diag else 0;

        forall idx in d {
            const i = idx[d.rank-2],
                  j = idx[d.rank-1];

            if i - iThresh < j - jThresh then a[idx] = 0 : t;
        }
    }

    // When upper = true, zero out the lower diagonal.

    private proc zeroTri(ref a: [?d] ?t, diag: int, param upper: bool)
        where d.rank >= 2 && upper == true
    {
        const iThresh = if diag < 0 then abs(diag) else 0,
              jThresh = if diag > 0 then diag else 0;

        forall idx in d {
            const i = idx[d.rank-2],
                  j = idx[d.rank-1];

            if i - iThresh > j - jThresh then a[idx] = 0 : t;
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

    @arkouda.instantiateAndRegister
    proc matmul(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd: int): MsgTuple throws {
        if array_nd < 2 {
            const errorMsg = "Matrix multiplication with arrays of dimension < 2 is not supported";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

	// Get the left and right arguments.

        const x1Name = msgArgs.getValueOf("x1"),
              x2Name = msgArgs.getValueOf("x2");

        var x1G: borrowed GenSymEntry = getGenericTypedArrayEntry(x1Name, st),
            x2G: borrowed GenSymEntry = getGenericTypedArrayEntry(x2Name, st);

        const rname = st.nextName();

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype1: %s dtype2: %s rname: %s".format(
            cmd,dtype2str(x1G.dtype),dtype2str(x2G.dtype),rname));

	//  The select block below (and in vecdot) implements the rule:
	//     if either input is real, the output is real
	//     else if either input is integer, the output is integer
	//     else if either input is uint(8), the output is uint(8)
	//     else if both inputs are bool, the output is bool
	//     else error condition.

        select (x1G.dtype, x2G.dtype) {
		// real cases
            when (DType.Float64, DType.Int64)   do return doMatMult(real,    int,     real);
            when (DType.Float64, DType.UInt8)   do return doMatMult(real,    uint(8), real);
            when (DType.Float64, DType.Float64) do return doMatMult(real,    real,    real);
            when (DType.Float64, DType.Bool)    do return doMatMult(real,    bool,    real);
            when (DType.Int64, DType.Float64)   do return doMatMult(int,     real,    real);
            when (DType.UInt8, DType.Float64)   do return doMatMult(uint(8), real,    real);
            when (DType.Bool, DType.Float64)    do return doMatMult(bool,    real,    real);
	    	// int cases
            when (DType.Int64, DType.Int64)     do return doMatMult(int,     int,     int);
            when (DType.Int64, DType.UInt8)     do return doMatMult(int,     uint(8), int);
            when (DType.Int64, DType.Bool)      do return doMatMult(int,     bool,    int);
            when (DType.UInt8, DType.Int64)     do return doMatMult(uint(8), int,     int);
            when (DType.Bool, DType.Int64)      do return doMatMult(bool,    int,     int);
	    	// uint cases
            when (DType.UInt8, DType.UInt8)     do return doMatMult(uint(8), uint(8), uint(8));
            when (DType.UInt8, DType.Bool)      do return doMatMult(uint(8), bool,    uint(8));
            when (DType.Bool, DType.UInt8)      do return doMatMult(bool,    uint(8), uint(8)); 
	    	// bool case
            when (DType.Bool, DType.Bool)       do return doMatMult(bool,    bool,    bool);
            otherwise {
                const errorMsg = notImplementedError(getRoutineName(), "matmul", x1G.dtype, x2G.dtype);
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

	// doMatMult is called once the types are worked out.

        proc doMatMult(type x1Type, type x2Type, type resultType): MsgTuple throws {
            var x1E = toSymEntry(x1G, x1Type, array_nd),
                x2E = toSymEntry(x2G, x2Type, array_nd);

            const (valid, outDims, err) = assertValidDims(x1E, x2E);
            if !valid then return err;

            var eOut = st.addEntry(rname, (...outDims), resultType); // create entry of deduced output type

            const x1 = x1E.a : resultType,
                  x2 = x2E.a : resultType;

            if array_nd == 2
                then matMult(x1, x2, eOut.a);
                else batchedMatMult(x1, x2, eOut.a);

            const repMsg = "created " + st.attrib(rname);
            linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

    }

    // Check that dimensions are valid for matrix multiplication.

    proc assertValidDims(ref x1, ref x2) throws {
        const (validInputDims, outDims) = matMultDims(x1.tupShape, x2.tupShape);
        if !validInputDims {
            const errorMsg = "Invalid dimensions for matrix multiplication";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return (false, outDims, new MsgTuple(errorMsg, MsgType.ERROR));
        } else {
            return (true, outDims, new MsgTuple("", MsgType.NORMAL));
        }
    }

    proc matMultDims(a: ?Na*int, b: ?Nb*int): (bool, Na*int) {
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

    proc batchedMatMult(in A: [?D], in B, ref C) throws {
        const BatchDom = domOffAxis(D, D.rank-2, D.rank-1);

        // for each matrix in the batch, perform matrix multiplication
        forall i in BatchDom {
            const matrixI = domOnAxis(D, i, D.rank-2, D.rank-1);

            // TODO: pass the 'matrixI' domain and a reference to the
            //  full matrices to a parallel matMult proc
            //  instead of creating new slices for each call
            var a = removeDegenRanks(A[matrixI], 2);
            var b = removeDegenRanks(B[matrixI], 2);
            var c = removeDegenRanks(C[matrixI], 2);

            matMult(a, b, c);
        }
    }

    // TODO: not performant at all -- use tiled and parallel matrix multiplication
    //  or maybe use the linear algebra module? (do we want to compile Arkouda with that?)

    proc matMult(in A: [?D1] ?t, in B: [?D2] t, ref C: [?D3] t)
        where D1.rank == 2 && D2.rank == 2 && D3.rank == 2
    {
        const (m      , k      ) = D1.shape,
              (_ /*k*/, n      ) = D2.shape,
              (_ /*m*/, _ /*n*/) = D2.shape;

        for i in 0..<m do
            for j in 0..<n do
                for l in 0..<k do
		    if t != bool {
                    	C[i, j] += A[i, l] * B[l, j];
		    } else {
			C[i,j] |= A[i, l] & B[l, j];
		    }
    }

    // Transpose an array.

    @arkouda.instantiateAndRegister
    proc transpose(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        if array_nd < 2 {
            const errorMsg = "Matrix transpose with arrays of dimension < 2 is not supported";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        const name = msgArgs.getValueOf("array"),
              rname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s rname: %s".format(
            cmd,type2str(array_dtype),rname));

	// get the input array, copy its shape

        var eIn = toSymEntry(gEnt, array_dtype, array_nd),
            outShape = eIn.tupShape;

	// switch the indices of the output shape

        outShape[outShape.size-2] <=> outShape[outShape.size-1];

	// create the output array

        var eOut = st.addEntry(rname, (...outShape), array_dtype);
        doTranspose(eIn.a, eOut.a); // do the transpose

        const repMsg = "created " + st.attrib(rname);
        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    // TODO: performance improvements. Should use tiling to keep data local

    proc doTranspose(ref A: [?D], ref B) {
        forall idx in D {
            var bIdx = idx;
            bIdx[D.rank-1] <=> bIdx[D.rank-2];  // bIdx is now the reverse of idx
            B[bIdx] = A[idx];			// making B the transpose of A
        }
    }

    /*
        Compute the generalized dot product of two tensors along the specified axis.

        Assumes that both tensors have already been broadcasted to have the same shape
        with 'nd' dimensions.
    */

    @arkouda.instantiateAndRegister
    proc vecdot(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd: int): MsgTuple throws {
        if array_nd < 2 {
            const errorMsg = "VecDot with arrays of dimension < 2 is not supported";
            linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        const x1Name = msgArgs.getValueOf("x1"),
              x2Name = msgArgs.getValueOf("x2"),
              bcShape = msgArgs.get("bcShape").getTuple(array_nd),
              axis = msgArgs.get("axis").getIntValue(),
              rname = st.nextName();

        var x1G: borrowed GenSymEntry = getGenericTypedArrayEntry(x1Name, st),
            x2G: borrowed GenSymEntry = getGenericTypedArrayEntry(x2Name, st);

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype1: %s dtype2: %s rname: %s".format(
            cmd,dtype2str(x1G.dtype),dtype2str(x2G.dtype),rname));

        // assumes both arrays have been broadcasted to ND dimensions

	// See the comment in matmul for a description of how this deduces the output type
	// from the input types.  The one exception here is that "both inputs are bool" is
	// not allowed.

        select (x1G.dtype, x2G.dtype) {
		// real cases
            when (DType.Float64, DType.Int64)   do return doVecdot(real,    int,     real);
            when (DType.Float64, DType.UInt8)   do return doVecdot(real,    uint(8), real);
            when (DType.Float64, DType.Float64) do return doVecdot(real,    real,    real);
            when (DType.Float64, DType.Bool)    do return doVecdot(real,    bool,    real);
            when (DType.Int64, DType.Float64)   do return doVecdot(int,     real,    real);
            when (DType.UInt8, DType.Float64)   do return doVecdot(uint(8), real,    real);
            when (DType.Bool, DType.Float64)    do return doVecdot(bool,    real,    real);
	    	// int cases
            when (DType.Int64, DType.Int64)     do return doVecdot(int,     int,     int);
            when (DType.Int64, DType.UInt8)     do return doVecdot(int,     uint(8), int);
            when (DType.Int64, DType.Bool)      do return doVecdot(int,     bool,    int);
            when (DType.UInt8, DType.Int64)     do return doVecdot(uint(8), int,     int);
            when (DType.Bool, DType.Int64)      do return doVecdot(bool,    int,     int);
	    	// uint cases
            when (DType.UInt8, DType.UInt8)     do return doVecdot(uint(8), uint(8), uint(8));
            when (DType.UInt8, DType.Bool)      do return doVecdot(uint(8), bool,    uint(8));
	    	// bool case
            when (DType.Bool, DType.UInt8)      do return doVecdot(bool,    uint(8), uint(8));
            otherwise {
                const errorMsg = notImplementedError(getRoutineName(), "vecdot", x1G.dtype, x2G.dtype);
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        proc doVecdot(type x1Type, type x2Type, type resultType): MsgTuple throws {
            var ex1 = toSymEntry(x1G, x1Type, array_nd),
                ex2 = toSymEntry(x2G, x2Type, array_nd);

            if ex1.tupShape != bcShape || ex2.tupShape != bcShape {
                const errorMsg = "Incompatible array shapes for VecDot: " + ex1.tupShape:string + ", " + ex2.tupShape:string +
                    ". VecDot assumes both arrays have been broadcasted to have the same shape, matching the bcShape argument.";
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }

            const _axis = if axis < 0 then array_nd + axis else axis;
            if _axis < 0 || _axis >= array_nd {
                const errorMsg = "Invalid axis for VecDot: " + axis:string;
                linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }

            const outShape = try! removeAxis(bcShape, _axis);
            var eOut = st.addEntry(rname, (...outShape), resultType);

            forall idx in eOut.a.domain {
                const opDom = try! domOnAxis(ex1.a.domain, appendAxis(idx, _axis, 0), _axis);

                var sum = 0: resultType;
                for i in opDom do
                    sum += ex1.a[i]:resultType * ex2.a[i]:resultType;

                eOut.a[idx] = sum;
            }

            const repMsg = "created " + st.attrib(rname);
            linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
    }

    // @arkouda.registerND(???)
    // proc tensorDotMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd1: int, param nd2: int): MsgTuple throws {
    //     if nd < 3 {
    //         const errorMsg = "TensorDot with arrays of dimension < 3 is not supported";
    //         linalgLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
    //         return new MsgTuple(errorMsg, MsgType.ERROR);
    //     }

    //     const x1Name = msgArgs.getValueOf("x1"),
    //           x2Name = msgArgs.getValueOf("x2"),
    //           rname = st.nextName();

    //     proc doTensorDot(type x1Type, type x2Type, type resultType): MsgTuple throws {
    //         var ex1 = toSymEntry(x1G, x1Type, nd1),
    //             ex2 = toSymEntry(x2G, x2Type, nd2);

    //         var tdDims;
    //         try {
    //             const n = msgArgs.get("n").getIntValue();
    //             tdDims = tensorDotDims(ex1.tupShape, ex2.tupShape, n);
    //         } catch {
    //             const a1 = msgArgs.get("axis1").getTuple(nd1),
    //                   a2 = msgArgs.get("axis2").getTuple(nd2);

    //             tdDims = tensorDotDims(ex1.tupShape, ex2.tupShape, a1, a2);
    //         }
    //     }


    // }

    // proc tensorDotDims(sa: ?Na*int, sb: ?Nb*int, param nCont): (bool, Na+Nb-2*nCont) {
    //     if Na+Nb-2*nCont < 0 then compilerError("Invalid number of contraction dimensions for tensor dot");

    //     var aCont: nCont*int,
    //         bCont: nCont*int;

    //     for param i in 0..<nCont {
    //         aCont[i] = Na-nCont+i;
    //         bCont[i] = i;
    //     }

    //     return tensorDotDims(sa, sb, aCont, bCont);
    // }

    // proc tensorDotDims(sa: ?Na*int, sb: ?Nb*int, aCont: ?nCont*int, bCont: nCont*int): (bool, (Na+Nb-2*nCont)*int) {
    //     if Na+Nb-2*nCont < 0 then compilerError("Invalid contraction dimensions for tensor dot");
    //     var s: (Na + Nb - 2*nCont)*int;

    //     for param i in 0..<nCont {
    //         if sa[aCont[i]] != sb[bCont[i]] then return (false, s);
    //     }

    //     var i = 0;
    //     for param ii in 0..<Na {
    //         if !contains(aCont, ii) {
    //             s[i] = sa[ii];
    //             i += 1;
    //         }
    //     }
    //     for param ii in 0..<Nb {
    //         if !contains(bCont, ii) {
    //             s[i] = sb[ii];
    //             i += 1;
    //         }
    //     }

    //     return (true, s);
    // }

    // proc contains(a: ?Na*int, param x: int): bool {
    //     for param i in 0..<Na {
    //         if a[i] == x then return true;
    //     }
    //     return false;
    // }
}
