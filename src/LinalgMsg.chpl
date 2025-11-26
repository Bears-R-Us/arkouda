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

    use CommAggregation; // needed for SrcAggregator

    use BigInteger;

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

        const N = msgArgs["N"].toScalar(int),
              M = msgArgs["M"].toScalar(int),
              k = msgArgs["k"].toScalar(int),
              shape = (N, M);

        // N, M = dimensions of 2 dimensional matrix.
        // k = 0 gives ones along center diagonal
        // k = nonzero moves the diagonal up/right for diag > 0, and down/left for diag < 0
        //        See comment below

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s aN: %i: aM: %i ak: %i".format(
            cmd,type2str(array_dtype),N,M,k));

        var e = createSymEntry((...shape), array_dtype);

        // Now put the ones where they go, on the main diagonal if k == 0, otherise
        // up-and-right by 'k' spaces) if k > 0,
        // or down-and-left by abs(k) spaces if k < 0

        if k == 0 {
            forall ij in 0..<min(N, M) do
                e.a[ij, ij] = 1 : array_dtype;
        } else if k > 0 {
            forall i in 0..<min(N, M-k) do
                e.a[i, i+k] = 1 : array_dtype;
        } else if k < 0 {
            forall j in 0..<min(N+k, M) do
                e.a[j-k, j] = 1 : array_dtype;
        }

        return st.insert(e);
    }

    //  tril and triu are identical except for the argument they pass to triluHandler (true for upper, false for lower)
    //  The zeros are written into the upper (or lower) triangle of the array, offset by the value of diag.

    //  Create an array from an existing array with its upper triangle zeroed out

    @arkouda.instantiateAndRegister
    proc tril(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws 
        where array_nd >= 2{
        return triluHandler(cmd, msgArgs, st, array_dtype, array_nd, false);
    }

    //  Create an array from an existing array with its lower triangle zeroed out

    @arkouda.instantiateAndRegister
    proc triu(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws 
    where array_nd >= 2 {
        return triluHandler(cmd, msgArgs, st, array_dtype, array_nd, true);
    }

    //  Fetch the arguments, call zeroTri, return result.
    // TODO: support instantiating param bools with 'true' and 'false' s.t. we'd have 'triluHandler<true>' and 'triluHandler<false>'
    //       cmds if this procedure were annotated instead of the two above.
    proc triluHandler(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                      type array_dtype, param array_nd: int, param upper: bool
    ): MsgTuple throws {
        const name = msgArgs["array"].toScalar(string),
              diag = msgArgs["diag"].toScalar(int);

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s aName: %s aDiag: %i".format(
            cmd,name,diag));

        var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd);
        var eOut = createSymEntry((...eIn.tupShape), array_dtype);

        eOut.a = eIn.a;
        zeroTri(eOut.a, diag, upper);

        return st.insert(eOut);
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
    proc matmul(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_x1, type array_dtype_x2, param array_nd: int): MsgTuple throws 
        where (array_nd >= 2) && (array_dtype_x1 != BigInteger.bigint) && (array_dtype_x2 != BigInteger.bigint) {

        // Get the left and right arguments.

        const x1Name = msgArgs["x1"].toScalar(string),
              x2Name = msgArgs["x2"].toScalar(string);

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype1: %s dtype2: %s".format(
            cmd,type2str(array_dtype_x1),type2str(array_dtype_x2)));

        var x1E = st[x1Name]: borrowed SymEntry(array_dtype_x1, array_nd),
            x2E = st[x2Name]: borrowed SymEntry(array_dtype_x2, array_nd);

        type resultType = numpyReturnType(array_dtype_x1, array_dtype_x2);

        const (valid, outDims, err) = assertValidDims(x1E, x2E);
        if !valid then return err;

        var eOut = createSymEntry((...outDims), resultType);  // create entry of deduced output type

        const x1 = x1E.a : resultType,
              x2 = x2E.a : resultType;

        if array_nd == 2
            then matMult(x1, x2, eOut.a);
            else batchedMatMult(x1, x2, eOut.a);

        return st.insert(eOut);

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

    //  batchedMatMult is now OBE with the addition of multidimmatmul, and will
    //  probably be removed in a future update.

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

    // Addendum: revised with foralls in the inmost loop.  There is probably still benefit
    // to trying to distribute the computation.

    proc matMult(in A: [?D1] ?t, in B: [?D2] t, ref C: [?D3] t)
        where D1.rank == 2 && D2.rank == 2 && D3.rank == 2
    {
        const (m      , k      ) = D1.shape,
              (_ /*k*/, n      ) = D2.shape,
              (_ /*m*/, _ /*n*/) = D2.shape;

        var sum: t;

        for i in 0..<m do
            for j in 0..<n {
                if t != bool {
                    sum = + reduce forall l in 0..<k do (A[i,l] * B[l,j]);
                } else {
                    sum = || reduce forall l in 0..<k do (A[i,l] && B[l,j]);
                }                
                C[i,j] = sum;
            }
    }

    // Transpose an array.

    @arkouda.registerCommand
    proc transpose(array: [?d] ?t): [] t throws
    where d.rank >= 2 {
        var outShape = array.shape;
        outShape[outShape.size-2] <=> outShape[outShape.size-1];
        var ret = makeDistArray((...outShape), t);
        
        // TODO: performance improvements. Should use tiling to keep data local
        forall idx in d {
            var bIdx = idx;
            bIdx[d.rank-1] <=> bIdx[d.rank-2];  // bIdx is now the reverse of idx
            ret[bIdx] = array[idx];                   // making B the transpose of A
        }
    
        return ret;
    }

    // For many numeric functions, the return type depends on the type of two inputs.  The
    // function numpyReturnType provides return types that match the behavior of numpy.

    // This function is used in dotProd and matMult.

    proc numpyReturnType(type ta, type tb) type {
        if ( (ta==real || tb==real) || (ta==int && tb==uint) || (ta==uint && tb==int) ) {
            return real ;
        } else if (ta==int || tb==int) {
            return int ;
        } else if (ta==uint || tb==uint) {
            return uint ;
        } else {
            return bool ;
        }
    }

    // TODO: add a special case of dotProd where b is 1-D.  This is currently handled by
    // padding a 1-D vector to 2-D client-side, doing the computation here, and removing
    // the padded dimension from the result.

    // TODO: add bigint handling

    // This implements the M-D x N-D case of ak.dot, aligning its funcionality to np.dot.
 
    @arkouda.registerCommand(name="dot")
    proc dotProd(a: [?d] ?ta, b: [?d2] ?tb): [] numpyReturnType(ta,tb) throws
    where ((d.rank >=2 && d2.rank>= 2)
        && (ta == int || ta == real || ta == bool || ta == uint )
        && (tb == int || tb == real || tb == bool || tb == uint )
           ) {
        param pn = Reflection.getRoutineName();

        // make an array of the appropriate shape and type to hold the output

        var eOut = makeDistArray((...dotShape(a.shape, b.shape)), numpyReturnType(ta,tb));

        // aOffset and bOffset will be used to iterate through A and B in the computation

        var aOffset: (d.rank)*int;
        for i in 0..<d.rank do aOffset[i] = (if i == d.rank - 1 then d.shape[d.rank - 1] else -1);

        var bOffset: (d2.rank)*int;
        for i in 0..<d2.rank do bOffset[i] = (if i == d2.rank - 2 then d2.shape[d2.rank - 2] else -1);

        // tmp1Domain and tmp2Domain look like 0..0 in all dimensions except the one where a and b
        // must have the same shape.  In that dimension, they look like 0..<k, where k is the size
        // of that dimension.

        var tmp1Domain = d.interior(aOffset);
        var tmp2Domain = d2.interior(bOffset);

        forall outIdx in eOut.domain with (
            var tmp1: [tmp1Domain] ta,
            var tmp2: [tmp2Domain] tb,
            const aOffsetTmp = aOffset,
            const bOffsetTmp = bOffset
        ) {
            var aIdx: (d.rank)*int;
            var bIdx: (d2.rank)*int;

            // Deduce the aIdx and bIdx from the output index.

            for i in 0..<d.rank-1 do aIdx[i] = outIdx[i];
            for i in 0..<d2.rank-2 do bIdx[i] = outIdx[d.rank-1+i];
            aIdx[d.rank-1] = 0;
            bIdx[d2.rank-1] = outIdx[outIdx.size-1];
            bIdx[d2.rank-2] = 0;

            // Set up the domains to grab the k elements at a time.

            var aDom = d;
            var bDom = d2;

            // Make the domains k long in the common dimension, 1 long in every other dimension.

            aDom = aDom.interior(aOffsetTmp);
            bDom = bDom.interior(bOffsetTmp);

            // Shift the domains by the index, which is set to zero in the common domain.

            aDom = aDom.translate(aIdx);
            bDom = bDom.translate(bIdx);

            // Grab the data with the domain slice.

            tmp1 = a[aDom];                 
            tmp2 = b[bDom];
            
            var total: numpyReturnType(ta,tb) = 0:numpyReturnType(ta,tb);

            // Perform the dot product.

            for (i, j) in zip(tmp1, tmp2) {
                if numpyReturnType(ta,tb) == bool {
                    total |= i:bool && j:bool;
                }
                else {
                    total += i:numpyReturnType(ta,tb) * j:numpyReturnType(ta,tb);
                }
            }
            eOut[outIdx] = total;
            
        }

        return eOut;
    }

    // dotShape creates a new shape based on the input shapes and the rules of
    // ak.dot (equivalently np.dot).
    // Note: error checking (e.g. ensuring the two ks are equal) was already
    // done python-side.

    // In general, if aShape is ( (front dims), k)
    //            and bShape is ( (back dims), k, m)
    //           then shapeOut is ( (front dims), (back dims), m)

    proc dotShape(aShape: ?N*int, bShape: ?N2*int): (N+N2-2)*int
        where N > 1 && N2 > 1 {
        var shapeOut: (N+N2-2)*int;
        for i in 0..<(N-1) do shapeOut[i] = aShape[i];
        for i in 0..<(N2-2) do shapeOut[(N - 1) + i] = bShape[i];
        shapeOut[N + N2 - 3] = bShape[N2 - 1];
        return shapeOut;
    }


    // What follows is code to distribute multi-dimensional matrix multiplication.

    // MatMulShape determines from the shapes of a and b what the shape of the product
    // should be.  Both ranks must be >= 2, and (in multiDimMatMul below) at least one
    // must be >= 3.

    proc MatMulShape (aShape: ?Na*int, bShape: ?Nb*int) : max(Na, Nb)*int throws
        where (Nb >= 2 && Na >= 2) {
        param outN = if Nb >= Na then Nb else Na;
        var shapeOut: outN*int;
	    shapeOut[outN-1] = bShape[Nb-1];
	    shapeOut[outN-2] = aShape[Na-2];
        if Nb == 2 || Na == 2 {
            if Nb >= Na {
                for i in 0..<outN-2 do shapeOut[i] = bShape[i];
            } else {
                for i in 0..<outN-2 do shapeOut[i] = aShape[i];
            }
            return shapeOut;
        }
        var aPreshape: (Na-2)*int;
        var bPreshape: (Nb-2)*int;
	    for i in 0..<(Na-2) {aPreshape[i] = aShape[i];}
		for i in 0..<(Nb-2) {bPreshape[i] = bShape[i];}
    	const cPreshape = try broadcastShape(aPreshape,bPreshape);
	    for i in 0..<Na-2 {shapeOut[i] = cPreshape[i];}
        return shapeOut;
    }

    //  Shapes were checked for broadcast compatibility on the python side.  This code
    //  checks that, uses the above function to determine the shape for c, but doesn't
    //  actually do any broadcasting -- it just adjusts the indices of a and b as if
    //  they had been broadcast.

    //  For example, if a has shape (2,1,30,15),
    //              and b has shape (1,3,15,20),
    //  then the output c has shape (2,3,30,20).

    //  If a and b had actually been broadcast, their new shapes would be:
    //                         a => (2,3,30,15)
    //                         b => (2,3,15,20)

    //  and computing c(i,j,m,n) would involves a sum over k of
    //                a(i,j,m,k) * b(i,j,k,n).

    //  Since actually doing the broadcast takes time and memory, we don't bother.
    //  In this example, that would mean the the sum for c(i,j,m,n) is the sum over k of
    //                a(i,0,m,k)*b(0,j,k,n).

    //  This is implemented in the makeAidx and makeBidx functions below.
        
    @arkouda.registerCommand(name="multidimmatmul")
    proc multiDimMatMul(a: [?da] ?ta, b: [?db] ?tb): [] numpyReturnType(ta,tb) throws
    where ( (da.rank >= 2 && db.rank >= 2 && (da.rank >= 3 || db.rank >= 3))
        && (ta == int || ta == real || ta == bool || ta == uint )
        && (tb == int || tb == real || tb == bool || tb == uint )
           ) {
        param pn = Reflection.getRoutineName();

        type RT = numpyReturnType(ta,tb);
        var c = makeDistArray((...MatMulShape(a.shape, b.shape)), RT);
        const kRange = 0..<a.shape[da.rank-1] ; // the loop index

        coforall loc in Locales do on loc {
           const ClocDom = c.localSubdomain();

           if !ClocDom.isEmpty() {

               var aRange: da.rank*range;
               var bRange: db.rank*range;
               var aoffset = c.rank - da.rank;
               var boffset = c.rank - db.rank;

               for i in 0..<da.rank {
                   aRange[i] =
                       if i == da.rank - 1 then kRange
                       else if a.shape[i] == 1 then 0..0
                       else ClocDom.dim[i + aoffset];
               }
               for i in 0..<db.rank {
                   bRange[i] =
                       if i == db.rank - 2 then kRange
                       else if b.shape[i] == 1 then 0..0
                       else ClocDom.dim[i + boffset];
               }

               // Now that we have ranges, we construct the subdomains, and pull the values we need.

               const SubDomA = {(...aRange)};
               const SubDomB = {(...bRange)};

               var Alocal = a[SubDomA];
               var Blocal = b[SubDomB];


               forall Cidx in ClocDom {

               // Helpers to build Aidx and Bidx for the given k

                   proc makeAidx(k: int): da.rank*int {
                       var idx: da.rank*int;
                       for param i in 0..<da.rank-1 do
                           idx[i] =
                               if aRange[i] == (0..0) then 0
                               else Cidx[i+aoffset];
                       idx[da.rank-1] = k;
                       return idx;
                   }

                   proc makeBidx(k: int): db.rank*int {
                       var idx: db.rank*int;
                       for param i in 0..<db.rank do
                           idx[i] =
                               if i == db.rank-2 then k
                               else if bRange[i] == (0..0) then 0
                               else Cidx[i+boffset];
                       return idx;
                   }

                   // The computation

                   var sum: RT;
                   if RT != bool {
                      sum = + reduce forall k in kRange do
                                Alocal[ makeAidx(k) ]:RT * Blocal[ makeBidx(k) ]:RT;
                   } else {
                      sum = || reduce forall k in kRange do
                                Alocal[ makeAidx(k) ]:bool && Blocal[ makeBidx(k) ]:bool;
                   }
                   c[Cidx] = sum;
               }
            }
        }
        return c;
    }

    /*
        Compute the generalized dot product of two tensors along the specified axis.

        Assumes that both tensors have already been broadcasted to have the same shape
        with 'nd' dimensions.
    */

    @arkouda.instantiateAndRegister
    proc vecdot(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_x1, type array_dtype_x2, param array_nd: int): MsgTuple throws 
        where (array_nd >= 2) && ((array_dtype_x1 != bool) || (array_dtype_x2 != bool)) 
            && (array_dtype_x1 != BigInteger.bigint) && (array_dtype_x2 != BigInteger.bigint) {

        const x1Name = msgArgs["x1"],
              x2Name = msgArgs["x2"],
              bcShape = msgArgs["bcShape"].toScalarTuple(int, array_nd),
              axis = msgArgs["axis"].toScalar(int);

        var x1 = st[x1Name]: borrowed SymEntry(array_dtype_x1, array_nd),
            x2 = st[x2Name]: borrowed SymEntry(array_dtype_x2, array_nd);

        linalgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype1: %s dtype2: %s".format(
            cmd,type2str(array_dtype_x1),type2str(array_dtype_x2)));

        if x1.tupShape != bcShape || x2.tupShape != bcShape {
            return MsgTuple.error("Incompatible array shapes for VecDot: " + x1.tupShape:string + ", " + x2.tupShape:string +
                ". VecDot assumes both arrays have been broadcasted to have the same shape, matching the bcShape argument.");
        }

        const _axis = if axis < 0 then array_nd + axis else axis;
        if _axis < 0 || _axis >= array_nd {
            return MsgTuple.error("Invalid axis for VecDot: " + axis:string);
        }

        const outShape = try! removeAxis(bcShape, _axis);
        type resultType = compute_result_type_vecdot(array_dtype_x1, array_dtype_x2);
        var eOut = createSymEntry((...outShape), resultType);

        forall idx in eOut.a.domain {
            const opDom = try! domOnAxis(x1.a.domain, appendAxis(idx, _axis, 0), _axis);

            var sum = 0: resultType;
            for i in opDom do
                sum += x1.a[i]:resultType * x2.a[i]:resultType;

            eOut.a[idx] = sum;
        }

        return st.insert(eOut);
    }

    proc compute_result_type_vecdot(type t1, type t2) type {
        if t1 == real || t2 == real then return real;
        if t1 == int || t2 == int then return int;
        if t1 == uint(8) || t2 == uint(8) then return uint(8);
        if t1 == uint(64) || t2 == uint(64) then return uint(64);
        return bool;
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

    //     proc doTensorDot(type array_dtype_x1, type array_dtype_x2, type resultType): MsgTuple throws {
    //         var ex1 = toSymEntry(x1G, array_dtype_x1, nd1),
    //             ex2 = toSymEntry(x2G, array_dtype_x2, nd2);

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
