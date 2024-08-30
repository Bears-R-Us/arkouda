module SparseMatrix {

  public use SpsMatUtil;


  // Quick and dirty, not permanent
  proc fillSparseMatrix(ref spsMat, const A: [?D] ?eltType) throws {
    if A.rank != 1 then
        throw getErrorWithContext(
                        msg="fill vals requires a 1D array; got a %iD array".format(A.rank),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IllegalArgumentError"
                        );
    if A.size != spsMat.domain.getNNZ() then
        throw getErrorWithContext(
                        msg="fill vals requires an array of the same size as the sparse matrix; got %i elements, expected %i".format(A.size, spsMat.domain.getNNZ()),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IllegalArgumentError"
                        );
    if eltType != spsMat.eltType then
        throw getErrorWithContext(
                        msg="fill vals requires an array of the same type as the sparse matrix; got %s, expected %s".format(eltType, spsMat.eltType),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IllegalArgumentError"
                        );
    for((i,j), idx) in zip(spsMat.domain,A.domain) {
        spsMat[i,j] = A[idx];
    }
  }

  proc sparseMatToPdarray(const ref spsMat, ref rows, ref cols, ref vals){
    for((i,j), idx) in zip(spsMat.domain,0..) {
      rows[idx] = i;
      cols[idx] = j;
      vals[idx] = spsMat[i, j];
    }
  }
  // sparse, outer, matrix-matrix multiplication algorithm; A is assumed
  // CSC and B CSR
  proc sparseMatMatMult(A, B) {
    var spsData: sparseMatDat;

    sparseMatMatMult(A, B, spsData);

    var C = makeSparseMat(A.domain.parentDom, spsData);
    return C;
  }

  // This version forms the guts of the above and permits a running set
  // of nonzeroes to be passed in and updated rather than assuming that
  // the multiplication is the first/only step.
  //
  proc sparseMatMatMult(A, B, ref spsData) {
    forall ac_br in A.cols() with (merge reduce spsData) do
      for (ar, a) in A.rowsAndVals(ac_br) do
        for (bc, b) in B.colsAndVals(ac_br) do
          spsData.add((ar, bc), a * b);
  }

  proc sparseMatMatMult(A, B) where (!A.chpl_isNonDistributedArray() &&
                                     !B.chpl_isNonDistributedArray()) {
    var CD = emptySparseDomLike(B);  // For now, hard-code C to use CSR, like B
    var C: [CD] int;

    ref targLocs = A.targetLocales();
    coforall (locRow, locCol) in targLocs.domain {
      on targLocs[locRow, locCol] {
        var spsData: sparseMatDat;

        for srcloc in targLocs.dim(0) {
          // Make a local copy of the remote blocks of A and B; on my branch
          // this will also make a local copy of the remote indices, so long
          // as these are 'const'/read-only
          //
          const aBlk = A.getLocalSubarray(locRow, srcloc),
                bBlk = B.getLocalSubarray(srcloc, locCol);

          // This local block is not strictly necessary but ensures that the
          // computation on the blocks will not require communication
          local {
            sparseMatMatMult(aBlk, bBlk, spsData);
          }
        }

        // Get my locale's local indices and create a sparse matrix
        // using them and the spsData computed above.
        //
        const locInds = A.domain.parentDom.localSubdomain();
        var cBlk = makeSparseMat(locInds, spsData);

        // Stitch the local portions back together into the global-view
        //
        CD.setLocalSubdomain(cBlk.domain);
        C.setLocalSubarray(cBlk);
      }
    }
    return C;
  }


  // dense, simple matrix-matrix multiplication algorithm; this is
  // wildly inefficient, both because it ignores the sparsity and
  // because it uses random access of the sparse arrays which tends to
  // be expensive.
  //
  proc denseMatMatMult(A, B) {
    const n = A.dim(0).size;

    var spsData: sparseMatDat;

    for i in 1..n {
      for j in 1..n {
        var prod = 0;

        forall k in 1..n with (+ reduce prod) do
          prod += A[i,k] * B[k,j];

        if prod != 0 then
          spsData.add((i,j), prod);
      }
    }

    var C = makeSparseMat(A.domain.parentDom, spsData);
    return C;
  }

  proc randSparseMatrix(size, density, param layout, param distributed=false, type eltType) {
    const Dom = {1..size, 1..size};

    // compute some random sparse index patterns for the matrices
    //
    const AD = randSparseDomain(Dom, density, layout, distributed);

    var A: [AD] eltType;
    return A;
  }

  module SpsMatUtil {
    // The following are routines that should arguably be supported directly
    // by the LayoutCS and SparseBlockDist modules themselves
    //
    //  public use LayoutCSUtil, SparseBlockDistUtil;

    use BlockDist, LayoutCS, Map, Random;

    enum layout {
      CSR,
      CSC
    };
    // public use layout;

    config const seed = 0;

    var rands = if seed == 0 then new randomStream(real)
                          else new randomStream(real, seed);

    record sparseMatDat {
      forwarding var m: map(2*int, int);

      proc ref add(idx: 2*int, val: int) {
        if val != 0 {
          if m.contains(idx) {
            m[idx] += val;
          } else {
            m.add(idx, val);
          }
        }
      }
    }


    // create a local random sparse matrix within the space of 'Dom' of
    // the given density and layout.  If distributed is true, this will
    // be a block-distributed sparse matrix, otherwise it'll be local.
    //
    proc randSparseDomain(parentDom, density, param matLayout, param distributed)
    where distributed == false {

      var SD: sparse subdomain(parentDom) dmapped new dmap(new CS(compressRows=(matLayout==layout.CSR)));

      for (i,j) in parentDom do
        if rands.next() <= density then
          SD += (i,j);

      return SD;
    }

    proc randSparseDomain(parentDom, density, param matLayout, param distributed)
    where distributed == true {
      const locsPerDim = sqrt(numLocales:real): int,
            grid = {0..<locsPerDim, 0..<locsPerDim},
            localeGrid = reshape(Locales[0..<grid.size], grid);

      type layoutType = CS(compressRows=(matLayout==layout.CSR));
      const DenseBlkDom = parentDom dmapped new blockDist(boundingBox=parentDom,
                                                    targetLocales=localeGrid,
                                                    sparseLayoutType=layoutType);

      var SD: sparse subdomain(DenseBlkDom);

      for (i,j) in parentDom do
        if rands.next() <= density then
          SD += (i,j);

      return SD;
    }


    proc emptySparseDomLike(Mat) {
      var ResDom: sparse subdomain(Mat.domain.parentDom);
      return ResDom;
    }


    // print out a sparse matrix (in a dense format)
    //
    proc writeSparseMatrix(msg, Arr) {
      const ref SparseDom = Arr.domain,
                DenseDom = SparseDom.parentDom;

      writeln(msg);

      for r in DenseDom.dim(0) {
        for c in DenseDom.dim(1) {
          write(Arr[r,c], " ");
        }
        writeln();
      }
      writeln();
    }


    // create a new sparse matrix from a map from sparse indices to values
    //
    proc makeSparseMat(parentDom, spsData) {
      use Sort;

      var CDom: sparse subdomain(parentDom) dmapped new dmap(new CS());
      var inds: [0..<spsData.size] 2*int;
      for (idx, i) in zip(spsData.keys(), 0..) do
        inds[i] = idx;

      sort(inds);

      for ij in inds do
        CDom += ij;

      var C: [CDom] int;
      for ij in inds do
        try! C[ij] += spsData[ij];  // TODO: Should this really throw?

      return C;
    }


    // create a new sparse matrix from a collection of nonzero indices
    // (nnzs) and values (vals)
    //
    proc makeSparseMat(parentDom, nnzs, vals) {
      var CDom: sparse subdomain(parentDom);
      for ij in nnzs do
        CDom += ij;

      var C: [CDom] int;
      for (ij, c) in zip(nnzs, vals) do
        C[ij] += c;
      return C;
    }


    // This is a custom reduction, and a good case study for why our custom
    // reduction interface needs a refresh
    //
    class merge: ReduceScanOp {
      type eltType = sparseMatDat;
      var value: eltType;  // TODO: lots of deep copying here to avoid

      proc identity {
        var ident: eltType;
        return ident;
      }

      proc accumulate(x) {
        // Why is this ever called with a sparseMatDat as the argument?!?
        for (k,v) in zip(x.keys(), x.values()) {
          if value.contains(k) {
            value[k] += v;
          } else {
            value.add(k, v);
          }
        }
      }

      proc accumulateOntoState(ref state, x) {
        halt("Error, shouldn't call merge.accumulateOntoState()");
      }

      proc combine(x) {
        accumulate(x.value);
      }

      proc generate() {
        return value;
      }

      inline proc clone() {
        return new unmanaged merge(eltType=eltType);
      }
    }
  }
}
