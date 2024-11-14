module SparseMatrix {

  public use SpsMatUtil;
  use ArkoudaSparseMatrixCompat;
  use BlockDist;
  use CommAggregation;

  // Quick and dirty, not permanent
  proc fillSparseMatrix(ref spsMat, const A: [?D] ?eltType, param l: Layout) throws {
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

    // Note: this simplified loop cannot be used because iteration over spsMat.domain
    //       occures one locale at a time (i.e., the first spsMat.domain.parDom.localSubdomain(Locales[0]).size
    //       values from 'A' are deposited on locale 0, and so on), rather than depositing
    //       them row-major or column-major globally
    // for ((i,j), idx) in zip(spsMat.domain,A.domain) {
    //   spsMat[i,j] = A[idx];
    // }

    if l == Layout.CSR {
      var idx = 0;
      for i in spsMat.domain.parentDom.dim(0) {
        for j in spsMat.domain.parentDom.dim(1) {
          if spsMat.domain.contains((i, j)) {
            spsMat[i,j] = A[idx];
            idx += 1;
          }
        }
      }
    } else {
      var idx = 0;
      for j in spsMat.domain.parentDom.dim(1) {
        for i in spsMat.domain.parentDom.dim(0) {
          if spsMat.domain.contains((i, j)) {
            spsMat[i,j] = A[idx];
            idx += 1;
          }
        }
      }
    }
  }

  proc getGrid(const ref spsMat) where spsMat.chpl_isNonDistributedArray() {
    return reshape([here,], {0..<1, 0..<1});
  }

  proc getGrid(const ref spsMat) where !spsMat.chpl_isNonDistributedArray() {
    return spsMat.domain.targetLocales();
  }

  proc getLSD(const ref spsMat) where spsMat.chpl_isNonDistributedArray() {
    return spsMat.domain;
  }

  proc getLSD(const ref spsMat) where !spsMat.chpl_isNonDistributedArray() {
    return spsMat.domain.localSubdomain();
  }

  proc getLSA(const ref spsMat, rowBlockIdx: int, colBlockIdx: int) const ref
    where spsMat.chpl_isNonDistributedArray()
  {
    return spsMat;
  }

  proc getLSA(const ref spsMat, rowBlockIdx: int, colBlockIdx: int) const ref
    where !spsMat.chpl_isNonDistributedArray()
  {
    return spsMat.getLocalSubarray(rowBlockIdx, colBlockIdx);
  }

  /*
    Fill the rows, cols, and vals arrays with the non-zero indices and values
    from the sparse matrix in row-major order.
  */
  proc sparseMatToPdarray(const ref spsMat, ref rows, ref cols, ref vals, param layout: Layout)
    where layout == Layout.CSR
  {
    // // serial algorithm (for reference):
    // var idx = 0;
    // for i in spsMat.domain.parentDom.dim(0) {
    //   for j in spsMat.domain.parentDom.dim(1) {
    //     if spsMat.domain.contains((i, j)) {
    //       rows[idx] = i;
    //       cols[idx] = j;
    //       vals[idx] = spsMat[i, j];
    //       idx += 1;
    //     }
    //   }
    // }

    const grid = getGrid(spsMat);

    // number of non-zeros in each row, for each column-block of the matrix
    // TODO: make this a sparse array ('spsMat.shape[0]' could be very large)
    const nnzDom = blockDist.createDomain({1..spsMat.shape[0], grid.dim(1)}, targetLocales=grid);
    var nnzPerColBlock: [nnzDom] int;

    // compute the number of non-zeros in each column-section of each row
    coforall (rowBlockIdx, colBlockIdx) in grid.domain with (ref nnzPerColBlock) {
      on grid[rowBlockIdx, colBlockIdx] {
        const lsd = getLSD(spsMat);

        forall i in lsd.rows() with (const ref lsd) {
          nnzPerColBlock.localAccess[i, colBlockIdx] =
            lsd.startIdx[i+1] - lsd.startIdx[i];
        }
      }
    }

    // scan operation to find the starting index (in the 1D output arrays) for each column-block of each row
    const colBlockStartOffsets: [nnzDom] int = rowMajorExScan(nnzPerColBlock, grid, spsMat.domain.parentDom);

    // deposit indices and values into output arrays in parallel
    coforall (rowBlockIdx, colBlockIdx) in grid.domain with (ref rows, ref cols, ref vals) {
      on grid[rowBlockIdx, colBlockIdx] {
        const lsd = getLSD(spsMat);
        const ref lsa = getLSA(spsMat, rowBlockIdx, colBlockIdx);

        forall i in lsd.rows() with (var idxAgg = newDstAggregator(int),
                                     var valAgg = newDstAggregator(spsMat.eltType),
                                     const ref lsa) {
          var idx = colBlockStartOffsets.localAccess[i, colBlockIdx];
          for (j, v) in lsa.colsAndVals(i) {
            idxAgg.copy(rows[idx], i);
            idxAgg.copy(cols[idx], j);
            valAgg.copy(vals[idx], v);
            idx += 1;
          }
        }
      }
    }
  }

  // /*
  //   Fill the rows, cols, and vals arrays with the non-zero indices and values
  //   from the sparse matrix in col-major order.
  // */
  proc sparseMatToPdarray(const ref spsMat, ref rows, ref cols, ref vals, param layout: Layout)
    where layout == Layout.CSC
  {
    // // serial algorithm (for reference):
    // var idx = 0;
    // for j in spsMat.domain.parentDom.dim(1) {
    //   for i in spsMat.domain.parentDom.dim(0) {
    //     if spsMat.domain.contains((i, j)) {
    //       rows[idx] = i;
    //       cols[idx] = j;
    //       vals[idx] = spsMat[i, j];
    //       idx += 1;
    //     }
    //   }
    // }

    const grid = getGrid(spsMat);

    // number of non-zeros in each column, for each row-block of the matrix
    // TODO: make this a sparse array ('spsMat.shape[1]' could be very large)
    const nnzDom = blockDist.createDomain({grid.dim(0), 1..spsMat.shape[1]}, targetLocales=grid);
    var nnzPerRowBlock: [nnzDom] int;

    // compute the number of non-zeros in each column-section of each row
    coforall (rowBlockIdx, colBlockIdx) in grid.domain with (ref nnzPerRowBlock) {
      on grid[rowBlockIdx, colBlockIdx] {
        const lsd = getLSD(spsMat);

        forall j in lsd.cols() with (const ref lsd) {
          nnzPerRowBlock.localAccess[rowBlockIdx, j] =
            lsd.startIdx[j+1] - lsd.startIdx[j];
        }
      }
    }

    // scan operation to find the starting index (in the 1D output arrays) for each row-block of each column
    const rowBlockStartOffsets: [nnzDom] int = colMajorExScan(nnzPerRowBlock, grid, spsMat.domain.parentDom);

    // compute the number of non-zeros in each column-section of each row
    coforall (rowBlockIdx, colBlockIdx) in grid.domain with (ref rows, ref cols, ref vals) {
      on grid[rowBlockIdx, colBlockIdx] {
        const lsd = getLSD(spsMat);
        const ref lsa = getLSA(spsMat, rowBlockIdx, colBlockIdx);

        forall j in lsd.cols() with (var idxAgg = newDstAggregator(int),
                                     var valAgg = newDstAggregator(spsMat.eltType),
                                     const ref lsa) {
          var idx = rowBlockStartOffsets.localAccess[rowBlockIdx, j];

          for (i, v) in lsa.rowsAndVals(j) {
            idxAgg.copy(rows[idx], i);
            idxAgg.copy(cols[idx], j);
            valAgg.copy(vals[idx], v);
            idx += 1;
          }
        }
      }
    }
  }

  // helper function for sparseMatToPdarrayCSR
  // computes a row-major flattened scan of a distributed 2D array in parallel
  proc rowMajorExScan(in nnzPerColBlock: [?d] int, grid: [] locale, const ref pdom) {
    var colBlockStartOffsets: [d] int;

    // // serial algorithm (for reference):
    // var sum = 0;
    // for i in 1..d.dim(0).size {
    //  for colBlockIdx in d.dim(1) {
    //     colBlockStartOffsets[i, colBlockIdx] = sum;
    //     sum += nnzPerColBlock[i, colBlockIdx];
    //   }
    // }

    // domain distributed over the first column of 'grid' only
    const interDom = blockDist.createDomain(
      grid.dim(0),
      targetLocales=reshape(grid[{grid.dim(0), 0..0}], {grid.dim(0)})
    );

    // array to store sum of values within each row block
    var intermediate: [interDom] int;

    // compute an exclusive scan within each row block
    coforall rowBlockIdx in grid.dim(0) with (ref colBlockStartOffsets) do on grid[rowBlockIdx, 0] {
      const iRange = pdom.localSubdomain().dim(0),
            nRowTasks = min(here.maxTaskPar, iRange.size),
            nRowsPerTask = iRange.size / nRowTasks;

      // array to store sum of values within each task's group of rows
      var rowBlockIntermediate: [0..<nRowTasks] int;

      // compute an exclusive scan within each task's group of rows
      coforall rt in 0..<nRowTasks with (ref rowBlockIntermediate) {
        const iStart = rt * nRowsPerTask + iRange.low,
              iStop = if rt == nRowTasks-1 then iRange.last else (rt+1) * nRowsPerTask + iRange.low - 1;

        var rtSum = 0;
        for i in iStart..iStop {
          for colBlockIdx in d.dim(1) {
            colBlockStartOffsets[i, colBlockIdx] = rtSum;

            // TODO: would aggregation we worthwhile here (for the non-local accesses of the non-zero columns)?
            rtSum += nnzPerColBlock[i, colBlockIdx];
          }
        }

        rowBlockIntermediate[rt] = rtSum;
      }

      // compute a scan on the intermediate results and update this row-block of the array
      rowBlockIntermediate = + scan rowBlockIntermediate;
      intermediate.localAccess[rowBlockIdx] = rowBlockIntermediate.last;

      coforall rt in 1..<nRowTasks {
        const iStart = rt * nRowsPerTask + iRange.low,
              iStop = if rt == nRowTasks-1 then iRange.last else (rt+1) * nRowsPerTask + iRange.low - 1;

        colBlockStartOffsets[{iStart..iStop, d.dim(1)}] += rowBlockIntermediate[rt-1];
      }
    }

    // compute a scan on the intermediate results and update the global array
    intermediate = + scan intermediate;

    coforall rowBlockIdx in 1..<grid.dim(0).size with (ref colBlockStartOffsets) do on grid[rowBlockIdx, 0] {
      const iRange = pdom.localSubdomain().dim(0);
      colBlockStartOffsets[{iRange, d.dim(1)}] += intermediate[rowBlockIdx-1];
    }

    return colBlockStartOffsets;
  }

  // helper function for sparseMatToPdarrayCSC
  // computes a col-major flattened scan of a distributed 2D array in parallel
  proc colMajorExScan(in nnzPerRowBlock: [?d] int, grid: [] locale, const ref pdom) {
    var rowBlockStartOffsets: [d] int;

    // // serial algorithm (for reference):
    // var sum = 0;
    // for j in 1..d.dim(1).size {
    //  for rowBlockIdx in d.dim(0) {
    //     rowBlockStartOffsets[rowBlockIdx, j] = sum;
    //     sum += nnzPerRowBlock[rowBlockIdx, j];
    //   }
    // }

    // domain distributed over the first column of 'grid' only
    const interDom = blockDist.createDomain(
      grid.dim(1),
      targetLocales=reshape(grid[{0..0, grid.dim(1)}], {grid.dim(1)})
    );

    // array to store sum of values within each column block
    var intermediate: [interDom] int;

    // compute an exclusive scan within each column block
    coforall colBlockIdx in grid.dim(1) with (ref rowBlockStartOffsets) do on grid[0, colBlockIdx] {
      const jRange = pdom.localSubdomain().dim(1),
            nColTasks = min(here.maxTaskPar, jRange.size),
            nColsPerTask = jRange.size / nColTasks;

      // array to store sum of values within each task's group of cols
      var colBlockIntermediate: [0..<nColTasks] int;

      // compute an exclusive scan within each task's group of rows
      coforall ct in 0..<nColTasks with (ref colBlockIntermediate) {
        const jStart = ct * nColsPerTask + jRange.low,
              jStop = if ct == nColTasks-1 then jRange.last else (ct+1) * nColsPerTask + jRange.low - 1;

        var ctSum = 0;
        for j in jStart..jStop {
          for rowBlockIdx in d.dim(0) {
            rowBlockStartOffsets[rowBlockIdx, j] = ctSum;

            // TODO: would aggregation we worthwhile here (for the non-local accesses of the non-zero rows)?
            ctSum += nnzPerRowBlock[rowBlockIdx, j];
          }
        }

        colBlockIntermediate[ct] = ctSum;
      }

      // compute a scan on the intermediate results and update this col-block of the array
      colBlockIntermediate = + scan colBlockIntermediate;
      intermediate.localAccess[colBlockIdx] = colBlockIntermediate.last;

      coforall ct in 1..<nColTasks {
        const jStart = ct * nColsPerTask + jRange.low,
              jStop = if ct == nColTasks-1 then jRange.last else (ct+1) * nColsPerTask + jRange.low - 1;

        rowBlockStartOffsets[{d.dim(0), jStart..jStop}] += colBlockIntermediate[ct-1];
      }
    }

    // compute a scan on the intermediate results and update the global array
    intermediate = + scan intermediate;

    coforall colBlockIdx in 1..<grid.dim(1).size with (ref rowBlockStartOffsets) do on grid[0,colBlockIdx] {
      const jRange = pdom.localSubdomain().dim(1);
      rowBlockStartOffsets[{d.dim(0), jRange}] += intermediate[colBlockIdx-1];
    }

    return rowBlockStartOffsets;
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

  proc randSparseMatrix(shape: 2*int, density, param layout, type eltType) {
    import SymArrayDmap.makeSparseDomain;
    var (SD, dense) = makeSparseDomain(shape, layout);

    // randomly include index pairs based on provided density
    for (i,j) in dense do
        if rands.next() <= density then
          SD += (i,j);

    var A: [SD] eltType;
    return A;
  }

  proc sparseMatFromArrays(rows, cols, vals, shape: 2*int, param layout,
                           type eltType) throws {
    import SymArrayDmap.makeSparseDomain;
    var (SD, dense) = makeSparseDomain(shape, layout);

    const minRow = min reduce rows;
    const maxRow = max reduce rows;
    const minCol = min reduce cols;
    const maxCol = max reduce cols;
    param e1 = "Sparse matrix indices must be greater than 0; got (%i, %i)";
    param e2 = "Sparse matrix indices must be less than the shape;"
               +"got (%i, %i) >= (%i, %i)";
    if minRow < 0 || minCol < 0 then
      throw getErrorWithContext(
         // TODO, change this when we start matrix from 0 instead of 1
        msg=e1.format(minRow, minCol),
        lineNumber=getLineNumber(),
        routineName=getRoutineName(),
        moduleName=getModuleName(),
        errorClass="InvalidArgumentError"
        );
    if maxRow >= shape[0] || maxCol >= shape[1] then
      throw getErrorWithContext(
        msg=e2.format(maxRow, maxCol, shape[0], shape[1]),
        lineNumber=getLineNumber(),
        routineName=getRoutineName(),
        moduleName=getModuleName(),
        errorClass="InvalidArgumentError"
        );

    var A: [SD] eltType;
    addElementsToSparseArray(A, SD, rows, cols, vals);

    return A;
  }

  proc addElementsToSparseArray(ref A, ref SD, const ref rows,const ref cols,
                                const ref vals) throws where
                                A.chpl_isNonDistributedArray() {
    for (r,c,v) in zip(rows, cols, vals) {
      if A.domain.contains(r,c) then
        throw getErrorWithContext(
          msg="Duplicate index (%i, %i) in sparse matrix".format(r, c),
          lineNumber=getLineNumber(),
          routineName=getRoutineName(),
          moduleName=getModuleName(),
          errorClass="InvalidArgumentError"
          );
      SD += (r,c);
      A[r,c] = v;
    }
  }


  proc addElementsToSparseArray(ref A, ref SD, const ref rows, const ref cols,
                                const ref vals) throws where
                                !A.chpl_isNonDistributedArray() {
    coforall (loc, locDom) in zip(getGrid(A),
                                  SD._value.locDoms) {
      on loc {
        for _srcLocId in loc.id..#numLocales {
          const srcLocId = _srcLocId % numLocales;
          var rowChunk = rows[rows.localSubdomain(Locales[srcLocId])];
          var colChunk = cols[rows.localSubdomain(Locales[srcLocId])];
          var valChunk = vals[rows.localSubdomain(Locales[srcLocId])];
          for (r,c,v) in zip(rowChunk, colChunk, valChunk) {
            if locDom!.parentDom.contains(r,c) {
              if locDom!.mySparseBlock.contains(r,c) then
                throw getErrorWithContext(
                  msg="Duplicate index (%i, %i) in sparse matrix".format(r, c),
                  lineNumber=getLineNumber(),
                  routineName=getRoutineName(),
                  moduleName=getModuleName(),
                  errorClass="InvalidArgumentError"
                  );


              locDom!.mySparseBlock += (r,c);
              A[r,c] = v;
            }
          }
        }
      }
    }

  }

  module SpsMatUtil {
    // The following are routines that should arguably be supported directly
    // by the LayoutCS and SparseBlockDist modules themselves
    //
    //  public use LayoutCSUtil, SparseBlockDistUtil;

    use BlockDist, Map, Random;

    enum Layout {
      CSR,
      CSC
    };

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
      use ArkoudaSparseMatrixCompat;
      use Sort;

      var CDom: sparse subdomain(parentDom) dmapped getSparseDom(Layout.CSR);
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
