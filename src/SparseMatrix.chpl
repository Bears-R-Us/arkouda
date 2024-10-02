module SparseMatrix {

  public use SpsMatUtil;
  use ArkoudaSparseMatrixCompat;
  use BlockDist;
  use CommAggregation;

  // Quick and dirty, not permanent
  proc fillSparseMatrix(ref spsMat, const A: [?D] ?eltType, param l: layout) throws {
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

    if l == layout.CSR {
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

  /*
    Fill the rows, cols, and vals arrays with the non-zero indices and values
    from the sparse matrix in row-major order.
  */
  proc sparseMatToPdarrayCSR(const ref spsMat, ref rows, ref cols, ref vals) {
    // // serial algorithm (for reference):
    // for((i,j), idx) in zip(spsMat.domain,0..) {
      // rows[idx] = i;
      // cols[idx] = j;
      // vals[idx] = spsMat[i, j];
    // }

    // info about matrix block distribution across a 2D grid of locales
    const grid = spsMat.domain.targetLocales(),
          nRowBlocks = grid.domain.dim(0).size,   // 2
          nColBlocks = grid.domain.dim(1).size;   // 2

    // number of non-zeros in each row, for each column-block of the matrix
    // TODO: make this a sparse array ('spsMat.shape[0]' could be very large)
    const nnzDom = blockDist.createDomain({1..spsMat.shape[0], 0..<nColBlocks}, targetLocales=grid);
    var nnzPerColBlock: [nnzDom] int;

    // compute the number of non-zeros in each column-section of each row
    coforall colBlockIdx in 0..<nColBlocks with (ref nnzPerColBlock) {
      coforall rowBlockIdx in 0..<nRowBlocks with (ref nnzPerColBlock) {
        on grid[rowBlockIdx, colBlockIdx] {
          const lsd = spsMat.domain.parentDom.localSubdomain(),
                iRange = lsd.dim(0),
                jRange = lsd.dim(1);

          // TODO: do all this with a single task if iRange.size < here.maxTaskPar (instead of spawning one task per row)
          const nRowTasks = min(here.maxTaskPar, iRange.size),
                nRowsPerTask = iRange.size / nRowTasks;

          coforall rt in 0..<nRowTasks {
            const iStart = rt * nRowsPerTask + iRange.low,
                  iStop = if rt == nRowTasks-1 then iRange.last else (rt+1) * nRowsPerTask + iRange.low - 1;

            for i in iStart..iStop {
              for j in jRange {
                if spsMat.domain.contains((i,j))
                  then nnzPerColBlock.localAccess[i, colBlockIdx] += 1;
              }
            }
          }
        }
      }
    }

    // scan operation to find the starting index (in the 1D output arrays) for each column-block of each row
    const colBlockStartOffsets: [nnzDom] int = rowMajorExScan(nnzPerColBlock, spsMat.domain.parentDom);

    // deposit indices and values into output arrays in parallel
    coforall colBlockIdx in 0..<nColBlocks with (ref rows, ref cols, ref vals) {
      coforall rowBlockIdx in 0..<nRowBlocks with (ref rows, ref cols, ref vals) {
        on grid[rowBlockIdx, colBlockIdx] {
          const lsd = spsMat.domain.parentDom.localSubdomain(),
                iRange = lsd.dim(0),
                jRange = lsd.dim(1);

          const nRowTasks = min(here.maxTaskPar, iRange.size),
                nRowsPerTask = iRange.size / nRowTasks;

          coforall rt in 0..<nRowTasks {
            const iStart = rt * nRowsPerTask + iRange.low,
                  iStop = if rt == nRowTasks-1 then iRange.last else (rt+1) * nRowsPerTask + iRange.low - 1;

            // aggregators to deposit indices and values into 1D output arrays
            var idxAgg = newDstAggregator(int),
                valAgg = newDstAggregator(spsMat.eltType);

            for i in iStart..iStop {
              var idx = colBlockStartOffsets.localAccess[i, colBlockIdx];
              for j in jRange {
                if spsMat.domain.contains((i,j)) {
                  idxAgg.copy(rows[idx], i);
                  idxAgg.copy(cols[idx], j);
                  valAgg.copy(vals[idx], spsMat[i, j]);
                  idx += 1;
                }
              }
            }
          }
        }
      }
    }
  }

  /*
    Fill the rows, cols, and vals arrays with the non-zero indices and values
    from the sparse matrix in col-major order.
  */
  proc sparseMatToPdarrayCSC(const ref spsMat, ref rows, ref cols, ref vals) {
    // // serial algorithm (for reference):
    // for((i,j), idx) in zip(spsMat.domain,0..) {
    //   rows[idx] = i;
    //   cols[idx] = j;
    //   vals[idx] = spsMat[i, j];
    // }

    // matrix shape
    const m = spsMat.shape[0],
          n = spsMat.shape[1];

    // info about matrix block distribution across a 2D grid of locales
    const grid = spsMat.domain.targetLocales(),
          nRowBlocks = grid.domain.dim(0).size,
          nColBlocks = grid.domain.dim(1).size;

    // number of non-zeros in each column, for each row-block of the matrix
    // TODO: use zero-based indexing for SparseSymEntry
    const nnzDom = blockDist.createDomain({0..<nRowBlocks, 1..n}, targetLocales=grid);
    var nnzPerRowBlock: [nnzDom] int;

    // details for splitting columns into groups for task-level parallelism
    const nTasksPerColBlock = here.maxTaskPar,
          nColGroups = min(nColBlocks * nTasksPerColBlock, n),
          nColsPerGroup = n / nColGroups,
          nRowsPerBlock = m / nRowBlocks;

    // compute the number of non-zeros in each row-section of each column
    coforall rowBlockIdx in 0..<nRowBlocks with (ref nnzPerRowBlock) {
      const iStart = rowBlockIdx * nRowsPerBlock + 1,
            iEnd = if rowBlockIdx == nRowBlocks-1 then m else (rowBlockIdx+1) * nRowsPerBlock;

      coforall cg in 0..<nColGroups with (ref nnzPerRowBlock) {
        const colBlockIdx = cg / nTasksPerColBlock;
        on grid[rowBlockIdx, colBlockIdx] {
          const jStart = cg * nColsPerGroup + 1,
                jEnd = if cg == nColGroups-1 then n else (cg+1) * nColsPerGroup;

          // TODO: there is probably a much smarter way to compute this information using the
          // underlying CSC data structures
          for j in jStart..jEnd {
            for i in iStart..iEnd {
              if spsMat.domain.contains((i,j)) {
                // TODO: this localAccess assumes that the `cg*nColsPerGroup` math lines up perfectly
                // with the matrix's block distribution; this is (probably) not guaranteed
                // - should use the parentDom to compute actual local indices
                // nnzPerRowBlock.localAccess[rowBlockIdx,j] += 1;
                nnzPerRowBlock[rowBlockIdx,j] += 1;
              }
            }
          }
        }
      }
    }

    // scan operation to find the starting index (in the 1D output arrays) for each row-block of each column
    const rowBlockStartOffsets = flattenedExScanCSC(nnzPerRowBlock, nColGroups, nTasksPerColBlock, nColsPerGroup);

    // store the non-zero indices and values in the output arrays
    coforall rowBlockIdx in 0..<nRowBlocks with (ref rows, ref cols, ref vals) {
      const iStart = rowBlockIdx * nRowsPerBlock + 1,
            iEnd = if rowBlockIdx == nRowBlocks-1 then m else (rowBlockIdx+1) * nRowsPerBlock;

      coforall cg in 0..<nColGroups with (ref rows, ref cols, ref vals) {
        const colBlockIdx = cg / nTasksPerColBlock;
        on grid[rowBlockIdx, colBlockIdx] {
          const jStart = cg * nColsPerGroup + 1,
                jEnd = if cg == nColGroups-1 then n else (cg+1) * nColsPerGroup;

          // aggregators to deposit indices and values into 1D output arrays
          var idxAgg = newSrcAggregator(int),
              valAgg = newSrcAggregator(spsMat.eltType);

          for j in jStart..jEnd {
            var idx = rowBlockStartOffsets[rowBlockIdx, j];
            for i in iStart..iEnd {
              if spsMat.domain.contains((i,j)) {
                rows[idx] = i;
                cols[idx] = j;
                vals[idx] = spsMat[i, j];
                // idxAgg.copy(rows[idx], i);
                // idxAgg.copy(cols[idx], j);
                // valAgg.copy(vals[idx], spsMat[i, j]); // TODO: (see above note about localAccess)
                idx += 1;
              }
            }
          }
        }
      }
    }
  }

  // helper function for sparseMatToPdarrayCSR
  // computes a row-major flattened scan of a distributed 2D array in parallel
  proc rowMajorExScan(in nnzPerColBlock: [?d] int, ref pdom) {
    const nColBlocks = d.dim(1).size,
          grid = d.targetLocales(),
          nRowBlocks = grid.dim(0).size;

    var colBlockStartOffsets: [d] int;

    // domain over the first column of 'grid'
    const interDom = blockDist.createDomain(
      0..<nRowBlocks,
      targetLocales=reshape(grid[{0..<nRowBlocks, 0..0}], {0..<nRowBlocks})
    );

    // array to store sum of values within each row block
    var intermediate: [interDom] int;

    // compute an exclusive scan within each row block
    coforall rowBlockIdx in 0..<nRowBlocks with (ref nnzPerColBlock) do on grid[rowBlockIdx, 0] {
      const iRange = pdom.localSubdomain().dim(0),
            nRowTasks = min(here.maxTaskPar, iRange.size),
            nRowsPerTask = iRange.size / nRowTasks;

      // array to store sum of values within each task's group of rows
      var rowBlockIntermediate: [0..<nRowTasks] int;

      // compute an exclusive scan within each task's group of rows
      coforall rt in 0..<nRowTasks {
        const iStart = rt * nRowsPerTask + iRange.low,
              iStop = if rt == nRowTasks-1 then iRange.last else (rt+1) * nRowsPerTask + iRange.low - 1;

        var rtSum = 0;
        for i in iStart..iStop {
          for colBlockIdx in 0..<nColBlocks {
            colBlockStartOffsets[i, colBlockIdx] = rtSum;

            // TODO: would aggregation we worthwhile here (for the non-local accesses of the non-zero columns)?
            rtSum += nnzPerColBlock[i, colBlockIdx];
          }
        }

        rowBlockIntermediate[rt] = rtSum;
      }

      // compute a scan on the intermediate results and upate this row-block of the array
      rowBlockIntermediate = + scan rowBlockIntermediate;
      intermediate.localAccess[rowBlockIdx] = rowBlockIntermediate.last;

      coforall rt in 1..<nRowTasks {
        const iStart = rt * nRowsPerTask + iRange.low,
              iStop = if rt == nRowTasks-1 then iRange.last else (rt+1) * nRowsPerTask + iRange.low - 1;

        colBlockStartOffsets[{iStart..iStop, 0..<nColBlocks}] += rowBlockIntermediate[rt-1];
      }
    }

    // compute a scan on the intermediate results and upate the global array
    intermediate = + scan intermediate;

    coforall rowBlockIdx in 1..<nRowBlocks with (ref nnzPerColBlock) do on grid[rowBlockIdx, 0] {
      const iRange = pdom.localSubdomain().dim(0);
      colBlockStartOffsets[{iRange, 0..<nColBlocks}] += intermediate[rowBlockIdx-1];
    }

    return colBlockStartOffsets;
  }

  // helper function for sparseMatToPdarrayCSC
  // computes a col-major flattened scan of a distributed 2D array in parallel
  proc flattenedExScanCSC(in nnzPerRowBlock: [?d] int, nColGroups: int, nTasksPerColBlock: int, nColsPerGroup: int) {
    const nRowBlocks = d.dim(0).size,
          n = d.dim(1).size,
          grid = d.targetLocales(),
          nColBlocks = grid.dim(1).size;

    var rowBlockStartOffsets: [d] int;

    // // serial algorithm (for reference):
    // var sum = 0;
    // for j in 1..n {
    //  for rowBlockIdx in 0..<nRowBlocks {
    //     rowBlockStartOffsets[rowBlockIdx, j] = sum;
    //     sum += nnzPerRowBlock[rowBlockIdx, j];
    //   }
    // }

    // 1D block distributed array representing intermediate result of scan for each column group
    // (distributed across grid's first row of locales only)
    const interDom = blockDist.createDomain(
      0..<nColGroups,
      targetLocales=reshape(grid[{0..0, 0..<nColBlocks}], {0..<nColBlocks})
    );
    var intermediate: [interDom] int;

    // compute an exclusive scan within each column group
    coforall cg in 0..<nColGroups with (ref rowBlockStartOffsets) {
      const colBlockIdx = cg / nTasksPerColBlock;
      on grid[0, colBlockIdx] {
        const jStart = cg * nColsPerGroup + 1,
              jEnd = if cg == nColGroups-1 then n else (cg+1) * nColsPerGroup;

        var cgSum = 0;
        for j in jStart..jEnd {
          for rowBlockIdx in 0..<nRowBlocks {
            rowBlockStartOffsets[rowBlockIdx, j] = cgSum;

            // TODO: would aggregation we worthwhile here (for the non-local accesses of the non-zero rows)?
            cgSum += nnzPerRowBlock[rowBlockIdx, j];
          }
        }

        // intermediate.localAccess[cg] = cgSum;
        intermediate[cg] = cgSum;
      }
    }

    // compute a scan of each column-group's sum
    intermediate = + scan intermediate;

    // update the column groups with the previous group's sum
    // (the 0'th column group is already correct)
    coforall cg in 1..<nColGroups with (ref rowBlockStartOffsets) {
      const colBlockIdx = cg / nTasksPerColBlock;
      on grid[0, colBlockIdx] {
        const jStart = cg * nColsPerGroup + 1,
              jEnd = if cg == nColGroups-1 then n else (cg+1) * nColsPerGroup;

        // TODO: explicit serial loop might be faster than slice assignment here
        rowBlockStartOffsets[{0..<nRowBlocks, jStart..jEnd}] +=
          intermediate[cg-1];
          // intermediate.localAccess[cg-1];
      }
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

  proc randSparseMatrix(size, density, param layout, type eltType) {
    import SymArrayDmap.makeSparseDomain;
    var (SD, dense) = makeSparseDomain(size, layout);

    // randomly include index pairs based on provided density
    for (i,j) in dense do
        if rands.next() <= density then
          SD += (i,j);

    var A: [SD] eltType;
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
