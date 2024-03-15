module ArkoudaAryUtilCompat {
  /*
    Get a domain that selects out the idx'th set of indices along the specified axes

    :arg D: the domain to slice
    :arg idx: the index to select along the specified axes (must have the same rank as D)
    :arg axes: the axes to slice along (must be a subset of the axes of D)

    For example, if D represents a stack of 1000 10x10 matrices (ex: {1..10, 1..10, 1..1000})
    Then, domOnAxis(D, (1, 1, 25), 0, 1) will return D sliced with {1..10, 1..10, 25..25}
    (i.e., the 25th matrix)
  */
  proc domOnAxis(D: domain(?), idx: D.rank*int, axes: int ...?NA): domain
    where NA < D.rank
  {
    var outDims: D.rank*range;
    label ranks for i in 0..<D.rank {
      for param j in 0..<NA {
        if i == axes[j] {
          outDims[i] = D.dim(i);
          continue ranks;
        }
      }
      outDims[i] = idx[i]..idx[i];
    }
    return D[{(...outDims)}];
  }

  proc domOnAxis(D: domain, idx: D.rank*int, axes: [?aD] int): domain throws {
    if axes.size >= D.rank then
      throw new Error("Cannot create a " + axes.size:string + "-dimensional slice from a " + D.rank:string + "-dimensional domain");

    var outDims: D.rank*range;
    label ranks for i in 0..<D.rank {
      for j in 0..<axes.size {
        if i == axes[j] {
          outDims[i] = D.dim(i);
          continue ranks;
        }
      }
      outDims[i] = idx[i]..idx[i];
    }
    return D[{(...outDims)}];
  }

  /*
    Get a domain over the set of indices orthogonal to the specified axes

    :arg D: the domain to slice
    :arg axes: the axes to slice along (must be a subset of the axes of D)

    For example, if D represents a stack of 1000 10x10 matrices (ex: {1..10, 1..10, 1..1000})
    Then, domOffAxis(D, 0, 1) will return D sliced with {0..0, 0..0, 1..1000}
    (i.e., a set of indices for the 1000 matrices)
  */
  proc domOffAxis(D: domain(?), axes: int ...?NA): domain(?)
    where NA < D.rank
  {
    var outDims: D.rank*range;
    label ranks for i in 0..<D.rank {
      for param j in 0..<NA {
        if i == axes[j] {
          outDims[i] = 0..0;
          continue ranks;
        }
      }
      outDims[i] = D.dim(i);
    }
    return D[{(...outDims)}];
  }

  proc domOffAxis(D: domain, axes: [?aD] int): domain throws {
    if axes.size >= D.rank then
      throw new Error("Cannot create a " + axes.size:string + "-dimensional slice from a " + D.rank:string + "-dimensional domain");

    var outDims: D.rank*range;
    label ranks for i in 0..<D.rank {
      for j in 0..<axes.size {
        if i == axes[j] {
          outDims[i] = 0..0;
          continue ranks;
        }
      }
      outDims[i] = D.dim(i);
    }
    return D[{(...outDims)}];
  }

  /*
    Naively create a domain over a chunk of the input domain

    Chunks are created by splitting the largest dimension of the input domain
    into 'nChunks' roughly equal-sized chunks, and then taking the
    'chunkIdx'-th chunk

    (if 'nChunks' is greater than the size of the largest dimension, the
    first 'nChunks-1' chunks will be empty, and the last chunk will contain
    the entire domain)
  */
  proc subDomChunk(dom: domain, chunkIdx: int, nChunks: int): domain {
    const dimSizes = [i in 0..<dom.rank] dom.dim(i).size,
          (maxDim, maxDimIdx) = maxloc reduce zip(dimSizes, dimSizes.domain);

    const chunkSize = maxDim / nChunks,
          start = chunkIdx * chunkSize + dom.dim(maxDimIdx).low,
          end = if chunkIdx == nChunks-1
            then dom.dim(maxDimIdx).high
            else (chunkIdx+1) * chunkSize + dom.dim(maxDimIdx).low - 1;

    var rngs: dom.rank*range;
    for i in 0..<dom.rank do rngs[i] = dom.dim(i);
    rngs[maxDimIdx] = start..end;
    return {(...rngs)};
  }
}
