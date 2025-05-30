module ManipulationMsg {
  use CommandMap;
  use Message;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerConfig;
  use NumPyDType;
  use Logging;
  use ServerErrorStrings;
  use CommAggregation;
  use AryUtil;
  use BigInteger;

  use Reflection;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const mLogger = new Logger(logLevel, logChannel);

  /*
    Create a "broadcasted" array (of rank 'nd') by copying an array into an
    array of the given shape.

    E.g., given the following broadcast:
    A      (4d array):  8 x 1 x 6 x 1
    B      (3d array):      7 x 1 x 5
    ---------------------------------
    Result (4d array):  8 x 7 x 6 x 5

    Two separate calls would be made to store 'A' and 'B' in arrays with
    result's shape.

    When copying from a singleton dimension, the value is repeated along
    that dimension (e.g., A's 1st and 3rd, or B's 2nd dimension above).
    For non singleton dimensions, the size of the two arrays must match,
    and the values are copied into the result array.

    When prepending a new dimension to increase an array's rank, the
    values from the other dimensions are repeated along the new dimension.

    !!! TODO: Avoid the promoted copies here by leaving the singleton
    dimensions in the result array, and making operations on arrays
    aware that promotion of singleton dimensions may be necessary. E.g.,
    make matrix multiplication aware that it can treat a singleton
    value as a vector of the appropriate length during multiplication.
    (this may require a modification of SymEntry to keep track of
    which dimensions are explicitly singletons)

    NOTE: registration of this procedure is handled specially in
    'serverModuleGen.py' because it has two param fields. The purpose of
    designing "broadcast" this way is to avoid the the need for multiple
    dimensionality param fields in **all** other message handlers (e.g.,
    matrix multiply can be designed to expect matrices of equal rank,
    requiring only one dimensionality param field. As such, the client
    implementation of matrix-multiply may be required to broadcast the array
    arguments up to some common rank (N) before issuing a 'matMult{N}D'
    command to the server)

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.broadcast_to.html#array_api.broadcast_to
  */
  @arkouda.instantiateAndRegister(prefix='broadcast')
  proc broadcastToMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    type array_dtype,
    param array_nd_in: int,
    param array_nd_out: int
  ): MsgTuple throws {
    const name = msgArgs["name"],
          shapeOut = msgArgs["shape"].toScalarTuple(int, array_nd_out);

    var eIn = toSymEntry(st[name]: borrowed GenSymEntry, array_dtype, array_nd_in),
        eOut = createSymEntry((...shapeOut), array_dtype);

    if array_nd_in == array_nd_out && eIn.tupShape == shapeOut {
      // no broadcast necessary, copy the array
      eOut.a = eIn.a;
    } else {
      // ensure that 'shapeOut' is a valid broadcast of 'eIn.tupShape'
      //   and determine which dimensions will require promoted assignment
      var (valid, bcDims) = checkValidBroadcast(eIn.tupShape, shapeOut);

      if !valid {
        const errorMsg = "Invalid broadcast: " + eIn.tupShape:string + " -> " + shapeOut:string;
        mLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      } else {
        // use List;
        // var bcDimsList = new list(int);
        // for i in 0..<ndIn do if bcDims[i] then bcDimsList.pushBack(i);

        // // iterate over each slice of the output array corresponding to one
        // // copy of the input array and perform the copy
        // /* Example:
        //   broadcast 5x1x3 array into 5x4x3 array:
        //   - the 5x1x3 array is copied into the 5x4x3 array 4 times
        //   - domOffAxis => {0..0, 0..<4, 0..0}
        //   - for 'nonBCIndex' = (0, 0, 0), outSliceIdx = (0..<5, 0..0, 0..<3)
        //   - for 'nonBCIndex' = (0, 1, 0), outSliceIdx = (0..<5, 1..1, 0..<3)
        //   - etc.
        // */
        // forall nonBCIndex in domOffAxis(eOut.a.domain, bcDimsList.toArray()) {
        //   const nbcT = if ndOut == 1 then (nonBCIndex,) else nonBCIndex;


        //   var outSliceIdx: ndOut*range;
        //   for i in 0..<ndOut do outSliceIdx[i] = 0..<shapeOut[i];
        //   for i in 0..<ndIn do if bcDims[i] then outSliceIdx[i] = nbcT[i];

        //   eOut.a[(...outSliceIdx)] = eIn.a; // !!! Doesn't work because of rank mismatch !!!
        // }

        inline proc imap(idx: array_nd_out*int, bc: array_nd_in*int): array_nd_in*int {
          var ret: array_nd_in*int;
          for param i in 0..<array_nd_in do ret[i] = if bc[i] then 0 else idx[i + (array_nd_out - array_nd_in)];
          return ret;
        }

        // copy values from the input array into the output array
        forall idx in eOut.a.domain with (var agg = newSrcAggregator(array_dtype), in bcDims) {
          const idxIn = imap(if array_nd_out==1 then (idx,) else idx, bcDims);
          agg.copy(eOut.a[idx], eIn.a[idxIn]);
        }
      }
    }

    return st.insert(eOut);
  }

  proc broadcastToMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    type array_dtype,
    param array_nd_in: int,
    param array_nd_out: int
  ): MsgTuple throws
    where array_nd_in > array_nd_out
  {
    return MsgTuple.error(
      "Cannot broadcast from higher (%i) dimensional to lower (%i) dimensional array"
      .format(array_nd_in, array_nd_out)
    );
  }

  proc checkValidBroadcast(from: ?Nf*int, to: ?Nt*int): (bool, Nf*bool) {
    var dimsToBroadcast: Nf*bool;
    if Nf > Nt then return (false, dimsToBroadcast);

    for param iIn in 0..<Nf {
      param iOut = Nt - Nf + iIn;
      if from[iIn] == 1 {
        dimsToBroadcast[iIn] = true;
      } else if from[iIn] != to[iOut] {
        return (false, dimsToBroadcast);
      }
    }

    return (true, dimsToBroadcast);
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html#array_api.concat
  @arkouda.instantiateAndRegister(prefix='concat')
  proc concatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const nArrays = msgArgs["n"].toScalar(int),
          names = msgArgs["names"].toScalarArray(string, nArrays),
          axis = msgArgs["axis"].getPositiveIntValue(array_nd);

    const eIns = for n in names do st[n]: borrowed SymEntry(array_dtype, array_nd),
          shapes = [i in 0..<nArrays] eIns[i].tupShape,
          (valid, shapeOut, startOffsets) = concatenatedShape(array_nd, shapes, axis);

    if !valid {
      const errMsg = "Arrays must have compatible shapes to concatenate: " +
        "attempt to concatenate arrays of shapes %? along axis %?".format(shapes, axis);
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    } else {
      var eOut = createSymEntry((...shapeOut), array_dtype);

      // copy the data from the input arrays to the output array
      // TODO: Are nested foralls with aggregators a good idea?
      forall (arrIdx, arr) in zip(eIns.domain, eIns) with (in startOffsets) {
        forall idx in arr.a.domain with (var agg = newDstAggregator(array_dtype)) {
          var outIdx = if array_nd == 1 then (idx,) else idx;
          outIdx[axis] += startOffsets[arrIdx];
          agg.copy(eOut.a[outIdx], arr.a[idx]);
        }
      }

      return st.insert(eOut);
    }
  }

  private proc concatenatedShape(param n: int, in shapes: [?d] n*int, axis: int): (bool, n*int, [d] int) {
    var shapeOut = shapes[0],
        axisSizes: [d] int;
    axisSizes[0] = shapes[0][axis];

    for s in 1..<shapes.size {
      for param i in 0..<n {
        if i == axis {
          shapeOut[i] += shapes[s][i];
          axisSizes[s] = shapes[s][i];
        } else {
          // all non-axis dimensions must match
          if shapes[s][i] != shapeOut[i] then
            return (false, shapeOut, axisSizes);
        }
      }
    }

    const startOffsets = (+ scan axisSizes) - axisSizes;
    return (true, shapeOut, startOffsets);
  }

  // alternative to 'concatMsg' to be used when the axis argument is 'None'
  @arkouda.instantiateAndRegister(prefix='concatFlat')
  proc concatFlatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const nArrays = msgArgs["n"].toScalar(int),
          names = msgArgs["names"].toScalarArray(string, nArrays);

    const eIns = for n in names do st[n]: borrowed SymEntry(array_dtype, array_nd),
          sizes = [i in 0..<nArrays] eIns[i].a.size,
          starts = (+ scan sizes) - sizes;

    // create a 1D output array
    var eOut = createSymEntry(+ reduce sizes, array_dtype);

    // copy the data from the input arrays to the output array
    forall arrIdx in 0..<nArrays {
      const a = if array_nd == 1 then eIns[arrIdx].a else flatten(eIns[arrIdx].a);
      eOut.a[starts[arrIdx]..#sizes[arrIdx]] = a;
    }

    return st.insert(eOut);
  }

  // This deletes by copying by index everything that isn't deleted.
  @arkouda.registerCommand(name="deleteAggCopy")
  proc deleteAxisAggCopyMsg (eIn: [?d] ?t, axis: int, del: [?d2] ?t2): [] t throws 
    where ((t2 == int || t2 == bool) &&
           (t == int || t == real || t == bool || t == uint || t == uint(8) || t == bigint) &&
           (d2.rank == 1)) {
    param pn = Reflection.getRoutineName();
    
    // Need to ensure we have a boolean array of what gets deleted.
    var delBool = makeDistArray(eIn.shape[axis], bool);
    
    if t2 == int {

      // Error handling
      var invalidIndex: atomic bool = false;
      var badIndex: atomic int = -1;
      
      forall i in del.domain with (
        var agg = newDstAggregator(bool)
      ) {

        // Index out of bounds checking
        if del[i] < eIn.shape[axis] && del[i] >= 0 {
          agg.copy(delBool[del[i]], true);
        } else if del[i] >= -eIn.shape[axis] && del[i] < 0 {
          // numpy allows negative indices in this way
          agg.copy(delBool[del[i] + eIn.shape[axis]], true);
        } else {
          invalidIndex.write(true);
          badIndex.write(del[i]);
        }
        
      }

      if invalidIndex.read() {
        var errorMsg = incompatibleArgumentsError(pn, 
                          "index %i is out of bounds for axis %i with size %i".format(badIndex.read(), axis, eIn.shape[axis])); 
        mLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                               
        throw new Error(errorMsg);
      }

    } else {
      if (delBool.size != del.size) { 
        var errorMsg = incompatibleArgumentsError(pn, 
                          "Boolean array argument obj to delete must be one dimensional and match the axis length of %i".format(delBool.size)); 
        mLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                               
        throw new Error(errorMsg);
      }
      delBool = del[..];
    }

    var eOut = makeDistArray((...deleteShape(eIn.shape, delBool, axis)), t);

    // Now true means it won't be deleted.
    var notDelBool: [delBool.domain] bool;
    forall i in delBool.domain do notDelBool[i] = !delBool[i];

    // Running sum of what isn't deleted, this increments whenever we hit something that stays
    var sumArray: [delBool.domain] int = + scan notDelBool;

    // This takes indices from eOut and converts them to indices in eIn
    var mapArray: [0..<(eIn.shape[axis] - (+ reduce delBool))] int;
    forall i in sumArray.domain {
      if notDelBool[i] then mapArray[sumArray[i] - 1] = i;
    }

    // Performs the copy into eOut
    forall idx in eOut.domain with (
      var agg = newSrcAggregator(t),
      ref mapArray
    ) {
      var inIdx = if eIn.shape.size == 1 then (idx,) else idx;
      inIdx[axis] = mapArray[inIdx[axis]];
      agg.copy(eOut[idx], eIn[if eIn.shape.size == 1 then inIdx[0] else inIdx]);
    }

    return eOut;
  }

  // This deletes by copying the slices that aren't deleted.
  @arkouda.registerCommand(name="deleteBulkCopy")
  proc deleteAxisBulkCopyMsg (eIn: [?d] ?t, axis: int, del: [?d2] ?t2): [] t throws 
    where ((t2 == int || t2 == bool) &&
           (t == int || t == real || t == bool || t == uint || t == uint(8) || t == bigint) &&
           (d2.rank == 1)) {
    param pn = Reflection.getRoutineName();

    // The best way for this to work is to make sure that we have a bool array of what gets deleted
    // and then convert that to ints.
    // If we get an int array that isn't sorted or has repeats, this will fix that.
    // It's also just convenient to work with an int array here.
    var delBool = makeDistArray(eIn.shape[axis], bool);

    if t2 == int {

      // Error handling
      var invalidIndex: atomic bool = false;
      var badIndex: atomic int = -1;
      
      forall i in del.domain with (
        var agg = newDstAggregator(bool)
      ) {

        // Index out of bounds checking
        if del[i] < eIn.shape[axis] && del[i] >= 0 {
          agg.copy(delBool[del[i]], true);
        } else if del[i] >= -eIn.shape[axis] && del[i] < 0 {
          // numpy allows negative indices in this way
          agg.copy(delBool[del[i] + eIn.shape[axis]], true);
        } else {
          invalidIndex.write(true);
          badIndex.write(del[i]);
        }
        
      }

      if invalidIndex.read() {
        var errorMsg = incompatibleArgumentsError(pn, 
                          "index %i is out of bounds for axis %i with size %i".format(badIndex.read(), axis, eIn.shape[axis])); 
        mLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                               
        throw new Error(errorMsg);
      }

    } else {
      if (eIn.shape[axis] != del.size) { 
        var errorMsg = incompatibleArgumentsError(pn, 
                          "Boolean array argument obj to delete must be one dimensional and match the axis length of %i".format(eIn.shape[axis])); 
        mLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                               
        throw new Error(errorMsg);
      }
      delBool = del[..];
    }
    
    // This will be where we put the unique and sorted indices we plan on deleting
    var deleteInds = makeDistArray(+ reduce delBool, int);

    // This gives the output indices
    var scannedBool: [delBool.domain] int = + scan delBool;

    // This performs the transformation
    forall i in scannedBool.domain with (
      var agg = newDstAggregator(int),
      ref deleteInds
    ) {
      if delBool[i] {
        agg.copy(deleteInds[scannedBool[i] - 1], i);
      }
    }

    var eOut = makeDistArray((...deleteShape(eIn.shape, delBool, axis)), t);
    
    // This will be the starting indices of the slices that we copy
    var startInds = makeDistArray(deleteInds.size + 1, int);
    startInds[1..] = deleteInds[0..<deleteInds.size] + 1;

    // This will be the lengths of the slices
    var diffs: [startInds.domain] int;
    diffs[..<diffs.size - 1] = deleteInds[..];
    diffs[diffs.size - 1] = eIn.shape[axis];

    // Finishes the calculation of the lengths
    forall i in startInds.domain do diffs[i] -= startInds[i];

    // This will be the index of the start of each slice in eOut
    var offsets: [diffs.domain] int;

    // It's just the running sum of the lengths
    offsets[1..] = + scan diffs[..diffs.size-2];

    // These are for domain transformations
    var shape = eIn.shape;
    var translation: (d.rank)*int;

    forall i in startInds.domain with (
        var myShape = shape,
        var myTranslation = translation
    ) {

      // If there's a slice to be copied
      if diffs[i] > 0 {

        // Start with the whole domain
        var dom = eIn.domain;

        // myShape is the shape of the slice to be copied, but negative in the axis we're deleting on
        myShape[axis] = -diffs[i];

        // This creates a domain that's the same size as the slice to be copied, starting from 0 on all axes
        dom = dom.interior(myShape);

        // We're going to translate dom by myTranslation to get the output domain
        myTranslation[axis] = offsets[i];
        var outDom = dom.translate(myTranslation);

        // Translate again to get the input domain
        myTranslation[axis] = startInds[i];
        var inDom = dom.translate(myTranslation);

        // Copy
        eOut[outDom] = eIn[inDom];
      }
    }

    return eOut;
  }

  proc deleteShape(shape: ?N*int, array: [] bool, axis: int): N*int {
    var shapeOut: N*int;
    for i in 0..<N do shapeOut[i] = shape[i];
    shapeOut[axis] -= + reduce array;
    return shapeOut;
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.expand_dims.html#array_api.expand_dims
  // insert a new singleton dimension at the given axis
  @arkouda.instantiateAndRegister(prefix='expandDims')
  proc expandDimsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();

    if array_nd == MaxArrayDims {
      const errMsg = "Cannot expand arrays with rank %i, as this would result an an array with rank %i".format(array_nd, array_nd+1) +
                     ", exceeding the server's configured maximum of %i. ".format(MaxArrayDims) +
                     "Please update the configuration and recompile to support higher-dimensional arrays.";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    const name = msgArgs["name"],
          axis = msgArgs["axis"].getPositiveIntValue(array_nd+1);

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        eOut = createSymEntry((...expandedShape(eIn.tupShape, axis)), array_dtype);

    // mapping between the input and output array indices
    inline proc imap(idx: (array_nd+1)*int, axis: int): array_nd*int {
      var ret: array_nd*int, ii = 0;
      for param io in 0..array_nd {
        if io != axis {
          ret[ii] = idx[io];
          ii += 1;
        }
      }
      return ret;
    }

    // copy the data from the input array to the output array
    forall idx in eOut.a.domain with (var agg = newSrcAggregator(array_dtype)) do
      agg.copy(eOut.a[idx], eIn.a[imap(idx, axis)]);

    return st.insert(eOut);
  }

  private proc expandedShape(shapeIn: ?N*int, axis: int): (N+1)*int {
    var shapeOut: (N+1)*int,
        ii = 0;
    for param io in 0..N {
      if io == axis {
        shapeOut[io] = 1;
      } else {
        shapeOut[io] = shapeIn[ii];
        ii += 1;
      }
    }
    return shapeOut;
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.flip.html#array_api.flip
  @arkouda.instantiateAndRegister(prefix='flip')
  proc flipMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          nAxes = msgArgs["nAxes"].toScalar(int),
          axesRaw = msgArgs["axis"].toScalarArray(int, nAxes);

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        (valid, axes) = validateAxes(axesRaw, array_nd);

    if !valid {
      const errMsg = "Unable to flip array with shape %? along axes %?".format(eIn.tupShape, axesRaw);
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    } else {
      var eOut = createSymEntry((...eIn.tupShape), array_dtype);

      // copy the data from the input array to the output array
      // while flipping along the specified axes
      forall idx in eOut.a.domain with (
        var agg = newSrcAggregator(array_dtype),
        const imap = new indexFlip(eIn.tupShape, axes)
      ) {
        const inIdx = imap(if array_nd == 1 then (idx,) else idx);
        agg.copy(eOut.a[idx], eIn.a[inIdx]);
      }

      return st.insert(eOut);
    }
  }

  record indexFlip {
    param nd;
    const shape: nd*int;
    const d: domain(rank=1, idxType=int, strides=strideKind.one);
    const axes: [d] int;

    proc init(shape: ?nd*int, in axes: [?d] int) {
      this.nd = nd;
      this.shape = shape;
      this.d = d;
      this.axes = axes;
    }

    proc this(idx: nd*int): nd*int {
      var ret = idx;
      for axis in axes do
        ret[axis] = shape[axis] - idx[axis] - 1;
      return ret;
    }
  }

  private proc validateAxes(axes: [?d] int, param nd: int): (bool, [d] int) {
    var ret: [d] int;
    if axes.size > nd then return (false, ret);
    for (i, a) in zip(d, axes) {
      if a >= 0 && a < nd {
        ret[i] = a;
      } else if a < 0 && a >= -nd {
        ret[i] = nd + a;
      } else {
        return (false, ret);
      }
    }
    return (true, ret);
  }

  // alternative to 'flipMsg' to be used when the axis argument is 'None'
  @arkouda.instantiateAndRegister(prefix='flipAll')
  proc flipAllMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"];

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        eOut = createSymEntry((...eIn.tupShape), array_dtype);

    forall idx in eOut.a.domain with (
      var agg = newSrcAggregator(array_dtype),
      const imap = new allIndexFlip(array_nd, eIn.tupShape)
    ) {
      const inIdx = imap(if array_nd == 1 then (idx,) else idx);
      agg.copy(eOut.a[idx], eIn.a[inIdx]);
    }

    return st.insert(eOut);
  }

  record allIndexFlip {
    param nd;
    const shape: nd*int;
    proc this(idx: nd*int): nd*int {
      var ret = idx;
      for param i in 0..<nd do
        ret[i] = shape[i] - idx[i] - 1;
      return ret;
    }
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.permute_dims.html#array_api.permute_dims
  @arkouda.instantiateAndRegister(prefix='permuteDims')
  proc permuteDims(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          axes = msgArgs["axes"].toScalarTuple(int, array_nd);

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        (valid, perm) = validateAxes(axes);

    if !valid {
      const errMsg = "Unable to permute array with shape %? using axes %?".format(eIn.tupShape, axes);
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    } else {
      const outShape = permuteTuple(eIn.tupShape, perm);
      var eOut = createSymEntry((...outShape), array_dtype);

      // copy the data from the input array to the output array while permuting the axes
      forall idx in eIn.a.domain with (var agg = newDstAggregator(array_dtype)) do
        agg.copy(eOut.a[permuteTuple(if array_nd == 1 then (idx,) else idx, perm)], eIn.a[idx]);

      return st.insert(eOut);
    }
  }

  private inline proc permuteTuple(tup: ?N*int, perm: N*int): N*int {
    var ret: N*int;
    for param i in 0..<N do ret[i] = tup[perm[i]];
    return ret;
  }

  // ensure all axis indices are in the range [-N, N-1]
  // convert negative indices to positive indices
  // return false if any axis index is out of range
  private proc validateAxes(axes: ?N*int): (bool, N*int) {
    var ret: N*int;
    for param i in 0..<N {
      const a = axes[i];
      if a >= 0 && a < N {
        ret[i] = a;
      } else if a < 0 && a >= -N {
        ret[i] = N + a;
      } else {
        return (false, ret);
      }
    }
    return (true, ret);
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html#array_api.reshape
  @arkouda.instantiateAndRegister(prefix='reshape')
  proc reshapeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    type array_dtype,
    param array_nd_in: int,
    param array_nd_out: int
  ): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          rawShape = msgArgs["shape"].toScalarTuple(int, array_nd_out);

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd_in),
        (valid, outShape) = validateShape(rawShape, eIn.a.size);

    if !valid {
      const errMsg = "Cannot reshape array of shape %? into shape %?. The total number of elements must match".format(eIn.tupShape, rawShape);
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    } else {
      if array_nd_in == 1 && array_nd_out == 1 {
        return st.insert(createSymEntry(eIn.a));
      } else if array_nd_in == 1 {
        // special case: unflatten a 1D array into a higher-dimensional array
        return st.insert(createSymEntry(unflatten(eIn.a, outShape)));
      } else if array_nd_out == 1 {
        // special case: flatten an array into a 1D array
        return st.insert(createSymEntry(flatten(eIn.a)));
      } else {
        // general case
        var eOut = createSymEntry((...outShape), array_dtype);

        // copy the data from the input array to the output array while reshaping
        forall idx in eIn.a.domain with (
          var agg = newDstAggregator(array_dtype),
          const output = eOut.a.domain,
          const input = new orderer(eIn.tupShape)
        ) {
          const outIdx = output.orderToIndex(input.indexToOrder(if array_nd_in == 1 then (idx,) else idx));
          agg.copy(eOut.a[outIdx], eIn.a[idx]);
        }

        return st.insert(eOut);
      }
    }
  }

  // ensures that the shape either:
  //  * has the same total size as the target size
  //  * has one negative dimension (in this case, that dimension's size
  //    is computed to make the total size match the target size)
  private proc validateShape(shape: ?N*int, targetSize: int): (bool, N*int) {
    var ret: N*int,
        neg = -1,
        size = 1;

    for param i in 0..<N {
      if shape[i] < 0 {
        if neg >=0 {
          // more than one negative dimension
          return (false, ret);
        } else {
          neg = i;
        }
      } else {
        size *= shape[i];
        ret[i] = shape[i];
      }
    }

    if neg >= 0 {
      if size > targetSize || targetSize % size != 0 {
        // cannot compute a valid size for the negative dimension
        return (false, ret);
      } else {
        ret[neg] = targetSize / size;
        return (true, ret);
      }
    } else {
      return (size == targetSize, ret);
    }
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.roll.html#array_api.roll
  @arkouda.instantiateAndRegister(prefix='roll')
  proc rollMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          nShifts = msgArgs["nShifts"].toScalar(int),  // number of elements in 'shift' argument
          nAxes = msgArgs["nAxes"].toScalar(int),      // number of elements in 'axis' argument
          shiftsRaw = msgArgs["shift"].toScalarArray(int, nShifts),
          axesRaw = msgArgs["axis"].toScalarArray(int, nAxes);

    if nShifts != 1 && nShifts != nAxes {
      const errMsg = "Unable to roll array; size of 'shift' must match size of 'axis' or be 1";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    }

    var shifts: [0..<nAxes] int;
    if nShifts == 1
      then shifts = [i in 0..<nAxes] shiftsRaw[0];
      else shifts = shiftsRaw;
    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        (valid, axes) = validateAxes(axesRaw, array_nd);

    if !valid {
      const errMsg = "Unable to roll array with shape %? along axes %?".format(eIn.tupShape, axesRaw);
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    } else {
      var eOut = createSymEntry((...eIn.tupShape), array_dtype);

      // copy the data from the input array to the output array while rolling along the specified axes
      forall idx in eIn.a.domain with (
        var agg = newDstAggregator(array_dtype),
        const imap = new rollIdxMapper(eIn.tupShape, axes, shifts)
      ) {
        agg.copy(eOut.a[imap(if array_nd == 1 then (idx, ) else idx)], eIn.a[idx]);
      }

      return st.insert(eOut);
    }
  }

  record rollIdxMapper {
    param nd;
    const shape: nd*int;
    const nAxes: int;
    const axes: [0..<nAxes] int;
    const shifts: [0..<nAxes] int;

    proc init(shape: ?nd*int, in axes: [?d] int, in shifts: [d] int) {
      this.nd = nd;
      this.shape = shape;
      this.nAxes = d.size;
      this.axes = axes;
      this.shifts = shifts;
    }

    proc this(idx: nd*int): nd*int {
      var ret = idx;
      for i in 0..<nAxes do
        ret[axes[i]] = (idx[axes[i]] + shifts[i] + shape[axes[i]]) % shape[axes[i]];
      return ret;
    }
  }

  // alternative to 'rollMsg' to be used when the axis argument is 'None'
  @arkouda.instantiateAndRegister(prefix='rollFlattened')
  proc rollFlattenedMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          shift = msgArgs["shift"].toScalarArray(int, 1)[0];

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        inFlat = if array_nd == 1 then eIn.a else flatten(eIn.a),
        rolled = unflatten(rollBy(shift, inFlat), eIn.tupShape);

    return st.insert(createSymEntry(rolled));
  }

  private proc rollBy(shift: int, in a: [?d] ?t): [d] t throws {
    var ret = makeDistArray(d, t);
    forall idx in d with (var agg = newDstAggregator(t)) do
      agg.copy(ret[(idx + shift + a.size) % a.size], a[idx]);
    return ret;
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.squeeze.html#array_api.squeeze
  @arkouda.instantiateAndRegister(prefix='squeeze')
  proc squeezeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    type array_dtype,
    param array_nd_in: int,
    param array_nd_out: int
  ): MsgTuple throws {
    if array_nd_out > array_nd_in {
      return MsgTuple.error(
        "Cannot squeeze %iD array into %iD array; output array must have fewer dimensions than input"
        .format(array_nd_in, array_nd_out)
      );
    }

    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          nAxes = msgArgs["nAxes"].toScalar(int),
          axes = msgArgs["axes"].toScalarArray(int, nAxes);

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd_in),
        (valid, shape, mapping) = validateSqueeze(eIn.tupShape, axes, array_nd_out);

    if !valid {
      const errMsg = "Unable to squeeze array with shape %? along axes %? into a %iD array".format(eIn.tupShape, axes, array_nd_out);
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    } else {
      var eOut = createSymEntry((...shape), array_dtype);

      // copy the data from the input array to the output array
      forall idx in eOut.a.domain with (
        var agg = newSrcAggregator(array_dtype),
        const imap = new squeezeIndexMapper(array_nd_in, array_nd_out, mapping)
      ) do
        agg.copy(eOut.a[idx], eIn.a[imap(if array_nd_out==1 then (idx,) else idx)]);

      return st.insert(eOut);
    }
  }

  record squeezeIndexMapper {
    param ndIn: int;
    param ndOut: int;
    const mapping: ndOut*int;

    proc this(idx: ndOut*int): ndIn*int {
      var ret: ndIn*int;
      for param i in 0..<ndOut do ret[mapping[i]] = idx[i];
      return ret;
    }
  }

  private proc validateSqueeze(shape: ?NIn*int, axes: [?d], param NOut: int): (bool, NOut*int, NOut*int) {
    var shapeOut: NOut*int,
        mapping: NOut*int;

    if NOut > NIn then return (false, shapeOut, mapping);
    if d.size >= NIn then return (false, shapeOut, mapping);

    const (valid, axes_) = validateNegativeAxes(axes, NIn);
    if !valid then return (false, shapeOut, mapping);

    var degenAxes: NIn*bool;
    for axis in axes_ {
      if shape[axis] != 1 then return (false, shapeOut, mapping);
      degenAxes[axis] = true;
    }

    if NOut != NIn - d.size then return (false, shapeOut, mapping);

    var i = 0;
    for param ii in 0..<NIn {
      if !degenAxes[ii] {
        mapping[i] = ii;
        shapeOut[i] = shape[ii];
        i += 1;
      }
    }

    return (true, shapeOut, mapping);
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.stack.html#array_api.stack
  @arkouda.instantiateAndRegister(prefix='stack')
  proc stackMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();

    if array_nd == MaxArrayDims {
      const errMsg = "Cannot stack arrays with rank %i, as this would result an an array with rank %i".format(array_nd, array_nd+1) +
                     ", exceeding the server's configured maximum of %i. ".format(MaxArrayDims) +
                     "Please update the configuration and recompile to support higher-dimensional arrays.";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    const nArrays = msgArgs["n"].toScalar(int),
          names = msgArgs["names"].toScalarArray(string, nArrays),
          axis = msgArgs["axis"].getPositiveIntValue(array_nd+1);

    const eIns = for n in names do st[n]: borrowed SymEntry(array_dtype, array_nd),
          shapes = [i in 0..<nArrays] eIns[i].tupShape,
          (valid, shapeOut, mapping) = stackedShape(shapes, axis, array_nd);
    var eOut = createSymEntry((...shapeOut), array_dtype);

    if !valid {
      const errMsg = "All arrays must have the same shape to stack";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    } else {
      // copy the data from the input arrays to the output array
      // TODO: does a nested forall with aggregators use too much memory for agg buffers?
      //       (maybe make outer loop be a 'for' or switch inner/outer loops?)
      forall (arrIdx, arr) in zip(eIns.domain, eIns) {
        forall idx in arr.a.domain with (
          var agg = newDstAggregator(array_dtype),
          const imap = new stackIndexMapper(array_nd+1, axis, arrIdx, mapping)
        ) do
          agg.copy(eOut.a[imap(if array_nd==1 then (idx,) else idx)], arr.a[idx]);
      }

      return st.insert(eOut);
    }
  }

  record stackIndexMapper {
    param ndOut: int;
    const axis: int;
    const arrIdx: int;
    const mapping: ndOut*int;

    proc this(idx: (ndOut-1)*int): ndOut*int {
      var ret: ndOut*int;
      for param i in 0..<ndOut do ret[i] = idx[mapping[i]];
      ret[axis] = arrIdx;
      return ret;
    }
  }

  private proc stackedShape(shapes: [?d] ?t, axis: int, param N: int): (bool, (N+1)*int, (N+1)*int)
    where isHomogeneousTuple(t)
  {
    var shapeOut: (N+1)*int,
        mapping: (N+1)*int,
        ii = 0;

    // confirm all arrays have the same shape in the non-axis dimensions
    const shape: N*int = shapes[0];
    for i in 1..d.last {
      for param j in 0..<N {
        if j != axis && shapes[i][j] != shape[j] {
          return (false, shapeOut, mapping);
        }
      }
    }

    // compute the shape of the output array
    for param i in 0..N {
      if i == axis {
        shapeOut[i] = d.size;
      } else {
        shapeOut[i] = shape[ii];
        mapping[i] = ii;
        ii += 1;
      }
    }
    return (true, shapeOut, mapping);
  }


  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.tile.html#array_api.tile
  // assumes that 'reps' is the same length as the array's shape
  // this is achieved on the client side by either:
  //  * reshaping the array to add singleton dimensions (if reps is longer than the array's shape)
  //  * prepending 1's to reps (if reps is shorter than the array's shape)
  @arkouda.instantiateAndRegister(prefix='tile')
  proc tileMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          reps = msgArgs["reps"].toScalarTuple(int, array_nd);

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        eOut = createSymEntry((...tiledShape(eIn.tupShape, reps)), array_dtype);

    // copy the data from the input array to the output array while tiling
    forall idx in eOut.a.domain with (
      var agg = newSrcAggregator(array_dtype),
      const imap = new tileIndexMapper(array_nd, eIn.tupShape)
    ) {
      const inIdx = imap(if array_nd == 1 then (idx,) else idx);
      agg.copy(eOut.a[idx], eIn.a[inIdx]);
    }

    return st.insert(eOut);
  }

  record tileIndexMapper {
    param nd: int;
    const shapeIn: nd*int;

    proc this(idx: nd*int): nd*int {
      var ret: nd*int;
      for param i in 0..<nd do ret[i] = idx[i] % shapeIn[i];
      return ret;
    }
  }


  proc tiledShape(shape: ?N*int, reps: N*int): N*int {
    var shapeOut: N*int;
    for i in 0..<N do shapeOut[i] = shape[i] * reps[i];
    return shapeOut;
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.repeat.html#array_api.repeat
  @arkouda.registerCommand (name="repeat")
  proc repeat (eIn: [?d] ?t, axis: int, ref reps: [?d2] ?t2): [] t throws 
    where ((t == int || t == real || t == bool || t == uint || t == uint(8) || t == bigint) &&
           (t2 == int || t2 == uint) &&
           (d2.rank == 1)) {
    param pn = Reflection.getRoutineName();

    const (valid, axis_) = validateNegativeAxes (axis,d.rank) ;
    if !valid then throw new Error ("Invalid axis %i in repeat".format(axis));

    var eOut = makeDistArray((...repeatShape(eIn.shape, reps, axis_)), t);

    // In this case, every element of eIn gets repeated the same number of times along the axis.
    if reps.size == 1 {
      forall idx in eOut.domain with (
        var agg = newSrcAggregator(t)
      ) {
        var idx2 = if eIn.shape.size == 1 then (idx,) else idx;

        // The integer division means that the index gets incremented in the axis every reps[0] times
        idx2[axis_] = idx2[axis_] / (reps[0]: int);

        agg.copy(eOut[idx], eIn[if eIn.shape.size == 1 then idx2[0] else idx2]);
      }
    } else if d.rank == 1 {
      // We can shortcut things a little bit if eIn is a one dimensional array
      const outSize = eOut.size;

      // Distribute the domain the same way as eIn
      var outStarts: [eIn.domain] int;

      // The running sum of repeats gives us the starting points of where we need to start repeating each element
      outStarts[1..] = (+ scan reps[..reps.size - 2]): int;
      forall (inIdx, outStart, numRepeats) in zip(eIn.domain, outStarts, reps) with (
        var agg = newDstAggregator(t)
      ) {
        for destIdx in outStart..#numRepeats {
          agg.copy(eOut[destIdx], eIn[inIdx]);
        }
      }
    } else {
      // Fundamentally this works the same way as above but we can't do zipped iteration
      // This is because eIn may be much larger.
      const outSize = eOut.shape[axis_];

      // Can't distribute the domain because we don't know how eIn is distributed
      var outStarts: [0..<eIn.shape[axis_]] int;

      outStarts[1..] = (+ scan reps[..reps.size - 2]): int;
      forall inIdx in eIn.domain with (
        var agg = newDstAggregator(t)
      ) {
        const outStart = outStarts[inIdx[axis_]];
        const numRepeats = reps[inIdx[axis_]];
        var destIdx = inIdx;
        for destIdxComponent in outStart..#numRepeats {
          destIdx[axis_] = destIdxComponent;
          agg.copy(eOut[destIdx], eIn[inIdx]);
        }
      }

    }

    return eOut;
  }

  proc repeatShape(shape: ?N*int, ref reps: [] ?t, axis: int): N*int 
    where (t == int || t == uint) {
    var shapeOut: N*int;
    for i in 0..<N do shapeOut[i] = shape[i];
    if reps.size == 1 {
      shapeOut[axis] = shapeOut[axis] * (reps[0]: int);
    } else {
      shapeOut[axis] = + reduce (reps: int);
    }
    return shapeOut;
  }

  //At some point when issue #4162 clears up we can use this instead in the where clause.
  proc repeatInputElemTypeCheck(type t) param: bool {
    return (t == int || t == real || t == bool || t == uint || t == uint(8) || t == bigint);
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.unstack.html
  // unstack an array into multiple arrays along a specified axis
  @arkouda.instantiateAndRegister(prefix='unstack')
  proc unstackMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();

    if array_nd == 1 {
      const errMsg = "Cannot unstack a 1D array";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    }

    const name = msgArgs["name"],
          axis = msgArgs["axis"].getPositiveIntValue(array_nd),
          numReturnArrays = msgArgs["numReturnArrays"].toScalar(int);

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        shapeOut = unstackedShape(eIn.tupShape, axis);

    if eIn.tupShape[axis] != numReturnArrays {
      const errMsg = "Cannot unstack array with shape %? along axis %? into %? arrays".format(eIn.tupShape, axis, numReturnArrays);
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    var eOuts = for i in 0..#numReturnArrays do createSymEntry((...shapeOut), array_dtype);

    // copy the data from the input array to the output arrays while unstacking
    for arrIdx in 0..<numReturnArrays {
      forall idx in eOuts[arrIdx].a.domain with (
        var agg = newSrcAggregator(array_dtype),
        const imap = new unstackIdxMapper(array_nd, arrIdx, axis)
      ) {
        const inIdx = imap(if array_nd == 2 then (idx,) else idx);
        agg.copy(eOuts[arrIdx].a[idx], eIn.a[inIdx]);
      }
    }

    // TODO: does the 'in' intent on 'insert' copy the symbols here?
    // (probably not since they are each 'shared' (i.e., only the managing record is copied?))
    const responses = for e in eOuts do st.insert(e);
    return MsgTuple.fromResponses(responses);
  }

  record unstackIdxMapper {
    param ndIn: int;
    const arrIdx: int;
    const axis: int;

    proc this(idx: (ndIn-1)*int): ndIn*int {
      var ret: ndIn*int;
      var i = 0;
      for param ii in 0..<ndIn {
        if ii == axis {
          ret[ii] = arrIdx;
        } else {
          ret[ii] = idx[i];
          i += 1;
        }
      }
      return ret;
    }
  }

  // TODO: should this reduce the array rank by 1, or introduce a singleton dimension for axis?
  // (the array-api docs are unclear on this point)
  proc unstackedShape(shape: ?N*int, axis: int): (N-1)*int
    where N > 1
  {
    var shapeOut: (N-1)*int,
        i = 0;
    for ii in 0..<N {
      if ii == axis {
        continue;
      } else {
        shapeOut[i] = shape[ii];
        i += 1;
      }
    }
    return shapeOut;
  }


  // see: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat
  // flattens the input array and repeats each element 'repeats' times
  // if 'repeats' is an array, it must have the same number of elements as the input array
  @arkouda.instantiateAndRegister(prefix='repeatFlat')
  proc repeatFlatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs["name"],
          repeats = msgArgs.getValueOf("repeats");

    var eIn = st[name]: borrowed SymEntry(array_dtype, array_nd),
        eRepeats = st[repeats]: borrowed SymEntry(int, 1),
        aFlat = if array_nd == 1 then eIn.a else flatten(eIn.a);

    if eRepeats.a.size == 1 {
      const rep = eRepeats.a[0],
            eOut = createSymEntry(aFlat.size * rep, array_dtype);

      // simple case: repeat each element of the input array 'rep' times
      forall i in aFlat.domain do eOut.a[i*rep..#rep] = aFlat[i];

      return st.insert(eOut);
    } else if eRepeats.a.size == aFlat.size {
      // repeat each element of the input array by the corresponding element of 'repeats'

      // // serial algorithm:
      // var start = 0;
      // for idx in aFlat.domain {
      //   eOut.a[start..#eRepeats.a[idx]] = aFlat[idx];
      //   start += eRepeats.a[idx];
      // }

      // compute the number of repeated elements in the output array owned by each task
      const nTasksPerLoc = here.maxTaskPar;
      var nRepsPerTask: [0..<numLocales] [0..<nTasksPerLoc] int;
      coforall loc in Locales with (ref nRepsPerTask) do on loc {
        const lsd = aFlat.localSubdomain(),
              indicesPerTask = lsd.size / nTasksPerLoc;
        coforall tid in 0..<nTasksPerLoc with (ref nRepsPerTask) {
          const startIdx = tid * indicesPerTask + lsd.low,
                stopIdx = if tid == nTasksPerLoc - 1 then lsd.high else indicesPerTask + startIdx - 1;

          var sum = 0;
          for i in startIdx..stopIdx do
            sum += eRepeats.a[i];
          nRepsPerTask[loc.id][tid] = sum;
        }
      }

      // compute the output array's size, and where in the output array each locale should start
      // depositing its repeated elements
      const nRepsPerLoc = [nt in nRepsPerTask] + reduce nt,
            locStarts = (+ scan nRepsPerLoc) - nRepsPerLoc,
            nTotal = + reduce nRepsPerLoc;
      var eOut = createSymEntry(nTotal, array_dtype);

      // copy the repeated elements into the output array
      coforall loc in Locales with (const ref nRepsPerTask, const ref locStarts) do on loc {
        const lsd = aFlat.localSubdomain(),
              indicesPerTask = lsd.size / nTasksPerLoc;

        // compute where in the output array each of this locale's tasks should start depositing
        // its repeated elements
        const taskStarts = ((+ scan nRepsPerTask[loc.id]) - nRepsPerTask[loc.id]) + locStarts[loc.id];
        coforall tid in 0..<nTasksPerLoc {
          const startIdx = tid * indicesPerTask + lsd.low,
                stopIdx = if tid == nTasksPerLoc - 1 then lsd.high else indicesPerTask + startIdx - 1;

          // copy this task's repeated elements into the output array
          var outStart = taskStarts[tid];
          for i in startIdx..stopIdx {
            eOut.a[outStart..#eRepeats.a[i]] = aFlat[i];
            outStart += eRepeats.a[i];
          }
        }
      }

      return st.insert(eOut);
    } else {
      const errMsg = "Unable to repeat array with shape %? using repeats %?. ".format(eIn.tupShape, eRepeats.tupShape) +
                      "Repeats must be a scalar or have the same number of elements as the input array";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return MsgTuple.error(errMsg);
    }
  }

  proc parSearch(a: [?d] ?t, x: t, sorted: bool): bool {
    use Search;
    var found = false;

    const nTasks=here.maxTaskPar;
    coforall loc in Locales with (|| reduce found) do on loc {
      const locDom = a.localSubdomain(),
            nPerTask = locDom.size / nTasks;

      coforall tid in 0..<nTasks with (|| reduce found) {
        const (lf, _) = search(
          a, x, sorted=sorted,
          lo = locDom.low + nPerTask*tid,
          hi = if tid == nTasks-1
            then locDom.high
            else locDom.low + nPerTask*(tid+1)
        );

        found ||= lf;
      }
    }
    return found;
  }
}
