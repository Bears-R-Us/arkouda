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

  use Reflection;
  use BigInteger;

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
  @arkouda.registerNDPermInc
  proc broadcastToMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    param ndIn: int, // rank of the array to be broadcast
    param ndOut: int // rank of the result array
  ): MsgTuple throws {
    const name = msgArgs.getValueOf("name"),
          shapeOut = msgArgs.get("shape").getTuple(ndOut),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doBroadcast(type t): MsgTuple throws {
      var eIn = toSymEntry(gEnt, t, ndIn),
          eOut = st.addEntry(rname, (...shapeOut), t);

      if ndIn == ndOut && eIn.tupShape == shapeOut {
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
          // define a mapping from the output array's indices to the input array's indices
          inline proc imap(idx: int ...ndOut): ndIn*int {
            var ret: ndIn*int;
            for param i in 0..<ndIn do
              ret[i] = if bcDims[i] then 0 else idx[i];
            return ret;
          }

          forall idx in eOut.a.domain with (var agg = newSrcAggregator(t)) do
            if ndOut == 1
              then eOut.a[idx] = eIn.a[imap(idx)];
              else eOut.a[idx] = eIn.a[imap((...idx))];
        }
      }

      const repMsg = "created " + st.attrib(rname);
      mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return doBroadcast(int);
      when DType.UInt8 do return doBroadcast(uint(8));
      when DType.UInt64 do return doBroadcast(uint);
      when DType.Float64 do return doBroadcast(real);
      when DType.Bool do return doBroadcast(bool);
      otherwise {
        var errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
        mLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  proc checkValidBroadcast(from: ?Nf*int, to: ?Nt*int): (bool, Nf*bool) {
    var dimsToBroadcast: Nf*bool;
    if Nf > Nt then return (false, dimsToBroadcast);

    for param iIn in 0..<Nf {
      param iOut = Nt - Nf + iIn;
      if from[iIn] == 1 && to[iOut] != 1 {
        dimsToBroadcast[iIn] = true;
      } else if from[iIn] != to[iOut] {
        return (false, dimsToBroadcast);
      }
    }

    return (true, dimsToBroadcast);
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html#array_api.concat
  @arkouda.registerND
  proc concatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const nArrays = msgArgs.get("n").getIntValue(),
          names = msgArgs.get("names").getList(nArrays),
          axis = msgArgs.get("axis").getPositiveIntValue(nd),
          rname = st.nextName();

    var gEnts: [0..<nArrays] borrowed GenSymEntry = getGenericEntries(names, st);

    // confirm that all arrays have the same dtype
    // (type promotion needs to be completed before calling 'concat')
    const dt = gEnts[0].dtype;
    for i in 1..<nArrays do if gEnts[i].dtype != dt {
      const errMsg = "All arrays must have the same dtype to concatenate";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    proc doConcat(type t): MsgTuple throws {
      const eIns = [i in 0..<nArrays] toSymEntry(gEnts[i], t, nd),
            shapes = [i in 0..<nArrays] eIns[i].tupShape,
            (valid, shapeOut, startOffsets) = concatenatedShape(nd, shapes, axis);

      if !valid {
        const errMsg = "Arrays must have compatible shapes to concatenate: " +
          "attempt to concatenate arrays of shapes %? along axis %?".doFormat(shapes, axis);
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        var eOut = st.addEntry(rname, (...shapeOut), t);

        // mapping between the input and output array indices
        inline proc imap(arrIdx: int, idx: nd*int): nd*int
          where nd > 1
        {
          var ret = idx;
          ret[axis] += startOffsets[arrIdx];
          return ret;
        }

        inline proc imap(arrIdx: int, idx: int): int
          where nd == 1
            do return idx + startOffsets[arrIdx];

        // copy the data from the input arrays to the output array
        for (arrIdx, arr) in zip(eIns.domain, eIns) do
          forall idx in arr.a.domain with (var agg = newDstAggregator(t)) do
            agg.copy(eOut.a[imap(arrIdx, idx)], arr.a[idx]);

        const repMsg = "created " + st.attrib(rname);
        mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
    }

    select dt {
      when DType.Int64 do return doConcat(int);
      when DType.UInt64 do return doConcat(uint);
      when DType.Float64 do return doConcat(real);
      when DType.Bool do return doConcat(bool);
      when DType.BigInt do return doConcat(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(dt));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
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
  @arkouda.registerND
  proc concatFlatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const nArrays = msgArgs.get("n").getIntValue(),
          names = msgArgs.get("names").getList(nArrays),
          rname = st.nextName();

    var gEnts: [0..<nArrays] borrowed GenSymEntry = getGenericEntries(names, st);

    // confirm that all arrays have the same dtype
    // (type promotion needs to be completed before calling 'concat')
    const dt = gEnts[0].dtype;
    for i in 1..<nArrays do if gEnts[i].dtype != dt {
      const errMsg = "All arrays must have the same dtype to concatenate";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    proc doFlatConcat(type t): MsgTuple throws {
      const eIns = [i in 0..<nArrays] toSymEntry(gEnts[i], t, nd),
            sizes = [i in 0..<nArrays] eIns[i].a.size,
            starts = (+ scan sizes) - sizes;

      // create a 1D output array
      var eOut = st.addEntry(rname, + reduce sizes, t);

      // copy the data from the input arrays to the output array
      for arrIdx in 0..<nArrays {
        const a = flatten(eIns[arrIdx].a);
        forall idx in a.domain with (var agg = newSrcAggregator(t)) do
          agg.copy(eOut.a[idx + starts[arrIdx]], a[idx]);
      }

      const repMsg = "created " + st.attrib(rname);
      mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select dt {
      when DType.Int64 do return doFlatConcat(int);
      when DType.UInt64 do return doFlatConcat(uint);
      when DType.Float64 do return doFlatConcat(real);
      when DType.Bool do return doFlatConcat(bool);
      when DType.BigInt do return doFlatConcat(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(dt));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  private proc flatten(const ref a: [?d] ?t): [] t throws {
    var flat = makeDistArray({0..<d.size}, t);
    forall idx in flat.domain with (var agg = newSrcAggregator(t)) do
      agg.copy(flat[idx], a[d.orderToIndex(idx)]);
    return flat;
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.expand_dims.html#array_api.expand_dims
  @arkouda.registerND
  proc expandDimsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    // TODO: add a check and error handling if nd+1 exceeds the maximum supported array rank
    const name = msgArgs.getValueOf("name"),
          axis = msgArgs.get("axis").getPositiveIntValue(nd+1),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc expandDims(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            shapeOut = expandedShape(eIn.tupShape, axis);

      var eOut = st.addEntry(rname, (...shapeOut), t);

      // mapping between the input and output array indices
      inline proc imap(idx: (nd+1)*int): nd*int {
        var ret: nd*int, ii = 0;
        for param io in 0..nd {
          if io != axis {
            ret[ii] = idx[io];
            ii += 1;
          }
        }
        return ret;
      }

      // copy the data from the input array to the output array
      forall idx in eOut.a.domain with (var agg = newSrcAggregator(t)) do
        agg.copy(eOut.a[idx], eIn.a[imap(idx)]);

      const repMsg = "created " + st.attrib(rname);
      mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return expandDims(int);
      when DType.UInt64 do return expandDims(uint);
      when DType.Float64 do return expandDims(real);
      when DType.Bool do return expandDims(bool);
      when DType.BigInt do return expandDims(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
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
  @arkouda.registerND
  proc flipMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          nAxes = msgArgs.get("nAxes").getIntValue(),
          axesRaw = msgArgs.get("axis").getListAs(int, nAxes),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doFlip(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            (valid, axes) = validateAxes(axesRaw, nd);
      var eOut = st.addEntry(rname, (...eIn.tupShape), t);

      if !valid {
        const errMsg = "Unable to flip array with shape %? along axes %?".doFormat(eIn.tupShape, axesRaw);
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        // mapping between the input and output array indices with the specified axes flipped
        inline proc imap(idx: nd*int): nd*int {
          var ret = idx;
          for axis in axes do
            ret[axis] = eIn.tupShape[axis] - idx[axis] - 1;
          return ret;
        }

        // copy the data from the input array to the output array
        // while flipping along the specified axis
        forall idx in eOut.a.domain with (var agg = newSrcAggregator(t)) do
          agg.copy(eOut.a[idx], eIn.a[imap(if nd == 1 then (idx,) else idx)]);

        const repMsg = "created " + st.attrib(rname);
        mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
    }

    select gEnt.dtype {
      when DType.Int64 do return doFlip(int);
      when DType.UInt64 do return doFlip(uint);
      when DType.Float64 do return doFlip(real);
      when DType.Bool do return doFlip(bool);
      when DType.BigInt do return doFlip(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
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
  @arkouda.registerND
  proc flipAllMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doFlip(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd);
      var eOut = st.addEntry(rname, (...eIn.tupShape), t);

      // mapping between the input and output array indices with all axes flipped
      inline proc imap(idx: nd*int): nd*int {
        var ret: nd*int;
        for param i in 0..<nd do
          ret[i] = eIn.tupShape[i] - idx[i] - 1;
        return ret;
      }

      // copy the data from the input array to the output array
      // while flipping along each axis
      forall idx in eOut.a.domain with (var agg = newSrcAggregator(t)) do
        agg.copy(eOut.a[idx], eIn.a[imap(if nd == 1 then (idx, ) else idx)]);

      const repMsg = "created " + st.attrib(rname);
      mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return doFlip(int);
      when DType.UInt64 do return doFlip(uint);
      when DType.Float64 do return doFlip(real);
      when DType.Bool do return doFlip(bool);
      when DType.BigInt do return doFlip(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.permute_dims.html#array_api.permute_dims
  @arkouda.registerND
  proc permuteDims(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          axes = msgArgs.get("axes").getTuple(nd),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doPermutation(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            (valid, perm) = validateAxes(axes);

      if !valid {
        const errMsg = "Unable to permute array with shape %? using axes %?".doFormat(eIn.tupShape, axes);
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        const outShape = permuteTuple(eIn.tupShape, perm);
        var eOut = st.addEntry(rname, (...outShape), t);

        // copy the data from the input array to the output array while permuting the axes
        forall idx in eOut.a.domain with (var agg = newSrcAggregator(t)) do
          agg.copy(eOut.a[idx], eIn.a[permuteTuple(if nd == 1 then (idx,) else idx, perm)]);

        const repMsg = "created " + st.attrib(rname);
        mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
    }

    select gEnt.dtype {
      when DType.Int64 do return doPermutation(int);
      when DType.UInt64 do return doPermutation(uint);
      when DType.Float64 do return doPermutation(real);
      when DType.Bool do return doPermutation(bool);
      when DType.BigInt do return doPermutation(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  private inline proc permuteTuple(tup: ?N*int, perm: N*int): N*int {
    var ret: N*int;
    for param i in 0..<N do ret[i] = tup[perm[i]];
    return ret;
  }

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
  @arkouda.registerNDPermAll
  proc reshapeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    param ndIn: int,
    param ndOut: int
  ): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          rawShape = msgArgs.get("shape").getTuple(ndOut),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doReshape(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, ndIn),
            (valid, shape) = validateShape(rawShape, eIn.a.size);

      if !valid {
        const errMsg = "Cannot reshape array of shape %? into shape %?. The total number of elements must match".doFormat(eIn.tupShape, rawShape);
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        var eOut = st.addEntry(rname, (...shape), t);

        const sizes = [i in 0..<ndOut by -1] eOut.tupShape[i],
              accumSizes = * scan sizes / sizes;

        // index -> order for the output array's indices
        // e.g., order = k + (nz * j) + (nz * ny * i)
        inline proc indexToOrder(idx: ndOut*int): int {
          var order = 0;
          for param i in 0..<ndOut do order += idx[i] * accumSizes[i];
          return order;
        }

        // copy the data from the input array to the output array while reshaping
        forall idx in eOut.a.domain with (var agg = newSrcAggregator(t)) {
          const inIdx = eIn.a.domain.orderToIndex(indexToOrder(if ndOut==1 then (idx,) else idx));
          agg.copy(eOut.a[idx], eIn.a[inIdx]);
        }

        const repMsg = "created " + st.attrib(rname);
        mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
    }

    select gEnt.dtype {
      when DType.Int64 do return doReshape(int);
      when DType.UInt64 do return doReshape(uint);
      when DType.Float64 do return doReshape(real);
      when DType.Bool do return doReshape(bool);
      when DType.BigInt do return doReshape(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
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
  @arkouda.registerND
  proc rollMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          nShifts = msgArgs.get("nShifts").getIntValue(), // number of elements in 'shift' argument
          nAxes = msgArgs.get("nAxes").getIntValue(),     // number of elements in 'axis' argument
          shiftsRaw = msgArgs.get("shift").getListAs(int, nShifts),
          axesRaw = msgArgs.get("axis").getListAs(int, nAxes),
          rname = st.nextName();

    if nShifts != 1 && nShifts != nAxes {
      const errMsg = "Unable to roll array; size of 'shift' must match size of 'axis' or be 1";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    var shifts: [0..<nAxes] int;
      if nShifts == 1
        then shifts = [i in 0..<nAxes] shiftsRaw[0];
        else shifts = shiftsRaw;
    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doRoll(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            (valid, axes) = validateAxes(axesRaw, nd);

      if !valid {
        const errMsg = "Unable to roll array with shape %? along axes %?".doFormat(eIn.tupShape, axesRaw);
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        var eOut = st.addEntry(rname, (...eIn.tupShape), t);

        // mapping between a starting index and the rolled index
        inline proc rollIdx(idx: nd*int): nd*int {
          var ret = idx;
          for i in 0..<nAxes do
            ret[axes[i]] = (idx[axes[i]] + shifts[i] + eIn.tupShape[axes[i]]) % eIn.tupShape[axes[i]];
          return ret;
        }

        // copy the data from the input array to the output array while rolling along the specified axes
        forall idx in eIn.a.domain with (var agg = newDstAggregator(t)) do
          agg.copy(eOut.a[rollIdx(if nd == 1 then (idx, ) else idx)], eIn.a[idx]);

        const repMsg = "created " + st.attrib(rname);
        mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
    }

    select gEnt.dtype {
      when DType.Int64 do return doRoll(int);
      when DType.UInt64 do return doRoll(uint);
      when DType.Float64 do return doRoll(real);
      when DType.Bool do return doRoll(bool);
      when DType.BigInt do return doRoll(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  // alternative to 'rollMsg' to be used when the axis argument is 'None'
  @arkouda.registerND
  proc rollFlattenedMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          shift = msgArgs.get("shift").getListAs(int, 1)[0],
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doRoll(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            eOut = st.addEntry(rname, (...eIn.tupShape), t),
            inFlatRolled = rollBy(shift, flatten(eIn.a));

      // copy the flattened/rolled array into the output array while unflattening
      forall idx in inFlatRolled.domain with (var agg = newDstAggregator(t)) do
        agg.copy(eOut.a[eOut.a.domain.orderToIndex(idx)], inFlatRolled[idx]);

      const repMsg = "created " + st.attrib(rname);
      mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return doRoll(int);
      when DType.UInt64 do return doRoll(uint);
      when DType.Float64 do return doRoll(real);
      when DType.Bool do return doRoll(bool);
      when DType.BigInt do return doRoll(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  private proc rollBy(shift: int, in a: [?d] ?t): [d] t throws {
    var ret = makeDistArray(d, t);
    forall idx in d with (var agg = newDstAggregator(t)) do
      agg.copy(ret[(idx + shift + a.size) % a.size], a[idx]);
    return ret;
  }

  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.squeeze.html#array_api.squeeze
  @arkouda.registerNDPermDec
  proc squeezeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    param ndIn: int,
    param ndOut: int
  ): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          nAxes = msgArgs.get("nAxes").getIntValue(),
          axes = msgArgs.get("axes").getListAs(int, nAxes),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doSqueeze(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, ndIn),
            (valid, shape, mapping) = validateSqueeze(eIn.tupShape, axes, ndOut);

      if !valid {
        const errMsg = "Unable to squeeze array with shape %? along axes %?".doFormat(eIn.tupShape, axes);
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        var eOut = st.addEntry(rname, (...shape), t);

        // mapping between the input and output array indices
        inline proc imap(idx: ndOut*int): ndIn*int {
          var ret: ndIn*int;
          for param ii in 0..<ndIn do ret[ii] = eIn.a.domain.dim(ii).low;
          for param io in 0..<ndOut do ret[mapping[io]] = idx[io];
          return ret;
        }

        // copy the data from the input array to the output array
        forall idx in eOut.a.domain with (var agg = newSrcAggregator(t)) do
          agg.copy(eOut.a[idx], eIn.a[imap(if ndOut==1 then (idx,) else idx)]);

        const repMsg = "created " + st.attrib(rname);
        mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
    }

    select gEnt.dtype {
      when DType.Int64 do return doSqueeze(int);
      when DType.UInt64 do return doSqueeze(uint);
      when DType.Float64 do return doSqueeze(real);
      when DType.Bool do return doSqueeze(bool);
      when DType.BigInt do return doSqueeze(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  private proc validateSqueeze(shape: ?NIn*int, axes: [?d], param NOut: int): (bool, NOut*int, NOut*int) {
    var shapeOut: NOut*int,
        mapping: NOut*int;

    if NOut > NIn then return (false, shapeOut, mapping);
    if d.size >= NIn then return (false, shapeOut, mapping);

    var degenAxes: NIn*bool;
    for axis in axes {
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
  @arkouda.registerND
  proc stackMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const nArrays = msgArgs.get("n").getIntValue(),
          names = msgArgs.get("names").getList(nArrays),
          axis = msgArgs.get("axis").getPositiveIntValue(nd),
          rname = st.nextName();

    var gEnts: [0..<nArrays] borrowed GenSymEntry = getGenericEntries(names, st);

    // confirm that all arrays have the same dtype and shape
    // (type promotion needs to be completed before calling 'stack')
    const dt = gEnts[0].dtype,
          sh = gEnts[0].shape;
    for i in 1..#nArrays do if gEnts[i].dtype != dt || gEnts[i].shape != sh {
      const errMsg = "All arrays must have the same dtype and shape to stack";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    proc doStack(type t): MsgTuple throws {
      const eIns = [i in 0..#nArrays] toSymEntry(gEnts[i], t, nd),
            (shapeOut, mapping) = stackedShape(eIns[0].tupShape, axis, nArrays);
      var eOut = st.addEntry(rname, (...shapeOut), t);

      // mapping between the input and output array indices
      inline proc imap(arrIdx: int, idx: nd*int): (nd+1)*int {
        var ret: (nd+1)*int;
        for i in 0..nd {
          if i == axis
            then ret[i] = arrIdx;
            else ret[i] = idx[mapping[i]];
        }
        return ret;
      }

      // copy the data from the input arrays to the output array
      forall (arrIdx, arr) in zip(eIns.domain, eIns) do
        forall idx in arr.a.domain with (var agg = newDstAggregator(t)) do
          agg.copy(eOut.a[imap(arrIdx, if nd==1 then (idx,) else idx)], arr.a[idx]);

      const repMsg = "created " + st.attrib(rname);
      mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select dt {
      when DType.Int64 do return doStack(int);
      when DType.UInt64 do return doStack(uint);
      when DType.Float64 do return doStack(real);
      when DType.Bool do return doStack(bool);
      when DType.BigInt do return doStack(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(dt));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  private proc stackedShape(shape: ?N*int, axis: int, nArrays: int): ((N+1)*int, (N+1)*int) {
    var shapeOut: (N+1)*int,
        mapping: (N+1)*int,
        ii = 0;

    for param i in 0..N {
      if i == axis {
        shapeOut[i] = nArrays;
      } else {
        shapeOut[i] = shape[ii];
        mapping[i] = ii;
        ii += 1;
      }
    }
    return (shapeOut, mapping);
  }

  proc getGenericEntries(names: [?d] string, st: borrowed SymTab): [] borrowed GenSymEntry throws {
    var gEnts: [d] borrowed GenSymEntry?;
    for (i, name) in zip(d, names) do gEnts[i] = getGenericTypedArrayEntry(name, st);
    const ret = [i in d] gEnts[i]!;
    return ret;
  }
}
