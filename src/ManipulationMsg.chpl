module ManipulationMsg {
  use CommandMap;
  use Message;
  use MultiTypeSymbolTable;
  use ServerConfig;
  use Logging;
  use CommAggregation;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const mLogger = new Logger(logLevel, logChannel);

  @arkouda.registerND
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.expand_dims.html#array_api.expand_dims
  proc expandDimsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    // TODO: add a check and error handling if nd+1 exceeds the maximum supported array rank
    const name = msgArgs.getValueOf("name"),
          axis = msgArgs.get("axis").getPositiveIntValue(nd),
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
      forall idx in eOut.a.domain with (var agg = newDstAggregator(t)) do
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

  @arkouda.registerND
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html#array_api.concat
  proc concatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const nArrays = msgArgs.get("n").getIntValue(),
          names = msgArgs.get("names").getList(nArrays),
          axis = msgArgs.get("axis").getPositiveIntValue(nd),
          rname = st.nextName();

    var gEnts: [] borrowed GenSymEntry = [name in names] getGenericTypedArrayEntry(name, st);

    // confirm that all arrays have the same dtype
    // (type promotion needs to be completed before calling 'concat')
    const dt = gEnts[0].dtype;
    for i in 1..#nArrays do if gEnts[i].dtype != dtype {
      const errMsg = "All arrays must have the same dtype to concatenate";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    proc doConcat(type t): MsgTuple throws {
      const eIns = [i in 0..#nArrays] toSymEntry(gEnts[i], t, nd),
            (validConcat, shapeOut, startOffsets) =
              concatenatedShape(nd, [e in eIns] e.tupShape, axis);

      if !validConcat {
        const errMsg = "Arrays must have compatible shapes to concatenate";
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        var eOut = st.addEntry(rname, (...shapeOut), t);

        // mapping between the input and output array indices
        inline proc imap(arrIdx: int, idx: nd*int): nd*int {
          var ret = idx;
          ret[axis] += startOffsets[arrIdx];
          return ret;
        }

        // copy the data from the input arrays to the output array
        forall (arrIdx, arr) in zip(eIns.domain, eIns) do
          forall idx in arr.a.domain with (var agg = newSrcAggregator(t)) do
            agg.copy(eOut.a[imap(arrIdx, idx)], arr.a[idx]);

        const repMsg = "created " + st.attrib(rname);
        mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
    }

    select gEnt.dtype {
      when DType.Int64 do return doConcat(int);
      when DType.UInt64 do return doConcat(uint);
      when DType.Float64 do return doConcat(real);
      when DType.Bool do return doConcat(bool);
      when DType.BigInt do return doConcat(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  inline proc concatenatedShape(param n: int, shapes: [?d] n*int, axis: int): (bool, n*int, [d] int) {
    var shapeOut = shapes[0],
        validConcat = true,
        startOffsets: [d] int;

    label shapes for s in 1..<shapes.size {
      for param i in 0..<n {
        if i == axis {
          shapeOut[i] += shapes[s][i];
          startOffsets[s] = shapeOut[i];
        } else {
          if shapes[s][i] != shapeOut[i] {
            validConcat = false;
            break shapes;
          }
        }
      }
    }
    return (validConcat, shapeOut, startOffsets);
  }

  @arkouda.registerND
  // alternative to 'concatMsg' to be used when the axis argument is 'None'
  proc flatConcatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const nArrays = msgArgs.get("n").getIntValue(),
          names = msgArgs.get("names").getList(nArrays),
          rname = st.nextName();

    var gEnts: [] borrowed GenSymEntry = [name in names] getGenericTypedArrayEntry(name, st);

    // confirm that all arrays have the same dtype
    // (type promotion needs to be completed before calling 'concat')
    const dt = gEnts[0].dtype;
    for i in 1..#nArrays do if gEnts[i].dtype != dtype {
      const errMsg = "All arrays must have the same dtype to concatenate";
      mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
      return new MsgTuple(errMsg,MsgType.ERROR);
    }

    proc doFlatConcat(type t): MsgTuple throws {
      const eIns = [i in 0..#nArrays] toSymEntry(gEnts[i], t, nd),
            flatArrays = [e in eIns] flatten(e.a),
            sizes = [a in flatArrays] a.size,
            starts = + scan sizes;

      // create a 1D output array
      var eOut = st.addEntry(rname, + reduce sizes, t);

      // copy the data from the input arrays to the output array
      forall (arrIdx, a) in zip(flatArrays.domain, flatArrays) do
        forall idx in a.domain with (var agg = newSrcAggregator(t)) do
          agg.copy(eOut.a[idx + starts[arrIdx]], a[idx]);

      const repMsg = "created " + st.attrib(rname);
      mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return doFlatConcat(int);
      when DType.UInt64 do return doFlatConcat(uint);
      when DType.Float64 do return doFlatConcat(real);
      when DType.Bool do return doFlatConcat(bool);
      when DType.BigInt do return doFlatConcat(bigint);
      otherwise {
        var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
        mLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      }
    }
  }

  private proc flatten(const ref a: [?d] ?t): [] t {
    var flat : [0..<d.size] t;
    forall idx in flat.domain with (var agg = newDstAggregator(t)) do
      agg.copy(flat[idx], a[d.orderToIndex(idx)]);
    return flat;
  }

  @arkouda.registerND
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.flip.html#array_api.flip
  proc flipMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          axis = msgArgs.get("axis").getPositiveIntValue(nd),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doFlip(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd);
      var eOut = st.addEntry(rname, (...eIn.tupShape), t);

      // mapping between the input and output array indices
      inline proc imap(idx: nd*int): nd*int {
        var ret = idx;
        ret[axis] = eIn.tupShape[axis] - idx[axis] - 1;
        return ret;
      }

      // copy the data from the input array to the output array
      // while flipping along the specified axis
      forall idx in eOut.a.domain with (var agg = newDstAggregator(t)) do
        agg.copy(eOut.a[idx], eIn.a[imap(idx)]);

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

  @arkouda.registerND
  // alternative to 'flipMsg' to be used when the axis argument is 'None'
  proc flipAllMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doFlip(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd);
      var eOut = st.addEntry(rname, (...eIn.tupShape), t);

      // mapping between the input and output array indices
      inline proc imap(idx: nd*int): nd*int {
        var ret: nd*int;
        for param i in 0..<nd do
          ret[i] = eIn.tupShape[i] - idx[i] - 1;
        return ret;
      }

      // copy the data from the input array to the output array
      // while flipping along each axis
      forall idx in eOut.a.domain with (var agg = newDstAggregator(t)) do
        agg.copy(eOut.a[idx], eIn.a[imap(idx)]);

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

  @arkouda.registerND
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.permute_dims.html#array_api.permute_dims
  proc permuteDims(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          perm = msgArgs.get("perm").getTuple(nd),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doPermutation(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            outShape = permuteTuple(eIn.tupShape, perm);
      var eOut = st.addEntry(rname, (...outShape), t);

      // copy the data from the input array to the output array
      // while permuting the axes
      forall idx in eOut.a.domain with (var agg = newDstAggregator(t)) do
        agg.copy(eOut.a[idx], eIn.a[permuteTuple(idx, perm)]);

      const repMsg = "created " + st.attrib(rname);
      mLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
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

  private inline proc permuteTuple(param tup: ?N*int, param perm: N*int): N*int {
    var ret: N*int;
    for param i in 0..<N do ret[i] = tup[perm[i]];
    return ret;
  }

  @arkouda.registerND
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html#array_api.reshape
  proc reshapeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    param ndIn: int,
    param ndOut: int,
  ): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          rawShape = msgArgs.get("shape").getTuple(ndOut),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doReshape(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, ndIn),
            (valid, shape) = validateShape(rawShape, eIn.a.size),

      if !valid {
        const errMsg = "Cannot reshape array of shape %? into shape %?. The total number of elements must match".doFormat(eIn.tupShape, rawShape);
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg,eIn.a.size,shape);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
        var eOut = st.addEntry(rname, (...shape), t);

        // index -> order for the output array's indices
        // e.g., order = k + (nz * j) + (nz * ny * i)
        inline proc indexToOrder(idx: ndOut*int): int {
          var order = 0,
              accum = 1;
          for param i in 0..<ndOut {
            order += idx[i] * accum;
            accum *= eOut.tupShape[i];
          }
          return order;
        }

        // copy the data from the input array to the output array
        forall idx in eOut.a.domain with (var agg = newDstAggregator(t)) do
          agg.copy(eOut.a[idx], eIn.a[eIn.a.domain.orderToIndex(indexToOrder(idx))]);

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

  @arkouda.registerND
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.roll.html#array_api.roll
  proc rollMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          nShifts = msgArgs.get("nShifts").getIntValue(), // number of elements in 'shift' argument
          nAxes = msgArgs.get("nAxes").getIntValue(),     // number of elements in 'axis' argument
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doRoll(type t): MsgTuple throws {
      var errMsg = "";

      // TODO: simplify the structure of this code
      // Implements the following logic from the spec:
      //    "If shift is a tuple, then axis must be a tuple of the same size,
      //     and each of the given axes must be shifted by the corresponding
      //     element in shift. If shift is an int and axis a tuple, then the
      //     same shift must be used for all specified axes ... If axis is
      //     None, the array must be flattened, shifted, and then restored to
      //     its original shape"
      if nAxes == 0 { // axis = None
        if nShifts == 1 {
          const shift = msgArgs.get("shift").getIntValue(),
                rolled = rollFlattened(t, shift, gEnt, nd);
          st.addEntry(rname, createSymEntry(rolled));
        } else {
          errMsg = "size of 'shift' must be 1 when 'axis' is None";
        }
      } else if nAxes == 1 {
        if nShifts == 1 {
          const shift = msgArgs.get("shift").getIntValue(),
                axis = msgArgs.get("axis").getIntValue(),
                rolled = rollAlongAxes(t, [shift], [axis], gEnt, nd);
          st.addEntry(rname, createSymEntry(rolled));
        } else {
          errMsg = "Cannot roll along a single axis with multiple shift values";
        }
      } else {
        if nShifts == nAxes {
          const shifts = msgArgs.get("shift").getList(nShifts),
                axes = msgArgs.get("axis").getList(nAxes),
                rolled = rollAlongAxes(t, shifts, axes, gEnt, nd);
          st.addEntry(rname, createSymEntry(rolled));
        } else if nShifts == 1 {
          const shift = msgArgs.get("shift").getIntValue(),
                shifts = [i in 0..nAxes] shift,
                axes = msgArgs.get("axis").getList(nAxes),
                rolled = rollAlongAxes(t, shifts, axes, gEnt, nd);
          st.addEntry(rname, createSymEntry(rolled));
        } else {
          errMsg = "size of 'shift' must match size of 'axis' or be 1";
        }
      }

      if errMsg != "" {
        mLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
        return new MsgTuple(errMsg,MsgType.ERROR);
      } else {
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

  // flatten the array, apply the shift, and then unflatten
  private proc rollFlattened(type t, shift: int, gEnt: borrowed GenSymEntry, param nd: int): [] t {
    const eIn = toSymEntry(gEnt, t, nd),
          rolled: [eIn.a.domain] t,
          shifted = shift1D(flatten(eIn.a), shift);

    forall idx in shifted.domain with (var agg = newSrcAggregator(t)) do
      agg.copy(eOut.a[eOut.a.domain.orderToIndex(idx)], shifted[idx]);

    return rolled;
  }

  private proc shift1D(in a: [?d] ?t, shift: int): [d] t {
    var ret: [d] t;

    forall idx in a.domain with (var agg = newDstAggregator(t)) do
      agg.copy(ret[idx], a[(idx + shift) % a.size]);

    return ret;
  }

  private proc rollAlongAxes(type t, shifts: [?d] int, axes: [?d] int, gEnt: borrowed GenSymEntry, param nd: int): [] t {
    const eIn = toSymEntry(gEnt, t, nd),
          rolled: [eIn.a.domain] t;

    inline proc rollIdx(idx: nd*int): nd*int {
      var ret = idx;
      for i in 0..<nShifts do ret[axes[i]] = (idx[axes[i]] + shifts[i]) % eIn.tupShape[axes[i]];
      return ret;
    }

    forall idx in rolled.domain with (var agg = newDstAggregator(t)) do
      agg.copy(rolled[idx], eIn.a[rollIdx(idx)]);

    return rolled;
  }
}
