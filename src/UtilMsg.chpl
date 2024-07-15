module UtilMsg {

  use ServerConfig;
  use ServerErrorStrings;

  use Reflection;
  use ServerErrors;
  use Logging;
  use Message;
  use AryUtil;
  use ArkoudaAryUtilCompat;

  use MultiTypeSymEntry;
  use MultiTypeSymbolTable;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const uLogger = new Logger(logLevel, logChannel);

  /*
    Constrain an array's values to be within a specified range [min, max]

    see: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
  */
  @arkouda.registerND
  proc clipMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();

    const name = msgArgs.getValueOf("name"),
          min = msgArgs.get("min"),
          max = msgArgs.get("max"),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doClip(type t): MsgTuple throws {
      const minVal = min.getScalarValue(t),
            maxVal = max.getScalarValue(t);

      const e = toSymEntry(gEnt, t, nd);
      var c = st.addEntry(rname, (...e.tupShape), t);

      forall i in e.a.domain {
        if e.a[i] < minVal then
          c.a[i] = minVal;
        else if e.a[i] > maxVal then
          c.a[i] = maxVal;
        else
          c.a[i] = e.a[i];
      }

      const repMsg = "created " + st.attrib(rname);
      uLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return doClip(int);
      when DType.UInt8 do return doClip(uint(8));
      when DType.UInt64 do return doClip(uint);
      when DType.Float64 do return doClip(real);
      when DType.Bool do return doClip(bool);
      otherwise {
        const errorMsg = notImplementedError(pn,gEnt.dtype);
        uLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  /*
    Compute the n'th order discrete difference along a given axis

    see: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
  */
  @arkouda.registerND
  proc diffMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();

    const name = msgArgs.getValueOf("name"),
          n = msgArgs.get("n").getIntValue(),
          axis = msgArgs.get("axis").getPositiveIntValue(nd),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doDiff(type t): MsgTuple throws {
      const e = toSymEntry(gEnt, t, nd);

      if n == 1 {
        // 1st order difference: requires no temporary storage
        const outDom = subDomain(e.tupShape, axis, 1);
        var d = st.addEntry(rname, (...outDom.shape), t);

        forall axisSliceIdx in domOffAxis(e.a.domain, axis) {
          const slice = domOnAxis(outDom, tuplify(axisSliceIdx), axis);
          for i in slice {
            var idxp = tuplify(i);
            idxp[axis] += 1;
            d.a[i] = e.a[idxp] - e.a[i];
          }
        }
      } else {
        // n'th order difference: requires 2 temporary arrays
        var d1 = makeDistArray(e.a);

        {
          var d2 = makeDistArray(e.a.domain, e.a.eltType);
          for m in 1..n {
            d1 <=> d2;
            const diffSubDom = subDomain(e.tupShape, axis, m);

            forall axisSliceIdx in domOffAxis(e.a.domain, axis) {
              const slice = domOnAxis(diffSubDom, tuplify(axisSliceIdx), axis);

              for i in slice {
                var idxp = tuplify(i);
                idxp[axis] += 1;
                d1[i] = d2[idxp] - d2[i];
              }
            }
          }
        } // d2 deinit here

        const outDom = subDomain(e.tupShape, axis, n),
              d = createSymEntry(d1[outDom]);
        st.addEntry(rname, d);
      }

      const repMsg = "created " + st.attrib(rname);
      uLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return doDiff(int);
      when DType.UInt8 do return doDiff(uint(8));
      when DType.UInt64 do return doDiff(uint);
      when DType.Float64 do return doDiff(real);
      otherwise {
        const errorMsg = notImplementedError(pn,gEnt.dtype);
        uLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  // helper to create a domain that's 'n' elements smaller in the 'axis' dimension
  private proc subDomain(shape: ?N*int, axis: int, n: int) {
    var rngs: N*range;
    for i in 0..<N {
      if i == axis
        // then rngs[i] = (n/2)..<(shape[i] - (n/2 + n%2));
        then rngs[i] = 0..<(shape[i] - n);
        else rngs[i] = 0..<shape[i];
    }
    return {(...rngs)};
  }

  private proc tuplify(x: int) do return (x,);
  private proc tuplify(t: ?N*int) do return t;


  /*
    Pad an array with a set of specified values

    Implements the 'constant' mode of numpy.pad: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
  */
  @arkouda.registerND
  proc padMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();

    const name = msgArgs.getValueOf("name"),
          padWidthBefore = msgArgs.get("padWidthBefore").getTuple(nd),
          padWidthAfter = msgArgs.get("padWidthAfter").getTuple(nd),
          padValsBefore = msgArgs.get("padValsBefore"),
          padValsAfter = msgArgs.get("padValsAfter"),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc doPad(type t): MsgTuple throws {
      const e = toSymEntry(gEnt, t, nd);

      const pvb = padValsBefore.toScalarArray(t, nd),
            pva = padValsAfter.toScalarArray(t, nd);

      // compute the padded shape
      var outShape: nd*int;
      for i in 0..<nd do outShape[i] = padWidthBefore[i] + e.tupShape[i] + padWidthAfter[i];

      var p = st.addEntry(rname, (...outShape), t);

      // copy the original array into the padded array
      const dOffset = e.a.domain.translate(padWidthBefore);
      p.a[dOffset] = e.a;

      // starting with the last dimension, pad the array (i.e., dimension 0 overwrites dimension 1 in the corners, etc.)
      for rank in 0..<nd {
        var beforeSlice, afterSlice: nd*range;
        for i in 0..<nd {
          // TODO: compute the exact slice for each pad-section so these assignments can be done
          // in parallel and to avoid accessing the corners of the array unnecessarily (which
          // could result in additional comm for large pad widths)
          if i == rank {
            beforeSlice[i] = 0..<padWidthBefore[i];
            afterSlice[i] = (outShape[i]-padWidthAfter[i])..<outShape[i];
          } else {
            beforeSlice[i] = 0..<outShape[i];
            afterSlice[i] = 0..<outShape[i];
          }
        }

        p.a[(...beforeSlice)] = pvb[rank];
        p.a[(...afterSlice)] = pva[rank];
      }

      const repMsg = "created " + st.attrib(rname);
      uLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return doPad(int);
      when DType.UInt8 do return doPad(uint(8));
      when DType.UInt64 do return doPad(uint);
      when DType.Float64 do return doPad(real);
      when DType.Bool do return doPad(bool);
      otherwise {
        const errorMsg = notImplementedError(pn,gEnt.dtype);
        uLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }
}
