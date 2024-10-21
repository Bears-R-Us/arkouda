module UtilMsg {

  use ServerConfig;
  use ServerErrorStrings;

  use Reflection;
  use ServerErrors;
  use Logging;
  use Message;
  use AryUtil;
  use List;
  use BigInteger;

  use MultiTypeSymEntry;
  use MultiTypeSymbolTable;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const uLogger = new Logger(logLevel, logChannel);

  /*
    Constrain an array's values to be within a specified range [min, max]

    see: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
  */
  @arkouda.registerCommand()
  proc clip(const ref x: [?d] ?t, min: real, max: real): [] t throws 
    where (t == int) || (t == real) || (t == uint(8)) || (t == uint(64)) {

      var y = makeDistArray(d, t);

      const minVal = min: t,
            maxVal = max: t;

      forall i in d {
        if x[i] < minVal then
          y[i] = minVal;
        else if x[i] > maxVal then
          y[i] = maxVal;
        else
          y[i] = x[i];
      }
      return y;
  }

  /*
    Compute the n'th order discrete difference along a given axis

    see: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
  */

  @arkouda.registerCommand()
  proc diff(x: [?d] ?t, n: int, axis: int): [] t throws 
    where (t == real) || (t == int) || (t == uint(8)) || (t == uint(64)){

    const outDom = subDomain(x.shape, axis, n);
    if n == 1 {
      // 1st order difference: requires no temporary storage
      var y = makeDistArray(outDom, t);
      for axisSliceIdx in domOffAxis(d, axis) {
        const slice = domOnAxis(outDom, tuplify(axisSliceIdx), axis);
        for i in slice {
          var idxp = tuplify(i);
          idxp[axis] += 1;
          y[i] = x[idxp] - x[i];
        }
      }
      return y;
    } else {
      // n'th order difference: requires 2 temporary arrays
      var d1 = makeDistArray(x);
      {
        var d2 = makeDistArray(d, t);
        for m in 1..n {
          d1 <=> d2;
          const diffSubDom = subDomain(x.shape, axis, m);

          forall axisSliceIdx in domOffAxis(d, axis) {
            const slice = domOnAxis(diffSubDom, tuplify(axisSliceIdx), axis);

            for i in slice {
              var idxp = tuplify(i);
              idxp[axis] += 1;
              d1[i] = d2[idxp] - d2[i];
            }
          }
        }
      } // d2 deinit here
      return d1[outDom];
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
  @arkouda.instantiateAndRegister
  proc pad(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws 
    where (array_dtype == int) || (array_dtype == uint(8)) || (array_dtype == uint(64)) || (array_dtype == real) || (array_dtype == bool) {

    const e = st[msgArgs["name"]]: SymEntry(array_dtype, array_nd),
          padWidthBefore = msgArgs["padWidthBefore"].toScalarTuple(int, array_nd),
          padWidthAfter = msgArgs["padWidthAfter"].toScalarTuple(int, array_nd),
          padValsBefore = msgArgs["padValsBefore"].toScalarArray(array_dtype, array_nd),
          padValsAfter = msgArgs["padValsAfter"].toScalarArray(array_dtype, array_nd);

    // compute the padded shape
    var outShape: array_nd*int;
    for i in 0..<array_nd do outShape[i] = padWidthBefore[i] + e.tupShape[i] + padWidthAfter[i];

    var paddedArray = makeDistArray((...outShape), array_dtype);

    // copy the original array into the padded array
    const dOffset = e.a.domain.translate(padWidthBefore);
    paddedArray[dOffset] = e.a;

    // starting with the last dimension, pad the array (i.e., dimension 0 overwrites dimension 1 in the corners, etc.)
    for rank in 0..<array_nd {
      var beforeSlice, afterSlice: array_nd*range;
      for i in 0..<array_nd {
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

      paddedArray[(...beforeSlice)] = padValsBefore[rank];
      paddedArray[(...afterSlice)] = padValsAfter[rank];
    }

    return st.insert(new shared SymEntry(paddedArray));
  }

}
