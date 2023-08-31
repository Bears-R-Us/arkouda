module ReductionMsg
{
    use ServerConfig;

    use ArkoudaTimeCompat as Time;
    use Math;
    use Reflection only;
    use CommAggregation;
    use BigInteger;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    use AryUtil;
    use PrivateDist;
    use RadixSortLSD;
    use ArkoudaMathCompat;
    use ArkoudaBlockCompat;

    private config const lBins = 2**25 * numLocales;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const rmLogger = new Logger(logLevel, logChannel);

    // this should never be called with the default value
    // it should always be overriden by the pdarray's max_bits attribute
    var class_lvl_max_bits = -1;

    // these functions take an array and produce a scalar
    // parse and respond to reduction message
    // scalar = reductionop(vector)
    proc reductionMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string = ""; // response message
        const reductionop = msgArgs.getValueOf("op");
        const name = msgArgs.getValueOf("array");
        rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s reductionop: %s name: %s".doFormat(cmd,reductionop,name));

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
       
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum != 0 {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum == e.a.domain.size {val = "True";} else {val = "False";}
                       repMsg = "bool %s".doFormat(val);
                    }
                    when "sum" {
                        var val = + reduce e.a;
                        repMsg = "int64 %i".doFormat(val);
                    }
                    when "prod" {
                        // Cast to real to avoid int64 overflow
                        var val = * reduce e.a:real;
                        // Return value is always float64 for prod
                        repMsg = "float64 %.17r".doFormat(val);
                    }
                    when "min" {
                      var val = min reduce e.a;
                      repMsg = "int64 %i".doFormat(val);
                    }
                    when "max" {
                        var val = max reduce e.a;
                        repMsg = "int64 %i".doFormat(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.a.domain);
                        repMsg = "int64 %i".doFormat(minLoc);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.a.domain);
                        repMsg = "int64 %i".doFormat(maxLoc);
                    }
                    when "is_sorted" {
                        ref ea = e.a;
                        var sorted = isSorted(ea);
                        var val: string;
                        if sorted {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "is_locally_sorted" {
                      var locSorted: [LocaleSpace] bool;
                      coforall loc in Locales with (ref locSorted) {
                        on loc {
                          ref myA = e.a[e.a.localSubdomain()];
                          locSorted[here.id] = isSorted(myA);
                        }
                      }

                      var val: string;
                      if (& reduce locSorted) {val = "True";} else {val = "False";}

                      repMsg = "bool %s".doFormat(val);
                      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                      return new MsgTuple(repMsg, MsgType.NORMAL); 
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,reductionop,gEnt.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);                      
                    }
                }
            }
            when (DType.UInt64) {
                var e = toSymEntry(gEnt,uint);
                select reductionop
                {
                    when "sum" {
                        var val = + reduce e.a;
                        repMsg = "uint64 %i".doFormat(val);
                    }
                    when "prod" {
                        // Cast to real to avoid int64 overflow
                        var val = * reduce e.a:real;
                        // Return value is always float64 for prod
                        repMsg = "float64 %.17r".doFormat(val);
                    }
                    when "min" {
                      var val = min reduce e.a;
                      repMsg = "uint64 %i".doFormat(val);
                    }
                    when "max" {
                        var val = max reduce e.a;
                        repMsg = "uint64 %i".doFormat(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.a.domain);
                        repMsg = "uint64 %i".doFormat(minLoc);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.a.domain);
                        repMsg = "uint64 %i".doFormat(maxLoc);
                    }
                    when "is_sorted" {
                        ref ea = e.a;
                        var sorted = isSorted(ea);
                        var val: string;
                        if sorted {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,reductionop,gEnt.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum != 0.0 {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum == e.a.domain.size {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "sum" {
                        var val = + reduce e.a;
                        repMsg = "float64 %.17r".doFormat(val);
                    }
                    when "prod" {
                        var val = * reduce e.a;
                        repMsg =  "float64 %.17r".doFormat(val);
                    }
                    when "min" {
                        var val = min reduce e.a;
                        repMsg = "float64 %.17r".doFormat(val);
                    }
                    when "max" {
                        var val = max reduce e.a;
                        repMsg = "float64 %.17r".doFormat(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.a.domain);
                        repMsg = "int64 %i".doFormat(minLoc);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.a.domain);
                        repMsg = "int64 %i".doFormat(maxLoc);
                    }
                    when "is_sorted" {
                        var sorted = isSorted(e.a);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,reductionop,gEnt.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);                    
                    }
                }
            }
            when (DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var any = | reduce e.a;
                        if any {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "all" {
                        var val:string;
                        var all = & reduce e.a;
                        if all {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "sum" {
                        var val = + reduce e.a:int;
                        repMsg = "int64 %i".doFormat(val);
                    }
                    when "prod" {
                        var val = * reduce e.a:int;
                        repMsg = "int64 %i".doFormat(val);
                    }
                    when "min" {
                        var val:string;
                        if (& reduce e.a) { val = "True"; } else { val = "False"; }
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "max" {
                        var val:string;
                        if (| reduce e.a) { val = "True"; } else { val = "False"; }
                        repMsg = "bool %s".doFormat(val);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.a.domain);
                        repMsg = "int64 %i".doFormat(maxLoc);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.a.domain);
                        repMsg = "int64 %i".doFormat(minLoc);
                    }
                    when "is_sorted" {
                        var sorted = isSorted(e.a);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        repMsg = "bool %s".doFormat(val);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,reductionop,gEnt.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);                        
                    }
                }
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn, dtype2str(gEnt.dtype));
                rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);       
                return new MsgTuple(errorMsg, MsgType.ERROR);       
            }
        }
        rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);       
        return new MsgTuple(repMsg, MsgType.NORMAL);          
    }

    proc countReductionMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
      // reqMsg: segmentedReduction values segments operator
      // 'segments_name' describes the segment offsets
      const segments_name = msgArgs.getValueOf("segments");
      const size = msgArgs.get("size").getIntValue();
      var rname = st.nextName();
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "cmd: %s segments_name: %s size: %s".doFormat(cmd,segments_name, size));

      var gSeg: borrowed GenSymEntry = getGenericTypedArrayEntry(segments_name, st);
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {    
          var errorMsg = "Array of segment offsets must be int dtype";
          rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);       
          return new MsgTuple(errorMsg, MsgType.ERROR); 
      }
      var counts = segCount(segments.a, size);
      st.addEntry(rname, createSymEntry(counts));

      var repMsg = "created " + st.attrib(rname);
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);       
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc segCount(segments:[?D] int, upper: int):[D] int {
      var counts = makeDistArray(D, int);
      if (D.size == 0) { return counts; }
      forall (c, low, i) in zip(counts, segments, D) {
        var high: int;
        if (i < D.high) {
          high = segments[i+1] - 1;
        } else {
          high = upper - 1;
        }
        c = high - low + 1;
      }
      return counts;
    }

    proc nanCounts(values:[] ?t, segments:[?D] int) throws {
      // count cumulative nans over all values
      var cumnans = isNan(values):int;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * values.size);
      cumnans = + scan cumnans;

      // find cumulative nans at segment boundaries
      var segnans = makeDistArray(D, int);
      forall (si, sn) in zip(D, segnans) with (var agg = newSrcAggregator(int)) {
        if si == D.high {
          agg.copy(sn, cumnans[cumnans.domain.high]);
        } else {
          agg.copy(sn, cumnans[segments[si+1]-1]);
        }
      }

      // take diffs of adjacent segments to find nan count in each segment
      var nancounts = makeDistArray(D, int);
      nancounts[D.low] = segnans[D.low];
      nancounts[D.low+1..] = segnans[D.low+1..] - segnans[..D.high-1];
      return nancounts;
    }
    
    proc segmentedReductionMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        // reqMsg: segmentedReduction values segments operator
        // 'values_name' is the segmented array of values to be reduced
        // 'segments_name' is the sement offsets
        // 'op' is the reduction operator
        const skipNan = msgArgs.get("skip_nan").getBoolValue();
        const values_name = msgArgs.getValueOf("values");
        const segments_name = msgArgs.getValueOf("segments");
        const op = msgArgs.getValueOf("op");
        const ddof = msgArgs.get("ddof").getIntValue();
      
        var rname = st.nextName();
        rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "cmd: %s values_name: %s segments_name: %s operator: %s skipNan: %s".doFormat(
                                       cmd,values_name,segments_name,op,skipNan));
        var gVal: borrowed GenSymEntry = getGenericTypedArrayEntry(values_name, st);
        var gSeg: borrowed GenSymEntry = getGenericTypedArrayEntry(segments_name, st);
        var segments = toSymEntry(gSeg, int);
        if (segments == nil) {
            var errorMsg = "Error: array of segment offsets must be int dtype";
            rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
            return new MsgTuple(errorMsg, MsgType.ERROR);        
        }
        select (gVal.dtype) {
            when (DType.Int64) {
                var values = toSymEntry(gVal, int);
                select op {
                    when "sum" {
                        var res = segSum(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "prod" {
                        var res = segProduct(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "var" {
                        var res = segVar(values.a, segments.a, ddof);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "std" {
                        var res = segStd(values.a, segments.a, ddof);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "mean" {
                        var res = segMean(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "median" {
                        var res = segMedian(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "min" {
                        var res = segMin(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "max" {
                        var res = segMax(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "argmin" {
                        var (vals, locs) = segArgmin(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    when "argmax" {
                        var (vals, locs) = segArgmax(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    when "or" {
                        var res = segOr(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "and" {
                        var res = segAnd(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "xor" {
                        var res = segXor(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "nunique" {
                        var res = segNumUnique(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,op,gVal.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64) {
                var values = toSymEntry(gVal,uint);
                select op {
                    when "sum" {
                        var res = segSum(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "prod" {
                        var res = segProduct(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "var" {
                        var res = segVar(values.a, segments.a, ddof);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "std" {
                        var res = segStd(values.a, segments.a, ddof);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "mean" {
                        var res = segMean(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "median" {
                        var res = segMedian(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "min" {
                        var res = segMin(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "max" {
                        var res = segMax(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "argmin" {
                        var (vals, locs) = segArgmin(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    when "argmax" {
                        var (vals, locs) = segArgmax(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    when "or" {
                        var res = segOr(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "and" {
                        var res = segAnd(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "xor" {
                        var res = segXor(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "nunique" {
                        var res = segNumUnique(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,op,gVal.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);  
                    }                       
                }    
            }
            when (DType.Float64) {
                var values = toSymEntry(gVal, real);
                select op {
                    when "sum" {
                        var res = segSum(values.a, segments.a, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "prod" {
                        var res = segProduct(values.a, segments.a, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    } 
                    when "var" {
                        var res = segVar(values.a, segments.a, ddof, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "std" {
                        var res = segStd(values.a, segments.a, ddof, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "mean" {
                        var res = segMean(values.a, segments.a, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "median" {
                        var res = segMedian(values.a, segments.a, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "min" {
                        var res = segMin(values.a, segments.a, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "max" {
                        var res = segMax(values.a, segments.a, skipNan);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "argmin" {
                        var (vals, locs) = segArgmin(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    when "argmax" {
                        var (vals, locs) = segArgmax(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,op,gVal.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);         
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
               }
            }
            when (DType.Bool) {
                var values = toSymEntry(gVal, bool);
                select op {
                    when "sum" {
                        var res = segSum(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "any" {
                        var res = segAny(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "all" {
                        var res = segAll(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "var" {
                        var res = segVar(values.a, segments.a, ddof);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "std" {
                        var res = segStd(values.a, segments.a, ddof);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "mean" {
                        var res = segMean(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "median" {
                        var res = segMedian(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "argmin" {
                        var (vals, locs) = segArgmin(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    when "argmax" {
                        var (vals, locs) = segArgmax(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(locs));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,op,gVal.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.BigInt) {
                var values = toSymEntry(gVal, bigint);
                var max_bits = values.max_bits;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                    class_lvl_max_bits = max_bits;
                }
                select op {
                    when "sum" {
                        var res = segSum(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "min" {
                        if !has_max_bits {
                          throw new Error("Must set max_bits to MIN");
                        }
                        var res = segMin(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "max" {
                        if !has_max_bits {
                          throw new Error("Must set max_bits to MAX");
                        }
                        var res = segMax(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "or" {
                        var res = segOr(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    when "and" {
                        if !has_max_bits {
                          throw new Error("Must set max_bits to AND");
                        }
                        var res = segAnd(values.a, segments.a);
                        st.addEntry(rname, createSymEntry(res));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,op,gVal.dtype);
                        rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
          otherwise {
              var errorMsg = unrecognizedTypeError(pn, dtype2str(gVal.dtype));
              rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
          }
       }
       var repMsg = "created " + st.attrib(rname);
       rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
       return new MsgTuple(repMsg, MsgType.NORMAL);
    }

          
    /* Segmented Reductions of the form: seg<Op>(values:[] t, segments: [] int)
       Use <segments> as the boundary indices to divide <values> into chunks, 
       and then reduce over each chunk using the operator <Op>. The return array 
       of reduced values is the same size as <segments>.
     */
    proc segSum(values:[?vD] ?intype, segments:[?D] int, skipNan=false) throws {
      type t = if intype == bool then int else intype;
      var res = makeDistArray(D, t);
      if (D.size == 0) { return res; }
      // Set reset flag at segment boundaries
      var flagvalues = makeDistArray(vD, (bool, t)); // = [v in values] (false, v);
      if isRealType(t) && skipNan {
        forall (fv, val) in zip(flagvalues, values) {
          fv = if isNan(val) then (false, 0.0) else (false, val);
        }
      } else {
        forall (fv, val) in zip(flagvalues, values) {
          fv = (false, val:t);
        }
      }
      forall s in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(flagvalues[s][0], true);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      if t != bigint {
        // TODO update when we have a better way to handle bigint mem estimation
        overMemLimit((numBytes(t)+1) * flagvalues.size);
      }
      // Scan with custom operator, which resets the bitwise AND
      // at segment boundaries.
      const scanresult = ResettingPlusScanOp scan flagvalues;
      // Read the results from the last element of each segment
      forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(t)) {
        agg.copy(r, scanresult[s-1](1));
      }
      res[D.high] = scanresult[vD.high](1);
      return res;
    }

    /* Performs a bitwise sum scan, controlled by a reset flag. While
     * the reset flag is false, the accumulation of values proceeds as 
     * normal. When a true is encountered, the state resets to the
     * identity. */
    class ResettingPlusScanOp: ReduceScanOp {
      type eltType;
      /* value is a tuple comprising a flag and the actual result of 
         segmented sum. 

         The meaning of the flag depends on whether it belongs to an 
         array element yet to be scanned or to an element that has 
         already been scanned (including the internal state of a class
         instance doing the scanning). For elements yet to be scanned,
         the flag means "reset to the identity here". For elements that
         have already been scanned, or for internal state, the flag means 
         "there has already been a reset in the computation of this value".
      */
      var value: eltType;

      proc identity {
        if eltType == (bool, real) then return (false, 0.0);
        else if eltType == (bool, uint) then return (false, 0:uint);
        else if eltType == (bool, bigint) then return (false, 0:bigint);
        else return (false, 0);
      }

      proc accumulate(x) {
        // Assume x is an element that has not yet been scanned, and
        // that it comes after the current state.
        const (reset, other) = x;
        const (hasReset, v) = value;
        // x's reset flag controls whether value gets replaced or combined
        // also update this instance's "hasReset" flag with x's reset flag
        value = (hasReset | reset, if reset then other else (v + other));
      }

      proc accumulateOntoState(ref state, x) {
        // Assume state is an element that has already been scanned,
        // and x is an update from a previous boundary.
        const (prevReset, other) = x;
        const (hasReset, v) = state;
        // absorb reset history
        // If state has already encountered a reset, then it should
        // ignore x's value
        state = (hasReset | prevReset, if hasReset then v else (v + other));
      }

      proc combine(x) {
        // Assume x is an instance that scanned a prior chunk.
        const (xHasReset, other) = x.value;
        const (hasReset, v) = value;
        // Since current instance is absorbing x's history,
        // xHasReset flag should be ORed in.
        // But if current instance has already encountered a reset,
        // then it should ignore x's value.
        value = (hasReset | xHasReset, if hasReset then v else (v + other));
      }

      proc generate() {
        return value;
      }

      proc clone() {
        return new unmanaged ResettingPlusScanOp(eltType=eltType);
      }
    }

    proc segProduct(values:[] ?t, segments:[?D] int, skipNan=false): [D] real throws {
      /* Compute the product of values in each segment. The logic here 
         is to convert the product into a sum in the log-domain. To 
         operate in the log-domain, signs and zeros must be removed and 
         handled separately at the end. Thus, we take the absolute 
         value of the array and replace all zeros with ones, keeping 
         track of the original signs and locations of zeros for later. 
         In computing the result, if there are any zeros in the segment, 
         the product is zero. Otherwise, the sign of the segment product 
         is the parity of the negative bits in the segment.
       */
      // Regardless of input type, the product is real
      var res = makeDistArray(D, real);
      if (D.size == 0) { return res; }
      const isZero = (values == 0);
      // Take absolute value, replacing zeros with ones
      // Ones will become zeros in log-domain and not affect + scan
      var magnitudes = makeDistArray(values.domain, real);
      if (isRealType(t) && skipNan) {
        forall (m, v, z) in zip(magnitudes, values, isZero) {
          if isNan(v) {
            m = 1.0;
          } else {
            m = abs(v) + z:real;
          }
        }
      } else {
        magnitudes = abs(values) + isZero:real;
      }
      var logs = log(magnitudes);
      var negatives = (sgn(values) == -1);
      forall (r, v, n, z) in zip(res,
                                 segSum(logs, segments),
                                 segSum(negatives, segments),
                                 segSum(isZero, segments)) {
        // if any zeros in segment, leave result zero
        if z == 0 {
          // n = number of negative values in segment
          // if n is even, product is positive; if odd, negative
          var sign = -2*(n%2) + 1;
          // v = the sum of log-magnitudes; must be exponentiated and signed
          r = sign * exp(v);
        }
      }
      return res;
    }

    proc segVar(ref values:[?vD] ?t, segments:[?D] int, ddof:int, skipNan=false): [D] real throws {
      var res = makeDistArray(D, real);
      if D.size == 0 { return res; }

      var counts = segCount(segments, values.size);
      const means = segMean(values, segments, skipNan);
      // expand mean per segment to be size of values
      const expandedMeans = [k in expandKeys(vD, segments)] means[k];
      var squaredDiffs = makeDistArray(vD, real);
      // First deal with any NANs and calculate squaredDiffs
      if isRealType(t) && skipNan {
        // calculate counts with nan values excluded and 0 out the nans
        squaredDiffs = [(v,m) in zip(values, expandedMeans)] if isNan(v) then 0:real else (v - m)**2;
        counts -= nanCounts(values, segments);
      }
      else {
        squaredDiffs = [(v,m) in zip(values, expandedMeans)] (v:real - m)**2;
      }
      forall (r, s, c) in zip(res, segSum(squaredDiffs, segments), counts) {
        r = if c-ddof > 0 then s / (c-ddof):real else nan;
      }
      return res;
    }

    proc segStd(ref values:[] ?t, segments:[?D] int, ddof:int, skipNan=false): [D] real throws {
      if D.size == 0 { return [D] 0.0; }
      return sqrt(segVar(values, segments, ddof, skipNan));
    }

    proc segMean(ref values:[] ?t, segments:[?D] int, skipNan=false): [D] real throws {
      var res = makeDistArray(D, real);
      if (D.size == 0) { return res; }
      // convert to real early to avoid int overflow
      overMemLimit(numBytes(real) * values.size);
      var real_values = values: real;
      var sums;
      var counts;
      if skipNan {
        // first verify that we can make a copy of real_values
        overMemLimit(numBytes(real) * real_values.size);
        // calculate sum and counts with nan real_values replaced with 0.0
        var arrCopy = [elem in real_values] if isNan(elem) then 0.0 else elem;
        sums = segSum(arrCopy, segments);
        counts = segCount(segments, real_values.size) - nanCounts(real_values, segments);
      } else {
        sums = segSum(real_values, segments);
        counts = segCount(segments, real_values.size);
      }
      forall (r, s, c) in zip(res, sums, counts) {
        if (c > 0) {
          r = s:real / c:real;
        }
      }
      return res;
    }

    proc segMedian(ref values:[?vD] ?intype, segments:[?D] int, skipNan=false): [D] real throws {
      type t = if intype == bool then int else intype;
      var res = makeDistArray(D, real);
      if (D.size == 0) { return res; }

      var counts = segCount(segments, values.size);
      var noNanVals = values: t;
      // First deal with any nans
      if isRealType(t) && skipNan {
        // calculate counts with nan values excluded and replace nan with max(real)
        // this will force them at the very end of the sorted segment and since
        // counts has been corrected, they won't affect the result
        noNanVals = [elem in values] if isNan(elem) then max(real) else elem;
        counts -= nanCounts(values, segments);
      }

      // then sort values within their segment (from segNumUnique)
      // keys will indicate which segment we are in
      const keys = expandKeys(vD, segments);
      const firstIV = radixSortLSD_ranks(noNanVals);
      var intermediate = makeDistArray(vD, int);
      forall (ii, idx) in zip(intermediate, firstIV) with (var agg = newSrcAggregator(int)) {
          agg.copy(ii, keys[idx]);
      }
      const deltaIV = radixSortLSD_ranks(intermediate);
      var IV = makeDistArray(vD, int);
      forall (IVi, idx) in zip(IV, deltaIV) with (var agg = newSrcAggregator(int)) {
          agg.copy(IVi, firstIV[idx]);
      }
      var sortedVals = makeDistArray(vD, t);
      forall (sv, idx) in zip(sortedVals, IV) with (var valsAgg = newSrcAggregator(t)) {
        valsAgg.copy(sv, noNanVals[idx]);
      }

      var tmp1 = makeDistArray(D, t);
      var tmp = makeDistArray(D, t);
      forall (s, c, r, t1, t2) in zip(segments, counts, res, tmp1, tmp2) with (var resAgg = newSrcAggregator(t)) {
        if c % 2 != 0 {
          // odd case: grab middle of sorted values
          resAgg.copy(t1, sortedVals[s + (c + 1)/2 - 1]);
        }
        else {
          // even case: average the 2 middles of the sorted values
          resAgg.copy(t1, sortedVals[s + c/2 - 1]);
          resAgg.copy(t2, sortedVals[s + c/2]);
        }
      }

      forall (c, r, t1, t2) in zip(counts, res, tmp1, tmp2) {
        if c % 2 != 0 {
          r = t1:real;
        } else {
          r = ((t1+t2):real)/2.0;
        }
      }

      return res;
    }

    proc segMin(values:[?vD] ?t, segments:[?D] int, skipNan=false): [D] t throws {
      var res: [D] t = if t != bigint then max(t) else (1:bigint << class_lvl_max_bits) - 1;
      if (D.size == 0) { return res; }
      var keys = expandKeys(vD, segments);
      var kv: [keys.domain] (int, t);
      if (isRealType(t) && skipNan) {
        var arrCopy = [elem in values] if isNan(elem) then max(real) else elem;
        kv = [(k, v) in zip(keys, arrCopy)] (-k, v);
      } else {
        kv = [(k, v) in zip(keys, values)] (-k, v);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      if t != bigint {
        // TODO update when we have a better way to handle bigint mem estimation
        overMemLimit((numBytes(t) + numBytes(int)) * kv.size);
      }
      var cummin = makeDistArray(keys.domain, (int,t));
      if t != bigint {
        cummin = min scan kv;
      } else {
        cummin = segmentedBigintMinScanOp scan kv;
      }
      forall (i, r, low) in zip(D, res, segments) with (var agg = newSrcAggregator(t)) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          agg.copy(r, cummin[vi][1]);
        }
      }
      return res;
    }

    class segmentedBigintMinScanOp: ReduceScanOp {
      type eltType;
      const max_val: bigint = (1:bigint << class_lvl_max_bits) - 1;
      var value = (max(int), max_val);

      proc identity {
        return (max(int), max_val);
      }
      proc accumulate(x) {
        value = min(x, value);
      }
      proc accumulateOntoState(ref state, x) {
        state = min(state, x);
      }
      proc combine(x) {
        value = min(value, x.value);
      }
      proc generate() {
        return value;
      }
      proc clone() {
        return new unmanaged segmentedBigintMinScanOp(eltType=eltType);
      }
    }

    proc segMax(values:[?vD] ?t, segments:[?D] int, skipNan=false): [D] t throws {
      var res: [D] t = if t != bigint then min(t) else -(1:bigint << class_lvl_max_bits);
      if (D.size == 0) { return res; }
      var keys = expandKeys(vD, segments);
      var kv = makeDistArray(keys.domain, (int, t));
      if (isRealType(t) && skipNan) {
        var arrCopy = [elem in values] if isNan(elem) then min(real) else elem;
        kv = [(k, v) in zip(keys, arrCopy)] (k, v);
      } else {
        kv = [(k, v) in zip(keys, values)] (k, v);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      if t != bigint {
        // TODO update when we have a better way to handle bigint mem estimation
        overMemLimit((numBytes(t) + numBytes(int)) * kv.size);
      }
      var cummax = makeDistArray(keys.domain, (int,t));
      if t != bigint {
        cummax = max scan kv;
      } else {
        cummax = segmentedBigintMaxScanOp scan kv;
      }
      
      forall (i, r, low) in zip(D, res, segments) with (var agg = newSrcAggregator(t)) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          agg.copy(r, cummax[vi][1]);
        }
      }
      return res;
    }

    class segmentedBigintMaxScanOp: ReduceScanOp {
      type eltType;
      const min_val: bigint = -(1:bigint << class_lvl_max_bits);
      var value = (min(int), min_val);

      proc identity {
        return (min(int), min_val);
      }
      proc accumulate(x) {
        value = max(x, value);
      }
      proc accumulateOntoState(ref state, x) {
        state = max(state, x);
      }
      proc combine(x) {
        value = max(value, x.value);
      }
      proc generate() {
        return value;
      }
      proc clone() {
        return new unmanaged segmentedBigintMaxScanOp(eltType=eltType);
      }
    }

    proc segArgmin(values:[?vD] ?t, segments:[?D] int): ([D] t, [D] int) throws {
      var locs = makeDistArray(D, int);
      var vals = makeDistArray(D, max(t));
      if (D.size == 0) { return (vals, locs); }
      var keys = expandKeys(vD, segments);
      var kvi = [(k, v, i) in zip(keys, values, vD)] ((-k, v), i);
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * kvi.size);
      var cummin = minloc scan kvi;
      forall (l, v, low, i) in zip(locs, vals, segments, D)
        with (var locagg = newSrcAggregator(int), var valagg = newSrcAggregator(t)) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          // ((_,v),_) = cummin[vi];
          valagg.copy(v, cummin[vi][0][1]);
          // (_    ,l) = cummin[vi];
          locagg.copy(l, cummin[vi][1]);
        }
      }
      return (vals, locs);
    }

    proc segArgmax(values:[?vD] ?t, segments:[?D] int): ([D] t, [D] int) throws {
      var locs = makeDistArray(D, int);
      var vals = makeDistArray(D, min(t));
      if (D.size == 0) { return (vals, locs); }
      var keys = expandKeys(vD, segments);
      var kvi = [(k, v, i) in zip(keys, values, vD)] ((k, v), i);
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * kvi.size);
      var cummax = maxloc scan kvi;
      forall (l, v, low, i) in zip(locs, vals, segments, D)
        with (var locagg = newSrcAggregator(int), var valagg = newSrcAggregator(t)) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          // ((_,v),_) = cummax[vi];
          valagg.copy(v, cummax[vi][0][1]);
          // (_,    l) = cummax[vi];
          locagg.copy(l, cummax[vi][1]);
        }
      }
      return (vals, locs);
    }

    proc segAny(values:[] bool, segments:[?D] int): [D] bool throws {
      var res = makeDistArray(D, bool);
      if (D.size == 0) { return res; }
      const sums = segSum(values, segments);
      res = (sums > 0);
      return res;
    }

    proc segAll(values:[] bool, segments:[?D] int): [D] bool throws {
      var res = makeDistArray(D, bool);
      if (D.size == 0) { return res; }
      const sums = segSum(values, segments);
      const lengths = segCount(segments, values.domain.high + 1);
      res = (sums == lengths);
      return res;
    }

    proc segOr(values:[?vD] ?t, segments:[?D] int): [D] t throws {
      var res = makeDistArray(D, t);
      if (D.size == 0) { return res; }
      // Set reset flag at segment boundaries
      var flagvalues: [vD] (bool, t) = [v in values] (false, v);
      forall s in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(flagvalues[s][0], true);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      if t != bigint {
        // TODO update when we have a better way to handle bigint mem estimation
        overMemLimit((numBytes(t)+1) * flagvalues.size);
      }
      // Scan with custom operator, which resets the bitwise AND
      // at segment boundaries.
      const scanresult = ResettingOrScanOp scan flagvalues;
      // Read the results from the last element of each segment
      forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(t)) {
        agg.copy(r, scanresult[s-1](1));
      }
      res[D.high] = scanresult[vD.high](1);
      return res;
    }

    /* Performs a bitwise OR scan, controlled by a reset flag. While
     * the reset flag is false, the accumulation of values proceeds as 
     * normal. When a true is encountered, the state resets to the
     * identity. */
    class ResettingOrScanOp: ReduceScanOp {
      type eltType;
      /* value is a tuple comprising a flag and the actual result of 
         segmented bitwise OR. 

         The meaning of the flag depends on whether it belongs to an 
         array element yet to be scanned or to an element that has 
         already been scanned (including the internal state of a class
         instance doing the scanning). For elements yet to be scanned,
         the flag means "reset to the identity here". For elements that
         have already been scanned, or for internal state, the flag means 
         "there has already been a reset in the computation of this value".
      */
      var value: eltType;

      proc identity {
        return if eltType == (bool, int) then (false, 0) else if eltType == (bool, uint) then (false, 0:uint) else (false, 0:bigint);
      }

      proc accumulate(x) {
        // Assume x is an element that has not yet been scanned, and
        // that it comes after the current state.
        const (reset, other) = x;
        const (hasReset, v) = value;
        // x's reset flag controls whether value gets replaced or combined
        // also update this instance's "hasReset" flag with x's reset flag
        value = (hasReset | reset, if reset then other else (v | other));
      }

      proc accumulateOntoState(ref state, x) {
        // Assume state is an element that has already been scanned,
        // and x is an update from a previous boundary.
        const (prevReset, other) = x;
        const (hasReset, v) = state;
        // absorb reset history
        // If state has already encountered a reset, then it should
        // ignore x's value
        state = (hasReset | prevReset, if hasReset then v else (v | other));
      }

      proc combine(x) {
        // Assume x is an instance that scanned a prior chunk.
        const (xHasReset, other) = x.value;
        const (hasReset, v) = value;
        // Since current instance is absorbing x's history,
        // xHasReset flag should be ORed in.
        // But if current instance has already encountered a reset,
        // then it should ignore x's value.
        value = (hasReset | xHasReset, if hasReset then v else (v | other));
      }

      proc generate() {
        return value;
      }

      proc clone() {
        return new unmanaged ResettingOrScanOp(eltType=eltType);
      }
    }

    proc segAnd(values:[?vD] ?t, segments:[?D] int): [D] t throws {
      var res = makeDistArray(D, int);
      if (D.size == 0) { return res; }
      // Set reset flag at segment boundaries
      var flagvalues: [vD] (bool, t) = [v in values] (false, v);
      forall s in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(flagvalues[s][0], true);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      if t != bigint {
        // TODO update when we have a better way to handle bigint mem estimation
        overMemLimit((numBytes(t)+1) * flagvalues.size);
      }
      // Scan with custom operator, which resets the bitwise AND
      // at segment boundaries.
      const scanresult = ResettingAndScanOp scan flagvalues;
      // Read the results from the last element of each segment
      forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(t)) {
        agg.copy(r, scanresult[s-1](1));
      }
      res[D.high] = scanresult[vD.high](1);
      return res;
    }

    /* Performs a bitwise AND scan, controlled by a reset flag. While
     * the reset flag is false, the accumulation of values proceeds as 
     * normal. When a true is encountered, the state resets to the
     * identity. */
    class ResettingAndScanOp: ReduceScanOp {
      type eltType;
      /* value is a tuple comprising a flag and the actual result of 
         segmented bitwise AND. 

         The meaning of the flag depends on
         whether it belongs to an array element yet to be scanned or 
         to an element that has already been scanned (or the state of
         an instance doing the scanning). For elements yet to be scanned,
         the flag means "reset to the identity here". For elements that
         have already been scanned, or for internal state, the flag means 
         "there has already been a reset in the computation of this value".
      */
      const max_val: bigint = (1:bigint << class_lvl_max_bits) - 1;
      var value = if eltType == (bool, int) then (false, 0xffffffffffffffff:int) else if eltType == (bool, uint) then (false, 0xffffffffffffffff:uint) else (false, max_val);

      proc identity {
        return if eltType == (bool, int) then (false, 0xffffffffffffffff:int) else if eltType == (bool, uint) then (false, 0xffffffffffffffff:uint) else (false, max_val);
      }

      proc accumulate(x) {
        // Assume x is an element that has not yet been scanned, and
        // that it comes after the current state.
        const (reset, other) = x;
        const (hasReset, v) = value;
        // x's reset flag controls whether value gets replaced or combined
        // also update this instance's "hasReset" flag with x's reset flag
        value = (hasReset | reset, if reset then other else (v & other));
      }

      proc accumulateOntoState(ref state, x) {
        // Assume state is an element that has already been scanned,
        // and x is an update from a previous boundary.
        const (prevReset, other) = x;
        const (hasReset, v) = state;
        // absorb reset history
        // If state has already encountered a reset, then it should
        // ignore x's value
        state = (hasReset | prevReset, if hasReset then v else (v & other));
      }

      proc combine(x) {
        // Assume x is an instance that scanned a prior chunk.
        const (xHasReset, other) = x.value;
        const (hasReset, v) = value;
        // Since current instance is absorbing x's history,
        // xHasReset flag should be ORed in.
        // But if current instance has already encountered a reset,
        // then it should ignore x's value.
        value = (hasReset | xHasReset, if hasReset then v else (v & other));
      }

      proc generate() {
        return value;
      }

      proc clone() {
        return new unmanaged ResettingAndScanOp(eltType=eltType);
      }
    }
    

    proc segXor(values:[] ?t, segments:[?D] int) throws {
      // Because XOR has an inverse (itself), this can be
      // done with a scan like segSum
      var res = makeDistArray(D, t);
      if (D.size == 0) { return res; }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(t) * values.size);
      var cumxor = ^ scan values;
      // Iterate over segments
      var rightvals = makeDistArray(D, t);
      forall (i, r) in zip(D, rightvals) with (var agg = newSrcAggregator(t)) {
        // Find the segment boundaries
        if (i == D.high) {
          agg.copy(r, cumxor[values.domain.high]);
        } else {
          agg.copy(r, cumxor[segments[i+1] - 1]);
        }
      }
      res[D.low] = rightvals[D.low];
      res[D.low+1..] = rightvals[D.low+1..] ^ rightvals[..D.high-1];
      return res;
    }

    proc expandKeys(kD, segments: [?sD] int): [kD] int throws {
      var truth = makeDistArray(kD, bool);
      forall i in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(truth[i], true);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * truth.size);
      var keys = (+ scan truth) - 1;
      return keys;
    }

    proc segNumUnique(values: [?kD] ?t, segments: [?sD] int) throws {
      var res = makeDistArray(sD, int);
      if (sD.size == 0) {
        return res;
      }
      var keys = expandKeys(kD, segments);
      // sort keys and values together
      var t1 = Time.timeSinceEpoch().totalSeconds();
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                         "Sorting keys and values...");
      /* var toSort = [(k, v) in zip(keys, values)] (k, v); */
      /* Sort.TwoArrayRadixSort.twoArrayRadixSort(toSort); */
      var firstIV = radixSortLSD_ranks(values);
      var intermediate = makeDistArray(kD, int);
      forall (ii, idx) in zip(intermediate, firstIV) with (var agg = newSrcAggregator(int)) {
          agg.copy(ii, keys[idx]);
      }
      var deltaIV = radixSortLSD_ranks(intermediate);
      var IV = makeDistArray(kD, int);
      forall (IVi, idx) in zip(IV, deltaIV) with (var agg = newSrcAggregator(int)) {
          agg.copy(IVi, firstIV[idx]);
      }
      var sortedKV = makeDistArray(kD, (int,t));
      forall ((kvi0,kvi1), idx) in zip(sortedKV, IV) with (var keysAgg = newSrcAggregator(int),
                                                           var valsAgg = newSrcAggregator(t)) {
        keysAgg.copy(kvi0, keys[idx]);
        valsAgg.copy(kvi1, values[idx]);
      }
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "sort time = %i".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Finding unique (key, value) pairs...");
      var truth = makeDistArray(kD, bool);
      // true where new (k, v) pair appears
      [(tr, (_,val), i) in zip(truth, sortedKV, kD)] if i > kD.low {
        const (_,sortedVal) = sortedKV[i-1];
        tr = (sortedVal != val);
      }
      // first value of every segment is automatically new
      [s in segments] truth[s] = true;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * truth.size);
      // count cumulative new values and take diffs at segment boundaries
      var count: [kD] int = (+ scan truth);
      var pop = count[kD.high];
      // find steps to get unique (key, val) pairs
      var hD: domain(1) dmapped blockDist(boundingBox={0..#pop}) = {0..#pop};
      // save off only the key from each pair (now there will be nunique of each key)
      var keyhits = makeDistArray(hD, int);
      forall i in truth.domain with (var agg = newDstAggregator(int)) {
        if (truth[i]) {
          var (key,_) = sortedKV[i];
          agg.copy(keyhits[count[i]-1], key);
        }
      }
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "time = %i".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Finding unique keys and num unique vals per key.");
      // find steps in keys
      var truth2 = makeDistArray(hD, bool);
      truth2[hD.low] = true;
      [(tr, k, i) in zip(truth2, keyhits, hD)] if (i > hD.low) { tr = (keyhits[i-1] != k); }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * truth2.size);
      var kiv: [hD] int = (+ scan truth2);
      var nKeysPresent = kiv[hD.high];
      var nD: domain(1) dmapped blockDist(boundingBox={0..#(nKeysPresent+1)}) = {0..#(nKeysPresent+1)};
      // get step indices and take diff to get number of times each key appears
      var stepInds = makeDistArray(nD, int);
      stepInds[nKeysPresent] = keyhits.size;
      forall i in hD with (var agg = newDstAggregator(int)) {
        if (truth2[i]) {
          var idx = i;
          agg.copy(stepInds[kiv[i]-1], idx);
        }
      }
      var nunique = stepInds[1..#nKeysPresent] - stepInds[0..#nKeysPresent];
      // if every key is present, we're done
      if (nKeysPresent == sD.size) {
        res = nunique;
      } else { // need to skip over non-present keys
        var segSizes = makeDistArray(sD, int);
        segSizes[sD.low..sD.high-1] = segments[sD.low+1..sD.high] - segments[sD.low..sD.high-1];
        segSizes[sD.high] = kD.high - segments[sD.high] + 1;
        var idx = 0;
        for (r, s) in zip(res, segSizes) {
          if (s > 0) {
            r = nunique[idx];
            idx += 1;
          }
        }
      }
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "time = %i".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
      return res;
    }

    use CommandMap;
    registerFunction("segmentedReduction", segmentedReductionMsg, getModuleName());
    registerFunction("reduction", reductionMsg, getModuleName());
    registerFunction("countReduction", countReductionMsg, getModuleName());
}
