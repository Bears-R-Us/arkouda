module ReductionMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;
    use CommAggregation;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use Reflection;
    use Errors;
    use Logging;
    use Message;

    use AryUtil;
    use PrivateDist;
    use RadixSortLSD;

    private config const lBins = 2**25 * numLocales;

    const rmLogger = new Logger();
  
    if v {
        rmLogger.level = LogLevel.DEBUG;
    } else {
        rmLogger.level = LogLevel.INFO;
    }

    // these functions take an array and produce a scalar
    // parse and respond to reduction message
    // scalar = reductionop(vector)
    proc reductionMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string = ""; // response message
        // split request into fields
        var (reductionop, name) = payload.splitMsgToTuple(2);
        rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s reductionop: %s name: %s".format(cmd,reductionop,name));

        var gEnt: borrowed GenSymEntry = st.lookup(name);
       
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum != 0 {val = "True";} else {val = "False";}
                        repMsg = "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                       repMsg = "bool %s".format(val);
                    }
                    when "sum" {
                        var val = + reduce e.a;
                        repMsg = "int64 %i".format(val);
                    }
                    when "prod" {
                        // Cast to real to avoid int64 overflow
                        var val = * reduce e.a:real;
                        // Return value is always float64 for prod
                        repMsg = "float64 %.17r".format(val);
                    }
                    when "min" {
                      var val = min reduce e.a;
                      repMsg = "int64 %i".format(val);
                    }
                    when "max" {
                        var val = max reduce e.a;
                        repMsg = "int64 %i".format(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        repMsg = "int64 %i".format(minLoc);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        repMsg = "int64 %i".format(maxLoc);
                    }
                    when "is_sorted" {
                        ref ea = e.a;
                        var sorted = isSorted(ea);
                        var val: string;
                        if sorted {val = "True";} else {val = "False";}
                        repMsg = "bool %s".format(val);
                    }
                    when "is_locally_sorted" {
                      var locSorted: [LocaleSpace] bool;
                      coforall loc in Locales {
                        on loc {
                          ref myA = e.a[e.a.localSubdomain()];
                          locSorted[here.id] = isSorted(myA);
                        }
                      }

                      var val: string;
                      if (& reduce locSorted) {val = "True";} else {val = "False";}

                      repMsg = "bool %s".format(val);
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
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum != 0.0 {val = "True";} else {val = "False";}
                        repMsg = "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                        repMsg = "bool %s".format(val);
                    }
                    when "sum" {
                        var val = + reduce e.a;
                        repMsg = "float64 %.17r".format(val);
                    }
                    when "prod" {
                        var val = * reduce e.a;
                        repMsg =  "float64 %.17r".format(val);
                    }
                    when "min" {
                        var val = min reduce e.a;
                        repMsg = "float64 %.17r".format(val);
                    }
                    when "max" {
                        var val = max reduce e.a;
                        repMsg = "float64 %.17r".format(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        repMsg = "int64 %i".format(minLoc);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        repMsg = "int64 %i".format(maxLoc);
                    }
                    when "is_sorted" {
                        var sorted = isSorted(e.a);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        repMsg = "bool %s".format(val);
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
                        repMsg = "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var all = & reduce e.a;
                        if all {val = "True";} else {val = "False";}
                        repMsg = "bool %s".format(val);
                    }
                    when "sum" {
                        var val = + reduce e.a:int;
                        repMsg = "int64 %i".format(val);
                    }
                    when "prod" {
                        var val = * reduce e.a:int;
                        repMsg = "int64 %i".format(val);
                    }
                    when "min" {
                        var val:string;
                        if (& reduce e.a) { val = "True"; } else { val = "False"; }
                        repMsg = "bool %s".format(val);
                    }
                    when "max" {
                        var val:string;
                        if (| reduce e.a) { val = "True"; } else { val = "False"; }
                        repMsg = "bool %s".format(val);
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

    proc countReductionMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
      // reqMsg: segmentedReduction values segments operator
      // 'segments_name' describes the segment offsets
      var (segments_name, sizeStr) = payload.splitMsgToTuple(2);
      var size = try! sizeStr:int;
      var rname = st.nextName();
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "cmd: %s segments_name: %s size: %s".format(cmd,segments_name, size));

      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {    
          var errorMsg = "Array of segment offsets must be int dtype";
          rmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);       
          return new MsgTuple(errorMsg, MsgType.ERROR); 
      }
      var counts = segCount(segments.a, size);
      st.addEntry(rname, new shared SymEntry(counts));

      var repMsg = "created " + st.attrib(rname);
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);       
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc segCount(segments:[?D] int, upper: int):[D] int {
      var counts:[D] int;
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
    
    proc segmentedReductionMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        // reqMsg: segmentedReduction values segments operator
        // 'values_name' is the segmented array of values to be reduced
        // 'segments_name' is the sement offsets
        // 'op' is the reduction operator
        var (values_name, segments_name, op, skip_nan) = payload.splitMsgToTuple(4);
        var skipNan = stringtobool(skip_nan);
      
        var rname = st.nextName();
        rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "cmd: %s values_name: %s segments_name: %s operator: %s skipNan: %s".format(
                                       cmd,values_name,segments_name,op,skipNan));
        var gVal: borrowed GenSymEntry = st.lookup(values_name);
        var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
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
                        st.addEntry(rname, new shared SymEntry(res));
                    } 
                    when "prod" {
                        var res = segProduct(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "mean" {
                        var res = segMean(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "min" {
                        var res = segMin(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "max" {
                        var res = segMax(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "argmin" {
                        var (vals, locs) = segArgmin(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(locs));
                    }
                    when "argmax" {
                        var (vals, locs) = segArgmax(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(locs));
                    }
                    when "or" {
                        var res = segOr(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "and" {
                        var res = segAnd(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "xor" {
                        var res = segXor(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "nunique" {
                        var res = segNumUnique(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(res));
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
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "prod" {
                        var res = segProduct(values.a, segments.a, skipNan);
                        st.addEntry(rname, new shared SymEntry(res));
                    } 
                    when "mean" {
                        var res = segMean(values.a, segments.a, skipNan);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "min" {
                        var res = segMin(values.a, segments.a, skipNan);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "max" {
                        var res = segMax(values.a, segments.a, skipNan);
                        st.addEntry(rname, new shared SymEntry(res));
                    }
                    when "argmin" {
                        var (vals, locs) = segArgmin(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(locs));
                    }
                    when "argmax" {
                        var (vals, locs) = segArgmax(values.a, segments.a);
                        st.addEntry(rname, new shared SymEntry(locs));
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
                      st.addEntry(rname, new shared SymEntry(res));
                   }
                   when "any" {
                      var res = segAny(values.a, segments.a);
                      st.addEntry(rname, new shared SymEntry(res));
                   }
                   when "all" {
                      var res = segAll(values.a, segments.a);
                      st.addEntry(rname, new shared SymEntry(res));
                   }
                   when "mean" {
                      var res = segMean(values.a, segments.a);
                      st.addEntry(rname, new shared SymEntry(res));
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
    proc segSum(values:[] ?intype, segments:[?D] int, skipNan=false) {
      type t = if intype == bool then int else intype;
      var res: [D] t;
      if (D.size == 0) { return res; }
      var cumsum;
      if (isFloatType(t) && skipNan) {
        var arrCopy = [elem in values] if isnan(elem) then 0.0 else elem;
        cumsum = + scan arrCopy;
      }
      else {
        cumsum = + scan values;
      }
      // Iterate over segments
      var rightvals: [D] t;
      forall (i, r) in zip(D, rightvals) with (var agg = newSrcAggregator(t)) {
        // Find the segment boundaries
        if (i == D.high) {
          agg.copy(r, cumsum[values.domain.high]);
        } else {
          agg.copy(r, cumsum[segments[i+1] - 1]);
        }
      }
      res[D.low] = rightvals[D.low];
      res[D.low+1..] = rightvals[D.low+1..] - rightvals[..D.high-1];
      return res;
    }

    proc segProduct(values:[] ?t, segments:[?D] int, skipNan=false): [D] real {
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
      var res: [D] real = 0.0;
      if (D.size == 0) { return res; }
      const isZero = (values == 0);
      // Take absolute value, replacing zeros with ones
      // Ones will become zeros in log-domain and not affect + scan
      var magnitudes: [values.domain] real;
      if (isFloatType(t) && skipNan) {
        forall (m, v, z) in zip(magnitudes, values, isZero) {
          if isnan(v) {
            m = 1.0;
          } else {
            m = Math.abs(v) + z:real;
          }
        }
      } else {
        magnitudes = Math.abs(values) + isZero:real;
      }
      var logs = Math.log(magnitudes);
      var negatives = (Math.sgn(values) == -1);
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
          r = sign * Math.exp(v);
        }
      }
      return res;
    }

    proc segMean(values:[] ?t, segments:[?D] int, skipNan=false): [D] real {
      var res: [D] real;
      if (D.size == 0) { return res; }
      var sums;
      var counts;
      if (isFloatType(t) && skipNan) {
        // count cumulative nans over all values
        var cumnans = isnan(values):int;
        cumnans = + scan cumnans;
        
        // find cumulative nans at segment boundaries
        var segnans: [D] int;
        forall (si, sn) in zip(D, segnans) with (var agg = newSrcAggregator(int)) {
          if si == D.high {
            agg.copy(sn, cumnans[cumnans.domain.high]); 
          } else {
            agg.copy(sn, cumnans[segments[si+1]-1]);
          }
        }
        
        // take diffs of adjacent segments to find nan count in each segment
        var nancounts: [D] int;
        nancounts[D.low] = segnans[D.low];
        nancounts[D.low+1..] = segnans[D.low+1..] - segnans[..D.high-1];
        
        // calculate sum and counts with nan values replaced with 0.0
        var arrCopy = [elem in values] if isnan(elem) then 0.0 else elem;
        sums = segSum(arrCopy, segments);
        counts = segCount(segments, values.size) - nancounts;
      } else {
        sums = segSum(values, segments);
        counts = segCount(segments, values.size);
      }
      forall (r, s, c) in zip(res, sums, counts) {
        if (c > 0) {
          r = s:real / c:real;
        }
      }
      return res;
    }

    proc segMin(values:[?vD] ?t, segments:[?D] int, skipNan=false): [D] t {
      var res: [D] t = max(t);
      if (D.size == 0) { return res; }
      var keys = expandKeys(vD, segments);
      var kv: [keys.domain] (int, t);
      if (isFloatType(t) && skipNan) {
        var arrCopy = [elem in values] if isnan(elem) then max(real) else elem;
        kv = [(k, v) in zip(keys, arrCopy)] (-k, v);
      } else {
        kv = [(k, v) in zip(keys, values)] (-k, v);
      }
      var cummin = min scan kv;
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
    
    proc segMax(values:[?vD] ?t, segments:[?D] int, skipNan=false): [D] t {
      var res: [D] t = min(t);
      if (D.size == 0) { return res; }
      var keys = expandKeys(vD, segments);
      var kv: [keys.domain] (int, t);
      if (isFloatType(t) && skipNan) {
        var arrCopy = [elem in values] if isnan(elem) then min(real) else elem;
        kv = [(k, v) in zip(keys, arrCopy)] (k, v);
      } else {
        kv = [(k, v) in zip(keys, values)] (k, v);
      }
      var cummax = max scan kv;
      
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

    proc segArgmin(values:[?vD] ?t, segments:[?D] int): ([D] t, [D] int) {
      var locs: [D] int;
      var vals: [D] t = max(t);
      if (D.size == 0) { return (vals, locs); }
      var keys = expandKeys(vD, segments);
      var kvi = [(k, v, i) in zip(keys, values, vD)] ((-k, v), i);
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

    proc segArgmax(values:[?vD] ?t, segments:[?D] int): ([D] t, [D] int) {
      var locs: [D] int;
      var vals: [D] t = min(t);
      if (D.size == 0) { return (vals, locs); }
      var keys = expandKeys(vD, segments);
      var kvi = [(k, v, i) in zip(keys, values, vD)] ((k, v), i);
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

    proc segAny(values:[] bool, segments:[?D] int): [D] bool {
      var res: [D] bool;
      if (D.size == 0) { return res; }
      const sums = segSum(values, segments);
      res = (sums > 0);
      return res;
    }

    proc segAll(values:[] bool, segments:[?D] int): [D] bool {
      var res: [D] bool;
      if (D.size == 0) { return res; }
      const sums = segSum(values, segments);
      const lengths = segCount(segments, values.domain.high + 1);
      res = (sums == lengths);
      return res;
    }

    proc segOr(values:[?vD] int, segments:[?D] int): [D] int {
      var res: [D] int;
      // Set reset flag at segment boundaries
      var flagvalues: [vD] (bool, int) = [v in values] (false, v);
      forall s in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(flagvalues[s][0], true);
      }
      // Scan with custom operator, which resets the bitwise AND
      // at segment boundaries.
      const scanresult = ResettingOrScanOp scan flagvalues;
      // Read the results from the last element of each segment
      forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(int)) {
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
      var value = (false, 0);

      proc identity return (false, 0);

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
        const (_, other) = x;
        const (hasReset, v) = state;
        // x's hasReset flag does not matter
        // If state has already encountered a reset, then it should
        // ignore x's value
        state = (hasReset, if hasReset then v else (v | other));
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

    proc segAnd(values:[?vD] int, segments:[?D] int): [D] int {
      var res: [D] int;
      // Set reset flag at segment boundaries
      var flagvalues: [vD] (bool, int) = [v in values] (false, v);
      forall s in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(flagvalues[s][0], true);
      }
      // Scan with custom operator, which resets the bitwise AND
      // at segment boundaries.
      const scanresult = ResettingAndScanOp scan flagvalues;
      // Read the results from the last element of each segment
      forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(int)) {
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
      var value = (false, 0xffffffffffffffff:int);

      proc identity return (false, 0xffffffffffffffff:int);

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
        const (_, other) = x;
        const (hasReset, v) = state;
        // x's hasReset flag does not matter
        // If state has already encountered a reset, then it should
        // ignore x's value
        state = (hasReset, if hasReset then v else (v & other));
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
    

    proc segXor(values:[] int, segments:[?D] int) {
      // Because XOR has an inverse (itself), this can be
      // done with a scan like segSum
      var res: [D] int;
      if (D.size == 0) { return res; }
      var cumxor = ^ scan values;
      // Iterate over segments
      var rightvals: [D] int;
      forall (i, r) in zip(D, rightvals) with (var agg = newSrcAggregator(int)) {
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

    proc expandKeys(kD, segments: [?sD] int): [kD] int {
      var truth: [kD] bool;
      forall i in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(truth[i], true);
      }
      var keys = (+ scan truth) - 1;
      return keys;
    }

    proc segNumUnique(values: [?kD] int, segments: [?sD] int) throws {
      var res: [sD] int;
      if (sD.size == 0) {
        return res;
      }
      var keys = expandKeys(kD, segments);
      // sort keys and values together
      var t1 = Time.getCurrentTime();
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                         "Sorting keys and values...");
      /* var toSort = [(k, v) in zip(keys, values)] (k, v); */
      /* Sort.TwoArrayRadixSort.twoArrayRadixSort(toSort); */
      var firstIV = radixSortLSD_ranks(values);
      var intermediate: [kD] int;
      forall (ii, idx) in zip(intermediate, firstIV) with (var agg = newSrcAggregator(int)) {
          agg.copy(ii, keys[idx]);
      }
      var deltaIV = radixSortLSD_ranks(intermediate);
      var IV: [kD] int;
      forall (IVi, idx) in zip(IV, deltaIV) with (var agg = newSrcAggregator(int)) {
          agg.copy(IVi, firstIV[idx]);
      }
      var sortedKV: [kD] (int, int);
      forall ((kvi0,kvi1), idx) in zip(sortedKV, IV) with (var agg = newSrcAggregator(int)) {
        agg.copy(kvi0, keys[idx]);
        agg.copy(kvi1, values[idx]);
      }
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "sort time = %i".format(Time.getCurrentTime() - t1));
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Finding unique (key, value) pairs...");
      var truth: [kD] bool;
      // true where new (k, v) pair appears
      [(t, (_,val), i) in zip(truth, sortedKV, kD)] if i > kD.low {
        const (_,sortedVal) = sortedKV[i-1];
        t = (sortedVal != val);
      }
      // first value of every segment is automatically new
      [s in segments] truth[s] = true;
      // count cumulative new values and take diffs at segment boundaries
      var count: [kD] int = (+ scan truth);
      var pop = count[kD.high];
      // find steps to get unique (key, val) pairs
      var hD: domain(1) dmapped Block(boundingBox={0..#pop}) = {0..#pop};
      // save off only the key from each pair (now there will be nunique of each key)
      var keyhits: [hD] int;
      forall i in truth.domain with (var agg = newDstAggregator(int)) {
        if (truth[i]) {
          var (key,_) = sortedKV[i];
          agg.copy(keyhits[count[i]-1], key);
        }
      }
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "time = %i".format(Time.getCurrentTime() - t1));
      rmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Finding unique keys and num unique vals per key.");
      // find steps in keys
      var truth2: [hD] bool;
      truth2[hD.low] = true;
      [(t, k, i) in zip(truth2, keyhits, hD)] if (i > hD.low) { t = (keyhits[i-1] != k); }
      var kiv: [hD] int = (+ scan truth2);
      var nKeysPresent = kiv[hD.high];
      var nD: domain(1) dmapped Block(boundingBox={0..#(nKeysPresent+1)}) = {0..#(nKeysPresent+1)};
      // get step indices and take diff to get number of times each key appears
      var stepInds: [nD] int;
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
        var segSizes: [sD] int;
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
                                                   "time = %i".format(Time.getCurrentTime() - t1));
      return res;
    }

    proc stringtobool(str: string): bool throws {
      if str == "True" then return true;
      else if str == "False" then return false;
      throw new owned ErrorWithMsg("message: skipNan must be of type bool");
    }
}

