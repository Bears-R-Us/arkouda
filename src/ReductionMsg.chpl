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
    use PerLocaleHelper;

    use AryUtil;
    use PrivateDist;
    use RadixSortLSD;

    private config const reductionDEBUG = false;
    private config const lBins = 2**25 * numLocales;
      
    // these functions take an array and produce a scalar
    // parse and respond to reduction message
    // scalar = reductionop(vector)
    proc reductionMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (reductionop, name) = payload.decode().splitMsgToTuple(2);
        if v {try! writeln("%s %s %s".format(cmd,reductionop,name));try! stdout.flush();}

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
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var val = + reduce e.a;
                        return try! "int64 %i".format(val);
                    }
                    when "prod" {
                      ref ea = e.a;
                      // If any element is zero, skip the computation and return 0.0
                      var val: real = 0.0;
                      if (&& reduce (ea != 0)) {
                        // Cast to real to avoid int64 overflow
                        val = * reduce ea:real;
                      }
                        // Return value is always float64 for prod
                        return try! "float64 %.17r".format(val);
                    }
                    when "min" {
                      var val = min reduce e.a;
                      return try! "int64 %i".format(val);
                    }
                    when "max" {
                        var val = max reduce e.a;
                        return try! "int64 %i".format(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        return try! "int64 %i".format(minLoc);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        return try! "int64 %i".format(maxLoc);
                    }
                    when "is_sorted" {
                        ref ea = e.a;
                        var sorted = isSorted(ea);
                        var val: string;
                        if sorted {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
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
                      return try! "bool %s".format(val);
                    }
                    otherwise {return notImplementedError(pn,reductionop,gEnt.dtype);}
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
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var val = + reduce e.a;
                        return try! "float64 %.17r".format(val);
                    }
                    when "prod" {
                        var val = * reduce e.a;
                        return try! "float64 %.17r".format(val);
                    }
                    when "min" {
                        var val = min reduce e.a;
                        return try! "float64 %.17r".format(val);
                    }
                    when "max" {
                        var val = max reduce e.a;
                        return try! "float64 %.17r".format(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        return try! "int64 %i".format(minLoc);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        return try! "int64 %i".format(maxLoc);
                    }
                    when "is_sorted" {
                        var sorted = isSorted(e.a);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    otherwise {return notImplementedError(pn,reductionop,gEnt.dtype);}
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
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var all = & reduce e.a;
                        if all {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var val = + reduce e.a:int;
                        return try! "int64 %i".format(val);
                    }
                    when "prod" {
                        var val = * reduce e.a:int;
                        return try! "int64 %i".format(val);
                    }
                    when "min" {
                        var val:string;
                        if (& reduce e.a) { val = "True"; } else { val = "False"; }
                        return try! "bool %s".format(val);
                    }
                    when "max" {
                        var val:string;
                        if (| reduce e.a) { val = "True"; } else { val = "False"; }
                        return try! "bool %s".format(val);
                    }
                    otherwise {return notImplementedError(pn,reductionop,gEnt.dtype);}
                }
            }
            otherwise {return unrecognizedTypeError(pn, dtype2str(gEnt.dtype));}
        }
    }

    proc countReductionMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
      // reqMsg: segmentedReduction values segments operator
      // 'segments_name' describes the segment offsets
      var (segments_name, sizeStr) = payload.decode().splitMsgToTuple(2);
      var size = try! sizeStr:int;
      var rname = st.nextName();
      if v {try! writeln("%s %s %s".format(cmd,segments_name, size));try! stdout.flush();}

      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      var counts = segCount(segments.a, size);
      st.addEntry(rname, new shared SymEntry(counts));
      return try! "created " + st.attrib(rname);
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
    
    proc countLocalRdxMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
      // reqMsg: countLocalRdx segments
      // segments.size = numLocales * numKeys
      // 'segments_name' describes the segment offsets
      // 'size[Str]' is the size of the original keys array
      var (segments_name, sizeStr) = payload.decode().splitMsgToTuple(2);
      var size = try! sizeStr:int; // size of original keys array
      var rname = st.nextName();
      if v {try! writeln("%s %s %s".format(cmd,segments_name, size));try! stdout.flush();}

      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      var counts = perLocCount(segments.a, size);
      st.addEntry(rname, new shared SymEntry(counts));
      return try! "created " + st.attrib(rname);
    }

    proc perLocCount(segments:[?D] int, size: int): [] int {
      var origD = makeDistDom(size);
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var localCounts: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
        on loc {
          localCounts[here.id] = segCount(segments.localSlice[D.localSubdomain()],
                                          origD.localSubdomain().high + 1);
        }
      }
      var counts: [keyDom] int = + reduce [i in PrivateSpace] localCounts[i];
      return counts;
    }


    proc segmentedReductionMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
      // reqMsg: segmentedReduction values segments operator
      // 'values_name' is the segmented array of values to be reduced
      // 'segments_name' is the sement offsets
      // 'operator' is the reduction operator
      var (values_name, segments_name, operator, skip_nan) = payload.decode().splitMsgToTuple(4);
      var skipNan = stringtobool(skip_nan);
      
      var rname = st.nextName();
      if v {try! writeln("%s %s %s %s %s".format(cmd,values_name,segments_name,operator,skipNan));try! stdout.flush();}
      var gVal: borrowed GenSymEntry = st.lookup(values_name);
      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      select (gVal.dtype) {
      when (DType.Int64) {
        var values = toSymEntry(gVal, int);
        select operator {
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
          when "nunique" {
            var res = segNumUnique(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          otherwise {return notImplementedError(pn,operator,gVal.dtype);}
          }
      }
      when (DType.Float64) {
        var values = toSymEntry(gVal, real);
        select operator {
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
          otherwise {return notImplementedError(pn,operator,gVal.dtype);}
          }
      }
      when (DType.Bool) {
        var values = toSymEntry(gVal, bool);
        select operator {
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
          otherwise {return notImplementedError(pn,operator,gVal.dtype);}
          }
      }
      otherwise {return unrecognizedTypeError(pn, dtype2str(gVal.dtype));}
      }
      return try! "created " + st.attrib(rname);
    }

    proc segmentedLocalRdxMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
      // reqMsg: segmentedReduction keys values segments operator
      // 'values_name' is the segmented array of values to be reduced
      // 'segments_name' is the segmented offsets
      // 'operator' is the reduction operator
      var (keys_name, values_name, segments_name, operator) = payload.decode().splitMsgToTuple(4);
      var rname = st.nextName();
      if v {try! writeln("%s %s %s %s %s".format(cmd,keys_name,values_name,segments_name,operator));try! stdout.flush();}

      var gKey: borrowed GenSymEntry = st.lookup(keys_name);
      if (gKey.dtype != DType.Int64) {return unrecognizedTypeError(pn, dtype2str(gKey.dtype));}
      var keys = toSymEntry(gKey, int);
      var gVal: borrowed GenSymEntry = st.lookup(values_name);
      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      select (gVal.dtype) {
      when (DType.Int64) {
        var values = toSymEntry(gVal, int);
        select operator {
          when "sum" {
            var res = perLocSum(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "prod" {
            var res = perLocProduct(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "mean" {
            var res = perLocMean(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "min" {
            var res = perLocMin(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "max" {
            var res = perLocMax(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "argmin" {
            var res = perLocArgmin(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "argmax" {
            var res = perLocArgmax(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "nunique" {
            var res = perLocNumUnique(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          otherwise {return notImplementedError(pn,operator,gVal.dtype);}
          }
      }
      when (DType.Float64) {
        var values = toSymEntry(gVal, real);
        select operator {
          when "sum" {
            var res = perLocSum(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "prod" {
            var res = perLocProduct(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "mean" {
            var res = perLocMean(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "min" {
            var res = perLocMin(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "max" {
            var res = perLocMax(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "argmin" {
            var res = perLocArgmin(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "argmax" {
            var res = perLocArgmax(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          otherwise {return notImplementedError(pn,operator,gVal.dtype);}
          }
      }
      when (DType.Bool) {
        var values = toSymEntry(gVal, bool);
        select operator {
          when "sum" {
            var res = perLocSum(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "any" {
            var res = perLocAny(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "all" {
            var res = perLocAll(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          when "mean" {
            var res = perLocMean(values.a, segments.a);
            st.addEntry(rname, new shared SymEntry(res));
          }
          otherwise {return notImplementedError(pn,operator,gVal.dtype);}
          }
      }
      otherwise {return unrecognizedTypeError(pn, dtype2str(gVal.dtype));}
      }
      return try! "created " + st.attrib(rname);
    }
          
    /* Segmented Reductions of the form: seg<Op>(values:[] t, segments: [] int)
       Use <segments> as the boundary indices to divide <values> into chunks, 
       and then reduce over each chunk uisng the operator <Op>. The return array 
       of reduced values is the same size as <segments>.
     */
    proc segSum(values:[] ?t, segments:[?D] int, skipNan=false): [D] t {
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
      forall (i, r) in zip(D, res) {
        // Find the segment boundaries
        var vl: t, vr: t;
        if (i > D.low) {
          vl = cumsum[segments[i] - 1]:t;
        }
        if (i == D.high) {
          vr = cumsum[values.domain.high]:t;
        } else {
          vr = cumsum[segments[i+1] -1]:t;
        }
        r = vr - vl;
      }
      return res;
    }

    /* Per-Locale Segmented Reductions have the same form as segmented reductions:
       perLoc<Op>(values:[] t, segments: [] int)
       However, in this case <segments> has length <numSegments>*<numLocales> and
       stores the segment boundaries for each locale's chunk of <values>. These
       reductions perform two stages: a local reduction (implemented via a call
       to seg<Op> on the local slice of values) and a global reduction of the 
       local results. The return is the same as seg<Op>: one reduced value per segment.
    */
    proc perLocSum(values:[] ?t, segments:[?D] int): [] t {
      // Infer the number of keys from size of <segments>
      var numKeys:int = segments.size / numLocales;
      // Make the distributed domain of the final result
      var keyDom = makeDistDom(numKeys);
      // Local reductions stored in a PrivateDist
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      coforall loc in Locales {
        on loc {
          // Each locale reduces its local slice of <values>
          perLocVals[here.id] = segSum(values.localSlice[values.localSubdomain()],
                                       segments.localSlice[D.localSubdomain()]);
        }
      }
      // The global result is a distributed array, computed as a vector reduction over local results
      var res:[keyDom] t = + reduce [i in PrivateSpace] perLocVals[i];
      return res;
    }

    proc segSum(values:[] bool, segments:[?D] int): [D] int {
      var res: [D] int;
      if (D.size == 0) { return res; }
      var cumsum = + scan values;
      // Iterate over segments
      forall (i, r) in zip(D, res) {
        // Find the values to the left of the segment boundaries
        var vl: int, vr: int;
        if (i > D.low) {
          vl = cumsum[segments[i] - 1];
        }
        if (i == D.high) {
          vr = cumsum[values.domain.high];
        } else {
          vr = cumsum[segments[i+1] -1];
        }
        r = vr - vl;
      }
      return res;
    }

    proc perLocSum(values:[] bool, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
        on loc {
          perLocVals[here.id] = segSum(values.localSlice[values.localSubdomain()],
                                       segments.localSlice[D.localSubdomain()]);
        }
      }
      var res:[keyDom] int = + reduce [i in PrivateSpace] perLocVals[i];
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

    proc perLocProduct(values:[] ?t, segments:[?D] int): [] real {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] real;
      coforall loc in Locales {
        on loc {
          perLocVals[here.id] = segProduct(values.localSlice[values.localSubdomain()],
                                           segments.localSlice[D.localSubdomain()]);
        }
      }
      var res: [keyDom] real = * reduce [i in PrivateSpace] perLocVals[i];
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
        forall si in D {
          if si == D.high {
              segnans[si] = cumnans[cumnans.domain.high];
          } else {
              segnans[si] = cumnans[segments[si+1]-1];
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
      /* var cumsum = + scan values; */
      /* forall (i, r) in zip(D, res) { */
      /*         // Find the values to the left of the segment boundaries */
      /*         var vl: t, vr: t; */
      /*         var j: int; */
      /*         if (i > D.low) { */
      /*           vl = cumsum[segments[i] - 1]:t; */
      /*         } */
      /*         if (i == D.high) { */
      /*           j = values.domain.high; */
      /*         } else { */
      /*           j = segments[i+1] - 1; */
      /*         } */
      /*         vr = cumsum[j]:t; */
      /*         r = (vr - vl):real / (j - i + 1):real; */
      /* } */
      return res;
    }

    proc perLocMean(values:[] ?t, segments:[?D] int): [] real {
      var numKeys:int = segments.size / numLocales;
      var keyCounts = perLocCount(segments, values.size);
      var res = perLocSum(values, segments);
      return res:real / keyCounts:real;
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
      forall (i, r, low) in zip(D, res, segments) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          (_,r) = cummin[vi];
        }
      }
      return res;
    }
    
    proc perLocMin(values:[] ?t, segments:[?D] int): [] t {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      coforall loc in Locales {
        on loc {
          perLocVals[here.id] = segMin(values.localSlice[values.localSubdomain()],
                                       segments.localSlice[D.localSubdomain()]);
        }
      }
      var res: [keyDom] t;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
        r = min reduce [i in PrivateSpace] perLocVals[i][keyInd];
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
      
      forall (i, r, low) in zip(D, res, segments) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          (_,r) = cummax[vi];
        }
      }
      return res;
    }

    proc perLocMax(values:[] ?t, segments:[?D] int): [] t {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      coforall loc in Locales {
        on loc {
          perLocVals[here.id] = segMax(values.localSlice[values.localSubdomain()],
                                       segments.localSlice[D.localSubdomain()]);
        }
      }
      var res: [keyDom] t;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
        r = max reduce [i in PrivateSpace] perLocVals[i][keyInd];
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
      forall (l, v, low, i) in zip(locs, vals, segments, D) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          ((_,v),_) = cummin[vi];
          (_    ,l) = cummin[vi];
        }
      }
      return (vals, locs);
    }

    proc perLocArgmin(values:[] ?t, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      var perLocLocs: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
        on loc {
          (perLocVals[here.id], perLocLocs[here.id]) = segArgmin(values.localSlice[values.localSubdomain()],
                                                                 segments.localSlice[D.localSubdomain()]);
        }
      }
      var res: [keyDom] int;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
        var val: t;
        (val, r) = minloc reduce zip([i in PrivateSpace] perLocVals[i][keyInd],
                                     [i in PrivateSpace] perLocLocs[i][keyInd]);
      }
      return res;
    }
    
    proc segArgmax(values:[?vD] ?t, segments:[?D] int): ([D] t, [D] int) {
      var locs: [D] int;
      var vals: [D] t = min(t);
      if (D.size == 0) { return (vals, locs); }
      var keys = expandKeys(vD, segments);
      var kvi = [(k, v, i) in zip(keys, values, vD)] ((k, v), i);
      var cummax = maxloc scan kvi;
      forall (l, v, low, i) in zip(locs, vals, segments, D) {
        var vi: int;
        if (i < D.high) {
          vi = segments[i+1] - 1;
        } else {
          vi = values.domain.high;
        }
        if (vi >= low) {
          ((_,v),_) = cummax[vi];
          (_,    l) = cummax[vi];
        }
      }
      return (vals, locs);
    }

    proc perLocArgmax(values:[] ?t, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      var perLocLocs: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
        on loc {
          (perLocVals[here.id], perLocLocs[here.id]) = segArgmax(values.localSlice[values.localSubdomain()],
                                                                 segments.localSlice[D.localSubdomain()]);
        }
      }
      var res: [keyDom] int;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
        var val: t;
        (val, r) = maxloc reduce zip([i in PrivateSpace] perLocVals[i][keyInd],
                                     [i in PrivateSpace] perLocLocs[i][keyInd]);
      }
      return res;
    }
    
    proc segAny(values:[] bool, segments:[?D] int): [D] bool {
      var res: [D] bool;
      if (D.size == 0) { return res; }
      forall (r, low, i) in zip(res, segments, D) {
        var high: int;
        if (i < D.high) {
          high = segments[i+1] - 1;
        } else {
          high = values.domain.high;
        }
        r = || reduce values[low..high];
      }
      return res;
    }

    proc perLocAny(values:[] bool, segments:[?D] int): [] bool {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] bool;
      coforall loc in Locales {
        on loc {
          perLocVals[here.id] = segAny(values.localSlice[values.localSubdomain()],
                                       segments.localSlice[D.localSubdomain()]);
        }
      }
      var res: [keyDom] bool;
      for (r, keyInd) in zip(res, 0..#numKeys) {
        r = || reduce [i in PrivateSpace] perLocVals[i][keyInd];
      }
      return res;
    }
    
    proc segAll(values:[] bool, segments:[?D] int): [D] bool {
      var res: [D] bool;
      if (D.size == 0) { return res; }
      forall (r, low, i) in zip(res, segments, D) {
        var high: int;
        if (i < D.high) {
          high = segments[i+1] - 1;
        } else {
          high = values.domain.high;
        }
        r = && reduce values[low..high];
      }
      return res;
    }

    proc perLocAll(values:[] bool, segments:[?D] int): [] bool {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] bool;
      coforall loc in Locales {
        on loc {
          perLocVals[here.id] = segAll(values.localSlice[values.localSubdomain()],
                                       segments.localSlice[D.localSubdomain()]);
        }
      }
      var res: [keyDom] bool;
      for (r, keyInd) in zip(res, 0..#numKeys) {
        r = && reduce [i in PrivateSpace] perLocVals[i][keyInd];
      }    
      return res;
    }

    /* proc segNumUnique(values:[?vD] int, segments:[?sD] int): [sD] int { */
    /*   // TO DO: this is not correct. Sometimes gives values that are too large. */
    /*   // var res: [sD] int; */
    /*   var sorted: [vD] int; */
    /*   var aMin = min reduce values; */
    /*   var aMax = max reduce values; */
    /*   forall (i, left) in zip(sD, segments) { */
    /*         var right: int; */
    /*         if (i == sD.high) { */
    /*           right = vD.high; */
    /*         } else { */
    /*           right = segments[i+1] - 1; */
    /*         } */
    /*         if (right > left) { */
    /*           //msbRadixSortWithScratchSpace(left, right, sorted, values, defaultComparator, aMin, aMax); */
    /*           sorted[left..right] = values[left..right]; */
    /*           ref seg = sorted[left..right]; */
    /*           sort(seg); */
    /*         } */
    /*   } */
    /*   // TO DO: make sure within-key sort worked properly */
    /*   if reductionDEBUG { */
    /*         var sortCheck = true; */
    /*         forall (i, left) in zip(sD, segments) with (&& reduce sortCheck) { */
    /*           var right: int; */
    /*           if (i == sD.high) { */
    /*             right = sorted.domain.high; */
    /*           } else { */
    /*             right = segments[i+1] - 1; */
    /*           } */
    /*           if (right > left) { */
    /*             if (i < 3) || (i > sD.high - 3) { */
    /*               if ((right - left) > 5) { */
    /*                 writeln(i, ": ", sorted[left..left+3], " ... ", sorted[right-3..right]); */
    /*               } else { */
    /*                 writeln(i, ": ", sorted[left..right]); */
    /*               } */
    /*             } */
    /*             var ascending = [j in left..right-1] (sorted[j] <= sorted[j+1]); */
    /*             sortCheck reduce= (&& reduce ascending); */
    /*           } */
    /*         } */
    /*         writeln("All segments of values are internally sorted? ", sortCheck); */
    /*   } */
      
    /*   var truth: [vD] bool; */
    /*   // true where new value appears */
    /*   [(t, s, i) in zip(truth, sorted, vD)] if i > vD.low { t = (sorted[i-1] != s); } */
    /*   // first value of every segment is automatically new */
    /*   [s in segments] truth[s] = true; */
    /*   // count cumulative new values and take diffs at segment boundaries */
    /*   var count: [vD] int = (+ scan truth); */
    /*   var pop = count[vD.high]; */
    /*   if reductionDEBUG { writeln("Total unique vals across all keys: ", pop); } */
    /*   var nunique: [sD] int; */
    /*   forall (i, s, n) in zip(sD, segments, nunique) { */
    /*         var high: int; */
    /*         if (i == sD.high) { */
    /*           n = pop + 1 - count[s]; */
    /*         } else { */
    /*           n = count[segments[i+1]] - count[s]; */
    /*         } */
    /*   } */
    /*   return nunique; */
    /* } */

    proc expandKeys(kD, segments: [?sD] int): [kD] int {
      var truth: [kD] bool;
      forall i in segments with (var agg = newDstAggregator(bool)) {
        agg.copy(truth[i], true);
      }
      var keys = (+ scan truth) - 1;
      return keys;
    }

    proc segNumUnique(values: [?kD] int, segments: [?sD] int) {
      var res: [sD] int;
      if (sD.size == 0) {
        return res;
      }
      var keys = expandKeys(kD, segments);
      // sort keys and values together
      var t1 = Time.getCurrentTime();
      if v {writeln("Sorting keys and values..."); try! stdout.flush();}
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
      if v {writeln("sort time = ", Time.getCurrentTime() - t1); try! stdout.flush();}
      if v {writeln("Finding unique (key, value) pairs..."); try! stdout.flush();}
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
      if v {writeln("time = ", Time.getCurrentTime() - t1); try! stdout.flush();}
      if v {writeln("Finding unique keys and num unique vals per key."); try! stdout.flush(); t1 = Time.getCurrentTime();}
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
      if v {writeln("time = ", Time.getCurrentTime() - t1); try! stdout.flush();}
      return res;
    }

    /* proc segNumUnique(values:[] int, segments:[?D] int): [D] int { */
    /*   var res: [D] int; */
    /*   forall (r, low, i) in zip(res, segments, D) { */
    /*         var high: int; */
    /*         if (i < D.high) { */
    /*           high = segments[i+1] - 1; */
    /*         } else { */
    /*           high = values.domain.high; */
    /*         } */
    /*         var unique: domain(int); */
    /*         var domLock$:sync bool = true; */
    /*         forall v in values[low..high] with (ref unique, ref domLock$) { */
    /*           if !unique.contains(v) { */
    /*             domLock$; */
    /*             if !unique.contains(v) { */
    /*               unique += v; */
    /*             } */
    /*             domLock$ = true; */
    /*           } */
    /*         } */
    /*         r = unique.size; */
    /*   } */
    /*   return res; */
    /* } */

    proc perLocNumUnique(values:[] int, segments:[?D] int): [] int {
      var minVal = min reduce values;
      var valRange = (max reduce values) - minVal + 1;
      var numKeys:int = segments.size / numLocales;
      if (numKeys*valRange <= lBins) {
        if v {try! writeln("bins %i <= %i; using perLocNumUniqueHist".format(numKeys*valRange, lBins)); try! stdout.flush();}
        return perLocNumUniqueHist(values, segments, minVal, valRange, numKeys);
      } else {
        if v {try! writeln("bins %i > %i; using perLocNumUniqueAssoc".format(numKeys*valRange, lBins)); try! stdout.flush();}
        return perLocNumUniqueAssoc(values, segments, numKeys);
      }
    }

    /* proc perLocNumUnique(values:[] int, segments:[?D] int, numKeys: int): [] int { */
    /*   // First get all per-locale sets of unique values */
    /*   var localUnique: [PrivateSpace] [0..#numKeys] domain(int); */
    /*   coforall loc in Locales { */
    /*         on loc { */
    /*           var myD = D.localSubdomain(); */
    /*           // Loop over keys */
    /*           forall (i, low, myU) in zip(myD, segments.localSlice[myD], localUnique[here.id]) { */
    /*             // Find segment boundaries */
    /*             var high: int; */
    /*             if (i < myD.high) { */
    /*               high = segments[i+1] - 1; */
    /*             } else { */
    /*               high = values.localSubdomain().high; */
    /*             } */
    /*             // Aggregate segment's values into a set */
    /*             var domLock$:sync bool = true; */
    /*             forall v in values.localSlice[low..high] with (ref myU, ref domLock$) { */
    /*               // This outer check is not logically necessary, but it reduces unnecesary acquisition of the lock, which saves a lot of time when keys are dense */
    /*               if !myU.contains(v) { */
    /*                 domLock$; */
    /*                 if !myU.contains(v) { */
    /*                   myU += v; */
    /*                 } */
    /*                 domLock$ = true; */
    /*               } */
    /*             } */
    /*           } */
    /*         } */
    /*   } */
    /*   // Union the local sets */
    /*   var keyDom = makeDistDom(numKeys); */
    /*   var globalUnique: [keyDom] domain(int) = + reduce [i in PrivateSpace] localUnique[i]; */
    /*   var res = [i in keyDom] globalUnique[i].size; */
    /*   return res; */
    /* } */

    proc perLocNumUniqueHist(values: [] int, segments: [?D] int, minVal: int, valRange: int, numKeys: int): [] int {
      var valDom = makeDistDom(numKeys*valRange);
      var globalValFlags: [valDom] bool;
      coforall loc in Locales {
        on loc {
          var myD = D.localSubdomain();
          forall (i, low) in zip(myD, segments.localSlice[myD]) {
            var high: int;
            if (i < myD.high) {
              high = segments[i+1] - 1;
            } else {
              high = values.localSubdomain().high;
            }
            if (high >= low) {
              var perm: [0..#(high-low+1)] int;
              ref myVals = values.localSlice[low..high];
              var myMin = min reduce myVals;
              var myRange = (max reduce myVals) - myMin + 1;
              localHistArgSort(perm, myVals, myMin, myRange);
              var sorted: [low..high] int;
              [(s, idx) in zip(sorted, perm)] s = myVals[idx];
              var (mySegs, myUvals) = segsAndUkeysFromSortedArray(sorted);
              var keyInd = i - myD.low;
              forall v in myUvals with (ref globalValFlags) {
                // Does not need to be atomic
                globalValFlags[keyInd*valRange + v - minVal] = true;
              }
            }        
          }
        }
      }
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] int;
      forall (keyInd, r) in zip(keyDom, res) {
        r = + reduce globalValFlags[keyInd*valRange..#valRange];
      }
      return res;
    }

    proc perLocNumUniqueAssoc(values: [] int, segments: [?D] int, numKeys: int): [] int {
      var localUvals: [PrivateSpace] [0..#numKeys] domain(int, parSafe=false);
      coforall loc in Locales {
        on loc {
          var myD = D.localSubdomain();
          forall (i, low) in zip(myD, segments.localSlice[myD]) {
            var high: int;
            if (i < myD.high) {
              high = segments[i+1] - 1;
            } else {
              high = values.localSubdomain().high;
            }
            if (high >= low) {
              var perm: [0..#(high-low+1)] int;
              ref myVals = values.localSlice[low..high];
              var myMin = min reduce myVals;
              var myRange = (max reduce myVals) - myMin + 1;
              localHistArgSort(perm, myVals, myMin, myRange);
              var sorted: [low..high] int;
              [(s, idx) in zip(sorted, perm)] s = myVals[idx];
              var (mySegs, myUvals) = segsAndUkeysFromSortedArray(sorted);
              var keyInd = i - myD.low;
              forall v in myUvals {
                localUvals[here.id][keyInd] += v;
              }
            }        
          }
        }
      }
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] int;
      forall (keyInd, r) in zip(keyDom, res) {
        r = (+ reduce [i in PrivateSpace] localUvals[i][keyInd]).size;
      }
      return res;
    }

    proc stringtobool(str: string): bool throws {
      if str == "True" then return true;
      else if str == "False" then return false;
      throw new owned ErrorWithMsg("message: skipNan must be of type bool");
    }
}

