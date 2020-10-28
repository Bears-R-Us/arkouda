module PerLocaleReduction {
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;
    use CommAggregation;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use ServerConfig;
    use Reflection;
    use Errors;

    use AryUtil;
    use PrivateDist;
    use RadixSortLSD;
    use ReductionMsg;

    private config const reductionDEBUG = false;
    private config const lBins = 2**25 * numLocales;

    /* localArgsort takes a pdarray and returns an index vector which sorts the array on a per-locale basis */
    proc localArgsortMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.decode().splitMsgToTuple(1);

        // get next symbol name
        var ivname = st.nextName();
        if v {writeln("%s %s : %s %s".format(cmd, name, ivname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var iv = perLocaleArgSort(e.a);
                st.addEntry(ivname, new shared SymEntry(iv));
            }
            otherwise {
                var errorMsg = notImplementedError(pn,gEnt.dtype);
                writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError"));                    
                return errorMsg;                 
            }
        }
        return try! "created " + st.attrib(ivname);
    }
    
    proc perLocaleArgSort(a:[?aD] int):[aD] int {
      var iv: [aD] int;
      coforall loc in Locales {
          on loc {
              var toSort = [(v, i) in zip(a.localSlice[a.localSubdomain()], a.localSubdomain())] (v, i);
              Sort.sort(toSort);
              iv.localSlice[iv.localSubdomain()] = [(v, i) in toSort] i;
          }
      }
      return iv;
    }

    proc perLocaleArgCountSort(a:[?aD] int):[aD] int {
      var iv: [aD] int;
      coforall loc in Locales {
        on loc {
          //ref myIV = iv[iv.localSubdomain()];
          var myIV: [0..#iv.localSubdomain().size] int;
          ref myA = a.localSlice[a.localSubdomain()];
          // Calculate number of histogram bins
          var locMin = min reduce myA;
          var locMax = max reduce myA;
          var bins = locMax - locMin + 1;
          if (bins <= mBins) {
            if (v && here.id==0) {try! writeln("bins %i <= %i; using localHistArgSort".format(bins, mBins));}
            localHistArgSort(myIV, myA, locMin, bins);
          } else {
            if (v && here.id==0) {try! writeln("bins %i > %i; using localAssocArgSort".format(bins, mBins));}
            localAssocArgSort(myIV, myA);
          }
          iv.localSlice[iv.localSubdomain()] = myIV;
        }
      }
      return iv;
    }

    proc localAssocArgSort(iv:[] int, a:[?D] int) {
      use Sort only;
      // a is sparse, so use an associative domain
      var binDom: domain(int);
      // Make counts for each value in a
      var hist: [binDom] atomic int;
      forall val in a with (ref hist, ref binDom) {
        if !binDom.contains(val) {
          binDom += val;
        }
        hist[val].add(1);
      }
      // Need the bins in sorted order as a dense array
      var sortedBins: [0..#binDom.size] int;
      for (s, b) in zip(sortedBins, binDom) {
        s = b;
      }
      Sort.sort(sortedBins);
      // Make an associative array that translates from value to dense, sorted bin index
      var val2bin: [binDom] int;
      forall (i, v) in zip(sortedBins.domain, sortedBins) {
        val2bin[v] = i;
      }
      // Get segment offsets in correct order
      var counts = [b in sortedBins] hist[b].read();
      var offsets = (+ scan counts) - counts;
      // Now insert the a_index into iv
      var binpos: [sortedBins.domain] atomic int;
      forall (aidx, val) in zip(D, a) with (ref binpos, ref iv) {
        // Use val's bin to determine where in iv to put a_index
        var bin = val2bin[val];
        // ividx is the offset of val's bin plus a running counter
        var ividx = offsets[bin] + binpos[bin].fetchAdd(1);
        iv[ividx] = aidx;
      }
    }

    record Segments {
      var n:int;
      var dom = {0..#n};
      var k:[dom] int;
      var s:[dom] int;

      proc init() {
        n = 0;
      }

      proc init(keys:[?D] int, segs:[D] int) {
        n = D.size;
        dom = D;
        k = keys;
        s = segs;
      }
    }
    
    proc findLocalSegmentsMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (kname) = payload.decode().splitMsgToTuple(1);

        // get next symbol name
        var sname = st.nextName(); // segments
        var uname = st.nextName(); // unique keys

        var kEnt: borrowed GenSymEntry = st.lookup(kname);

        select (kEnt.dtype) {
            when (DType.Int64) {
                var k = toSymEntry(kEnt,int); // key array
                var minKey = min reduce k.a;
                var keyRange = (max reduce k.a) - minKey + 1;
                var (segs, ukeys) = perLocFindSegsAndUkeys(k.a, minKey, keyRange);
                st.addEntry(sname, new shared SymEntry(segs));
                st.addEntry(uname, new shared SymEntry(ukeys));
            }
            otherwise {
                var errorMsg = notImplementedError(pn,"("+dtype2str(kEnt.dtype)+")");
                writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError"));                 
                return errorMsg;   
            }
        }
        
        return try! "created " + st.attrib(sname) + " +created " + st.attrib(uname);
    }

    proc perLocFindSegsAndUkeys(perLocSorted:[?D] int, minKey:int, keyRange:int) {
      var timer = new Time.Timer();
      if v {writeln("finding local segments and keys..."); try! stdout.flush(); timer.start();}
      var keyDom = makeDistDom(keyRange);
      var globalRelKeys:[keyDom] bool;
      var localSegments: [PrivateSpace] Segments;
      coforall loc in Locales {
        on loc {
          var (locSegs, locUkeys) = segsAndUkeysFromSortedArray(perLocSorted.localSlice[perLocSorted.localSubdomain()]);
          localSegments[here.id] = new Segments(locUkeys, locSegs);
          forall k in locUkeys with (ref globalRelKeys, var agg = newDstAggregator(bool)) {
            // This does not need to be atomic, because race conditions will result in the correct answer
            agg.copy(globalRelKeys[k - minKey], true);
          }
        }
      }
      if v {timer.stop(); writeln("time = ", timer.elapsed(), " sec"); try! stdout.flush(); timer.clear();}

      if v {writeln("aggregating globally unique keys..."); try! stdout.flush(); timer.start();}
      var relKey2ind = (+ scan globalRelKeys) - 1;
      var numKeys = relKey2ind[keyDom.high] + 1;
      if v {writeln("Global unique keys: ", numKeys); try! stdout.flush();}
      var globalUkeys = makeDistArray(numKeys, int);
      forall (relKey, present, ind) in zip(keyDom, globalRelKeys, relKey2ind) 
        with (var agg = newDstAggregator(int)) {
        if present {
          // globalUkeys guaranteed to be sorted because ind and relKey monotonically increasing
          agg.copy(globalUkeys[ind], relKey + minKey);
        }
      }
      if v {timer.stop(); writeln("time = ", timer.elapsed(), " sec"); try! stdout.flush(); timer.clear();}

      if v {writeln("creating global segment array..."); try! stdout.flush(); timer.start();}
      var globalSegments = makeDistArray(numKeys*numLocales, int);
      globalSegments = -1;
      coforall loc in Locales {
        on loc {
          ref mySeg = localSegments[here.id].s;
          ref myKey = localSegments[here.id].k;
          forall (offset, key) in zip(mySeg, myKey) {
            var segInd = globalSegments.localSubdomain().low + relKey2ind[key - minKey];
            globalSegments[segInd] = offset;
          }
          var last = D.localSubdomain().high + 1;
          for i in globalSegments.localSubdomain() by -1 {
            if globalSegments[i] == -1 {
              globalSegments[i] = last;
            } else {
              last = globalSegments[i];
            }
          }
        }
      }
      if v {timer.stop(); writeln("time = ", timer.elapsed(), " sec"); try! stdout.flush(); timer.clear();}
      return (globalSegments, globalUkeys);
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
      if v {writeln("%s %s %s".format(cmd,segments_name, size));try! stdout.flush();}

      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {
          var errorMsg = "Error: array of segment offsets must be int dtype";
          writeln(generateErrorContext(
                       msg=errorMsg, 
                       lineNumber=getLineNumber(), 
                       moduleName=getModuleName(), 
                       routineName=getRoutineName(), 
                       errorClass="IncompatibleArgumentsError"));           
          return errorMsg;          
      }
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

    proc segmentedLocalRdxMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
      // reqMsg: segmentedReduction keys values segments operator
      // 'values_name' is the segmented array of values to be reduced
      // 'segments_name' is the segmented offsets
      // 'operator' is the reduction operator
      var (keys_name, values_name, segments_name, operator) = payload.decode().splitMsgToTuple(4);
      var rname = st.nextName();
      if v {writeln("%s %s %s %s %s".format(cmd,keys_name,values_name,segments_name,operator));try! stdout.flush();}

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
          otherwise {
               var errorMsg = notImplementedError(pn,operator,gVal.dtype);
                writeln(generateErrorContext(
                     msg=errorMsg, 
                     lineNumber=getLineNumber(), 
                     moduleName=getModuleName(), 
                     routineName=getRoutineName(), 
                     errorClass="IncompatibleArgumentsError")); 
                return errorMsg;         
             }
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
    
    proc perLocMean(values:[] ?t, segments:[?D] int): [] real {
      var numKeys:int = segments.size / numLocales;
      var keyCounts = perLocCount(segments, values.size);
      var res = perLocSum(values, segments);
      return res:real / keyCounts:real;
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

}