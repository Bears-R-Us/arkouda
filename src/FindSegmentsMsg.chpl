module FindSegmentsMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use PerLocaleHelper;
    use SegmentedArray;

    use PrivateDist;
    use CommAggregation;

    /*

    :arg reqMsg: request containing (cmd,kname,pname)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) 
    :throws: `UndefinedSymbolError(name)`

    */
    proc findSegmentsMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (pname, nkeysStr, rest) = payload.decode().splitMsgToTuple(3);
        var nkeys = nkeysStr:int; // number of key arrays
        var fields = rest.split(); // split request into fields
        if (fields.size != 2*nkeys) { return incompatibleArgumentsError(pn, "Expected %i arrays but got %i".format(nkeys, (fields.size - 3)/2));}
        var low = fields.domain.low;
        var knames = fields[low..#nkeys]; // key arrays
        var ktypes = fields[low+nkeys..#nkeys]; // objtypes
        var size: int;
        // Check all the argument arrays before doing anything
        var gPerm = st.lookup(pname);
        if (gPerm.dtype != DType.Int64) { return notImplementedError(pn,"(permutation dtype "+dtype2str(gPerm.dtype)+")"); }        
        // var keyEntries: [0..#nkeys] borrowed GenSymEntry;
        for (name, objtype, i) in zip(knames, ktypes, 1..) {
      // var g: borrowed GenSymEntry;
      var thisSize: int;
      var thisType: DType;
      select objtype {
        when "pdarray" {
          var g = st.lookup(name);
          thisSize = g.size;
          thisType = g.dtype;
        }
        when "str" {
          var (myNames,_) = name.splitMsgToTuple('+', 2);
          var g = st.lookup(myNames);
          thisSize = g.size;
          thisType = g.dtype;
        }
        otherwise {return unrecognizedTypeError(pn, objtype);}
      }
      if (i == 1) {
        size = thisSize;
      } else {
        if (thisSize != size) { return try! incompatibleArgumentsError(pn, "Expected array of size %i, got size %i".format(size, thisSize)); }
      }
          if (thisType != DType.Int64) { return notImplementedError(pn,"(key array dtype "+dtype2str(thisType)+")");}
        }
        
        // At this point, all arg arrays exist, have the same size, and are int64 or string dtype
        if (size == 0) {
          // Return two empty integer entries
          var n1 = st.nextName();
          st.addEntry(n1, 0, int);
          var n2 = st.nextName();
          st.addEntry(n2, 0, int);
          return try! "created " + st.attrib(n1) + " +created " + st.attrib(n1);
        }
        // Permutation that groups the keys
        var perm = toSymEntry(gPerm, int);
        ref pa = perm.a; // ref to actual permutation array
        ref paD = perm.aD; // ref to domain
        // Unique key indices; true where first value of each unique key occurs
        var ukeylocs: [paD] bool;
        // First key is automatically present
        ukeylocs[0] = true;
        for (name, objtype) in zip(knames, ktypes) {
      select objtype {
        when "pdarray" {
          var g: borrowed GenSymEntry = st.lookup(name);
          var k = toSymEntry(g,int); // key array
          ref ka = k.a; // ref to key array
          // Permute the key array to grouped order
          var permKey: [paD] int;
          forall (s, p) in zip(permKey, pa) with (var agg = newSrcAggregator(int)) { 
            agg.copy(s, ka[p]);
          }
          // Find steps and update ukeylocs
          [(u, s, i) in zip(ukeylocs, permKey, paD)] if ((i > paD.low) && (permKey[i-1] != s))  { u = true; }
        }
        when "str" {
          var (myNames1,myNames2) = name.splitMsgToTuple('+', 2);
          var str = new owned SegString(myNames1, myNames2, st);
          var (permOffsets, permVals) = str[pa];
          const ref D = permOffsets.domain;
          var permLengths: [D] int;
          permLengths[D.interior(-(D.size-1))] = permOffsets[D.interior(D.size-1)] - permOffsets[D.interior(-(D.size-1))];
          permLengths[D.high] = permVals.domain.high - permOffsets[D.high] + 1;
          // Find steps and update ukeylocs
          // [(u, s, i) in zip(ukeylocs, permKey, paD)] if ((i > paD.low) && (permKey[i-1] != s))  { u = true; }
          forall (u, o, l, i) in zip(ukeylocs, permOffsets, permLengths, D) {
            if (i > D.low) {
              // If string lengths don't match, mark a step
              if (permLengths[i-1] != l) {
                u = true;
              } else {
                // Have to compare bytes of previous string to current string
                // If any bytes differ, mark a step
                for pos in 0..#l {
                  if permVals[permOffsets[i-1]+pos] != permVals[o+pos] {
                    u = true;
                    break;
                  }
                }
              }
            }
          } // forall
        } // when "str"
      } // select objtype
        }
    // All keys have been processed, all steps have been found
        // +scan to compute segment position... 1-based because of inclusive-scan
        var iv: [ukeylocs.domain] int = (+ scan ukeylocs);
        // compute how many segments
        var pop = iv[iv.size-1];
        if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}
        // Create SymEntry to hold segment boundaries
        var sname = st.nextName(); // segments
        var segs = st.addEntry(sname, pop, int);
        ref sa = segs.a;
        ref saD = segs.aD;
        // segment position... 1-based needs to be converted to 0-based because of inclusive-scan
        // where ever a segment break (true value) is... that index is a segment start index
        forall i in ukeylocs.domain with (var agg = newDstAggregator(int)) {
          if (ukeylocs[i] == true) {
            var idx = i; 
            agg.copy(sa[iv[i]-1], idx);
          }
        }
        // Create SymEntry to hold indices of unique keys in original (unpermuted) key arrays
        var uname = st.nextName(); // unique key indices
        var ukeyinds = st.addEntry(uname, pop, int);
        ref uka = ukeyinds.a;
        // Segment boundaries are in terms of permuted arrays, so invert the permutation to get back to original index
        forall (s, i) in zip(sa, saD) {
          // TODO convert to aggreation, which side is remote though?
          use UnorderedCopy;
          unorderedCopy(uka[i], pa[s]);
        }
        // Return entry names of segments and unique key indices
        return try! "created " + st.attrib(sname) + " +created " + st.attrib(uname);
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
            otherwise {return notImplementedError(pn,"("+dtype2str(kEnt.dtype)+")");}
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
}
