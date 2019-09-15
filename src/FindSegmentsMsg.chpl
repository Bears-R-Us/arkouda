module FindSegmentsMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use PerLocaleHelper;

    use PrivateDist;
    // experimental
    use UnorderedCopy;

    /*

    :arg reqMsg: request containing (cmd,kname,pname)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) 

    */
    proc findSegmentsMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "findSegments";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var kname = fields[2]; // sorted key array
        
        // get next symbol name
        var sname = st.nextName(); // segments
        var uname = st.nextName(); // unique keys

        var kEnt_: borrowed GenSymEntry? = st.lookup(kname);
        if (kEnt_ == nil) {return unknownSymbolError(pn,kname);}
        var kEnt = kEnt_!;

        select (kEnt.dtype) {
            when (DType.Int64) {
                var k = toSymEntry(kEnt,int); // key array
                
                ref ka = k.a; // ref to key array
                ref kad = k.aD; // ref to key array domain
                                
                var truth: [k.aD] bool;
                // truth array to hold segment break points
                truth[0] = true;
                [(t, s, i) in zip(truth, ka, kad)] if i > kad.low { t = (ka[i-1] != s); }

                // +scan to compute segment position... 1-based because of inclusive-scan
                var iv: [truth.domain] int = (+ scan truth);
                // compute how many segments
                var pop = iv[iv.size-1];
                if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}

                var segs = makeDistArray(pop, int);
                var ukeys = makeDistArray(pop, int);

                // segment position... 1-based needs to be converted to 0-based because of inclusive-scan
                // where ever a segment break (true value) is... that index is a segment start index
                [i in truth.domain] if (truth[i] == true) {var idx = i; unorderedCopy(segs[iv[i]-1], idx);}
                // pull out the first key in each segment as a unique key
		// unique keys guaranteed to be sorted because keys are sorted
                [i in segs.domain] ukeys[i] = ka[segs[i]];
                
                st.addEntry(sname, new shared SymEntry(segs));
                st.addEntry(uname, new shared SymEntry(ukeys));
            }
            otherwise {return notImplementedError(pn,"("+dtype2str(kEnt.dtype)+")");}
        }
        
        return try! "created " + st.attrib(sname) + " +created " + st.attrib(uname);
    }

    proc findLocalSegmentsMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "findLocalSegments";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var kname = fields[2]; // key array

        // get next symbol name
        var sname = st.nextName(); // segments
        var uname = st.nextName(); // unique keys

        var kEnt_: borrowed GenSymEntry? = st.lookup(kname);
        if (kEnt_ == nil) {return unknownSymbolError(pn,kname);}
        var kEnt = kEnt_!;

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
	  forall k in locUkeys with (ref globalRelKeys) {
	    // This does not need to be atomic, because race conditions will result in the correct answer
	    globalRelKeys[k - minKey] = true;
	    // would like to use unorderedCopy here but get non-lvalue error
	    // unorderedCopy(globalRelKeys[k - minKey], 1);
	  }
	}
      }
      if v {timer.stop(); writeln("time = ", timer.elapsed(), " sec"); try! stdout.flush(); timer.clear();}

      if v {writeln("aggregating globally unique keys..."); try! stdout.flush(); timer.start();}
      var relKey2ind = (+ scan globalRelKeys) - 1;
      var numKeys = relKey2ind[keyDom.high] + 1;
      if v {writeln("Global unique keys: ", numKeys); try! stdout.flush();}
      var globalUkeys = makeDistArray(numKeys, int);
      forall (relKey, present, ind) in zip(keyDom, globalRelKeys, relKey2ind) {
      	if present {
	  // globalUkeys guaranteed to be sorted because ind and relKey monotonically increasing
      	  globalUkeys[ind] = relKey + minKey;
	  // would like to use unorderedCopy here but get non-lvalue error
	  // unorderedCopy(globalUkeys[ind], relKey + minKey);
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
