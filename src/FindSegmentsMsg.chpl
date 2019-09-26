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
	var pname = fields[2]; // permutation array
	var nkeys = try! fields[3]:int; // number of key arrays
	var size = try! fields[4]:int; // size of key arrays
        var knames = fields[5..]; // key arrays
	if (nkeys != knames.size) {
	  return try! incompatibleArgumentsError(pn, "Expected %i key arrays, but got %i".format(nkeys, knames.size));
	}
	// Check all the argument arrays before doing anything
	var gPerm = st.lookup(pname);
	if (gPerm == nil) { return unknownSymbolError(pn, pname); }
	if (gPerm.dtype != DType.Int64) { return notImplementedError(pn,"(permutation dtype "+dtype2str(gPerm.dtype)+")"); }	
	// var keyEntries: [0..#nkeys] borrowed GenSymEntry;
	for (name, i) in zip(knames, 0..) {
	  var g = st.lookup(name);
	  if (g == nil) { return unknownSymbolError(pn, name); }
	  if (g.size != size) { return try! incompatibleArgumentsError(pn, "Expected array of size %i, got size %i".format(size, g.size)); }
	  if (g.dtype != DType.Int64) { return notImplementedError(pn,"(key array dtype "+dtype2str(g.dtype)+")");}
	}
	
	// At this point, all arg arrays exist, have the same size, and are int64 dtype
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
	var permKey: [paD] int;
	for name in knames {
	  var g: borrowed GenSymEntry = st.lookup(name);
	  var k = toSymEntry(g,int); // key array
	  ref ka = k.a; // ref to key array
	  // Permute the key array to grouped order
	  [(s, p) in zip(permKey, pa)] { unorderedCopy(s, ka[p]); }
	  // Find steps and update ukeylocs
	  [(u, s, i) in zip(ukeylocs, permKey, paD)] if ((i > paD.low) && (permKey[i-1] != s))  { u = true; }
	}
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
	[i in ukeylocs.domain] if (ukeylocs[i] == true) {var idx = i; unorderedCopy(sa[iv[i]-1], idx);}
	// Create SymEntry to hold indices of unique keys in original (unpermuted) key arrays
        var uname = st.nextName(); // unique key indices
	var ukeyinds = st.addEntry(uname, pop, int);
	ref uka = ukeyinds.a;
	// Segment boundaries are in terms of permuted arrays, so invert the permutation to get back to original index
	[(s, i) in zip(sa, saD)] { unorderedCopy(uka[i], pa[s]); }
	// Return entry names of segments and unique key indices
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

        var kEnt: borrowed GenSymEntry = st.lookup(kname);
        if (kEnt == nil) {return unknownSymbolError(pn,kname);}

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
