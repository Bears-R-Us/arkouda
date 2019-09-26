/* arg sort algorithm
these pass back an index vector which can be used
to permute the original array into sorted order */

module ArgSortMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
    use PerLocaleHelper;
    
    use PrivateDist;

    // experimental
    use UnorderedCopy;
    use UnorderedAtomics;

    use Sort only;
    use RadixSortLSD;
    
    // thresholds for different sized sorts
    var lgSmall = 10;
    var small = 2**lgSmall;
    var lgMedium = 20;
    var medium = 2**lgMedium;
    var lgLarge = 30;
    var large = 2**lgLarge;

    // thresholds for ranges of values in the sorts
    var sBins = 2**10;
    var mBins = 2**25;
    var lBins = 2**25 * numLocales;

    /* // defined for reduction and scan on atomics */
    /* proc +(x: atomic int, y: atomic int) { */
    /*     return x.read() + y.read(); */
    /* } */
    
    /* // defined for reduction and scan on atomics */
    /* proc +=(X: [?D] int, Y: [D] atomic int) { */
    /*     [i in D] {X[i] += Y[i].read();} */
    /* } */

    /* /\* do a counting sort on a (an array of integers) */
    /*    returns iv an array of indices that would sort the array original array *\/ */
    /* proc argCountSortGlobHist(a: [?aD] int, aMin: int, aMax: int): [aD] int { */
    /*     // index vector to hold permutation */
    /*     var iv: [aD] int; */

    /*     // how many bins in histogram */
    /*     var bins = aMax-aMin+1; */
    /*     if v {try! writeln("bins = %t".format(bins));} */

    /*     // histogram domain size should be equal to a_nvals */
    /*     var hD = makeDistDom(bins); */

    /*     // atomic histogram */
    /*     var atomic_hist: [hD] atomic int; */

    /*     // normal histogram for + scan */
    /*     var hist: [hD] int; */

    /*     // count number of each value into atomic histogram */
    /*     //[e in a] atomic_hist[e-aMin].add(1); */
    /*     [e in a] atomic_hist[e-aMin].unorderedAdd(1); */
        
    /*     // copy from atomic histogram to normal histogram */
    /*     [(e,ae) in zip(hist, atomic_hist)] e = ae.read(); */
    /*     if v {printAry("hist =",hist);} */

    /*     // calc starts and ends of buckets */
    /*     var ends: [hD] int = + scan hist; */
    /*     if v {printAry("ends =",ends);} */
    /*     var starts: [hD] int = ends - hist; */
    /*     if v {printAry("starts =",starts);} */

    /*     // atomic position in output array for buckets */
    /*     var atomic_pos: [hD] atomic int; */
        
    /*     // copy in start positions */
    /*     [(ae,e) in zip(atomic_pos, starts)] ae.write(e); */

    /*     // permute index vector */
    /*     forall (e,i) in zip(a,aD) { */
    /*         var pos = atomic_pos[e-aMin].fetchAdd(1);// get position to deposit element */
    /*         //iv[pos] = i; */
    /*         var idx = i; */
    /*         unorderedCopy(iv[pos], idx); */
    /*     } */
        
    /*     // return the index vector */
    /*     return iv; */
    /* } */

    /* /\* do a counting sort on a (an array of integers) */
    /*    returns iv an array of indices that would sort the array original array *\/ */
    /* proc argCountSortLocHistGlobHist(a: [?aD] int, aMin: int, aMax: int): [aD] int { */
    /*     // index vector to hold permutation */
    /*     var iv: [aD] int; */

    /*     // how many bins in histogram */
    /*     var bins = aMax-aMin+1; */
    /*     if v {try! writeln("bins = %t".format(bins));} */

    /*     // create a global count array to scan */
    /*     var globalCounts = makeDistArray(bins * numLocales, int); */

    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // histogram domain size should be equal to bins */
    /*             var hD = {0..#bins}; */

    /*             // atomic histogram */
    /*             var atomicHist: [hD] atomic int; */

    /*             // count number of each value into local atomic histogram */
    /*             [i in a.localSubdomain()] atomicHist[a[i]-aMin].add(1); */

    /*             // put counts into globalCounts array */
    /*             [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[i].read(); */
    /*         } */
    /*     } */

    /*     // scan globalCounts to get bucket ends on each locale */
    /*     var globalEnds: [globalCounts.domain] int = + scan globalCounts; */
    /*     if v {printAry("globalCounts =",globalCounts);try! stdout.flush();} */
    /*     if v {printAry("globalEnds =",globalEnds);try! stdout.flush();} */
        
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // histogram domain size should be equal to bins */
    /*             var hD = {0..#bins}; */
    /*             var localCounts: [hD] int; */
    /*             [i in hD] localCounts[i] = globalCounts[i * numLocales + here.id]; */
    /*             var localEnds: [hD] int = + scan localCounts; */
                
    /*             // atomic histogram */
    /*             var atomicHist: [hD] atomic int; */

    /*             // local storage to sort into */
    /*             var localBuffer: [0..#(a.localSubdomain().size)] int; */

    /*             // put locale-bucket-ends into atomic hist */
    /*             [i in hD] atomicHist[i].write(localEnds[i] - localCounts[i]); */
                
    /*             // get position in localBuffer of each element and place it there */
    /*             // counting up to local-bucket-end */
    /*             [idx in a.localSubdomain()] { */
    /*                 var pos = atomicHist[a[idx]-aMin].fetchAdd(1); // local pos in localBuffer */
    /*                 localBuffer[pos] = idx; // should be local pos and global idx */
    /*             } */

    /*             // move blocks to output array */
    /*             [i in hD] { */
    /*                 var gEnd = globalEnds[i * numLocales + here.id]; */
    /*                 var gHigh = gEnd - 1; */
    /*                 var gLow =  gEnd - localCounts[i]; */
    /*                 var lHigh = localEnds[i] - 1; */
    /*                 var lLow = localEnds[i] - localCounts[i]; */
    /*                 if (gLow..gHigh).size != (lLow..lHigh).size { */
    /*                     writeln(gLow..gHigh, " ", lLow..lHigh); */
    /*                     writeln((gLow..gHigh).size, " != ", (lLow..lHigh).size); */
    /*                     try! stdout.flush(); */
    /*                     exit(1); */
    /*                 } */
    /*                 if localCounts[i] > 0 {iv[gLow..gHigh] = localBuffer[lLow..lHigh];} */
    /*             } */
    /*         } */
    /*     } */
        
    /*     // return the index vector */
    /*     return iv; */
    /* } */
    
    /* /\* do a counting sort on a (an array of integers) */
    /* returns iv an array of indices that would sort the array original array */
    /* PD == PrivateDist */
    /* IW == Indirect write to local array then block copy to output array *\/ */
    /* proc argCountSortLocHistGlobHistPDIW(a: [?aD] int, aMin: int, aMax: int): [aD] int { */
    /*     // index vector to hold permutation */
    /*     var iv: [aD] int; */

    /*     // how many bins in histogram */
    /*     var bins = aMax-aMin+1; */
    /*     if v {try! writeln("bins = %t".format(bins));} */

    /*     // create a global count array to scan */
    /*     var globalCounts = makeDistArray(bins * numLocales, int); */

    /*     // histogram domain size should be equal to bins */
    /*     var hD = {0..#bins}; */
        
    /*     // atomic histogram */
    /*     var atomicHist: [PrivateSpace] [hD] atomic int; */
        
    /*     // start timer */
    /*     var t1 = Time.getCurrentTime(); */
    /*     // count number of each value into local atomic histogram */
    /*     [val in a] atomicHist[here.id][val-aMin].add(1); */
    /*     if v {writeln("done atomicHist time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */

    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // put counts into globalCounts array */
    /*             [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[here.id][i].read(); */
    /*         } */
    /*     } */
    /*     if v {writeln("done copy to globalCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */

    /*     // scan globalCounts to get bucket ends on each locale */
    /*     var globalEnds: [globalCounts.domain] int = + scan globalCounts; */
    /*     if v {printAry("globalCounts =",globalCounts);try! stdout.flush();} */
    /*     if v {printAry("globalEnds =",globalEnds);try! stdout.flush();} */

    /*     var localCounts: [PrivateSpace] [hD] int; */
    /*     var localEnds: [PrivateSpace] [hD] int; */
        
    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             [i in hD] localCounts[here.id][i] = globalCounts[i * numLocales + here.id]; */
    /*         } */
    /*     } */
    /*     if v {writeln("done copy back to localCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */
        
    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             localEnds[here.id] = + scan localCounts[here.id]; */

    /*             // put locale-bucket-ends into atomic hist */
    /*             [i in hD] atomicHist[here.id][i].write(localEnds[here.id][i] - localCounts[here.id][i]); */
                
    /*             // local storage to sort into */
    /*             var localBuffer: [0..#(a.localSubdomain().size)] int; */

    /*             // get position in localBuffer of each element and place it there */
    /*             // counting up to local-bucket-end */
    /*             [idx in a.localSubdomain()] { */
    /*                 var pos = atomicHist[here.id][a[idx]-aMin].fetchAdd(1); // local pos in localBuffer */
    /*                 localBuffer[pos] = idx; // should be local pos and global idx */
    /*             } */

    /*             // move blocks to output array */
    /*             [i in hD] { */
    /*                 var gEnd = globalEnds[i * numLocales + here.id]; */
    /*                 var gHigh = gEnd - 1; */
    /*                 var gLow =  gEnd - localCounts[here.id][i]; */
    /*                 var lHigh = localEnds[here.id][i] - 1; */
    /*                 var lLow = localEnds[here.id][i] - localCounts[here.id][i]; */
    /*                 if (gLow..gHigh).size != (lLow..lHigh).size { */
    /*                     writeln(gLow..gHigh, " ", lLow..lHigh); */
    /*                     writeln((gLow..gHigh).size, " != ", (lLow..lHigh).size); */
    /*                     try! stdout.flush(); */
    /*                     exit(1); */
    /*                 } */
    /*                 if localCounts[here.id][i] > 0 {iv[gLow..gHigh] = localBuffer[lLow..lHigh];} */
    /*             } */
    /*         } */
    /*     } */
    /*     if v {writeln("done sort locally and move segments time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */
        
    /*     // return the index vector */
    /*     return iv; */
    /* } */
    
    /* /\* do a counting sort on a (an array of integers) */
    /* returns iv an array of indices that would sort the array original array */
    /* PD = PrivateDist */
    /* DW = Direct Write into output array *\/ */
    /* proc argCountSortLocHistGlobHistPDDW(a: [?aD] int, aMin: int, aMax: int): [aD] int { */
    /*     // index vector to hold permutation */
    /*     var iv: [aD] int; */

    /*     // how many bins in histogram */
    /*     var bins = aMax-aMin+1; */
    /*     if v {try! writeln("bins = %t".format(bins));} */

    /*     // create a global count array to scan */
    /*     var globalCounts = makeDistArray(bins * numLocales, int); */

    /*     // histogram domain size should be equal to bins */
    /*     var hD = {0..#bins}; */
        
    /*     // atomic histogram */
    /*     var atomicHist: [PrivateSpace] [hD] atomic int; */
        
    /*     // start timer */
    /*     var t1 = Time.getCurrentTime(); */
    /*     // count number of each value into local atomic histogram */
    /*     [val in a] atomicHist[here.id][val-aMin].add(1); */
    /*     if v {writeln("done atomicHist time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */

    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // put counts into globalCounts array */
    /*             [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[here.id][i].read(); */
    /*         } */
    /*     } */
    /*     if v {writeln("done copy to globalCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */

    /*     // scan globalCounts to get bucket ends on each locale */
    /*     var globalEnds: [globalCounts.domain] int = + scan globalCounts; */
    /*     if v {printAry("globalCounts =",globalCounts);try! stdout.flush();} */
    /*     if v {printAry("globalEnds =",globalEnds);try! stdout.flush();} */

    /*     var localCounts: [PrivateSpace] [hD] int; */
        
    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             [i in hD] localCounts[here.id][i] = globalCounts[i * numLocales + here.id]; */
    /*         } */
    /*     } */
    /*     if v {writeln("done copy back to localCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */
        
    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // put locale-subbin-starts into atomic hist */
    /*             [i in hD] atomicHist[here.id][i].write(globalEnds[i * numLocales + here.id] - localCounts[here.id][i]); */
    /*         } */
    /*     } */
    /*     if v {writeln("done init atomic counts time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */
        
    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // fetch-and-inc to get per-locale-subbin-position */
    /*             // and directly write index to output array */
    /*             forall i in a.localSubdomain() { */
    /*                 var idx = i; */
    /*                 var pos = atomicHist[here.id][a[idx]-aMin].fetchAdd(1); // local pos in localBuffer */
    /*                 unorderedCopy(iv[pos],idx); // iv[pos] = idx; // should be global pos and global idx */
    /*             } */
    /*         } */
    /*     } */
    /*     if v {writeln("done move time = ",Time.getCurrentTime() - t1);try! stdout.flush();} */

    /*     // return the index vector */
    /*     return iv; */
    /* } */

    /* Perform one step in a multi-step argsort, starting with an initial 
       permutation vector and further permuting it in the manner required
       to sort an array of keys.
     */
    proc incrementalArgSort(g: GenSymEntry, iv: [?aD] int): [] int throws {
      // Store the incremental permutation to be applied on top of the initial perm
      var deltaIV: [aD] int;
      // Discover the dtype of the entry holding the keys array
      select g.dtype {
	when DType.Int64 {
	  var e = toSymEntry(g, int);
	  // Permute the keys array with the initial iv
	  var newa: [e.aD] int;
	  ref olda = e.a;
	  // Effectively: newa = olda[iv]
	  [(newai, idx) in zip(newa, iv)] unorderedCopy(newai, olda[idx]);
	  // Generate the next incremental permutation
	  deltaIV = radixSortLSD_ranks(newa);
	}
	when DType.Float64 {
	  var e = toSymEntry(g, real);
	  var newa: [e.aD] real;
	  ref olda = e.a;
	  [(newai, idx) in zip(newa, iv)] unorderedCopy(newai, olda[idx]);
	  deltaIV = radixSortLSD_ranks(newa);
	}
	otherwise { throw new owned ErrorWithMsg(dtype2str(g.dtype)); }
      }
      // The output permutation is the composition of the initial and incremental permutations
      var newIV: [aD] int;
      // Effectively: newIV = iv[deltaIV] 
      [(newIVi, idx) in zip(newIV, deltaIV)] unorderedCopy(newIVi, iv[idx]);
      return newIV;
    }

    /* Do a LSD radix sort across multiple arrays, where each array represents a digit.
     */
    /* proc coArgSort(arrays: [?D] GenSymEntry): [] int throws { */
    /*   // Calling function already checked that all arrays have same size */
    /*   const aD = makeDistDom(arrays[D.low].size); */
    /*   // Initialize permutation to the identity */
    /*   var cumulativeIV: [aD] int = aD.low..aD.high; */
    /*   // Starting with the last array, incrementally permute the IV by sorting each array */
    /*   for i in D.low..D.high-1 by -1 { */
    /* 	try cumulativeIV = incrementalArgSort(arrays[i], cumulativeIV); */
    /*   } */
    /*   return cumulativeIV; */
    /* } */

    /* Find the permutation that sorts multiple arrays, treating each array as a
       new level of the sorting key.
     */
    proc coargsortMsg(reqMsg: string, st: borrowed SymTab) {
      var pn = "coargsort";
      var repMsg: string;
      var fields = reqMsg.split();
      var cmd = fields[1];
      var n = try! fields[2]:int; // number of arrays to sort
      var names = fields[3..];
      // Check that fields contains the stated number of arrays
      if (n != names.size) { return try! incompatibleArgumentsError(pn, "Expected %i arrays but got %i".format(n, names.size)); }
      /* var arrays: [0..#n] borrowed GenSymEntry; */
      var size: int;
      // Check that all arrays exist in the symbol table and have the same size
      for (name, i) in zip(names, 1..) {
	// arrays[i] = st.lookup(name): borrowed GenSymEntry;
	var g: borrowed GenSymEntry = st.lookup(name);
	if (g == nil) { return unknownSymbolError(pn, name); }
	if (i == 1) {
	  size = g.size;
	} else {
	  if (g.size != size) { return incompatibleArgumentsError(pn, "Arrays must all be same size"); }
	}
      }
      // Initialize the permutation vector in the symbol table with the identity perm
      var rname = st.nextName();
      st.addEntry(rname, size, int);
      var iv = toSymEntry(st.lookup(rname), int);
      iv.a = 0..#size;
      // Starting with the last array, incrementally permute the IV by sorting each array
      for i in names.domain.low..names.domain.high by -1 {
	var g: borrowed GenSymEntry = st.lookup(names[i]);
	try {
	  // Perform the coArgSort and store in the new SymEntry
	  iv.a = incrementalArgSort(g, iv.a);
	} catch e:ErrorWithMsg {
	  // The only error thrown is for an unsupported dtype
	  return notImplementedError(pn, e.msg);
	} catch {
	  return try! "Error: %s unknown cause".format(pn);
	}
      }
      return try! "created " + st.attrib(rname);
    }
    
    proc argsortDefault(A:[?D] ?t):[D] int {
      var t1 = Time.getCurrentTime();
      //var AI = [(a, i) in zip(A, D)] (a, i);
      //Sort.TwoArrayRadixSort.twoArrayRadixSort(AI);
      //var iv = [(a, i) in AI] i;
      var iv = radixSortLSD_ranks(A);
      if v {writeln("argsort time = ", Time.getCurrentTime() - t1); try! stdout.flush();}
      return iv;
    }
    
    /* argsort takes pdarray and returns an index vector iv which sorts the array */
    proc argsortMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "argsort";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var ivname = st.nextName();
        if v {try! writeln("%s %s : %s %s".format(cmd, name, ivname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError(pn,name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var iv = argsortDefault(e.a);
                st.addEntry(ivname, new shared SymEntry(iv));
            }
	    when (DType.Float64) {
	        var e = toSymEntry(gEnt, real);
		var iv = argsortDefault(e.a);
		st.addEntry(ivname, new shared SymEntry(iv));
	    }
            otherwise {return notImplementedError(pn,gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(ivname);
    }

    /* localArgsort takes a pdarray and returns an index vector which sorts the array on a per-locale basis */
    proc localArgsortMsg(reqMsg: string, st: borrowed SymTab): string {
      var pn = "localArgsort";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var ivname = st.nextName();
        if v {try! writeln("%s %s : %s %s".format(cmd, name, ivname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError(pn,name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
		var iv = perLocaleArgSort(e.a);
		st.addEntry(ivname, new shared SymEntry(iv));
	    }
	    otherwise {return notImplementedError(pn,gEnt.dtype);}
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
}
