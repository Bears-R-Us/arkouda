/* arg sort algorithm
these pass back an index vector which can be used
to permute the original array into sorted order */

module ArgSortMsg
{
    use ServerConfig;
    
    use CPtr;

    use Time only;
    use Math only;
    use Sort only;
    use Reflection only;
    
    use PrivateDist;

    use CommAggregation;

    use AryUtil;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use RadixSortLSD;
    use SegmentedArray;
    use Reflection;
    use Errors;
    use Logging;

    const asLogger = new Logger();

    if v {
        asLogger.level = LogLevel.DEBUG;
    } else {
        asLogger.level = LogLevel.INFO;
    }
    
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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                    "bins = %t".format(bins)); */

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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                   "bins = %t".format(bins)); */

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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                             "bins = %t".format(bins));*/

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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                      "done atomicHist time = %i".format(Time.getCurrentTime() - t1);try! stdout.flush();}*/

    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // put counts into globalCounts array */
    /*             [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[here.id][i].read(); */
    /*         } */
    /*     } */
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                  "done copy to globalCounts time = %i".format(Time.getCurrentTime() - t1));*/

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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                  "done copy back to localCounts time = %i".format(Time.getCurrentTime() - t1));*/
        
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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "done sort locally and move segments time = %i".format(Time.getCurrentTime() - t1));*/
        
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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"bins = %t".format(bins));*/

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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                      "done atomicHist time = %i".format(Time.getCurrentTime() - t1));*/

    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // put counts into globalCounts array */
    /*             [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[here.id][i].read(); */
    /*         } */
    /*     } */
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "done copy to globalCounts time = %i".format(Time.getCurrentTime() - t1));*/

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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "done copy back to localCounts time = %i".format(Time.getCurrentTime() - t1));*/
        
    /*     // start timer */
    /*     t1 = Time.getCurrentTime(); */
    /*     coforall loc in Locales { */
    /*         on loc { */
    /*             // put locale-subbin-starts into atomic hist */
    /*             [i in hD] atomicHist[here.id][i].write(globalEnds[i * numLocales + here.id] - localCounts[here.id][i]); */
    /*         } */
    /*     } */
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                             "done init atomic counts time = %i".format(Time.getCurrentTime() - t1));*/
        
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
    /*     asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "done move time = %i".format(Time.getCurrentTime() - t1));*/

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
              forall (newai, idx) in zip(newa, iv) with (var agg = newSrcAggregator(int)) {
                  agg.copy(newai, olda[idx]);
              }
              // Generate the next incremental permutation
              deltaIV = radixSortLSD_ranks(newa);
          }
          when DType.Float64 {
              var e = toSymEntry(g, real);
              var newa: [e.aD] real;
              ref olda = e.a;
              forall (newai, idx) in zip(newa, iv) with (var agg = newSrcAggregator(real)) {
                  agg.copy(newai, olda[idx]);
              }
              deltaIV = radixSortLSD_ranks(newa);
          }
          otherwise { throw getErrorWithContext(
                                msg="Unsupported DataType: %t".format(dtype2str(g.dtype)),
                                lineNumber=getLineNumber(),
                                routineName=getRoutineName(),
                                moduleName=getModuleName(),
                                errorClass="IllegalArgumentError"
                                ); 
          }
      }
      // The output permutation is the composition of the initial and incremental permutations
      var newIV: [aD] int;
      // Effectively: newIV = iv[deltaIV] 
      forall (newIVi, idx) in zip(newIV, deltaIV) with (var agg = newSrcAggregator(int)) {
        agg.copy(newIVi, iv[idx]);
      }
      return newIV;
    }

    proc incrementalArgSort(s: SegString, iv: [?aD] int): [] int throws {
      var hashes = s.hash();
      var newHashes: [aD] 2*uint;
      forall (nh, idx) in zip(newHashes, iv) with (var agg = newSrcAggregator((2*uint))) {
        agg.copy(nh, hashes[idx]);
      }
      var deltaIV = radixSortLSD_ranks(newHashes);
      // var (newOffsets, newVals) = s[iv];
      // var deltaIV = newStr.argGroup();
      var newIV: [aD] int;
      forall (newIVi, idx) in zip(newIV, deltaIV) with (var agg = newSrcAggregator(int)) {
        agg.copy(newIVi, iv[idx]);
      }
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
    /*         try cumulativeIV = incrementalArgSort(arrays[i], cumulativeIV); */
    /*   } */
    /*   return cumulativeIV; */
    /* } */

    /* Find the permutation that sorts multiple arrays, treating each array as a
       new level of the sorting key.
     */
    proc coargsortMsg(cmd: string, payload: string, st: borrowed SymTab) throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string;
      var (nstr, rest) = payload.splitMsgToTuple(2);
      var n = nstr:int; // number of arrays to sort
      var fields = rest.split();
      asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                  "number of arrays: %i fields: %t".format(n,fields));
      // Check that fields contains the stated number of arrays
      if (fields.size != 2*n) { 
          var errorMsg = incompatibleArgumentsError(pn, 
                        "Expected %i arrays but got %i".format(n, fields.size/2 - 1));
          asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return errorMsg;
      }
      const low = fields.domain.low;
      var names = fields[low..#n];
      var types = fields[low+n..#n];
      /* var arrays: [0..#n] borrowed GenSymEntry; */
      var size: int;
      // Check that all arrays exist in the symbol table and have the same size
      var hasStr = false;
      for (name, objtype, i) in zip(names, types, 1..) {
        // arrays[i] = st.lookup(name): borrowed GenSymEntry;
        var thisSize: int;
        select objtype {
          when "pdarray" {
            var g = st.lookup(name);
            thisSize = g.size;
          }
          when "str" {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = st.lookup(myNames);
            thisSize = g.size;
            hasStr = true;
          }
          otherwise {return unrecognizedTypeError(pn, objtype);
              var errorMsg = unrecognizedTypeError(pn, objtype);
              asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
              return errorMsg;
          }
        }
        
        if (i == 1) {
            size = thisSize;
        } else {
            if (thisSize != size) { 
                var errorMsg = incompatibleArgumentsError(pn, 
                                               "Arrays must all be same size");
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return errorMsg;
            }
        }
        
      }

      // If there were no string arrays, merge the arrays into a single array and sort
      // that. This eliminates having to merge index vectors, but has a memory overhead
      // and increases the size of the comm we have to do since the KEY is larger). We
      // merge the elements into a `uint(RSLSD_bitsPerDigit)` tuple. This wastes space
      // (e.g. when merging 2 arrays that use 7 and 9 bits), but it allows us to use
      // `getDigit`, which changes the bit patterns to correctly sort negatives. We
      // consider tuple[1] to be the most significant digit.
      //
      // TODO support string? This further increases size (128-bits for each hash), so we
      // need to be OK with memory overhead and comm from the KEY)
      if !hasStr {
        param bitsPerDigit = RSLSD_bitsPerDigit;
        var bitWidths: [names.domain] int;
        var negs: [names.domain] bool;
        var totalDigits: int;

        for (bitWidth, name, neg) in zip(bitWidths, names, negs) {
          // TODO checkSorted and exclude array if already sorted?
          var g: borrowed GenSymEntry = st.lookup(name);
          select g.dtype {
              when DType.Int64   { (bitWidth, neg) = getBitWidth(toSymEntry(g, int ).a); }
              when DType.Float64 { (bitWidth, neg) = getBitWidth(toSymEntry(g, real).a); }
              otherwise          { 
                                     throw getErrorWithContext(
                                         msg=dtype2str(g.dtype),
                                         lineNumber=getLineNumber(),
                                         routineName=getRoutineName(),
                                         moduleName=getModuleName(),
                                         errorClass="ErrorWithContext"
                                     );
                                 }
          }
          totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;
        }

        // TODO support arbitrary size with array-of-arrays or segmented array
        proc mergedArgsort(param numDigits) throws {

          overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
                       + (2 * here.maxTaskPar * numLocales * 2**16 * 8));

          var ivname = st.nextName();
          var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
          var curDigit = RSLSD_tupleLow + numDigits - totalDigits;
          for (name, nBits, neg) in zip(names, bitWidths, negs) {
              var g: borrowed GenSymEntry = st.lookup(name);
              proc mergeArray(type t) {
                var e = toSymEntry(g, t);
                ref A = e.a;

                const r = 0..#nBits by bitsPerDigit;
                for rshift in r {
                  const myDigit = (r.high - rshift) / bitsPerDigit;
                  const last = myDigit == 0;
                  forall (m, a) in zip(merged, A) {
                    m[curDigit+myDigit] =  getDigit(a, rshift, last, neg):uint(bitsPerDigit);
                  }
                }
                curDigit += r.size;
              }
              select g.dtype {
                when DType.Int64   { mergeArray(int); }
                when DType.Float64 { mergeArray(real); }
                otherwise          { 
                                       throw getErrorWithContext(
                                                msg=dtype2str(g.dtype),
                                                lineNumber=getLineNumber(),
                                                routineName=getRoutineName(),
                                                moduleName=getModuleName(),
                                                errorClass="IllegalArgumentError"
                                       ); 
                                     
                                   }
              }
          }

          var iv = argsortDefault(merged);
          st.addEntry(ivname, new shared SymEntry(iv));
          return try! "created " + st.attrib(ivname);
        }

        // Since we're using tuples, we have to stamp out for each size we want to
        // support. For now support 8, 16, and 32 byte sorting.
        if totalDigits <=  4 { return mergedArgsort( 4); }
        if totalDigits <=  8 { return mergedArgsort( 8); }
        if totalDigits <= 16 { return mergedArgsort(16); }
      }

      // check and throw if over memory limit
      overMemLimit(((4 + 3) * size * numBytes(int))
                   + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
      
      // Initialize the permutation vector in the symbol table with the identity perm
      var rname = st.nextName();
      st.addEntry(rname, size, int);
      var iv = toSymEntry(st.lookup(rname), int);
      iv.a = 0..#size;
      // Starting with the last array, incrementally permute the IV by sorting each array
      for (i, j) in zip(names.domain.low..names.domain.high by -1,
                        types.domain.low..types.domain.high by -1) {
        if (types[j] == "str") {
          var (myNames1,myNames2) = names[i].splitMsgToTuple('+', 2);
          var strings = new owned SegString(myNames1, myNames2, st);
          iv.a = incrementalArgSort(strings, iv.a);
        } else {
          var g: borrowed GenSymEntry = st.lookup(names[i]);
          // Perform the coArgSort and store in the new SymEntry
          iv.a = incrementalArgSort(g, iv.a);
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
      try! asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                             "argsort time = %i".format(Time.getCurrentTime() - t1));
      return iv;
    }
    
    /* argsort takes pdarray and returns an index vector iv which sorts the array */
    proc argsortMsg(cmd: string, payload: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (objtype, name) = payload.splitMsgToTuple(2);

        // get next symbol name
        var ivname = st.nextName();
        asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "cmd: %s name: %s ivname: %s".format(cmd, name, ivname));

        select objtype {
          when "pdarray" {
            var gEnt: borrowed GenSymEntry = st.lookup(name);
            // check and throw if over memory limit
            overMemLimit(((4 + 1) * gEnt.size * gEnt.itemsize)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
        
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
                otherwise {
                    var errorMsg = notImplementedError(pn,gEnt.dtype);
                    asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);               
                    return errorMsg;
                }
            }
          }
          when "str" {
            var (names1, names2) = name.splitMsgToTuple('+', 2);
            var strings = new owned SegString(names1, names2, st);
            // check and throw if over memory limit
            overMemLimit((8 * strings.size * 8)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
            var iv = strings.argsort();
            st.addEntry(ivname, new shared SymEntry(iv));
          }
          otherwise {
              var errorMsg = notImplementedError(pn, objtype);
              asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                    
              return errorMsg;
          }
        }
        repMsg = "created " + st.attrib(ivname);
        asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return repMsg;
    }
}
