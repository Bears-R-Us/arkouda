module SegmentedArray {
  use AryUtil;
  use CPtr;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use CommAggregation;
  use UnorderedCopy;
  use SipHash;
  use SegStringSort;
  use RadixSortLSD only radixSortLSD_ranks;
  use PrivateDist;
  use ServerConfig;
  use Unique;
  use Time only Timer, getCurrentTime;
  use Reflection;
  use Logging;
  use ServerErrors;
  use ArkoudaRegexCompat;

  private config const logLevel = ServerConfig.logLevel;
  const saLogger = new Logger(logLevel);

  private config param useHash = true;
  param SegmentedArrayUseHash = useHash;

  private config param regexMaxCaptures = ServerConfig.regexMaxCaptures;
  
  class OutOfBoundsError: Error {}

  /* 
   * This version of the getSegString method is the most common and is only used 
   * when the names of the segments (offsets) and values SymEntries are known.
   */
  proc getSegString(offsetName: string, valName: string, 
                                          st: borrowed SymTab): owned SegString throws {
      var offsetEntry = st.lookup(offsetName);
      var valEntry = st.lookup(valName);
      return new owned SegString(offsetEntry, offsetName, valEntry, valName, st);
  }

  /*
   * This version of the getSegString method takes segments and values arrays as
   * inputs, generates the SymEntry objects for each and passes the
   * offset and value SymTab lookup names to the alternate init method
   */
  proc getSegString(segments: [] int, values: [] uint(8), 
                                          st: borrowed SymTab): SegString throws {
      var offsetName = st.nextName();
      var offsetEntry = new shared SymEntry(segments);
      st.addEntry(offsetName, offsetEntry);

      var valName = st.nextName();
      var valEntry = new shared SymEntry(values);
      st.addEntry(valName, valEntry);

      return new SegString(offsetEntry, offsetName, valEntry, valName, st);
  }

  /**
   * Represents an array of strings, implemented as a segmented array of bytes.
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining string-relevant
   * operations.
   */
  class SegString {
 
    /**
     * The name of the SymEntry corresponding to the pdarray containing
     * the offsets, which are start indices for each string bytearray
     */
    var offsetName: string;

    /**
     * The pdarray containing the offsets, which are the start indices of
     * the bytearrays, each of which corresponds to an individual string.
     */ 
    var offsets: borrowed SymEntry(int);

    /**
     * The name of the SymEntry corresponding to the pdarray containing
     * the string values where each value is byte array.
     */
    var valueName: string;

    /**
     * The pdarray containing the complete byte array composed of bytes
     * corresponding to each string, joined by nulls. Note: the null byte
     * is uint(8) value of zero.
     */ 
    var values: borrowed SymEntry(uint(8));
    
    /**
     * The number of strings in the segmented array
     */
    var size: int;
    
    /**
     * The total number of bytes in the entire segmented array including
     * the bytes corresonding to the strings as well as the nulls
     * separating the string bytes.
     */ 
    var nBytes: int;

    /* 
     * This method should not be called directly. Instead, call one of the
     * getSegString factory methods.
     */
    proc init(offsetEntry: borrowed GenSymEntry, segsName: string, 
                   valEntry: borrowed GenSymEntry, valName: string, st: borrowed SymTab) {
      offsetName = segsName;
      //Must be unmanaged because borrowed throws a lifetime error
      offsets = toSymEntry(offsetEntry, int): unmanaged SymEntry(int);

      valueName = valName;
      //Must be unmanaged because borrowed throws a lifetime error
      values = toSymEntry(valEntry, uint(8)): unmanaged SymEntry(uint(8));

      size = offsets.size;
      nBytes = values.size;
    }

    proc show(n: int = 3) throws {
      if (size >= 2*n) {
        for i in 0..#n {
            saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
        for i in size-n..#n {
            saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
      } else {
        for i in 0..#size {
            saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
      }
    }

    /* Retrieve one string from the array */
    proc this(idx: int): string throws {
      if (idx < offsets.aD.low) || (idx > offsets.aD.high) {
        throw new owned OutOfBoundsError();
      }
      // Start index of the string
      var start = offsets.a[idx];
      // Index of last (null) byte in string
      var end: int;
      if (idx == size - 1) {
        end = nBytes - 1;
      } else {
        end = offsets.a[idx+1] - 1;
      }
      // Take the slice of the bytearray and "cast" it to a chpl string
      var s = interpretAsString(values.a, start..end);
      return s;
    }

    /* Take a slice of strings from the array. The slice must be a 
       Chapel range, i.e. low..high by stride, not a Python slice.
       Returns arrays for the segment offsets and bytes of the slice.*/
    proc this(const slice: range(stridable=true)) throws {
      if (slice.low < offsets.aD.low) || (slice.high > offsets.aD.high) {
          saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
          "Array is out of bounds");
          throw new owned OutOfBoundsError();
      }
      // Early return for zero-length result
      if (size == 0) || (slice.size == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      // Start of bytearray slice
      var start = offsets.a[slice.low];
      // End of bytearray slice
      var end: int;
      if (slice.high == offsets.aD.high) {
        // if slice includes the last string, go to the end of values
        end = values.aD.high;
      } else {
        end = offsets.a[slice.high+1] - 1;
      }
      // Segment offsets of the new slice
      var newSegs = makeDistArray(slice.size, int);
      ref oa = offsets.a;
      // newSegs = offsets.a[slice] - start;
      forall (i, ns) in zip(newSegs.domain, newSegs) with (var agg = newSrcAggregator(int)) {
        agg.copy(ns, oa[slice.low + i]);
      }
      // Offsets need to be re-zeroed
      newSegs -= start;
      // Bytearray of the new slice
      var newVals = makeDistArray(end - start + 1, uint(8));
      ref va = values.a;
      // newVals = values.a[start..end];
      forall (i, nv) in zip(newVals.domain, newVals) with (var agg = newSrcAggregator(uint(8))) {
        agg.copy(nv, va[start + i]);
      }
      return (newSegs, newVals);
    }

    /* Gather strings by index. Returns arrays for the segment offsets
       and bytes of the gathered strings.*/
    proc this(iv: [?D] int) throws {
      // Early return for zero-length result
      if (D.size == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      // Check all indices within bounds
      var ivMin = min reduce iv;
      var ivMax = max reduce iv;
      if (ivMin < 0) || (ivMax >= offsets.size) {
          saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "Array out of bounds");
          throw new owned OutOfBoundsError();
      }
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                              "Computing lengths and offsets");
      var t1 = getCurrentTime();
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // Gather the right and left boundaries of the indexed strings
      // NOTE: cannot compute lengths inside forall because agg.copy will
      // experience race condition with loop-private variable
      var right: [D] int, left: [D] int;
      forall (r, l, idx) in zip(right, left, iv) with (var agg = newSrcAggregator(int)) {
        if (idx == high) {
          agg.copy(r, values.size);
        } else {
          agg.copy(r, oa[idx+1]);
        }
        agg.copy(l, oa[idx]);
      }
      // Lengths of segments including null bytes
      var gatheredLengths: [D] int = right - left;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * gatheredLengths.size);
      // The returned offsets are the 0-up cumulative lengths
      var gatheredOffsets = (+ scan gatheredLengths);
      // The total number of bytes in the gathered strings
      var retBytes = gatheredOffsets[D.high];
      gatheredOffsets -= gatheredLengths;
      
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                "aggregation in %i seconds".format(getCurrentTime() - t1));
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Copying values");
      if logLevel == LogLevel.DEBUG {
          t1 = getCurrentTime();
      }
      var gatheredVals = makeDistArray(retBytes, uint(8));
      // Multi-locale requires some extra localization work that is not needed
      // in CHPL_COMM=none
      if CHPL_COMM != 'none' {
        // Compute the src index for each byte in gatheredVals
        /* For performance, we will do this with a scan, so first we need an array
           with the difference in index between the current and previous byte. For
           the interior of a segment, this is just one, but at the segment boundary,
           it is the difference between the src offset of the current segment ("left")
           and the src index of the last byte in the previous segment (right - 1).
        */
        var srcIdx = makeDistArray(retBytes, int);
        srcIdx = 1;
        var diffs: [D] int;
        diffs[D.low] = left[D.low]; // first offset is not affected by scan
        if (D.size > 1) {
          // This expression breaks when D.size == 1, resulting in strange behavior
          // However, this logic is only necessary when D.size > 1 anyway
          diffs[D.interior(D.size-1)] = left[D.interior(D.size-1)] - (right[D.interior(-(D.size-1))] - 1);
        }
        // Set srcIdx to diffs at segment boundaries
        forall (go, d) in zip(gatheredOffsets, diffs) with (var agg = newDstAggregator(int)) {
          agg.copy(srcIdx[go], d);
        }
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * srcIdx.size);
        srcIdx = + scan srcIdx;
        // Now srcIdx has a dst-local copy of the source index and vals can be efficiently gathered
        ref va = values.a;
        forall (v, si) in zip(gatheredVals, srcIdx) with (var agg = newSrcAggregator(uint(8))) {
          agg.copy(v, va[si]);
        }
      } else {
        ref va = values.a;
        // Copy string data to gathered result
        forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, iv) {
          for pos in 0..#gl {
            gatheredVals[go+pos] = va[oa[idx]+pos];
          }
        }
      }
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "Gathered offsets and vals in %i seconds".format(
                                           getCurrentTime() -t1));
      return (gatheredOffsets, gatheredVals);
    }

    /* Logical indexing (compress) of strings. */
    proc this(iv: [?D] bool) throws {
      // Index vector must be same domain as array
      if (D != offsets.aD) {
          saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "Array out of bounds");
          throw new owned OutOfBoundsError();
      }
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                 "Computing lengths and offsets");
      var t1 = getCurrentTime();
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * iv.size);
      // Calculate the destination indices
      var steps = + scan iv;
      var newSize = steps[high];
      steps -= iv;
      // Early return for zero-length result
      if (newSize == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      var segInds = makeDistArray(newSize, int);
      forall (t, dst, idx) in zip(iv, steps, D) with (var agg = newDstAggregator(int)) {
        if t {
          agg.copy(segInds[dst], idx);
        }
      }
      return this[segInds];
      
      /* // Lengths of dest segments including null bytes */
      /* var gatheredLengths = makeDistArray(newSize, int); */
      /* forall (idx, present, i) in zip(D, iv, steps) { */
      /*   if present { */
      /*     segInds[i-1] = idx; */
      /*     if (idx == high) { */
      /*       gatheredLengths[i-1] = values.size - oa[high]; */
      /*     } else { */
      /*       gatheredLengths[i-1] = oa[idx+1] - oa[idx]; */
      /*     } */
      /*   } */
      /* } */
      /* // Make dest offsets from lengths */
      /* var gatheredOffsets = (+ scan gatheredLengths); */
      /* var retBytes = gatheredOffsets[newSize-1]; */
      /* gatheredOffsets -= gatheredLengths; */
      /* if logLevel == LogLevel.DEBUG { */
      /*     saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                  "%i seconds".format(getCurrentTime() - t1)); */
      /*     saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Copying values"));*/
      /*     t1 = getCurrentTime(); */
      /* } */
      /* var gatheredVals = makeDistArray(retBytes, uint(8)); */
      /* ref va = values.a; */
      /* if logLevel == LogLevel.DEBUG { */
      /*   printAry("gatheredOffsets: ", gatheredOffsets); */
      /*   printAry("gatheredLengths: ", gatheredLengths); */
      /*   printAry("segInds: ", segInds); */
      /* } */
      /* // Copy string bytes from src to dest */
      /* forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, segInds) { */
      /*   gatheredVals[{go..#gl}] = va[{oa[idx]..#gl}]; */
      /* } */
      /* saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "%i seconds".format(getCurrentTime() - t1));*/
      /* return (gatheredOffsets, gatheredVals); */
    }

    /* Apply a hash function to all strings. This is useful for grouping
       and set membership. The hash used is SipHash128.*/
    proc hash() throws {
      // 128-bit hash values represented as 2-tuples of uint(64)
      var hashes: [offsets.aD] 2*uint(64);
      // Early exit for zero-length result
      if (size == 0) {
        return hashes;
      }
      ref oa = offsets.a;
      ref va = values.a;
      // Compute lengths of strings
      var lengths = getLengths();
      // Hash each string
      // TO DO: test on clause with aggregator
      forall (o, l, h) in zip(oa, lengths, hashes) {
        const myRange = o..#l;
        h = sipHash128(va, myRange);
        /* // localize the string bytes */
        /* const myBytes = va[{o..#l}]; */
        /* h = sipHash128(myBytes, hashKey); */
        /* // Perf Note: localizing string bytes is ~3x faster on IB multilocale than this: */
        /* // h = sipHash128(va[{o..#l}]); */
      }
      return hashes;
    }

    /* Return a permutation that groups the strings. Because hashing is used,
       this permutation will not sort the strings, but all equivalent strings
       will fall in one contiguous block. */
    proc argGroup() throws {
      var t = new Timer();
      if useHash {
        // Hash all strings
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Hashing strings"); 
        if logLevel == LogLevel.DEBUG { t.start(); }
        var hashes = this.hash();

        if logLevel == LogLevel.DEBUG { 
            t.stop();    
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "hashing took %t seconds\nSorting hashes".format(t.elapsed())); 
            t.clear(); t.start(); 
        }

        // Return the permutation that sorts the hashes
        var iv = radixSortLSD_ranks(hashes);
        if logLevel == LogLevel.DEBUG { 
            t.stop(); 
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "sorting took %t seconds".format(t.elapsed())); 
        }
        if logLevel == LogLevel.DEBUG {
          var sortedHashes = [i in iv] hashes[i];
          var diffs = sortedHashes[(iv.domain.low+1)..#(iv.size-1)] - 
                                                 sortedHashes[(iv.domain.low)..#(iv.size-1)];
          printAry("diffs = ", diffs);
          var nonDecreasing = [(d0,d1) in diffs] ((d0 > 0) || ((d0 == 0) && (d1 >= 0)));
          saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                    "Are hashes sorted? %i".format(&& reduce nonDecreasing));
        }
        return iv;
      } else {
        var iv = argsort();
        return iv;
      }
    }

    /* Return lengths of all strings, including null terminator. */
    proc getLengths() {
      var lengths: [offsets.aD] int;
      if (size == 0) {
        return lengths;
      }
      ref oa = offsets.a;
      const low = offsets.aD.low;
      const high = offsets.aD.high;
      forall (i, o, l) in zip(offsets.aD, oa, lengths) {
        if (i == high) {
          l = values.size - o;
        } else {
          l = oa[i+1] - o;
        }
      }
      /* lengths[low..high-1] = (oa[low+1..high] - oa[low..high-1]); */
      /* lengths[high] = values.size - oa[high]; */
      return lengths;
    }

    proc findSubstringInBytes(const substr: string) {
      // Find the start position of every occurence of substr in the flat bytes array
      // Start by making a right-truncated subdomain representing all valid starting positions for substr of given length
      var D: subdomain(values.aD) = values.aD[values.aD.low..#(values.size - substr.numBytes + 1)];
      // Every start position is valid until proven otherwise
      var truth: [D] bool = true;
      // Shift the flat values one byte at a time and check against corresponding byte of substr
      for (i, b) in zip(0.., substr.chpl_bytes()) {
        truth &= (values.a[D.translate(i)] == b);
      }
      return truth;
    }

    /*
      Returns Regexp.compile if pattern can be compiled without an error
    */
    proc checkCompile(const pattern: ?t) throws where t == bytes || t == string {
      try {
        return compile(pattern);
      }
      catch {
        var errorMsg = "re2 could not compile pattern: %s)".format(pattern);
        saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        throw new owned IllegalArgumentError(errorMsg);
      }
    }

    proc _unsafeCompileRegex(const pattern: ?t) where t == bytes || t == string {
      // This is a private function and should not be called to compile pattern. Use checkCompile instead

      // This proc is a workaound to allow declaring regexps using a with clause in forall loops
      // since using declarations with throws are illegal
      // It is only called after checkCompile so the try! will not result in a server crash
      return try! compile(pattern);
    }

    /*
      Given a SegString, finds pattern matches and returns pdarrays containing the number, start postitions, and lengths of matches
      :arg pattern: The regex pattern used to find matches
      :type pattern: string
      :arg groupNum: The number of the capture group to be returned
      :type groupNum: int
      :returns: int64 pdarray – For each original string, the number of pattern matches and int64 pdarray – The start positons of pattern matches and int64 pdarray – The lengths of pattern matches
    */
    proc findMatchLocations(const pattern: string, groupNum: int) throws {
      checkCompile(pattern);
      ref origOffsets = this.offsets.a;
      ref origVals = this.values.a;
      const lengths = this.getLengths();

      overMemLimit((4 * this.offsets.size * numBytes(int)) + (3 * this.values.size * numBytes(int)));
      var numMatches: [this.offsets.aD] int;
      var matchStartBool: [this.values.aD] bool = false;
      var sparseLens: [this.values.aD] int;
      var sparseStarts: [this.values.aD] int;
      var searchBools: [this.offsets.aD] bool = false;
      var matchBools: [this.offsets.aD] bool = false;
      var fullMatchBools: [this.offsets.aD] bool = false;
      forall (i, off, len) in zip(this.offsets.aD, origOffsets, lengths) with (var myRegex = _unsafeCompileRegex(pattern),
                                                                               var lenAgg = newDstAggregator(int),
                                                                               var startPosAgg = newDstAggregator(int),
                                                                               var startBoolAgg = newDstAggregator(bool),
                                                                               var searchBoolAgg = newDstAggregator(bool),
                                                                               var matchBoolAgg = newDstAggregator(bool),
                                                                               var fullMatchBoolAgg = newDstAggregator(bool),
                                                                               var numMatchAgg = newDstAggregator(int)) {
        var matchessize = 0;
        for m in myRegex.matches(interpretAsString(origVals, off..#len, borrow=true), captures=regexMaxCaptures) {
          var match = m[0]; // v1.24.x -> reMatch, v1.25.x -> regexMatch
          var group = m[groupNum];
          if group.offset != -1 {
            lenAgg.copy(sparseLens[off + group.offset:int], group.size);
            startPosAgg.copy(sparseStarts[off + group.offset:int], group.offset:int);
            startBoolAgg.copy(matchStartBool[off + group.offset:int], true);
            matchessize += 1;
            searchBoolAgg.copy(searchBools[i], true);
            if match.offset == 0 {
              matchBoolAgg.copy(matchBools[i], true);
            }
            if match.size == len-1 {
              fullMatchBoolAgg.copy(fullMatchBools[i], true);
            }
          }
        }
        numMatchAgg.copy(numMatches[i], matchessize);
      }
      var totalMatches = + reduce numMatches;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit((2 * numBytes(int) * matchStartBool.size) + (3 * numBytes(int) * searchBools.size));

      // the matchTransform starts at 0 and increment after hitting a matchStart
      // when looping over the origVals domain, matchTransform acts as a function: origVals.domain -> makeDistDom(totalMatches)
      var matchTransform = + scan matchStartBool - matchStartBool;
      var matchesIndicies = (+ scan numMatches) - numMatches;
      var matchStarts: [makeDistDom(totalMatches)] int;
      var matchLens: [makeDistDom(totalMatches)] int;
      [i in this.values.aD] if (matchStartBool[i] == true) {
        matchStarts[matchTransform[i]] = sparseStarts[i];
        matchLens[matchTransform[i]] = sparseLens[i];
      }
      var searchScan = (+ scan searchBools) - searchBools;
      var matchScan = (+ scan matchBools) - matchBools;
      var fullMatchScan = (+ scan fullMatchBools) - fullMatchBools;
      return (numMatches, matchStarts, matchLens, matchesIndicies, searchBools, searchScan, matchBools, matchScan, fullMatchBools, fullMatchScan);
    }

    /*
      Given a SegString, return a new SegString only containing matches of the regex pattern,
      If returnMatchOrig is set to True, return a pdarray containing the index of the original string each pattern match is from
      :arg numMatchesEntry: For each string in SegString, the number of pattern matches
      :type numMatchesEntry: borrowed SymEntry(int) or borrowed SysmEntry(bool)
      :arg startsEntry: The starting postions of pattern matches
      :type startsEntry: borrowed SymEntry(int)
      :arg lensEntry: The lengths of pattern matches
      :type lensEntry: borrowed SymEntry(int)
      :arg returnMatchOrig: If True, return a pdarray containing the index of the original string each pattern match is from
      :type returnMatchOrig: bool
      :returns: Strings – Only the portions of Strings which match pattern and (optional) int64 pdarray – For each pattern match, the index of the original string it was in
    */
    proc findAllMatches(const numMatchesEntry: ?t, const startsEntry: borrowed SymEntry(int), const lensEntry: borrowed SymEntry(int), const indicesEntry: borrowed SymEntry(int), const returnMatchOrig: bool) throws where t == borrowed SymEntry(int) || t == borrowed SymEntry(bool) {
      ref origVals = this.values.a;
      ref origOffsets = this.offsets.a;
      ref numMatches = numMatchesEntry.a;
      ref matchStarts = startsEntry.a;
      ref matchLens = lensEntry.a;
      ref indices = indicesEntry.a;

      overMemLimit(matchLens.size * numBytes(int));
      var absoluteStarts: [makeDistDom(matchLens.size)] int;
      forall (off, numMatch, matchInd) in zip(origOffsets, numMatches, indices) with (var absAgg = newDstAggregator(int)) {
        var localizedStarts = new lowLevelLocalizingSlice(matchStarts, matchInd..#numMatch);
        for k in 0..#numMatch {
          // Each string has numMatches[stringInd] number of pattern matches, so matchOrigins needs to repeat stringInd for numMatches[stringInd] times
          absAgg.copy(absoluteStarts[matchInd+k], localizedStarts.ptr[k] + off);
        }
      }

      // matchesValsSize is the total length of all matches + the number of matches (to account for null bytes)
      var matchesValsSize = (+ reduce matchLens) + matchLens.size;
      // check there's enough room to create a copy for scan and to allocate matchesVals/Offsets
      overMemLimit((matchesValsSize * numBytes(uint(8))) + (2 * matchLens.size * numBytes(int)));
      var matchesVals: [makeDistDom(matchesValsSize)] uint(8);
      var matchesOffsets: [makeDistDom(matchLens.size)] int;
      // + current index to account for null bytes
      var matchesIndicies = + scan matchLens - matchLens + lensEntry.aD;
      forall (i, start, len, matchesInd) in zip(lensEntry.aD, absoluteStarts, matchLens, matchesIndicies) with (var valAgg = newDstAggregator(uint(8)), var offAgg = newDstAggregator(int)) {
        for j in 0..#len {
          // copy in match
          valAgg.copy(matchesVals[matchesInd + j], origVals[start + j]);
        }
        // write null byte after each match
        valAgg.copy(matchesVals[matchesInd + len], 0:uint(8));
        if i == 0 {
          offAgg.copy(matchesOffsets[i], 0);
        }
        if i != lensEntry.aD.high {
          offAgg.copy(matchesOffsets[i+1], matchesInd + len + 1);
        }
      }

      // build matchOrigins mapping from matchesStrings (pattern matches) to the original Strings they were found in
      const matchOriginsDom = if returnMatchOrig then makeDistDom(matchesOffsets.size) else makeDistDom(0);
      var matchOrigins: [matchOriginsDom] int;
      if returnMatchOrig {
        forall (stringInd, matchInd) in zip(this.offsets.aD, indices) with (var originAgg = newDstAggregator(int)) {
          for k in matchInd..#numMatches[stringInd] {
            // Each string has numMatches[stringInd] number of pattern matches, so matchOrigins needs to repeat stringInd for numMatches[stringInd] times
            originAgg.copy(matchOrigins[k], stringInd);
          }
        }
      }
      return (matchesOffsets, matchesVals, matchOrigins);
    }

    /*
      Substitute pattern matches with repl. If count is nonzero, at most count substitutions occur
      If returnNumSubs is set to True, the number of substitutions per string will be returned

      :arg pattern: regex pattern used to find matches
      :type pattern: string

      :arg replStr: the string to replace pattern matches with
      :type replStr: string

      :arg initCount: If count is nonzero, at most count splits occur. If zero, substitute all occurences of pattern
      :type initCount: int

      :arg returnNumSubs: If True, also return the number of substitutions per string
      :type returnNumSubs: bool

      :returns: Strings – Substrings with pattern matches substituted and (optional) int64 pdarray – For each original string, the number of susbstitutions
    */
    proc sub(pattern: string, replStr: string, initCount: int, returnNumSubs: bool) throws {
      checkCompile(pattern);
      ref origOffsets = this.offsets.a;
      ref origVals = this.values.a;
      const lengths = this.getLengths();

      overMemLimit((2 * this.offsets.size * numBytes(int)) + (2 * this.values.size * numBytes(int)));
      var numReplacements: [this.offsets.aD] int;
      var replacedLens: [this.offsets.aD] int;
      var nonMatch: [this.values.aD] bool = true;
      var matchStartBool: [this.values.aD] bool = false;

      var repl = replStr:bytes;
      // count = 0 means substitute all occurances, so we set count equal to 10**9
      var count = if initCount == 0 then 10**9:int else initCount;
      // since the pattern matches are variable length, we don't know what the size of subbedVals should be until we've found the matches
      forall (i, off, len) in zip(this.offsets.aD, origOffsets, lengths) with (var myRegex = _unsafeCompileRegex(pattern),
                                                                               var numReplAgg = newDstAggregator(int),
                                                                               var LenAgg = newDstAggregator(int),
                                                                               var nonMatchAgg = newDstAggregator(bool),
                                                                               var startAgg = newDstAggregator(bool)) {
        var replacementCounter = 0;
        var replLen = 0;
        for m in myRegex.matches(interpretAsString(origVals, off..#len, borrow=true)) {
          var match = m[0];  // v1.24.x -> reMatch, v1.25.x -> regexMatch
          for k in (off + match.offset:int)..#match.size {
            nonMatchAgg.copy(nonMatch[k], false);
          }
          startAgg.copy(matchStartBool[off + match.offset:int], true);
          replacementCounter += 1;
          replLen += match.size;
          if replacementCounter == count { break; }
        }
        numReplAgg.copy(numReplacements[i], replacementCounter);
        LenAgg.copy(replacedLens[i], replLen);
      }
      // new val size is the original - (total length of replacements) + (repl.size * total numReplacements)
      const valSize = this.values.size - (+ reduce replacedLens) + (repl.size * (+ reduce numReplacements));
      const valDom = makeDistDom(valSize);
      var subbedVals: [valDom] uint(8);
      overMemLimit(2 * this.offsets.size * numBytes(int));
      // new offsets can be directly calculated
      // new offsets is the original - (running sum of replaced lens) + (running sum of replacements * repl.size)
      var subbedOffsets = origOffsets - ((+ scan replacedLens) - replacedLens) + (repl.size * ((+ scan numReplacements) - numReplacements));

      forall (subOff, origOff, origLen) in zip(subbedOffsets, origOffsets, lengths) with (var valAgg = newDstAggregator(uint(8))) {
        var j = 0;
        var localizedVals = new lowLevelLocalizingSlice(origVals, origOff..#origLen);
        for i in 0..#origLen {
          if nonMatch[origOff + i] {
            valAgg.copy(subbedVals[subOff+j], localizedVals.ptr[i]);
            j += 1;
          }
          else if matchStartBool[origOff + i] {
            for k in repl {
              valAgg.copy(subbedVals[subOff+j], k:uint(8));
              j += 1;
            }
          }
        }
      }
      return (subbedOffsets, subbedVals, numReplacements);
    }

    /*
      Returns list of bools where index i indicates whether the regular expression, pattern, matched string i of the SegString

      Note: the regular expression engine used, re2, does not support lookahead/lookbehind

      :arg pattern: regex pattern to be applied to strings in SegString
      :type pattern: string

      :arg mode: mode of search being performed (contains, startsWith, endsWith, match)
      :type mode: SearchMode enum

      :returns: [domain] bool where index i indicates whether the regular expression, pattern, matched string i of the SegString
    */
    // DEPRECATED - All substringSearchRegex calls now handled by Match objects on Client
    // TODO: Remove substringSearchRegex
    proc substringSearchRegex(const pattern: string, mode: SearchMode) throws {
      var hits: [offsets.aD] bool = false;  // the answer
      checkCompile(pattern);

      // should we do len check here? re2.compile('') is valid regex and matches everything
      ref oa = offsets.a;
      ref va = values.a;
      var lengths = getLengths();

      select mode {
        when SearchMode.contains {
          forall (o, l, h) in zip(oa, lengths, hits) with (var myRegex = _unsafeCompileRegex(pattern)) {
            // regexp.search searches the receiving string for matches at any offset
            h = myRegex.search(interpretAsString(va, o..#l, borrow=true)).matched;
          }
        }
        when SearchMode.startsWith {
          forall (o, l, h) in zip(oa, lengths, hits) with (var myRegex = _unsafeCompileRegex(pattern)) {
            // regexp.match only returns a match if the start of the string matches the pattern
            h = myRegex.match(interpretAsString(va, o..#l, borrow=true)).matched;
          }
        }
        when SearchMode.endsWith {
          forall (o, l, h) in zip(oa, lengths, hits) with (var myRegex = _unsafeCompileRegex(pattern)) {
            var matches = myRegex.matches(interpretAsString(va, o..#l, borrow=true));
            var lastMatch = matches[matches.size-1][0];  // v1.24.x reMatch, 1.25.x regexMatch
            // h = true iff start(lastMatch) + len(lastMatch) == len(string) (-1 to account for null byte)
            h = lastMatch.offset + lastMatch.size == l-1;
          }
        }
      }
      return hits;
    }
    
    proc substringSearch(const substr: string, mode: SearchMode, regex: bool = false) throws {
      if regex || mode == SearchMode.match {
        // match always uses substringSearchRegex
        return substringSearchRegex(substr, mode);
      }
      var hits: [offsets.aD] bool;  // the answer
      if (size == 0) || (substr.size == 0) {
        return hits;
      }
      var t = new Timer();

      if logLevel == LogLevel.DEBUG {
           saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "Checking bytes of substr"); 
           t.start();
      }
      const truth = findSubstringInBytes(substr);
      const D = truth.domain;
      if logLevel == LogLevel.DEBUG {
            t.stop(); 
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                  "took %t seconds\nTranslating to segments...".format(t.elapsed())); 
            t.clear(); 
            t.start();
      }
      // Need to ignore segment(s) at the end of the array that are too short to contain substr
      const tail = + reduce (offsets.a > D.high);
      // oD is the right-truncated domain representing segments that are candidates for containing substr
      var oD: subdomain(offsets.aD) = offsets.aD[offsets.aD.low..#(offsets.size - tail)];
      if logLevel == LogLevel.DEBUG {
             t.stop(); 
             saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                   "took %t seconds\ndetermining answer...".format(t.elapsed())); 
             t.clear(); 
             t.start();
      }
      ref oa = offsets.a;
      if mode == SearchMode.contains {
        // Determine whether each segment contains a hit
        // Do this by taking the difference in the cumulative number of hits at the end vs the beginning of the segment  
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * truth.size);
        // Cumulative number of hits up to (and excluding) this point
        var numHits = (+ scan truth) - truth;
        hits[oD.interior(-(oD.size-1))] = (numHits[oa[oD.interior(oD.size-1)]] - numHits[oa[oD.interior(-(oD.size-1))]]) > 0;
        hits[oD.high] = (numHits[D.high] + truth[D.high] - numHits[oa[oD.high]]) > 0;
      } else if mode == SearchMode.startsWith {
        // First position of segment must be a hit
        hits[oD] = truth[oa[oD]];
      } else if mode == SearchMode.endsWith {
        // Position where substr aligns with end of segment must be a hit
        // -1 for null byte
        hits[oD.interior(-(oD.size-1))] = truth[oa[oD.interior(oD.size-1)] - substr.numBytes - 1];
        hits[oD.high] = truth[D.high-1];
      }
      if logLevel == LogLevel.DEBUG {
          t.stop(); 
          saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                   "took %t seconds".format(t.elapsed()));
      }
      return hits;
    }

    /*
      Peel off one or more fields matching the regular expression, delimiter, from each string (similar
      to string.partition), returning two new arrays of strings.
      *Warning*: This function is experimental and not guaranteed to work.

      Note: the regular expression engine used, re2, does not support lookahead/lookbehind

      :arg delimter: regex delimter where the split in SegString will occur
      :type delimter: string

      :arg times: The number of times the delimiter is sought, i.e. skip over the first (times-1) delimiters
      :type times: int

      :arg includeDelimiter: If true, append the delimiter to the end of the first return array
                              By default, it is prepended to the beginning of the second return array.
      :type includeDelimiter: bool

      :arg keepPartial: If true, a string that does not contain <times> instances of
                        the delimiter will be returned in the first array. By default,
                        such strings are returned in the second array.
      :type keepPartial: bool

      :arg left: If true, peel from the left
      :type left: bool

      :returns: Components to build 2 SegStrings (leftOffsets, leftVals, rightOffsets, rightVals)
    */
    proc peelRegex(const delimiter: string, const times: int, const includeDelimiter: bool, const keepPartial: bool, const left: bool) throws {
      checkCompile(delimiter);
      // should we do len check here? re2.compile('') is valid regex and matches everything
      ref oa = offsets.a;
      ref va = values.a;
      const lengths = getLengths() - 1;
      var leftEnd: [offsets.aD] int;
      var rightStart: [offsets.aD] int;

      forall (o, len, i) in zip(oa, lengths, offsets.aD) with (var myRegex = _unsafeCompileRegex(delimiter)) {
        var matches = myRegex.matches(interpretAsString(va, o..#len, borrow=true));
        if matches.size < times {
          // not enough occurances of delim, the entire string stays together, and the param args
          // determine whether it ends up on the left or right
          if left == keepPartial {  // they counteract each other
            // if both true or both false
            // Goes on the left
            leftEnd[i] = o + len - 1;
            rightStart[i] = o + len;
          }
          else {
            // if one is true but not the other
            // Goes on the right
            leftEnd[i] = o - 1;
            rightStart[i] = o;
          }
        }
        else {
          // The string can be peeled; figure out where to split
          var match_index: int = if left then (times - 1) else (matches.size - times);
          var match = matches[match_index][0]; // v1.24.x -> reMatch, v1.25.x -> regexMatch
          var j: int = o + match.offset: int;
          // j is now the start of the correct delimiter
          // tweak leftEnd and rightStart based on includeDelimiter
          if includeDelimiter {
            if left {
              leftEnd[i] = j + match.size - 1;
              rightStart[i] = j + match.size;
            }
            else {
              leftEnd[i] = j - 1;
              rightStart[i] = j;
            }
          }
          else {
            leftEnd[i] = j - 1;
            rightStart[i] = j + match.size;
          }
        }
      }
      // this section is the same as `peel`
      // Compute lengths and offsets for left and right return arrays
      const leftLengths = leftEnd - oa + 2;
      const rightLengths = lengths - (rightStart - oa) + 1;
      // check there's enough room to create copies for the scans and throw if creating copies would go over memory limit
      overMemLimit(numBytes(int) * (leftLengths.size + rightLengths.size));
      const leftOffsets = (+ scan leftLengths) - leftLengths;
      const rightOffsets = (+ scan rightLengths) - rightLengths;
      // Allocate values and fill
      var leftVals = makeDistArray((+ reduce leftLengths), uint(8));
      var rightVals = makeDistArray((+ reduce rightLengths), uint(8));
      // Fill left values
      forall (srcStart, dstStart, len) in zip(oa, leftOffsets, leftLengths) {
        for i in 0..#(len-1) {
          unorderedCopy(leftVals[dstStart+i], va[srcStart+i]);
        }
      }
      // Fill right values
      forall (srcStart, dstStart, len) in zip(rightStart, rightOffsets, rightLengths) {
        for i in 0..#(len-1) {
          unorderedCopy(rightVals[dstStart+i], va[srcStart+i]);
        }
      }
      return (leftOffsets, leftVals, rightOffsets, rightVals);
    }

    proc peel(const delimiter: string, const times: int, param includeDelimiter: bool, param keepPartial: bool, param left: bool) throws {
      param stride = if left then 1 else -1;
      const dBytes = delimiter.numBytes;
      const lengths = getLengths() - 1;
      var leftEnd: [offsets.aD] int;
      var rightStart: [offsets.aD] int;
      const truth = findSubstringInBytes(delimiter);
      const D = truth.domain;
      ref oa = offsets.a;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * truth.size);
      var numHits = (+ scan truth) - truth;
      const high = offsets.aD.high;
      forall i in offsets.aD {
        // First, check whether string contains enough instances of delimiter to peel
        var hasEnough: bool;
        if oa[i] > D.high {
          // When the last string(s) is/are shorter than the substr
          hasEnough = false;
        } else if ((i == high) || (oa[i+1] > D.high)) {
          hasEnough = ((+ reduce truth) - numHits[oa[i]]) >= times;
        } else {
          hasEnough = (numHits[oa[i+1]] - numHits[oa[i]]) >= times;
        }
        if !hasEnough {
          // If not, then the entire string stays together, and the param args
          // determine whether it ends up on the left or right
          if left {
            if keepPartial {
              // Goes on the left
              leftEnd[i] = oa[i] + lengths[i] - 1;
              rightStart[i] = oa[i] + lengths[i];
            } else {
              // Goes on the right
              leftEnd[i] = oa[i] - 1;
              rightStart[i] = oa[i];
            }
          } else {
            if keepPartial {
              // Goes on the right
              leftEnd[i] = oa[i] - 1;
              rightStart[i] = oa[i];
            } else {
              // Goes on the left
              leftEnd[i] = oa[i] + lengths[i] - 1;
              rightStart[i] = oa[i] + lengths[i];
            }
          }
        } else {
          // The string can be peeled; figure out where to split
          var nDelim = 0;
          var j: int;
          if left {
            j = oa[i];
          } else {
            // If coming from the right, need to handle edge case of last string
            if i == high {
              j = values.aD.high - 1;
            } else {
              j = oa[i+1] - 2;
            }
          }
          // Step until the delimiter is encountered the exact number of times
          while true {
            if (j <= D.high) && truth[j] {
              nDelim += 1;
            }
            if nDelim == times {
              break;
            }
            j += stride;
          }
          // j is now the start of the correct delimiter
          // tweak leftEnd and rightStart based on includeDelimiter
          if left {
            if includeDelimiter {
              leftEnd[i] = j + dBytes - 1;
              rightStart[i] = j + dBytes;
            } else {
              leftEnd[i] = j - 1;
              rightStart[i] = j + dBytes;
            }
          } else {
            if includeDelimiter {
              leftEnd[i] = j - 1;
              rightStart[i] = j;
            } else {
              leftEnd[i] = j - 1;
              rightStart[i] = j + dBytes;
            }
          }
        }
      }
      // Compute lengths and offsets for left and right return arrays
      const leftLengths = leftEnd - oa + 2;
      const rightLengths = lengths - (rightStart - oa) + 1;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * leftLengths.size);
      const leftOffsets = (+ scan leftLengths) - leftLengths;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * rightLengths.size);
      const rightOffsets = (+ scan rightLengths) - rightLengths;
      // Allocate values and fill
      var leftVals = makeDistArray((+ reduce leftLengths), uint(8));
      var rightVals = makeDistArray((+ reduce rightLengths), uint(8));
      ref va = values.a;
      // Fill left values
      forall (srcStart, dstStart, len) in zip(oa, leftOffsets, leftLengths) {
        for i in 0..#(len-1) {
          unorderedCopy(leftVals[dstStart+i], va[srcStart+i]);
        }
      }
      // Fill right values
      forall (srcStart, dstStart, len) in zip(rightStart, rightOffsets, rightLengths) {
        for i in 0..#(len-1) {
          unorderedCopy(rightVals[dstStart+i], va[srcStart+i]);
        }
      }
      return (leftOffsets, leftVals, rightOffsets, rightVals);
    }

    proc stick(other: SegString, delim: string, param right: bool) throws {
        if (offsets.aD != other.offsets.aD) {
            throw getErrorWithContext(
                           msg="The SegString offsets to not match",
                           lineNumber = getLineNumber(),
                           routineName = getRoutineName(),
                           moduleName = getModuleName(),
                           errorClass="ArgumentError");
        }
      // Combine lengths and compute new offsets
      var leftLen = getLengths() - 1;
      var rightLen = other.getLengths() - 1;
      const newLengths = leftLen + rightLen + delim.numBytes + 1;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * newLengths.size);
      var newOffsets = (+ scan newLengths);
      const newBytes = newOffsets[offsets.aD.high];
      newOffsets -= newLengths;
      // Allocate new values array
      var newVals = makeDistArray(newBytes, uint(8));
      // Copy in the left and right-hand values, separated by the delimiter
      ref va1 = values.a;
      ref va2 = other.values.a;
      forall (o1, o2, no, l1, l2) in zip(offsets.a, other.offsets.a, newOffsets, leftLen, rightLen) {
        var pos = no;
        // Left side
        if right {
          for i in 0..#l1 {
            unorderedCopy(newVals[pos+i], va1[o1+i]);
          }
          pos += l1;
        } else {
          for i in 0..#l2 {
            unorderedCopy(newVals[pos+i], va2[o2+i]);
          }
          pos += l2;
        }
        // Delimiter
        for (i, b) in zip(0..#delim.numBytes, delim.chpl_bytes()) {
          unorderedCopy(newVals[pos+i], b);
        }
        pos += delim.numBytes;
        // Right side
        if right {
          for i in 0..#l2 {
            unorderedCopy(newVals[pos+i], va2[o2+i]);
          }
        } else {
          for i in 0..#l1 {
            unorderedCopy(newVals[pos+i], va1[o1+i]);
          }
        }
      }
      return (newOffsets, newVals);
    }

    proc ediff():[offsets.aD] int {
      var diff: [offsets.aD] int;
      if (size < 2) {
        return diff;
      }
      ref oa = offsets.a;
      ref va = values.a;
      const high = offsets.aD.high;
      forall (i, a) in zip(offsets.aD, diff) {
        if (i < high) {
          var asc: bool;
          const left = oa[i]..oa[i+1]-1;
          if (i < high - 1) {
            const right = oa[i+1]..oa[i+2]-1;
            a = -memcmp(va, left, va, right);
          } else { // i == high - 1
            const right = oa[i+1]..values.aD.high;
            a = -memcmp(va, left, va, right);
          }
        } else { // i == high
          a = 0;
        } 
      }
      return diff;
    }

    proc isSorted():bool {
      if (size < 2) {
        return true;
      }
      return (&& reduce (ediff() >= 0));
    }

    proc argsort(checkSorted:bool=false): [offsets.aD] int throws {
      const ref D = offsets.aD;
      const ref va = values.a;
      if checkSorted && isSorted() {
          saLogger.warn(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "argsort called on already sorted array");
          var ranks: [D] int = [i in D] i;
          return ranks;
      }
      var ranks = twoPhaseStringSort(this);
      return ranks;
    }

  } // class SegString


  inline proc memcmp(const ref x: [] uint(8), const xinds, const ref y: [] uint(8), const yinds): int {
    const l = min(xinds.size, yinds.size);
    var ret: int = 0;
    for (i, j) in zip(xinds.low..#l, yinds.low..#l) {
      ret = x[i]:int - y[j]:int;
      if (ret != 0) {
        break;
      }
    }
    if (ret == 0) {
      ret = xinds.size - yinds.size;
    }
    return ret;
  }


  enum SearchMode { contains, startsWith, endsWith, match }
  class UnknownSearchMode: Error {}
  
  /* Test for equality between two same-length arrays of strings. Returns
     a boolean vector of the same length. */
  operator ==(lss:SegString, rss:SegString) throws {
    return compare(lss, rss, true);
  }

  /* Test for inequality between two same-length arrays of strings. Returns
     a boolean vector of the same length. */
  operator !=(lss:SegString, rss:SegString) throws {
    return compare(lss, rss, false);
  }

  /* Element-wise comparison of two same-length arrays of strings. The
     polarity parameter determines whether the comparison checks for 
     equality (polarity=true, result is true where elements are equal) 
     or inequality (polarity=false, result is true where elements differ). */
  private proc compare(lss:SegString, rss:SegString, param polarity: bool) throws {
    // String arrays must be same size
    if (lss.size != rss.size) {
        throw getErrorWithContext(
                           msg="The String arrays must be the same size",
                           lineNumber = getLineNumber(),
                           routineName = getRoutineName(),
                           moduleName = getModuleName(),
                           errorClass="ArgumentError");
    }
    ref oD = lss.offsets.aD;
    // Start by assuming all elements differ, then correct for those that are equal
    // This translates to an initial value of false for == and true for !=
    var truth: [oD] bool = !polarity;
    // Early exit for zero-length result
    if (lss.size == 0) {
      return truth;
    }
    ref lvalues = lss.values.a;
    ref loffsets = lss.offsets.a;
    ref rvalues = rss.values.a;
    ref roffsets = rss.offsets.a;
    // Compare segments in parallel
    // Segments are guaranteed to be on same locale, but bytes are not
    forall (t, lo, ro, idx) in zip(truth, loffsets, roffsets, oD) 
      with (var agg = newDstAggregator(bool)) {
      var llen: int;
      var rlen: int;
      if (idx == oD.high) {
        llen = lvalues.size - lo - 1;
        rlen = rvalues.size - ro - 1;
      } else {
        llen = loffsets[idx+1] - lo - 1;
        rlen = roffsets[idx+1] - ro - 1;
      }
      // Only compare bytes if lengths are equal
      if (llen == rlen) {
        var allEqual = true;
        // TO DO: consider an on clause here to ensure at least one access is local
        for pos in 0..#llen {
          if (lvalues[lo+pos] != rvalues[ro+pos]) {
            allEqual = false;
            break;
          }
        }
        // Only if lengths and all bytes are equal, override the default value
        if allEqual {
          // For ==, the output should be true; for !=, false
          agg.copy(t, polarity);
        }
      }
    }
    return truth;
  }

  /* Test an array of strings for equality against a constant string. Return a boolean
     vector the same size as the array. */
  operator ==(ss:SegString, testStr: string) {
    return compare(ss, testStr, true);
  }
  
  /* Test an array of strings for inequality against a constant string. Return a boolean
     vector the same size as the array. */
  operator !=(ss:SegString, testStr: string) {
    return compare(ss, testStr, false);
  }

  /* Element-wise comparison of an arrays of string against a target string. 
     The polarity parameter determines whether the comparison checks for 
     equality (polarity=true, result is true where elements equal target) 
     or inequality (polarity=false, result is true where elements differ from 
     target). */
  proc compare(ss:SegString, testStr: string, param polarity: bool) {
    ref oD = ss.offsets.aD;
    // Initially assume all elements equal the target string, then correct errors
    // For ==, this means everything starts true; for !=, everything starts false
    var truth: [oD] bool = polarity;
    // Early exit for zero-length result
    if (ss.size == 0) {
      return truth;
    }
    ref values = ss.values.a;
    ref vD = ss.values.aD;
    ref offsets = ss.offsets.a;
    // Use a whole-array strategy, where the ith byte from every segment is checked simultaneously
    // This will do len(testStr) parallel loops, but loops will have low overhead
    for (b, i) in zip(testStr.chpl_bytes(), 0..) {
      forall (t, o, idx) in zip(truth, offsets, oD) with (var agg = newDstAggregator(bool)) {
        if ((o+i > vD.high) || (b != values[o+i])) {
          // Strings are not equal, so change the output
          // For ==, output is now false; for !=, output is now true
          agg.copy(t, !polarity);
        }
      }
    }
    // Check the length by checking that the next byte is null
    forall (t, o, idx) in zip(truth, offsets, oD) with (var agg = newDstAggregator(bool)) {
      if ((o+testStr.size > vD.high) || (0 != values[o+testStr.size])) {
        // Strings are not equal, so change the output
        // For ==, output is now false; for !=, output is now true
        agg.copy(t, !polarity);
      }
    }
    return truth;
  }

  private config const in1dAssocSortThreshold = 10**6;

  /* Test array of strings for membership in another array (set) of strings. Returns
     a boolean vector the same size as the first array. */
  proc in1d(mainStr: SegString, testStr: SegString, invert=false, forceSort=false) throws where useHash {
    var truth: [mainStr.offsets.aD] bool;
    // Early exit for zero-length result
    if (mainStr.size == 0) {
      return truth;
    }
    // Hash all strings for fast comparison
    var t = new Timer();
    saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Hashing strings");
    if logLevel == LogLevel.DEBUG { t.start(); }
    const hashes = mainStr.hash();
    if logLevel == LogLevel.DEBUG {
        t.stop(); 
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                "%t seconds".format(t.elapsed())); 
        t.clear();
    }

    if (testStr.size <= in1dAssocSortThreshold) && !forceSort {
      if logLevel == LogLevel.DEBUG {
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                       "Making associative domains for test set on each locale");
        t.start();
      }
      // On each locale, make an associative domain with the hashes of the second array
      // parSafe=false because we are adding in serial and it's faster
      var localTestHashes: [PrivateSpace] domain(2*uint(64), parSafe=false);
      coforall loc in Locales {
        on loc {
          // Local hashes of second array
          ref mySet = localTestHashes[here.id];
          mySet.requestCapacity(testStr.size);
          var testHashes: [{0..#testStr.size}] 2*uint(64);
          testHashes = testStr.hash();
          for h in testHashes {
            mySet += h;
          }
          /* // Check membership of hashes in this locale's chunk of the array */
          /* [i in truth.localSubdomain()] truth[i] = mySet.contains(hashes[i]); */
        }
      }
      if logLevel == LogLevel.DEBUG {
        t.stop(); 
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "%t seconds".format(t.elapsed())); 
        t.clear();
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),      
                       "Testing membership"); 
        t.start();
      }
      [i in truth.domain] truth[i] = localTestHashes[here.id].contains(hashes[i]);
      if logLevel == LogLevel.DEBUG {
        t.stop(); 
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "%t seconds".format(t.elapsed()));
      }
    } else {
      if logLevel == LogLevel.DEBUG {
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                       "Using sort-based strategy");
        t.start();
      }
      const testHashes = testStr.hash();
      // Unique the hashes of each array, preserving reverse index for main
      var (umain, cmain, revIdx) = uniqueSortWithInverse(hashes);
      var utest = uniqueSort(testHashes, needCounts=false);
      // Concat unique hashes
      var combinedDom = makeDistDom(umain.size + utest.size);
      var combined: [combinedDom] 2*uint(64);
      combined[combinedDom.interior(-umain.size)] = umain;
      combined[combinedDom.interior(utest.size)] = utest;
      // Sort
      var iv = radixSortLSD_ranks(combined);
      var sorted: [combinedDom] 2*uint(64);
      forall (i, s) in zip(iv, sorted) with (var agg = newSrcAggregator(2*uint(64))) {
        agg.copy(s, combined[i]);
      }
      // Find duplicates
      // Dupe parallels unique hashes of mainStr and is true when value is in testStr
      var dupe: [combinedDom] bool = false;
      forall (sortedIdx, origIdx, s) in zip(combinedDom, iv, sorted) with (var agg = newDstAggregator(bool)){
        // When next hash is same as current, string exists in both arrays
        if sortedIdx < combinedDom.high && (s == sorted[sortedIdx+1]){
          // Use the iv to scatter back to pre-sorted order
          agg.copy(dupe[origIdx], true);
        }
      }
      // Use revIdx to broadcast dupe to original non-unique domain
      forall (t, ri) in zip(truth, revIdx) with (var agg = newSrcAggregator(bool)) {
        agg.copy(t, dupe[ri]);
      }
      if logLevel == LogLevel.DEBUG {
        t.stop(); 
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "%t seconds".format(t.elapsed()));
      }
    }
    return truth;
  }

  proc concat(s1: [] int, v1: [] uint(8), s2: [] int, v2: [] uint(8)) throws {
    // TO DO: extend to axis == 1
    var segs = makeDistArray(s1.size + s2.size, int);
    var vals = makeDistArray(v1.size + v2.size, uint(8));
    ref sD = segs.domain;
    segs[sD.interior(-s1.size)] = s1;
    segs[sD.interior(s2.size)] = s2 + v1.size;
    ref vD = vals.domain;
    vals[vD.interior(-v1.size)] = v1;
    vals[vD.interior(v2.size)] = v2;
    return (segs, vals);
  }

  private config const in1dSortThreshold = 64;
  
  proc in1d(mainStr: SegString, testStr: SegString, invert=false, forceSort=false) throws where !useHash {
    var truth: [mainStr.offsets.aD] bool;
    // Early exit for zero-length result
    if (mainStr.size == 0) {
      return truth;
    }
    if (testStr.size <= in1dSortThreshold) && !forceSort {
      for i in 0..#testStr.size {
        truth |= (mainStr == testStr[i]);
      }
      return truth;
    } else {
      // This is inspired by numpy in1d
      const (uoMain, uvMain, cMain, revIdx) = uniqueGroup(mainStr, returnInverse=true);
      const (uoTest, uvTest, cTest, revTest) = uniqueGroup(testStr);
      const (segs, vals) = concat(uoMain, uvMain, uoTest, uvTest);
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
           "Unique strings in first array: %t\nUnique strings in second array: %t\nConcat length: %t".format(
                                             uoMain.size, uoTest.size, segs.size));
      var st = new owned SymTab();
      const ar = new owned SegString(segs, vals, st);
      const order = ar.argsort();
      const (sortedSegs, sortedVals) = ar[order];
      const sar = new owned SegString(sortedSegs, sortedVals, st);
      if logLevel == LogLevel.DEBUG { 
          saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "Sorted concatenated unique strings:"); 
          sar.show(10); 
          stdout.flush(); 
      }
      const D = sortedSegs.domain;
      // First compare lengths and only check pairs whose lengths are equal (because gathering them is expensive)
      var flag: [D] bool;
      const lengths = sar.getLengths();
      const ref saro = sar.offsets.a;
      const ref sarv = sar.values.a;
      const high = D.high;
      forall (i, f, o, l) in zip(D, flag, saro, lengths) {
        if (i < high) && (l == lengths[i+1]) {
          const left = o..saro[i+1]-1;
          var eq: bool;
          if (i < high - 1) {
            const right = saro[i+1]..saro[i+2]-1;
            eq = (memcmp(sarv, left, sarv, right) == 0);
          } else {
            const ref right = saro[i+1]..sar.values.aD.high;
            eq = (memcmp(sarv, left, sarv, right) == 0);
          }
          if eq {
            f = true;
            flag[i+1] = true;
          }
        }
      }
      
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "Flag pop: %t".format(+ reduce flag));

      // Now flag contains true for both elements of duplicate pairs
      if invert {flag = !flag;}
      // Permute back to unique order
      var ret: [D] bool;
      forall (o, f) in zip(order, flag) with (var agg = newDstAggregator(bool)) {
        agg.copy(ret[o], f);
      }
      if logLevel == LogLevel.DEBUG {
          saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "Ret pop: %t".format(+ reduce ret));
      }
      // Broadcast back to original (pre-unique) order
      var truth: [mainStr.offsets.aD] bool;
      forall (t, i) in zip(truth, revIdx) with (var agg = newSrcAggregator(bool)) {
        agg.copy(t, ret[i]);
      }
      return truth;
    }
  }

  /*
     Interpret a region of a byte array as a Chapel string. If `borrow=false` a
     new string is returned, otherwise the string borrows memory from the array
     (reduces memory allocations if the string isn't needed after array)
   */
  proc interpretAsString(bytearray: [?D] uint(8), region: range(?), borrow=false): string {
    var localSlice = new lowLevelLocalizingSlice(bytearray, region);
    // Byte buffer is null-terminated, so length is region.size - 1
    try {
      if localSlice.isOwned {
        localSlice.isOwned = false;
        return createStringWithOwnedBuffer(localSlice.ptr, region.size-1, region.size);
      } else if borrow {
        return createStringWithBorrowedBuffer(localSlice.ptr, region.size-1, region.size);
      } else {
        return createStringWithNewBuffer(localSlice.ptr, region.size-1, region.size);
      }
    } catch {
      return "<error interpreting bytes as string>";
    }
  }
}
