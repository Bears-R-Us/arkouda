module SegmentedArray {
  use AryUtil;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use CommAggregation;
  use SipHash;
  use SegStringSort;
  use RadixSortLSD only radixSortLSD_ranks;
  use Reflection;
  use PrivateDist;
  use ServerConfig;
  use Unique;
  use Time only Timer, getCurrentTime;

  private config const DEBUG = false;
  private config param useHash = true;
  param SegmentedArrayUseHash = useHash;
  
  class OutOfBoundsError: Error {}

  /* Represents an array of strings, implemented as a segmented array of bytes.
     Instances are ephemeral, not stored in the symbol table. Instead, attributes
     of this class refer to symbol table entries that persist. This class is a
     convenience for bundling those persistent objects and defining string-relevant
     operations.
   */
  class SegString {
    // Start indices of individual strings
    var offsets: borrowed SymEntry(int);
    // Bytes of all strings, joined by nulls
    var values: borrowed SymEntry(uint(8));
    // Number of strings
    var size: int;
    // Total number of bytes in all strings, including nulls
    var nBytes: int;

    /* This initializer is used when the SymEntries for offsets and values are
       already in the namespace. */
    proc init(segments: borrowed SymEntry(int), values: borrowed SymEntry(uint(8))) {
      offsets = segments;
      values = values;
      size = segments.size;
      nBytes = values.size;
    }

    /* This initializer is the most common, and is used when only the server
       names of the SymEntries are known. It handles the lookup. */
    proc init(segName: string, valName: string, st: borrowed SymTab) {
      // The try! is needed here because init cannot throw
      var gs = try! st.lookup(segName);
      // I want this to be borrowed, but that give a lifetime error
      var segs = toSymEntry(gs, int): unmanaged SymEntry(int);
      offsets = segs;
      var vs = try! st.lookup(valName);
      var vals = toSymEntry(vs, uint(8)): unmanaged SymEntry(uint(8));
      values = vals;
      size = segs.size;
      nBytes = vals.size;
    }

    proc init(segments: [] int, values: [] uint(8), st: borrowed SymTab) {
      var segName = st.nextName();
      var valName = st.nextName();
      var segEntry = new shared SymEntry(segments);
      var valEntry = new shared SymEntry(values);
      try! st.addEntry(segName, segEntry);
      try! st.addEntry(valName, valEntry);
      this.init(segName, valName, st);
    }

    proc show(n: int = 3) throws {
      if (size >= 2*n) {
        for i in 0..#n {
          writeln(this[i]);
        }
        writeln("...");
        for i in size-n..#n {
          writeln(this[i]);
        }
      } else {
        for i in 0..#size {
          writeln(this[i]);
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
      var s = interpretAsString(values.a[start..end]);
      return s;
    }

    /* Take a slice of strings from the array. The slice must be a 
       Chapel range, i.e. low..high by stride, not a Python slice.
       Returns arrays for the segment offsets and bytes of the slice.*/
    proc this(slice: range(stridable=true)) throws {
      if (slice.low < offsets.aD.low) || (slice.high > offsets.aD.high) {
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
      // Offsets need to be re-zeroed
      newSegs = offsets.a[slice] - start;
      // Bytearray of the new slice
      var newVals = makeDistArray(end - start + 1, uint(8));
      newVals = values.a[start..end];
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
        throw new owned OutOfBoundsError();
      }
      if v {writeln("Computing lengths and offsets"); stdout.flush();}
      var t1 = getCurrentTime();
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // Gather the right and left boundaries of the indexed strings
      // NOTE: cannot compute lengths inside forall because agg.copy will
      // experience race condition with loop-private variable
      var right: [D] int, left: [D] int;
      forall (r, l, idx) in zip(right, left, iv) with (var agg = new SrcAggregator(int)) {
        if (idx == high) {
          agg.copy(r, values.size);
        } else {
          agg.copy(r, oa[idx+1]);
        }
        agg.copy(l, oa[idx]);
      }
      // Lengths of segments including null bytes
      var gatheredLengths: [D] int = right - left;
      // The returned offsets are the 0-up cumulative lengths
      var gatheredOffsets = (+ scan gatheredLengths);
      // The total number of bytes in the gathered strings
      var retBytes = gatheredOffsets[D.high];
      gatheredOffsets -= gatheredLengths;
      if v {
        writeln(getCurrentTime() - t1, " seconds");
        writeln("Copying values"); stdout.flush();
        t1 = getCurrentTime();
      }
      var gatheredVals = makeDistArray(retBytes, uint(8));
      // For comm layer with poor small-message performance, use aggregation
      // at the expense of memory. Otherwise, unorderedCopy is faster and smaller.
      if !(CHPL_COMM == 'none' || CHPL_COMM == 'ugni') {
        // Compute the src index for each byte in gatheredVals
        /* For performance, we will do this with a scan, so first we need an array
           with the difference in index between the current and previous byte. For
           the interior of a segment, this is just one, but at the segment boundary,
           it is the difference between the src offset of the current segment ("left")
           and the src index of the last byte in the previous segment (right - 1).
        */
        if DEBUG { writeln("Using aggregation at the expense of memory"); stdout.flush(); }
        var srcIdx = makeDistArray(retBytes, int);
        srcIdx = 1;
        var diffs: [D] int;
        diffs[D.low] = left[D.low]; // first offset is not affected by scan
        diffs[{D.low+1..D.high}] = left[{D.low+1..D.high}] - (right[{D.low..D.high-1}] - 1);
        // Set srcIdx to diffs at segment boundaries
        forall (go, d) in zip(gatheredOffsets, diffs) with (var agg = newDstAggregator(int)) {
          agg.copy(srcIdx[go], d);
        }
        srcIdx = + scan srcIdx;
        // Now srcIdx has a dst-local copy of the source index and vals can be efficiently gathered
        ref va = values.a;
        forall (v, si) in zip(gatheredVals, srcIdx) with (var agg = newSrcAggregator(uint(8))) {
          agg.copy(v, va[si]);
        }
      } else {
        if DEBUG { writeln("Using unorderedCopy"); stdout.flush(); }
        ref va = values.a;
        // Copy string data to gathered result
        forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, iv) {
          for pos in 0..#gl {
            // Note: do not replace this unorderedCopy with aggregation
            use UnorderedCopy;
            unorderedCopy(gatheredVals[go+pos], va[oa[idx]+pos]);
          }
        }
      }
      if v {writeln(getCurrentTime() - t1, " seconds"); stdout.flush();}
      return (gatheredOffsets, gatheredVals);
    }

    /* Logical indexing (compress) of strings. */
    proc this(iv: [?D] bool) throws {
      // Index vector must be same domain as array
      if (D != offsets.aD) {
        throw new owned OutOfBoundsError();
      }
      if v {writeln("Computing lengths and offsets"); stdout.flush();}
      var t1 = getCurrentTime();
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
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
      /* if v { */
      /*   writeln(getCurrentTime() - t1, " seconds"); */
      /*   writeln("Copying values"); stdout.flush(); */
      /*   t1 = getCurrentTime(); */
      /* } */
      /* var gatheredVals = makeDistArray(retBytes, uint(8)); */
      /* ref va = values.a; */
      /* if DEBUG { */
      /*   printAry("gatheredOffsets: ", gatheredOffsets); */
      /*   printAry("gatheredLengths: ", gatheredLengths); */
      /*   printAry("segInds: ", segInds); */
      /* } */
      /* // Copy string bytes from src to dest */
      /* forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, segInds) { */
      /*   gatheredVals[{go..#gl}] = va[{oa[idx]..#gl}]; */
      /* } */
      /* if v {writeln(getCurrentTime() - t1, " seconds"); stdout.flush();} */
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
        if v { writeln("Hashing strings"); stdout.flush(); t.start(); }
        var hashes = this.hash();
        if v { t.stop(); writeln("hashing took %t seconds\nSorting hashes".format(t.elapsed())); stdout.flush(); t.clear(); t.start(); }
        // Return the permutation that sorts the hashes
        var iv = radixSortLSD_ranks(hashes);
        if v { t.stop(); writeln("sorting took %t seconds".format(t.elapsed())); stdout.flush(); }
        if DEBUG {
          var sortedHashes = [i in iv] hashes[i];
          var diffs = sortedHashes[(iv.domain.low+1)..#(iv.size-1)] - sortedHashes[(iv.domain.low)..#(iv.size-1)];
          printAry("diffs = ", diffs);
          var nonDecreasing = [d in diffs] ((d[1] > 0) || ((d[1] == 0) && (d[2] >= 0)));
          writeln("Are hashes sorted? ", && reduce nonDecreasing);
        }
        return iv;
      } else {
        var iv = argsort();
        return iv;
      }
    }

    proc getLengths() {
      var lengths: [offsets.aD] int;
      if (size == 0) {
        return lengths;
      }
      ref oa = offsets.a;
      const low = offsets.aD.low;
      const high = offsets.aD.high;
      lengths[low..high-1] = (oa[low+1..high] - oa[low..high-1]);
      lengths[high] = values.size - oa[high];
      return lengths;
    }

    proc substringSearch(const substr: string, mode: SearchMode) throws {
      var hits: [offsets.aD] bool;  // the answer
      if (size == 0) || (substr.size == 0) {
        return hits;
      }
      var t = new Timer();
      // Find the start position of every occurence of substr in the flat bytes array
      // Start by making a right-truncated subdomain representing all valid starting positions for substr of given length
      var D: subdomain(values.aD) = values.aD[values.aD.low..#(values.size - substr.numBytes)];
      // Every start position is valid until proven otherwise
      var truth: [D] bool = true;
      if DEBUG {writeln("Checking bytes of substr"); stdout.flush(); t.start();}
      // Shift the flat values one byte at a time and check against corresponding byte of substr
      for (i, b) in zip(0.., substr.chpl_bytes()) {
        truth &= (values.a[D.translate(i)] == b);
      }
      // Determine whether each segment contains a hit
      // Do this by taking the difference in the cumulative number of hits at the end vs the beginning of the segment
      if DEBUG {t.stop(); writeln("took %t seconds\nscanning...".format(t.elapsed())); stdout.flush(); t.clear(); t.start();}
      // Cumulative number of hits up to (and excluding) this point
      var numHits = (+ scan truth) - truth;
      if DEBUG {t.stop(); writeln("took %t seconds\nTranslating to segments...".format(t.elapsed())); stdout.flush(); t.clear(); t.start();}
      // Need to ignore segment(s) at the end of the array that are too short to contain substr
      const tail = + reduce (offsets.a > D.high);
      // oD is the right-truncated domain representing segments that are candidates for containing substr
      var oD: subdomain(offsets.aD) = offsets.aD[offsets.aD.low..#(offsets.size - tail)];
      ref oa = offsets.a;
      if mode == SearchMode.contains {
        // Find segments where at least one hit occurred between the start and end of the segment
        // hits[oD] = (numHits[oa[oD.translate(1)]] - numHits[oa[oD]]) > 0;
        hits[{oD.low..oD.high-1}] = (numHits[oa[{oD.low+1..oD.high}]] - numHits[oa[{oD.low..oD.high-1}]]) > 0;
        hits[oD.high] = (numHits[D.high] + truth[D.high] - numHits[oa[oD.high]]) > 0;
      } else if mode == SearchMode.startsWith {
        // First position of segment must be a hit
        hits[oD] = truth[oa[oD]];
      } else if mode == SearchMode.endsWith {
        // Position where substr aligns with end of segment must be a hit
        // -1 for null byte
        hits[{oD.low..oD.high-1}] = truth[oa[{oD.low+1..oD.high}] - substr.numBytes - 1];
        hits[oD.high] = truth[D.high];
      }
      return hits;
    }

    proc isAscending():[offsets.aD] int {
      var ascending: [offsets.aD] int;
      if (size < 2) {
        return ascending;
      }
      ref oa = offsets.a;
      ref va = values.a;
      const high = offsets.aD.high;
      forall (i, a) in zip(offsets.aD, ascending) {
        if (i < high) {
          var asc: bool;
          const left = oa[i]..oa[i+1]-1;
          if (i < high - 1) {
            const right = oa[i+1]..oa[i+2]-1;
            a = memcmp(va, left, va, right);
          } else { // i == high - 1
            const right = oa[i+1]..values.aD.high;
            a = memcmp(va, left, va, right);
          }
        } else { // i == high
          a = 0;
        } 
      }
      return ascending;
    }

    proc isSorted():bool {
      if (size < 2) {
        return true;
      }
      return (&& reduce (isAscending() <= 0));
    }
    
    /* proc isSorted(): bool { */
    /*   var res = true; // strings are sorted? */
    /*   // Is this position done comparing with its predecessor? */
    /*   var done: [offsets.aD] bool; */
    /*   // First string has no predecessor, so comparison is automatically done */
    /*   done[offsets.aD.low] = true; */
    /*   // Do not check null terminators */
    /*   const lengths = getLengths() - 1; */
    /*   const maxLen = max reduce lengths; */
    /*   ref oa = offsets.a; */
    /*   ref va = values.a; */
    /*   // Compare each pair of strings byte-by-byte */
    /*   for pos in 0..#maxLen { */
    /*     forall (o, l, d, i) in zip(oa, lengths, done, offsets.aD)  */
    /*       with (ref res) { */
    /*       if (!d) { */
    /*         // If either of the strings is exhausted, mark this entry done */
    /*         if (pos >= l) || (pos >= lengths[i-1]) { */
    /*           d = true; */
    /*         } else { */
    /*           const prevByte = va[oa[i-1] + pos]; */
    /*           const currByte = va[o + pos]; */
    /*           // If we can already tell the pair is sorted, mark done */
    /*           if (prevByte < currByte) { */
    /*             d = true; */
    /*           // If we can tell the pair is not sorted, the return is false */
    /*           } else if (prevByte > currByte) { */
    /*             res = false; */
    /*           } // If we can't tell yet, keep checking */
    /*         } */
    /*       } */
    /*     } */
    /*     // If some pair is not sorted, return false */
    /*     if !res { */
    /*       return false; */
    /*     // If all comparisons are conclusive, return true */
    /*     } */
    /*     /\* else if (&& reduce done) { *\/ */
    /*     /\*   return true; *\/ */
    /*     /\* } // else keep going *\/ */
    /*   } */
    /*   // If we get to this point, it's because there is at least one pair of strings with length maxLen that are the same up to the last byte. That last byte determines res. */
    /*   return res; */
    /* } */

    proc argsort(checkSorted:bool=true): [offsets.aD] int throws {
      const ref D = offsets.aD;
      const ref va = values.a;
      if checkSorted && isSorted() {
        if DEBUG { writeln("argsort called on already sorted array"); stdout.flush(); }
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


  enum SearchMode { contains, startsWith, endsWith }
  class UnknownSearchMode: Error {}
  
  /* Test for equality between two same-length arrays of strings. Returns
     a boolean vector of the same length. */
  proc ==(lss:SegString, rss:SegString) throws {
    return compare(lss, rss, true);
  }

  /* Test for inequality between two same-length arrays of strings. Returns
     a boolean vector of the same length. */
  proc !=(lss:SegString, rss:SegString) throws {
    return compare(lss, rss, false);
  }

  /* Element-wise comparison of two same-length arrays of strings. The
     polarity parameter determines whether the comparison checks for 
     equality (polarity=true, result is true where elements are equal) 
     or inequality (polarity=false, result is true where elements differ). */
  private proc compare(lss:SegString, rss:SegString, param polarity: bool) throws {
    // String arrays must be same size
    if (lss.size != rss.size) {
      throw new owned ArgumentError();
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
  proc ==(ss:SegString, testStr: string) {
    return compare(ss, testStr, true);
  }
  
  /* Test an array of strings for inequality against a constant string. Return a boolean
     vector the same size as the array. */
  proc !=(ss:SegString, testStr: string) {
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

  /* Test array of strings for membership in another array (set) of strings. Returns
     a boolean vector the same size as the first array. */
  proc in1d(mainStr: SegString, testStr: SegString, invert=false) throws where useHash {
    var truth: [mainStr.offsets.aD] bool;
    // Early exit for zero-length result
    if (mainStr.size == 0) {
      return truth;
    }
    // Hash all strings for fast comparison
    var t = new Timer();
    if v {writeln("Hashing strings"); stdout.flush(); t.start();}
    const hashes = mainStr.hash();
    if v {
      t.stop(); writeln("%t seconds".format(t.elapsed())); t.clear();
      writeln("Making associative domains for test set on each locale"); stdout.flush(); t.start();
    }
    // On each locale, make an associative domain with the hashes of the second array
    // parSafe=false because we are adding in serial and it's faster
    var localTestHashes: [PrivateSpace] domain(2*uint(64), parSafe=false);
    coforall loc in Locales {
      on loc {
        // Local hashes of second array
        ref mySet = localTestHashes[here.id];
        mySet.requestCapacity(testStr.size);
        const testHashes = testStr.hash();
        for h in testHashes {
          mySet += h;
        }
        /* // Check membership of hashes in this locale's chunk of the array */
        /* [i in truth.localSubdomain()] truth[i] = mySet.contains(hashes[i]); */
      }
    }
    if v {
      t.stop(); writeln("%t seconds".format(t.elapsed())); t.clear();
      writeln("Testing membership"); stdout.flush(); t.start();
    }
    [i in truth.domain] truth[i] = localTestHashes[here.id].contains(hashes[i]);
    if v {t.stop(); writeln("%t seconds".format(t.elapsed())); stdout.flush();}
    return truth;
  }

  proc concat(s1: [] int, v1: [] uint(8), s2: [] int, v2: [] uint(8)) throws {
    // TO DO: extend to axis == 1
    var segs = makeDistArray(s1.size + s2.size, int);
    var vals = makeDistArray(v1.size + v2.size, uint(8));
    segs[{0..#s1.size}] = s1;
    segs[{s1.size..#s2.size}] = s2 + v1.size;
    vals[{0..#v1.size}] = v1;
    vals[{v1.size..#v2.size}] = v2;
    return (segs, vals);
  }

  private config const in1dSortThreshold = 64;
  
  proc in1d(mainStr: SegString, testStr: SegString, invert=false) throws where !useHash {
    var truth: [mainStr.offsets.aD] bool;
    // Early exit for zero-length result
    if (mainStr.size == 0) {
      return truth;
    }
    if (testStr.size <= in1dSortThreshold) {
      for i in 0..#testStr.size {
        truth |= (mainStr == testStr[i]);
      }
      return truth;
    } else {
      // This is inspired by numpy in1d
      const (uoMain, uvMain, cMain, revIdx) = uniqueGroup(mainStr, returnInverse=true);
      const (uoTest, uvTest, cTest, revTest) = uniqueGroup(testStr);
      const (segs, vals) = concat(uoMain, uvMain, uoTest, uvTest);
      if DEBUG {writeln("Unique strings in first array: %t\nUnique strings in second array: %t\nConcat length: %t".format(uoMain.size, uoTest.size, segs.size)); try! stdout.flush();}
      var st = new owned SymTab();
      const ar = new owned SegString(segs, vals, st);
      const order = ar.argsort();
      const (sortedSegs, sortedVals) = ar[order];
      const sar = new owned SegString(sortedSegs, sortedVals, st);
      if DEBUG { writeln("Sorted concatenated unique strings:"); sar.show(10); stdout.flush(); }
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
      if DEBUG {writeln("Flag pop: ", + reduce flag); try! stdout.flush();}
      // Now flag contains true for both elements of duplicate pairs
      if invert {flag = !flag;}
      // Permute back to unique order
      var ret: [D] bool;
      forall (o, f) in zip(order, flag) with (var agg = newDstAggregator(bool)) {
        agg.copy(ret[o], f);
      }
      if DEBUG {writeln("Ret pop: ", + reduce ret); try! stdout.flush();}
      // Broadcast back to original (pre-unique) order
      var truth: [mainStr.offsets.aD] bool;
      forall (t, i) in zip(truth, revIdx) with (var agg = newSrcAggregator(bool)) {
        agg.copy(t, ret[i]);
      }
      return truth;
    }
  }

  /* Convert an array of raw bytes into a Chapel string. */
  inline proc interpretAsString(bytearray: [?D] uint(8)): string {
    // Byte buffer must be local in order to make a C pointer
    var localBytes: [0..#D.size] uint(8) = bytearray;
    var cBytes = c_ptrTo(localBytes);
    // Byte buffer is null-terminated, so length is buffer.size - 1
    // The contents of the buffer should be copied out because cBytes will go out of scope
    // var s = new string(cBytes, D.size-1, D.size, isowned=false, needToCopy=true);
    var s = try! createStringWithNewBuffer(cBytes, D.size-1, D.size);
    return s;
  }
}
