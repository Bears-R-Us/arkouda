module SegmentedArray {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use UnorderedCopy;
  use SipHash;
  use RadixSortLSD;
  use Reflection;
  use PrivateDist;

  private config const DEBUG = false;
  
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
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // Lengths of segments including null bytes
      var gatheredLengths: [D] int;
      [(gl, idx) in zip(gatheredLengths, iv)] {
        var l: int;
        if (idx == high) {
          l = values.size - oa[high];
        } else {
          l = oa[idx+1] - oa[idx];
        }
        unorderedCopy(gl, l);
      }
      // The returned offsets are the 0-up cumulative lengths
      var gatheredOffsets = (+ scan gatheredLengths);
      // The total number of bytes in the gathered strings
      var retBytes = gatheredOffsets[D.high];
      gatheredOffsets -= gatheredLengths;
      var gatheredVals = makeDistArray(retBytes, uint(8));
      ref va = values.a;
      // Copy string data to gathered result
      forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, iv) {
        for pos in 0..#gl {
          unorderedCopy(gatheredVals[go+pos], va[oa[idx]+pos]);
        }
      }
      return (gatheredOffsets, gatheredVals);
    }

    /* Logical indexing (compress) of strings. */
    proc this(iv: [?D] bool) throws {
      // Index vector must be same domain as array
      if (D != offsets.aD) {
        throw new owned OutOfBoundsError();
      }
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // Calculate the destination indices
      var steps = + scan iv;
      var newSize = steps[high];
      // Early return for zero-length result
      if (newSize == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      var segInds = makeDistArray(newSize, int);
      // Lengths of dest segments including null bytes
      var gatheredLengths = makeDistArray(newSize, int);
      forall (idx, present, i) in zip(D, iv, steps) {
        if present {
          segInds[i-1] = idx;
          if (idx == high) {
            gatheredLengths[i-1] = values.size - oa[high];
          } else {
            gatheredLengths[i-1] = oa[idx+1] - oa[idx];
          }
        }
      }
      // Make dest offsets from lengths
      var gatheredOffsets = (+ scan gatheredLengths);
      var retBytes = gatheredOffsets[newSize-1];
      gatheredOffsets -= gatheredLengths;
      var gatheredVals = makeDistArray(retBytes, uint(8));
      ref va = values.a;
      if DEBUG {
        printAry("gatheredOffsets: ", gatheredOffsets);
        printAry("gatheredLengths: ", gatheredLengths);
        printAry("segInds: ", segInds);
      }
      // Copy string bytes from src to dest
      forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, segInds) {
        gatheredVals[{go..#gl}] = va[{oa[idx]..#gl}];
      }
      return (gatheredOffsets, gatheredVals);
    }

    /* Apply a hash function to all strings. This is useful for grouping
       and set membership. The hash used is SipHash128.*/
    proc hash(hashKey=defaultSipHashKey) throws {
      // 128-bit hash values represented as 2-tuples of uint(64)
      var hashes: [offsets.aD] 2*uint(64);
      // Early exit for zero-length result
      if (size == 0) {
        return hashes;
      }
      ref oa = offsets.a;
      ref va = values.a;
      // Compute lengths of strings
      var lengths: [offsets.aD] int;
      forall (idx, l) in zip(offsets.aD, lengths) {
        if (idx == offsets.aD.high) {
          l = values.size - oa[idx];
        } else {
          l = oa[idx+1] - oa[idx];
        }
      }
      // Hash each string
      forall (o, l, h) in zip(oa, lengths, hashes) {
        h = sipHash128(va[{o..#l}], hashKey);
      }
      return hashes;
    }

    /* Return a permutation that groups the strings. Because hashing is used,
       this permutation will not sort the strings, but all equivalent strings
       will fall in one contiguous block. */
    proc argGroup() throws {
      // Hash all strings
      var hashes = this.hash();
      // Return the permutation that sorts the hashes
      var iv = radixSortLSD_ranks(hashes);
      if DEBUG {
        var sortedHashes = [i in iv] hashes[i];
        var diffs = sortedHashes[(iv.domain.low+1)..#(iv.size-1)] - sortedHashes[(iv.domain.low)..#(iv.size-1)];
        printAry("diffs = ", diffs);
        var nonDecreasing = [d in diffs] ((d[1] > 0) || ((d[1] == 0) && (d[2] >= 0)));
        writeln("Are hashes sorted? ", && reduce nonDecreasing);
      }
      return iv;
    }
  }

  /* Test for equality between two same-length arrays of strings. Returns
     a boolean vector of the same length. */
  proc ==(lss:SegString, rss:SegString) throws {
    // String arrays must be same size
    if (lss.size != rss.size) {
      throw new owned ArgumentError();
    }
    ref oD = lss.offsets.aD;
    var truth: [oD] bool;
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
    forall (t, lo, ro, idx) in zip(truth, loffsets, roffsets, oD) {
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
        // Only if lengths and all bytes are equal, set result to true
        if allEqual {
          unorderedCopy(t, true);
        }
      }
    }
    return truth;
  }

  /* Test an array of strings for equality against a constant string. Return a boolean
     vector the same size as the array. */
  proc ==(ss:SegString, testStr: string) {
    ref oD = ss.offsets.aD;
    // Initialize to true, then set non-matching entries to false along the way
    var truth: [oD] bool = true;
    // Early exit for zero-length result
    if (ss.size == 0) {
      return truth;
    }
    ref values = ss.values.a;
    ref offsets = ss.offsets.a;
    // Use a whole-array strategy, where the ith byte from every segment is checked simultaneously
    // This will do len(testStr) parallel loops, but loops will have low overhead
    for (b, i) in zip(testStr.chpl_bytes(), 0..) {
      [(t, o, idx) in zip(truth, offsets, oD)] if (b != values[o+i]) {unorderedCopy(t, false);}
    }
    // Check the length by checking that the next byte is null
    [(t, o, idx) in zip(truth, offsets, oD)] if (0 != values[o+testStr.size]) {unorderedCopy(t, false);}
    return truth;
  }

  /* Test array of strings for membership in another array (set) of strings. Returns
     a boolean vector the same size as the first array. */
  proc in1d(mainStr: SegString, testStr: SegString, invert=false, hashKey=defaultSipHashKey) throws {
    var truth: [mainStr.offsets.aD] bool;
    // Early exit for zero-length result
    if (mainStr.size == 0) {
      return truth;
    }
    // Hash all strings for fast comparison
    const hashes = mainStr.hash(hashKey);
    // On each locale, make an associative domain with the hashes of the second array
    // parSafe=false because we are adding in serial and it's faster
    var localTestHashes: [PrivateSpace] domain(2*uint(64), parSafe=false);
    coforall loc in Locales {
      on loc {
        // Local hashes of second array
        ref mySet = localTestHashes[here.id];
        mySet.requestCapacity(testStr.size);
        const testHashes = testStr.hash(hashKey);
        for h in testHashes {
          mySet += h;
        }
        // Check membership of hashes in this locale's chunk of the array
        [i in truth.localSubdomain()] truth[i] = mySet.contains(hashes[i]);
      }
    }
    return truth;
  }

  /* Convert an array of raw bytes into a Chapel string. */
  inline proc interpretAsString(bytearray: [?D] uint(8)): string {
    // Byte buffer must be local in order to make a C pointer
    var localBytes: [0..#D.size] uint(8) = bytearray;
    var cBytes = c_ptrTo(localBytes);
    // Byte buffer is null-terminated, so length is buffer.size - 1
    // The contents of the buffer should be copied out because cBytes will go out of scope
    // var s = new string(cBytes, D.size-1, D.size, isowned=false, needToCopy=true);
    var s = createStringWithNewBuffer(cBytes, D.size-1, D.size);
    return s;
  }
}