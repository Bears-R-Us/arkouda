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

  class SegString {
    var offsets: borrowed SymEntry(int);
    var values: borrowed SymEntry(uint(8));
    var size: int;
    var nBytes: int;
   
    proc init(segments: borrowed SymEntry(int), values: borrowed SymEntry(uint(8))) {
      // offsetDom = segments.aD;
      offsets = segments;
      // valueDom = values.aD;
      values = values;
      size = segments.size;
      nBytes = values.size;
    }

    proc init(segName: string, valName: string, st: borrowed SymTab) {
      var gs = try! st.lookup(segName);
      var segs = toSymEntry(gs, int): unmanaged SymEntry(int);
      // offsetDom = segs.aD;
      offsets = segs;
      var vs = try! st.lookup(valName);
      var vals = toSymEntry(vs, uint(8)): unmanaged SymEntry(uint(8));
      // valueDom = vals.aD;
      values = vals;
      size = segs.size;
      nBytes = vals.size;
    }

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

    proc this(slice: range(stridable=true)) throws {
      if (slice.low < offsets.aD.low) || (slice.high > offsets.aD.high) {
        throw new owned OutOfBoundsError();
      }
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

    proc this(iv: [?D] int) throws {
      if (D.size == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
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
      var gatheredOffsets = (+ scan gatheredLengths);
      var retBytes = gatheredOffsets[D.high];
      gatheredOffsets -= gatheredLengths;
      var gatheredVals = makeDistArray(retBytes, uint(8));
      ref va = values.a;
      forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, iv) {
        // gatheredVals[go..#gl] = va[oa[idx]..#lengths[idx]];
        for pos in 0..#gl {
          unorderedCopy(gatheredVals[go+pos], va[oa[idx]+pos]);
        }
      }
      return (gatheredOffsets, gatheredVals);
    }

    proc this(iv: [?D] bool) throws {
      if (D != offsets.aD) {
        throw new owned OutOfBoundsError();
      }
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // BUG: this is not segInds, this is steps. Need to compress and translate the index.
      var steps = + scan iv;
      var newSize = steps[high];
      if (newSize == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      var segInds = makeDistArray(newSize, int);
      // Lengths of segments including null bytes
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
      forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, segInds) {
        gatheredVals[{go..#gl}] = va[{oa[idx]..#gl}];
      }
      return (gatheredOffsets, gatheredVals);
    }

    proc hash(hashKey=defaultSipHashKey) throws {
      var hashes: [offsets.aD] 2*uint(64);
      if (size == 0) {
        return hashes;
      }
      ref oa = offsets.a;
      ref va = values.a;
      var lengths: [offsets.aD] int;
      forall (idx, l) in zip(offsets.aD, lengths) {
        if (idx == offsets.aD.high) {
          l = values.size - oa[idx];
        } else {
          l = oa[idx+1] - oa[idx];
        }
      }
      const maxLen = max reduce lengths;
      const empty: [0..#maxLen] uint(8);
      forall (o, l, h) in zip(oa, lengths, hashes) {
        h = sipHash128(va[{o..#l}], hashKey);
      }
      return hashes;
    }
    
    proc argGroup() throws {
      var hashes = this.hash();
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
  
  proc ==(lss:SegString, rss:SegString) throws {
    if (lss.size != rss.size) {
      throw new owned ArgumentError();
    }
    ref oD = lss.offsets.aD;
    var truth: [oD] bool;
    if (lss.size == 0) {
      return truth;
    }
    ref lvalues = lss.values.a;
    ref loffsets = lss.offsets.a;
    ref rvalues = rss.values.a;
    ref roffsets = rss.offsets.a;
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
      if (llen == rlen) {
        var allEqual = true;
        for pos in 0..#llen {
          if (lvalues[lo+pos] != rvalues[ro+pos]) {
            allEqual = false;
            break;
          }
        }
        if allEqual {
          unorderedCopy(t, true);
        }
      }
    }
    return truth;
  }

  
  proc ==(ss:SegString, testStr: string) {
    ref oD = ss.offsets.aD;
    var truth: [oD] bool = true;
    if (ss.size == 0) {
      return truth;
    }
    ref values = ss.values.a;
    ref offsets = ss.offsets.a;
    for (b, i) in zip(testStr.chpl_bytes(), 0..) {
      [(t, o, idx) in zip(truth, offsets, oD)] if (b != values[o+i]) {unorderedCopy(t, false);}
    }
    // Check the length by checking that the next byte is null
    [(t, o, idx) in zip(truth, offsets, oD)] if (0 != values[o+testStr.size]) {unorderedCopy(t, false);}
    return truth;
  }

  proc in1d(mainStr: SegString, testStr: SegString, invert=false, hashKey=defaultSipHashKey) throws {
    var truth: [mainStr.offsets.aD] bool;
    if (mainStr.size == 0) {
      return truth;
    }
    const hashes = mainStr.hash(hashKey);
    var localTestHashes: [PrivateSpace] domain(2*uint(64), parSafe=false);
    coforall loc in Locales {
      on loc {
        ref mySet = localTestHashes[here.id];
        mySet.requestCapacity(testStr.size);
        const testHashes = testStr.hash(hashKey);
        for h in testHashes {
          mySet += h;
        }
        [i in truth.localSubdomain()] truth[i] = mySet.contains(hashes[i]);
      }
    }
    return truth;
  }

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