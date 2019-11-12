module SegmentedArray {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use UnorderedCopy;
  use Crypto;
  use MurmurHash;
  use RadixSortLSD;
  use SHA256Implementation only SHA256Digest;

  private config const DEBUG = false;
  
  class OutOfBoundsError: Error {}

  class SegString {
    var offsets: borrowed SymEntry(int);
    var values: borrowed SymEntry(uint(8));
    var size: int;
    var nBytes: int;

    proc init(segments: borrowed SymEntry(int), values: borrowed SymEntry(uint(8))) {
      offsets = segments;
      values = values;
      size = segments.size;
      nBytes = values.size;
    }

    proc init(segName: string, valName: string, st: borrowed SymTab) {
      var gs = try! st.lookup(segName);
      var segs = toSymEntry(gs, int): unmanaged SymEntry(int);
      offsets = segs;
      var vs = try! st.lookup(valName);
      var vals = toSymEntry(vs, uint(8)): unmanaged SymEntry(uint(8));
      values = vals;
      size = segs.size;
      nBytes = vals.size;
    }

    proc this(idx: int): string throws {
      if (idx < 0) || (idx >= offsets.size) {
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
      if (slice.low < 0) || (slice.high >= offsets.size) {
        throw new owned OutOfBoundsError();
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
      var ivMin = min reduce iv;
      var ivMax = max reduce iv;
      if (ivMin < 0) || (ivMax >= offsets.size) {
        throw new owned OutOfBoundsError();
      }
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // Lengths of segments including null bytes
      /* var lengths: [offsets.aD] int; */
      /* lengths[low..#(size-1)] = oa[(low+1)..#(size-1)] - oa[low..#(size-1)]; */
      /* lengths[high] = values.size - oa[high]; */
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
      var segInds = makeDistArray(newSize, int);
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
      // Lengths of segments including null bytes
      /* var lengths: [offsets.aD] int; */
      /* lengths[low..#(size-1)] = oa[(low+1)..#(size-1)] - oa[low..#(size-1)]; */
      /* lengths[high] = values.size - oa[high]; */
      /* [idx in offsets.aD] if (iv[idx] == true) { */
      /*   var l: int; */
      /*   if (idx == high) { */
      /*     l = values.size - oa[high]; */
      /*   } else { */
      /*     l = oa[idx+1] - oa[idx]; */
      /*   } */
      /*   unorderedCopy(gatheredLengths[segInds[idx]], l); */
      /* } */
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

    inline proc hash2tuple(h: CryptoBuffer): 2*uint(64) {
      var d = h.getBuffData();
      var res: 2*uint(64);
      for param i in 0..7 {
        res[1] |= (d[i]: uint(64)) << (7-i)*8;
        res[2] |= (d[i+8]: uint(64)) << (7-i)*8;
      }
      return res;
    }

    inline proc chplHash(buf: [?D] uint(8)): 2*uint(64) {
      var res: 2*uint(64);
      for chunk in D by 16 {
        var word: 2*uint(64);
        for i in chunk..min(chunk+16, D.high) {
          var shift = 7 - ((i - chunk) / 2);
          word[((i - chunk) % 2) + 1] |= buf[i] << shift;
        }
        res[1] ^= _gen_key(word[1]);
        res[2] ^= _gen_key(word[2]);
      }
      return res;
    }

    proc cryptoHash(b: [] uint(8), hashFxn): 2*uint(64) {
      var hasher = new owned Hash(hashFxn);
      var buf = new owned CryptoBuffer(b);
      const hashed = hasher.getDigest(buf);
      return hash2tuple(hashed);
    }

    inline proc murmurHash(b: [] uint(8)): 2*uint(64) {
      return MurmurHash3_128(b);
    }

    inline proc SHA256Hash(b: [] uint(8)): 2*uint(64) {
      const h = SHA256Digest(b);
      var res: 2*uint(64);
      res[1] = ((h[1]:uint(64)) << 32) | (h[2]:uint(64));
      res[2] = ((h[3]:uint(64)) << 32) | (h[4]:uint(64));
      return res;
    }

    proc hash() {
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
      var hashes: [offsets.aD] 2*uint(64);
      forall (o, l, h) in zip(oa, lengths, hashes) {
        // h = cryptoHash(va[{o..#l}], Digest.SHA1);
        h = SHA256Hash(va[{o..#l}]);
      }
      return hashes;
    }
    
    proc argGroup() {
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
  
  proc ==(lss:SegString, rss:SegString) {
    ref oD = lss.offsets.aD;
    ref lvalues = lss.values.a;
    ref loffsets = lss.offsets.a;
    ref rvalues = rss.values.a;
    ref roffsets = rss.offsets.a;
    var truth: [oD] bool;
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
    ref values = ss.values.a;
    ref offsets = ss.offsets.a;
    for (b, i) in zip(testStr.chpl_bytes(), 0..) {
      [(t, o, idx) in zip(truth, offsets, oD)] if (b != values[o+i]) {unorderedCopy(t, false);}
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

  proc segmentedIndexMsg(reqMsg: string, st: borrowed SymTab): string {
    var pn = "segmentedIndexMsg";
    var repMsg: string;
    var fields = reqMsg.split();
    var cmd = fields[1];
    var subcmd = fields[2];
    var objtype = fields[3];
    var args: [1..#(fields.size-3)] string = fields[4..];
    try {
      select subcmd {
        when "intIndex" {
          return segIntIndex(objtype, args, st);
        }
        when "sliceIndex" {
          return segSliceIndex(objtype, args, st);
        }
        when "pdarrayIndex" {
          return segPdarrayIndex(objtype, args, st);
        }
        otherwise {
          return try! "Error: in %s, unknown subcommand %s".format(pn, subcmd);
        }
        }
    } catch e: OutOfBoundsError {
      return "Error: index out of bounds";
    } catch {
      return "Error: unknown cause";
    }
  }
  
  proc segIntIndex(objtype: string, args: [] string, st: borrowed SymTab): string throws {
    var pn = "segIntIndex";
    select objtype {
      when "str" {
        // args = (segName, valName, index)
        var strings = new owned SegString(args[1], args[2], st);
        var idx = try! args[3]:int;
        idx = convertPythonIndexToChapel(idx, strings.size);
        var s = strings[idx];
        return try! "item %s %jt".format("str", s);
      }
      otherwise { return notImplementedError(pn, objtype); }
      }
  }

  proc convertPythonIndexToChapel(pyidx: int, high: int): int {
    var chplIdx: int;
    if (pyidx < 0) {
      chplIdx = high + 1 + pyidx;
    } else {
      chplIdx = pyidx;
    }
    return chplIdx;
  }

  proc segSliceIndex(objtype: string, args: [] string, st: borrowed SymTab): string throws {
    var pn = "segSliceIndex";
    select objtype {
      when "str" {
        /* var gsegs = st.lookup(args[1]); */
        /* var segs = toSymEntry(gsegs, int); */
        /* var gvals = st.lookup(args[2]); */
        /* var vals = toSymEntry(gvals, uint(8)); */
        /* var strings = new owned SegString(segs, vals); */
        var strings = new owned SegString(args[1], args[2], st);
        var start = try! args[3]:int;
        var stop = try! args[4]:int;
        var stride = try! args[5]:int;
        if (stride != 1) { return notImplementedError(pn, "stride != 1"); }
        var slice: range(stridable=true) = convertPythonSliceToChapel(start, stop, stride);
        var newSegName = st.nextName();
        var newValName = st.nextName();
        var (newSegs, newVals) = strings[slice];
        var newSegsEntry = new shared SymEntry(newSegs);
        var newValsEntry = new shared SymEntry(newVals);
        st.addEntry(newSegName, newSegsEntry);
        st.addEntry(newValName, newValsEntry);
        return try! "created " + st.attrib(newSegName) + " +created " + st.attrib(newValName);
      }
      otherwise {return notImplementedError(pn, objtype);}
      }
  }

  proc convertPythonSliceToChapel(start:int, stop:int, stride:int=1): range(stridable=true) {
    var slice: range(stridable=true);
    // convert python slice to chapel slice
    // backwards iteration with negative stride
    if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
    // forward iteration with positive stride
    else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
    // BAD FORM start < stop and stride is negative
    else {slice = 1..0;}
    return slice;
  }

  proc segPdarrayIndex(objtype: string, args: [] string, st: borrowed SymTab): string throws {
    var pn = "segPdarrayIndex";
    var newSegName = st.nextName();
    var newValName = st.nextName();
    select objtype {
      when "str" {
        /* var gsegs = st.lookup(args[1]); */
        /* var segs = toSymEntry(gsegs, int); */
        /* var gvals = st.lookup(args[2]); */
        /* var vals = toSymEntry(gvals, uint(8)); */
        /* var strings = new owned SegString(segs, vals); */
        var strings = new owned SegString(args[1], args[2], st);
        var iname = args[3];
        var gIV: borrowed GenSymEntry = st.lookup(iname);
        select gIV.dtype {
          when DType.Int64 {
            var iv = toSymEntry(gIV, int);
            var (newSegs, newVals) = strings[iv.a];
            var newSegsEntry = new shared SymEntry(newSegs);
            var newValsEntry = new shared SymEntry(newVals);
            st.addEntry(newSegName, newSegsEntry);
            st.addEntry(newValName, newValsEntry);
          }
          when DType.Bool {
            var iv = toSymEntry(gIV, bool);
            var (newSegs, newVals) = strings[iv.a];
            var newSegsEntry = new shared SymEntry(newSegs);
            var newValsEntry = new shared SymEntry(newVals);
            st.addEntry(newSegName, newSegsEntry);
            st.addEntry(newValName, newValsEntry);
          }
          otherwise {return notImplementedError(pn,
                                                "("+objtype+","+dtype2str(gIV.dtype)+")");}
          }
      }
      otherwise {return notImplementedError(pn, objtype);}
      }
    return try! "created " + st.attrib(newSegName) + "+created " + st.attrib(newValName);
  }

  proc segBinopvvMsg(reqMsg: string, st: borrowed SymTab): string throws {
    var pn = "segBinopvv";
    var repMsg: string;
    var fields = reqMsg.split();
    var cmd = fields[1];
    var op = fields[2];
    var ltype = fields[3];
    var lsegName = fields[4];
    /* var glsegs = st.lookup(lsegName); */
    /* var lsegs = toSymEntry(glsegs, int); */
    var lvalName = fields[5];
    /* var glvals = st.lookup(lvalName); */
    var rtype = fields[6];
    var rsegName = fields[7];
    /* var grsegs = st.lookup(rsegName); */
    /* var rsegs = toSymEntry(grsegs, int); */
    var rvalName = fields[8];
    /* var grvals = st.lookup(rvalName); */
    var rname = st.nextName();
    select (ltype, rtype) {
    when ("str", "str") {
      /* var lvals = toSymEntry(glvals, uint(8)); */
      /* var rvals = toSymEntry(grvals, uint(8)); */
      /* var lstrings = SegString(lsegs, lvals); */
      /* var rstrings = SegString(rsegs, rvals); */
      var lstrings = SegString(lsegName, lvalName, st);
      var rstrings = SegString(rsegName, rvalName, st);
      select op {
        when "==" {
          var e = st.addEntry(rname, lstrings.size, bool);
          e.a = (lstrings == rstrings);
        }
        otherwise {return notImplementedError(pn, ltype, op, rtype);}
        }
    }
    otherwise {return unrecognizedTypeError(pn, "("+ltype+", "+rtype+")");} 
    }
    return try! "created " + st.attrib(rname);
  }

  proc segBinopvsMsg(reqMsg: string, st: borrowed SymTab): string throws {
    var pn = "segBinopvs";
    var repMsg: string;
    var fields = reqMsg.split();
    var cmd = fields[1];
    var op = fields[2];
    var objtype = fields[3];
    var segName = fields[4];
    /* var gsegs = st.lookup(segName); */
    /* var segs = toSymEntry(gsegs, int); */
    var valName = fields[5];
    /* var gvals = st.lookup(valName); */
    var valtype = fields[6];
    var value = fields[7];
    var rname = st.nextName();
    select (objtype, valtype) {
    when ("str", "str") {
      /* var vals = toSymEntry(gvals, uint(8)); */
      /* var strings = new owned SegString(segs, vals); */
      var strings = new owned SegString(segName, valName, st);
      select op {
        when "==" {
          var e = st.addEntry(rname, strings.size, bool);
          e.a = (strings == value);
        }
        otherwise {return notImplementedError(pn, objtype, op, valtype);}
        }
    }
    otherwise {return unrecognizedTypeError(pn, "("+objtype+", "+valtype+")");} 
    }
    return try! "created " + st.attrib(rname);
  }

  proc segGroupMsg(reqMsg: string, st: borrowed SymTab): string {
    var pn = "segGroupMsg";
    var fields = reqMsg.split();
    var cmd = fields[1];
    var objtype = fields[2];
    var segName = fields[3];
    var valName = fields[4];
    var rname = st.nextName();
    select (objtype) {
    when "str" {
      var strings = new owned SegString(segName, valName, st);
      var iv = st.addEntry(rname, strings.size, int);
      iv.a = strings.argGroup();
    }
    otherwise {return notImplementedError(pn, "("+objtype+")");}
    }
    return try! "created " + st.attrib(rname);
  }
}