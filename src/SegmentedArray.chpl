module SegmentedArray {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use UnorderedCopy;

  class SegString {
    var offsets: unmanaged SymEntry(int);
    var values: unmanaged SymEntry(uint(8));
    var size: int;
    var nBytes: int;

    proc init(segments: borrowed SymEntry(int), values: borrowed SymEntry(uint(8))) {
      offsets = segments;
      values = values;
      size = segments.size;
      nBytes = values.size;
    }

    proc init(segName: string, valName: string, st: borrowed SymTab) {
      var gs = st.lookup(segName);
      var segs = toSymEntry(gs, int): unmanaged SymEntry(int);
      offsets = segs;
      var vs = st.lookup(valName);
      var vals = toSymEntry(vs, uint(8)): unmanaged SymEntry(uint(8));
      values = vals;
      size = segs.size;
      nBytes = vals.size;
    }

    proc this(idx: int): string {
      // TO DO: Error handling for out of bounds
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

    proc this(slice: range(stridable=true)) {
      // TO DO: Error handling for out of bounds
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

    proc this(iv: [?D] int) {
      /* var ivMin = min reduce iv; */
      /* var ivMax = max reduce iv; */
      // TO DO: Error handling for out of bounds
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

    proc this(iv: [offsets.aD] bool) {
      // TO DO: Error handling for out of bounds
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      var segInds = + scan iv;
      var newSize = segInds[high];
      segInds -= 1;
      var gatheredLengths = makeDistArray(newSize, int);
      // Lengths of segments including null bytes
      /* var lengths: [offsets.aD] int; */
      /* lengths[low..#(size-1)] = oa[(low+1)..#(size-1)] - oa[low..#(size-1)]; */
      /* lengths[high] = values.size - oa[high]; */
      [idx in offsets.aD] if (iv[idx] == true) {
        var l: int;
        if (idx == high) {
          l = values.size - oa[high];
        } else {
          l = oa[idx+1] - oa[idx];
        }
        unorderedCopy(gatheredLengths[segInds[idx]], l);
      }
      var gatheredOffsets = (+ scan gatheredLengths);
      var retBytes = gatheredOffsets[newSize-1];
      gatheredOffsets -= gatheredLengths;
      var gatheredVals = makeDistArray(retBytes, uint(8));
      ref va = values.a;
      forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, segInds) {
        // gatheredVals[go..#gl] = va[oa[idx]..#lengths[idx]];
        for pos in 0..#gl {
          unorderedCopy(gatheredVals[go+pos], va[oa[idx]+pos]);
        }
      }
      return (gatheredOffsets, gatheredVals);
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
    var s = new string(cBytes, D.size-1, D.size, isowned=false, needToCopy=true);
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
    select subcmd {
      when "intIndex" {
        return segIntIndex(objtype, args, st);
      }
      when "sliceIndex" {
        return segSliceIndex(objtype, args, st);
      }
      /* when "pdarrayIndex" { */
      /*   return segPdarrayIndex(objtype, args, st); */
      /* } */
      otherwise {
        return try! "Error: in %s, unknown subcommand %s".format(pn, subcmd);
      }
      }
  }
  
  proc segIntIndex(objtype: string, args: [] string, st: borrowed SymTab): string {
    var pn = "segIntIndex";
    select objtype {
      when "string" {
        var segName = args[1];
        var valName = args[2];
        var strings = new owned SegString(segName, valName, st);
        var idx = try! args[3]:int;
        idx = convertPythonIndexToChapel(idx, strings.size);
        var s = strings[idx];
        return try! "item %s %jt".format("string", s);
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

  proc segSliceIndex(objtype: string, args: [] string, st: borrowed SymTab): string {
    var pn = "segSliceIndex";
    select objtype {
      when "string" {
        var segName = args[1];
        var valName = args[2];
        var strings = new owned SegString(segName, valName, st);
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

  proc segPdarrayIndex(objtype: string, args: [] string, st: borrowed SymTab): string {
    var pn = "segPdarrayIndex";
    var newSegName = st.nextName();
    var newValName = st.nextName();
    select objtype {
      when "string" {
        var segName = args[1];
        var valName = args[2];
        var strings = new owned SegString(segName, valName, st);
        var iname = args[3];
        var gIV: borrowed GenSymEntry = st.lookup(iname);
        if (gIV == nil) {return unknownSymbolError(pn, iname);}
        select gIV.dtype {
          when DType.Int64 {
            var iv = toSymEntry(gIV, int);
            var (newSegs, newVals) = strings[iv];
            var newSegsEntry = new shared SymEntry(newSegs);
            var newValsEntry = new shared SymEntry(newVals);
            st.addEntry(newSegName, newSegsEntry);
            st.addEntry(newValName, newValsEntry);
          }
          when DType.Bool {
            var iv = toSymEntry(gIV, bool);
            var (newSegs, newVals) = strings[iv];
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

  proc segBinopvvMsg(reqMsg: string, st: borrowed SymTab): string {
    var pn = "segBinopvs";
    var repMsg: string;
    var fields = reqMsg.split();
    var cmd = fields[1];
    var op = fields[2];
    var ltype = fields[3];
    var lsegName = fields[4];
    var lvalName = fields[5];
    var rtype = fields[6];
    var rsegName = fields[7];
    var rvalName = fields[8];
    var rname = st.nextName();
    select (ltype, rtype) {
    when ("string", "string") {
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

  proc segBinopvsMsg(reqMsg: string, st: borrowed SymTab): string {
    var pn = "segBinopvs";
    var repMsg: string;
    var fields = reqMsg.split();
    var cmd = fields[1];
    var op = fields[2];
    var objtype = fields[3];
    var segName = fields[4];
    var valName = fields[5];
    var valtype = fields[6];
    var value = fields[7];
    var rname = st.nextName();
    select (objtype, valtype) {
    when ("string", "string") {
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
}