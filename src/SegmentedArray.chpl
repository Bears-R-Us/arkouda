module SegmentedArray {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;

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

  proc convertPythonIndexToChapel(pyidx: int, high: int): int {
    var chplIdx: int;
    if (pyidx < 0) {
      chplIdx = high + 1 + pyidx;
    } else {
      chplIdx = pyidx;
    }
    return chplIdx;
  }

}