module StringType {
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
      var start = offsets.a[idx];
      var end: int;
      if (idx == size - 1) {
	end = nBytes - 1;
      } else {
	end = offsets.a[idx+1] - 1;
      }
      var s = interpretAsString(values.a[start..end]);
      return s;
    }

    proc this(slice: range(stridable=true)) {
      var newSegs = makeDistArray(slice.size, int);
      newSegs = offsets.a[slice];
      var newNBytes: int;
      var start = offsets.a[slice.low];
      var end: int;
      if (slice.high == offsets.aD.high) {
	end = values.aD.high;
      } else {
	end = offsets.a[slice.high+1] - 1;
      }
      var newVals = makeDistArray(end - start + 1, uint(8));
      newVals = values.a[start..end];
      return (newSegs, newVals);
    }
  }

  proc interpretAsString(bytearray: [?D] uint(8)): string {
    var localBytes: [0..#D.size] uint(8) = bytearray;
    var cBytes = c_ptrTo(localBytes);
    var s = new string(cBytes, D.size-1, D.size, isowned=false, needToCopy=true);
    return s;
  }
}