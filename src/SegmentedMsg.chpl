module SegmentedMsg {
  use SegmentedArray;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  
  proc segmentedIndexMsg(reqMsg: string, st: borrowed SymTab): string {
    var pn = Reflection.getRoutineName();
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
    var pn = Reflection.getRoutineName();
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
    var pn = Reflection.getRoutineName();
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
    var pn = Reflection.getRoutineName();
    var newSegName = st.nextName();
    var newValName = st.nextName();
    select objtype {
      when "str" {
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
    var pn = Reflection.getRoutineName();
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
      var lstrings = new owned SegString(lsegName, lvalName, st);
      var rstrings = new owned SegString(rsegName, rvalName, st);
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
    var pn = Reflection.getRoutineName();
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
    when ("str", "str") {
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
    return "created " + st.attrib(rname);
  }

  proc segIn1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var fields = reqMsg.split();
    var cmd = fields[1];
    var mainObjtype = fields[2];
    var mainSegName = fields[3];
    var mainValName = fields[4];
    var testObjtype = fields[5];
    var testSegName = fields[6];
    var testValName = fields[7];
    var rname = st.nextName();
    select (mainObjtype, testObjtype) {
    when ("str", "str") {
      var mainStr = new owned SegString(mainSegName, mainValName, st);
      var testStr = new owned SegString(testSegName, testValName, st);
      var e = st.addEntry(rname, mainStr.size, bool);
      e.a = in1d(mainStr, testStr);
    }
    otherwise {return unrecognizedTypeError(pn, "("+mainObjtype+", "+testObjtype+")");}
    }
    return "created " + st.attrib(rname);
  }

  proc segGroupMsg(reqMsg: string, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
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
    return "created " + st.attrib(rname);
  }
}