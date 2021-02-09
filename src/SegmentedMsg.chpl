module SegmentedMsg {
  use Reflection;
  use Errors;
  use Logging;
  use SegmentedArray;
  use ServerErrorStrings;
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use RandArray;
  use IO;
  use GenSymIO only jsonToPdArray,jsonToPdArrayInt;

  use SymArrayDmap;
  use SACA;
  use Random;
  use RadixSortLSD;
  use Set;
  use DistributedBag;
  public use ArgSortMsg;
  use Time;

  private config const DEBUG = false;
  const smLogger = new Logger();
  
  if v {
      smLogger.level = LogLevel.DEBUG;
  } else {
      smLogger.level = LogLevel.INFO;
  }

  proc randomStringsMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var (lenStr, dist, charsetStr, arg1str, arg2str, seedStr)
          = payload.decode().splitMsgToTuple(6);
      var len = lenStr: int;
      var charset = str2CharSet(charsetStr);
      var segName = st.nextName();
      var valName = st.nextName();
      var repMsg: string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
             "dist: %s segName: %t valName: %t".format(dist.toLower(),segName,valName));
      select dist.toLower() {
          when "uniform" {
              var minLen = arg1str:int;
              var maxLen = arg2str:int;
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + (maxLen + minLen)*len);
              var (segs, vals) = newRandStringsUniformLength(len, minLen, maxLen, charset, seedStr);
              var segEntry = new shared SymEntry(segs);
              var valEntry = new shared SymEntry(vals);
              st.addEntry(segName, segEntry);
              st.addEntry(valName, valEntry);
              repMsg = 'created ' + st.attrib(segName) + '+created ' + st.attrib(valName);
          }
          when "lognormal" {
              var logMean = arg1str:real;
              var logStd = arg2str:real;
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + exp(logMean + (logStd**2)/2):int*len);
              var (segs, vals) = newRandStringsLogNormalLength(len, logMean, logStd, charset, seedStr);
              var segEntry = new shared SymEntry(segs);
              var valEntry = new shared SymEntry(vals);
              st.addEntry(segName, segEntry);
              st.addEntry(valName, valEntry);
              repMsg = 'created ' + st.attrib(segName) + '+created ' + st.attrib(valName);
          }
          otherwise { 
              repMsg = notImplementedError(pn, dist);       
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
          }
      }
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return repMsg;
  }



  proc segmentLengthsMsg(cmd: string, payload: bytes, 
                                          st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var (objtype, segName, valName) = payload.decode().splitMsgToTuple(3);

    // check to make sure symbols defined
    st.check(segName);
    st.check(valName);
    
    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s objtype: %t segName: %t valName: %t".format(
                   cmd,objtype,segName,valName));

    select objtype {
      when "str" {
        var strings = new owned SegString(segName, valName, st);
        var lengths = st.addEntry(rname, strings.size, int);
        // Do not include the null terminator in the length
        lengths.a = strings.getLengths() - 1;
      }
      when "int" {
        var sarrays = new owned SegSArray(segName, valName, st);
        var lengths = st.addEntry(rname, sarrays.size, int);
        // Do not include the null terminator in the length
        lengths.a = sarrays.getLengths() - 1;
      }
      otherwise {
          var errorMsg = notImplementedError(pn, "%s".format(objtype));
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                      
          return errorMsg;
      }
    }
    var returnMsg = "created "+st.attrib(rname);
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
    return returnMsg;
  }

  proc segmentedEfuncMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (subcmd, objtype, segName, valName, valtype, valStr) = 
                                              payload.decode().splitMsgToTuple(6);

      // check to make sure symbols defined
      st.check(segName);
      st.check(valName);

      var json = jsonToPdArray(valStr, 1);
      var val = json[json.domain.low];
      var rname = st.nextName();
    
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));
    
        select (objtype, valtype) {
          when ("str", "str") {
            var strings = new owned SegString(segName, valName, st);
            select subcmd {
                when "contains" {
                var truth = st.addEntry(rname, strings.size, bool);
                truth.a = strings.substringSearch(val, SearchMode.contains);
                repMsg = "created "+st.attrib(rname);
            }
            when "startswith" {
                var truth = st.addEntry(rname, strings.size, bool);
                truth.a = strings.substringSearch(val, SearchMode.startsWith);
                repMsg = "created "+st.attrib(rname);
            }
            when "endswith" {
                var truth = st.addEntry(rname, strings.size, bool);
                truth.a = strings.substringSearch(val, SearchMode.endsWith);
                repMsg = "created "+st.attrib(rname);
            }
            otherwise {
               return notImplementedError(pn, "subcmd: %s, (%s, %s)".format(
                         subcmd, objtype, valtype));
            }
          }
        }
        otherwise {
          var errorMsg = "(%s, %s)".format(objtype, valtype);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
          return notImplementedError(pn, errorMsg);
        }
      }
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return repMsg;
  }

proc segmentedPeelMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (subcmd, objtype, segName, valName, valtype, valStr,
         idStr, kpStr, lStr, jsonStr) = payload.decode().splitMsgToTuple(10);

    // check to make sure symbols defined
    st.check(segName);
    st.check(valName);

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));

    select (objtype, valtype) {
    when ("str", "str") {
      var strings = new owned SegString(segName, valName, st);
      select subcmd {
        when "peel" {
          var times = valStr:int;
          var includeDelimiter = (idStr.toLower() == "true");
          var keepPartial = (kpStr.toLower() == "true");
          var left = (lStr.toLower() == "true");
          var json = jsonToPdArray(jsonStr, 1);
          var val = json[json.domain.low];
          var loname = st.nextName();
          var lvname = st.nextName();
          var roname = st.nextName();
          var rvname = st.nextName();
          select (includeDelimiter, keepPartial, left) {
          when (false, false, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, false, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (false, false, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, false, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (false, true, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, true, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (false, true, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, true, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, false, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, false, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, false, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, false, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, true, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, true, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, true, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, true, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } otherwise {
              var errorMsg = notImplementedError(pn, 
                               "subcmd: %s, (%s, %s)".format(subcmd, objtype, valtype));
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return errorMsg;                            
              }
          }
          repMsg = "created %s+created %s+created %s+created %s".format(st.attrib(loname),
                                                                        st.attrib(lvname),
                                                                        st.attrib(roname),
                                                                        st.attrib(rvname));
        }
        otherwise {
            var errorMsg = notImplementedError(pn, 
                              "subcmd: %s, (%s, %s)".format(subcmd, objtype, valtype));
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return errorMsg;                                          
        }
      }
    }
    otherwise {
        var errorMsg = notImplementedError(pn, "(%s, %s)".format(objtype, valtype));
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return errorMsg;       
      }
    }
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return repMsg;
  }

  proc segmentedHashMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, segName, valName) = payload.decode().splitMsgToTuple(3);

    // check to make sure symbols defined
    st.check(segName);
    st.check(valName);

    select objtype {
        when "str" {
            var strings = new owned SegString(segName, valName, st);
            var hashes = strings.hash();
            var name1 = st.nextName();
            var hash1 = st.addEntry(name1, hashes.size, int);
            var name2 = st.nextName();
            var hash2 = st.addEntry(name2, hashes.size, int);
            forall (h, h1, h2) in zip(hashes, hash1.a, hash2.a) {
                (h1,h2) = h:(int,int);
            }
            var repMsg = "created " + st.attrib(name1) + "+created " + st.attrib(name2);
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return repMsg;
        }
/*
        when "int" {
            var sarrays = new owned SegSArray(segName, valName, st);
            var hashes = sarrays.hash();
            var name1 = st.nextName();
            var hash1 = st.addEntry(name1, hashes.size, int);
            var name2 = st.nextName();
            var hash2 = st.addEntry(name2, hashes.size, int);
            forall (h, h1, h2) in zip(hashes, hash1.a, hash2.a) {
                (h1,h2) = h:(int,int);
            }
            return "created " + st.attrib(name1) + "+created " + st.attrib(name2);
        }
*/
        otherwise {
            var errorMsg = notImplementedError(pn, objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return errorMsg;
        }
    }
  }


  /*
   * Assigns a segIntIndex, sliceIndex, or pdarrayIndex to the incoming payload
   * consisting of a sub-command, object type, offset SymTab key, array SymTab
   * key, and index value for the incoming payload.
   * 
   * Note: the sub-command indicates the index type which can be one of the following:
   * 1. intIndex : setIntIndex
   * 2. sliceIndex : segSliceIndex
   * 3. pdarrayIndex : segPdarrayIndex
  */ 
  proc segmentedIndexMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    // 'subcmd' is the type of indexing to perform
    // 'objtype' is the type of segmented array
    var (subcmd, objtype, rest) = payload.decode().splitMsgToTuple(3);
    var fields = rest.split();
    var args: [1..#fields.size] string = fields; // parsed by subroutines
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "subcmd: %s objtype: %s rest: %s".format(subcmd,objtype,rest));
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
                var errorMsg = "Error: in %s, nknown subcommand %s".format(pn, subcmd);
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return errorMsg;
            }
        }
    } catch e: OutOfBoundsError {
        var errorMsg = "Error: index out of bounds";
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return errorMsg;
    } catch e: Error {
        var errorMsg = "Error: unknown cause %t".format(e);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return errorMsg;
    }
  }
 
  /*
  Returns the object corresponding to the index
  */ 
  proc segIntIndex(objtype: string, args: [] string, 
                                         st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();

      // check to make sure symbols defined
      st.check(args[1]);
      st.check(args[2]);
      
      select objtype {
          when "str" {
              // Make a temporary strings array
              var strings = new owned SegString(args[1], args[2], st);
              // Parse the index
              var idx = args[3]:int;
              // TO DO: in the future, we will force the client to handle this
              idx = convertPythonIndexToChapel(idx, strings.size);
              var s = strings[idx];

              var repMsg = "item %s %jt".format("str", s);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
              return repMsg;
          }
          when "int" {
              // Make a temporary int array
              var arrays = new owned SegSArray(args[1], args[2], st);
              // Parse the index
              var idx = args[3]:int;
              // TO DO: in the future, we will force the client to handle this
              idx = convertPythonIndexToChapel(idx, arrays.size);
              var s = arrays[idx];
              return "item %s %jt".format("int", s);
          }
          otherwise { 
              var errorMsg = notImplementedError(pn, objtype); 
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return errorMsg;                          
          }
      }
  }

  /* Allow Python-style negative indices. */
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

    // check to make sure symbols defined
    st.check(args[1]);
    st.check(args[2]);

    select objtype {
      when "str" {
        // Make a temporary string array
        var strings = new owned SegString(args[1], args[2], st);
        // Parse the slice parameters
        var start = args[3]:int;
        var stop = args[4]:int;
        var stride = args[5]:int;
        // Only stride-1 slices are allowed for now
        if (stride != 1) { 
            var errorMsg = notImplementedError(pn, "stride != 1"); 
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return errorMsg;
        }
        // TO DO: in the future, we will force the client to handle this
        var slice: range(stridable=true) = convertPythonSliceToChapel(start, stop, stride);
        var newSegName = st.nextName();
        var newValName = st.nextName();
        // Compute the slice
        var (newSegs, newVals) = strings[slice];

        // Store the resulting offsets and bytes arrays
        var newSegsEntry = new shared SymEntry(newSegs);
        var newValsEntry = new shared SymEntry(newVals);
        st.addEntry(newSegName, newSegsEntry);
        st.addEntry(newValName, newValsEntry);
        return "created " + st.attrib(newSegName) + " +created " + st.attrib(newValName);
      }
      when "int" {
        // Make a temporary integer  array
        var sarrays = new owned SegSArray(args[1], args[2], st);
        // Parse the slice parameters
        var start = args[3]:int;
        var stop = args[4]:int;
        var stride = args[5]:int;
        // Only stride-1 slices are allowed for now
        if (stride != 1) { 
            var errorMsg = notImplementedError(pn, "stride != 1"); 
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return errorMsg;
        }
        // TO DO: in the future, we will force the client to handle this
        var slice: range(stridable=true) = convertPythonSliceToChapel(start, stop, stride);
        var newSegName = st.nextName();
        var newValName = st.nextName();
        // Compute the slice
        var (newSegs, newVals) = sarrays[slice];
        // Store the resulting offsets and bytes arrays
        var newSegsEntry = new shared SymEntry(newSegs);
        var newValsEntry = new shared SymEntry(newVals);
        st.addEntry(newSegName, newSegsEntry);
        st.addEntry(newValName, newValsEntry);
        
        var repMsg = "created " + st.attrib(newSegName) + " +created " + st.attrib(newValName);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
        return repMsg;
      }
      otherwise {
          var errorMsg = notImplementedError(pn, objtype);
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
          return errorMsg;          
        }
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

    // check to make sure symbols defined
    st.check(args[1]);
    st.check(args[2]);

    var newSegName = st.nextName();
    var newValName = st.nextName();
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "objtype:%s".format(objtype));
    
    select objtype {
        when "str" {
            var strings = new owned SegString(args[1], args[2], st);
            var iname = args[3];
            var gIV: borrowed GenSymEntry = st.lookup(iname);
            try {
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
                    otherwise {
                        var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                        smLogger.error(getModuleName(),getRoutineName(),
                                                      getLineNumber(),errorMsg); 
                        return notImplementedError(pn,errorMsg);
                    }
                }
            } catch e: Error {
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      e.message());
                return "Error: %t".format(e.message());
            }
        }
        when "int" {
            var sarrays = new owned SegSArray(args[1], args[2], st);
            var iname = args[3];
            var gIV: borrowed GenSymEntry = st.lookup(iname);
            try {
                select gIV.dtype {
                    when DType.Int64 {
                        var iv = toSymEntry(gIV, int);
                        var (newSegs, newVals) = sarrays[iv.a];
                        var newSegsEntry = new shared SymEntry(newSegs);
                        var newValsEntry = new shared SymEntry(newVals);
                        st.addEntry(newSegName, newSegsEntry);
                        st.addEntry(newValName, newValsEntry);
                    }
                    when DType.Bool {
                        var iv = toSymEntry(gIV, bool);
                        var (newSegs, newVals) = sarrays[iv.a];
                        var newSegsEntry = new shared SymEntry(newSegs);
                        var newValsEntry = new shared SymEntry(newVals);
                        st.addEntry(newSegName, newSegsEntry);
                        st.addEntry(newValName, newValsEntry);
                    }
                    otherwise {
                        var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                        smLogger.error(getModuleName(),getRoutineName(),
                                                      getLineNumber(),errorMsg); 
                        return notImplementedError(pn,errorMsg);
                    }
                }
            } catch e: Error {
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      e.message());
                return "Error: %t".format(e.message());
            }
        }
        otherwise {
            var errorMsg = "unsupported objtype: %t".format(objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return notImplementedError(pn, objtype);
        }
    }
    var repMsg = "created " + st.attrib(newSegName) + "+created " + st.attrib(newValName);

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return "created " + st.attrib(newSegName) + "+created " + st.attrib(newValName);
  }

  proc segBinopvvMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op,
         // Type and attrib names of left segmented array
         ltype, lsegName, lvalName,
         // Type and attrib names of right segmented array
         rtype, rsegName, rvalName, leftStr, jsonStr)
           = payload.decode().splitMsgToTuple(9);

    // check to make sure symbols defined
    st.check(lsegName);
    st.check(lvalName);
    st.check(rsegName);
    st.check(rvalName);

    select (ltype, rtype) {
    when ("str", "str") {
      var lstrings = new owned SegString(lsegName, lvalName, st);
      var rstrings = new owned SegString(rsegName, rvalName, st);
      select op {
        when "==" {
          var rname = st.nextName();
          var e = st.addEntry(rname, lstrings.size, bool);
          e.a = (lstrings == rstrings);
          repMsg = "created " + st.attrib(rname);
        }
        when "!=" {
          var rname = st.nextName();
          var e = st.addEntry(rname, lstrings.size, bool);
          e.a = (lstrings != rstrings);
          repMsg = "created " + st.attrib(rname);
        }
        when "stick" {
          var left = (leftStr.toLower() != "false");
          var json = jsonToPdArray(jsonStr, 1);
          const delim = json[json.domain.low];
          var oname = st.nextName();
          var vname = st.nextName();
          if left {
            var (newOffsets, newVals) = lstrings.stick(rstrings, delim, false);
            st.addEntry(oname, new shared SymEntry(newOffsets));
            st.addEntry(vname, new shared SymEntry(newVals));
          } else {
            var (newOffsets, newVals) = lstrings.stick(rstrings, delim, true);
            st.addEntry(oname, new shared SymEntry(newOffsets));
            st.addEntry(vname, new shared SymEntry(newVals));
          }
          repMsg = "created %s+created %s".format(st.attrib(oname), st.attrib(vname));
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        }
        otherwise {return notImplementedError(pn, ltype, op, rtype);}
        }
    }
    otherwise {return unrecognizedTypeError(pn, "("+ltype+", "+rtype+")");} 
    }
    return repMsg;
  }

  proc segBinopvvIntMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op,
         // Type and attrib names of left segmented array
         ltype, lsegName, lvalName,
         // Type and attrib names of right segmented array
         rtype, rsegName, rvalName, leftStr, jsonStr)
           = payload.decode().splitMsgToTuple(9);

    // check to make sure symbols defined
    st.check(lsegName);
    st.check(lvalName);
    st.check(rsegName);
    st.check(rvalName);

    select (ltype, rtype) {
    when ("int", "int") {
      var lsa = new owned SegSArray(lsegName, lvalName, st);
      var rsa = new owned SegString(rsegName, rvalName, st);
      select op {
        when "==" {
          var rname = st.nextName();
          var e = st.addEntry(rname, lsa.size, bool);
          e.a = (lsa == rsa);
          repMsg = "created " + st.attrib(rname);
        }
        when "!=" {
          var rname = st.nextName();
          var e = st.addEntry(rname, lsa.size, bool);
          e.a = (lsa != rsa);
          repMsg = "created " + st.attrib(rname);
        }
        otherwise {return notImplementedError(pn, ltype, op, rtype);}
        }
    }
    otherwise {return unrecognizedTypeError(pn, "("+ltype+", "+rtype+")");} 
    }
    return repMsg;
  }

  proc segBinopvsMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op, objtype, segName, valName, valtype, encodedVal)
          = payload.decode().splitMsgToTuple(6);

    // check to make sure symbols defined
    st.check(segName);
    st.check(valName);
    var json = jsonToPdArray(encodedVal, 1);
    var value = json[json.domain.low];
    var rname = st.nextName();
    select (objtype, valtype) {
    when ("str", "str") {
      var strings = new owned SegString(segName, valName, st);
      select op {
        when "==" {
          var e = st.addEntry(rname, strings.size, bool);
          e.a = (strings == value);
        }
        when "!=" {
          var e = st.addEntry(rname, strings.size, bool);
          e.a = (strings != value);
        }
        otherwise {return notImplementedError(pn, objtype, op, valtype);}
        }
    }
    otherwise {return unrecognizedTypeError(pn, "("+objtype+", "+valtype+")");} 
    }
    return "created " + st.attrib(rname);
  }

  proc segBinopvsIntMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op, objtype, segName, valName, valtype, encodedVal)
          = payload.decode().splitMsgToTuple(6);

    // check to make sure symbols defined
    st.check(segName);
    st.check(valName);
    var json = jsonToPdArrayInt(encodedVal, 1);
    var value = json[json.domain.low];
    var rname = st.nextName();
    select (objtype, valtype) {
    when ("int", "int") {
      var sarrays  = new owned SegSArray(segName, valName, st);
      select op {
        when "==" {
          var e = st.addEntry(rname, sarrays.size, bool);
          var tmp=sarrays[sarrays.offsets.aD.low]:int;
          e.a = (tmp == value);
//          e.a = (sarrays == value);
        }
        when "!=" {
          var e = st.addEntry(rname, sarrays.size, bool);
          var tmp=sarrays[sarrays.offsets.aD.low]:int;
          e.a = (tmp != value);
//          e.a = (sarrays != value);
        }
        otherwise {return notImplementedError(pn, objtype, op, valtype);}
        }
    }
    otherwise {return unrecognizedTypeError(pn, "("+objtype+", "+valtype+")");} 
    }
    return "created " + st.attrib(rname);
  }

  proc segIn1dMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (mainObjtype, mainSegName, mainValName, testObjtype, testSegName,
         testValName, invertStr) = payload.decode().splitMsgToTuple(7);

    // check to make sure symbols defined
    st.check(mainSegName);
    st.check(mainValName);
    st.check(testSegName);
    st.check(testValName);

    var invert: bool;
    if invertStr == "True" {invert = true;}
    else if invertStr == "False" {invert = false;}
    else {return "Error: Invalid argument in %s: %s (expected True or False)".format(pn, invertStr);}
    
    var rname = st.nextName();
    select (mainObjtype, testObjtype) {
    when ("str", "str") {
      var mainStr = new owned SegString(mainSegName, mainValName, st);
      var testStr = new owned SegString(testSegName, testValName, st);
      var e = st.addEntry(rname, mainStr.size, bool);
      if invert {
        e.a = !in1d(mainStr, testStr);
      } else {
        e.a = in1d(mainStr, testStr);
      }
    }
    otherwise {
        var errorMsg = unrecognizedTypeError(pn, "("+mainObjtype+", "+testObjtype+")");
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return errorMsg;            
      }
    }
    return "created " + st.attrib(rname);
  }

  proc segIn1dIntMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (mainObjtype, mainSegName, mainValName, testObjtype, testSegName,
         testValName, invertStr) = payload.decode().splitMsgToTuple(7);

    // check to make sure symbols defined
    st.check(mainSegName);
    st.check(mainValName);
    st.check(testSegName);
    st.check(testValName);

    var invert: bool;
    if invertStr == "True" {invert = true;}
    else if invertStr == "False" {invert = false;}
    else {return "Error: Invalid argument in %s: %s (expected True or False)".format(pn, invertStr);}
    
    var rname = st.nextName();
    select (mainObjtype, testObjtype) {
    when ("int", "int") {
      var mainSA = new owned SegSArray(mainSegName, mainValName, st);
      var testSA = new owned SegSArray(testSegName, testValName, st);
      var e = st.addEntry(rname, mainSA.size, bool);
      if invert {
        e.a = !in1d_Int(mainSA, testSA);
      } else {
        e.a = in1d_Int(mainSA, testSA);
      }
    }
    otherwise {
        var errorMsg = unrecognizedTypeError(pn, "("+mainObjtype+", "+testObjtype+")");
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return errorMsg;            
      }
    }
    return "created " + st.attrib(rname);
  }
  proc segGroupMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var (objtype, segName, valName) = payload.decode().splitMsgToTuple(3);

      // check to make sure symbols defined
      st.check(segName);
      st.check(valName);
      
      var rname = st.nextName();
      select (objtype) {
          when "str" {
              var strings = new owned SegString(segName, valName, st);
              var iv = st.addEntry(rname, strings.size, int);
              iv.a = strings.argGroup();
          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype+")");
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return errorMsg;            
          }
      }
      return "created " + st.attrib(rname);
  }



  proc segSuffixArrayMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var (objtype, segName, valName) = payload.decode().splitMsgToTuple(3);
      var repMsg: string;

      // check to make sure symbols defined
      st.check(segName);
      st.check(valName);

      var strings = new owned SegString(segName, valName, st);
      var size=strings.size;
      var nBytes = strings.nBytes;
      var length=strings.getLengths();
      var offsegs = (+ scan length) - length;
      var algorithmNum=2:int; //2:"divsufsort";1:SuffixArraySkew
      select (objtype) {
          when "str" {
              // To be checked, I am not sure if this formula can estimate the total memory requirement
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*size + 16*size + nBytes);

              //allocate an offset array
              var sasoff = offsegs;
              //allocate an values array
              var sasval:[0..(nBytes-1)] int;
              //              var lcpval:[0..(nBytes-1)] int; now we will not build the LCP array at the same time

              var i:int;
              var j:int;
              forall i in 0..(size-1) do {
              // the start position of ith string in value array

                var startposition:int;
                var endposition:int;
                startposition = offsegs[i];
                endposition = startposition+length[i]-1;
                // what we do in the select structure is filling the sasval array with correct index
                select (algorithmNum) {
                    when 1 {
                       var sasize=length[i]:int;
                       ref strArray=strings.values.a[startposition..endposition];
                       var tmparray:[0..sasize+2] int;
                       var intstrArray:[0..sasize+2] int;
                       var x:int;
                       var y:int;
                       forall (x,y) in zip ( intstrArray[0..sasize-1],
                                strings.values.a[startposition..endposition]) do x=y;
                       intstrArray[sasize]=0;
                       intstrArray[sasize+1]=0;
                       intstrArray[sasize+2]=0;
                       SuffixArraySkew(intstrArray,tmparray,sasize,256);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[0..sasize-1]) do
                               x = y;
                    }
                    when 2 {
                       var sasize=length[i]:int(32);
                       var localstrArray:[0..endposition-startposition] uint(8);
                       var a:int(8);
                       var b:int(8);
                       ref strArray=strings.values.a[startposition..endposition];
                       localstrArray=strArray;
                       //for all (a,b) in zip (localstrArray[0..sasize-1],strArray) do a=b;
                       var tmparray:[1..sasize] int(32);
                       divsufsort(localstrArray,tmparray,sasize);
                       //divsufsort(strArray,tmparray,sasize);
                       var x:int;
                       var y:int(32);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[1..sasize]) do
                            x = y;
                    }
                }

/*
// Here we calculate the lcp(Longest Common Prefix) array value
                forall j in startposition+1..endposition do{
                        var tmpcount=0:int;
                        var tmpbefore=sasval[j-1]:int;
                        var tmpcur=sasval[j]:int;
                        var tmplen=min(sasize-tmpcur, sasize-tmpbefore);
                        var tmpi:int;
                        for tmpi in 0..tmplen-1 do {
                            if (intstrArray[tmpbefore]!=intstrArray[tmpcur]) {
                                 break;
                            }                        
                            tmpcount+=1;
                        } 
                        lcpval[j]=tmpcount;
                }
*/
              }
              var segName2 = st.nextName();
              var valName2 = st.nextName();
              //              var lcpvalName = st.nextName();

              var segEntry = new shared SymEntry(sasoff);
              var valEntry = new shared SymEntry(sasval);
              //              var lcpvalEntry = new shared SymEntry(lcpval);
              /*
              valEntry.enhancedInfo=lcpvalName;
              lcpvalEntry.enhancedInfo=valName2;
              we have removed enchancedInfo.
              */
              st.addEntry(segName2, segEntry);
              st.addEntry(valName2, valEntry);
//              st.addEntry(lcpvalName, lcpvalEntry);
              repMsg = 'created ' + st.attrib(segName2) + '+created ' + st.attrib(valName2);
              return repMsg;


          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype+")");
              writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError")); 
              return errorMsg;            
          }
      }

  }

  proc segLCPMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var (objtype, segName1, valName1,segName2,valName2) = payload.decode().splitMsgToTuple(5);
      var repMsg: string;

      // check to make sure symbols defined
      st.check(segName1);
      st.check(valName1);
      st.check(segName2);
      st.check(valName2);

      var suffixarrays = new owned SegSArray(segName1, valName1, st);
      var size=suffixarrays.size;
      var nBytes = suffixarrays.nBytes;
      var length=suffixarrays.getLengths();
      var offsegs = (+ scan length) - length;


      var strings = new owned SegString(segName2, valName2, st);

      select (objtype) {
          when "int" {
              // To be checked, I am not sure if this formula can estimate the total memory requirement
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*size + 16*size + nBytes);

              //allocate an offset array
              var sasoff = offsegs;
              //allocate an values array
              var lcpval:[0..(nBytes-1)] int;

              var i:int;
              var j:int;
              forall i in 0..(size-1) do {
              // the start position of ith surrix array  in value array
                var startposition:int;
                var endposition:int;
                startposition = offsegs[i];
                endposition = startposition+length[i]-1;




                var sasize=length[i]:int;
                ref sufArray=suffixarrays.values.a[startposition..endposition];
                ref strArray=strings.values.a[startposition..endposition];
// Here we calculate the lcp(Longest Common Prefix) array value
                forall j in startposition+1..endposition do{
                        var tmpcount=0:int;
                        var tmpbefore=sufArray[j-1]:int;
                        var tmpcur=sufArray[j]:int;
                        var tmplen=min(sasize-tmpcur, sasize-tmpbefore);
                        var tmpi:int;
                        for tmpi in 0..tmplen-1 do {
                            if (strArray[tmpbefore]!=strArray[tmpcur]) {
                                 break;
                            }                        
                            tmpbefore+=1;
                            tmpcur+=1;
                            tmpcount+=1;
                        } 
                        lcpval[j]=tmpcount;
                }
              }
              var lcpsegName = st.nextName();
              var lcpvalName = st.nextName();

              var lcpsegEntry = new shared SymEntry(sasoff);
              var lcpvalEntry = new shared SymEntry(lcpval);
              /*
              valEntry.enhancedInfo=lcpvalName;
              lcpvalEntry.enhancedInfo=valName2;
              we have removed enchancedInfo.
              */
              st.addEntry(lcpsegName, lcpsegEntry);
              st.addEntry(lcpvalName, lcpvalEntry);
              repMsg = 'created ' + st.attrib(lcpsegName) + '+created ' + st.attrib(lcpvalName);
              return repMsg;


          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype+")");
              writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError")); 
              return errorMsg;            
          }
      }

  }

// directly read a string from given file and generate its suffix array
  proc segSAFileMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
//      var (FileName) = payload.decode().splitMsgToTuple(1);
      var FileName = payload.decode();
      var repMsg: string;

//      var filesize:int(32);
      var filesize:int;
      var f = open(FileName, iomode.r);
      var size=1:int;
      var nBytes = f.size;
      var length:[0..0] int  =nBytes;
      var offsegs:[0..0] int =0 ;

      var sasize=nBytes:int;
      var startposition:int;
      var endposition:int;
      startposition = 0;
      endposition = nBytes-1;
      var strArray:[startposition..endposition]uint(8);
      var r = f.reader(kind=ionative);
      r.read(strArray);
      r.close();

      var segName = st.nextName();
      var valName = st.nextName();

      var segEntry = new shared SymEntry(offsegs);
      var valEntry = new shared SymEntry(strArray);
      st.addEntry(segName, segEntry);
      st.addEntry(valName, valEntry);

      var algorithmNum=2:int; //2:"divsufsort";1:SuffixArraySkew

      select ("str") {
          when "str" {
              // To be checked, I am not sure if this formula can estimate the total memory requirement
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*size + 16*size + nBytes);

              //allocate an offset array
              var sasoff = offsegs;
              //allocate a suffix array  values array and lcp array
              var sasval:[0..(nBytes-1)] int;
//              var lcpval:[0..(nBytes-1)] int;

              var i:int;
              forall i in 0..(size-1) do {
              // the start position of ith string in value array
                select (algorithmNum) {
                    when 1 {
                       var sasize=length[i]:int;
                       var tmparray:[0..sasize+2] int;
                       var intstrArray:[0..sasize+2] int;
                       var x:int;
                       var y:int;
                       forall (x,y) in zip ( intstrArray[0..sasize-1],strArray[startposition..endposition]) do x=y;
                       intstrArray[sasize]=0;
                       intstrArray[sasize+1]=0;
                       intstrArray[sasize+2]=0;
                       SuffixArraySkew(intstrArray,tmparray,sasize,256);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[0..sasize-1]) do
                               x = y;
                    }
                    when 2 {
                       var sasize=length[i]:int(32);
                       //ref strArray=strings.values.a[startposition..endposition];
                       var tmparray:[1..sasize] int(32);
                       divsufsort(strArray,tmparray,sasize);
                       var x:int;
                       var y:int(32);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[1..sasize]) do
                            x = y;
                    }
                }// end of select 
              } // end of forall
              var segName2 = st.nextName();
              var valName2 = st.nextName();
//              var lcpvalName = st.nextName();

              var segEntry = new shared SymEntry(sasoff);
              var valEntry = new shared SymEntry(sasval);
//              var lcpvalEntry = new shared SymEntry(lcpval);
              /*
              valEntry.enhancedInfo=lcpvalName;
              lcpvalEntry.enhancedInfo=valName2;
              We have removed enhancedInfo.
              */
              st.addEntry(segName2, segEntry);
              st.addEntry(valName2, valEntry);
//              st.addEntry(lcpvalName, lcpvalEntry);
              repMsg = 'created ' + st.attrib(segName2) + '+created ' + st.attrib(valName2) 
                        + '+created ' + st.attrib(segName) + '+created ' + st.attrib(valName);
              return repMsg;

          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+FileName+")");
              writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError")); 
              return errorMsg;            
          }
      }

  }


// directly read a graph from given file and build the SegGraph class in memory
  proc segGraphFileMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      //var pn = Reflection.getRoutineName();
      var (NeS,NvS,ColS,DirectedS, FileName) = payload.decode().splitMsgToTuple(5);
      //writeln("======================Graph Reading=====================");
      //writeln(NeS,NvS,ColS,DirectedS, FileName);
      var Ne=NeS:int;
      var Nv=NvS:int;
      var NumCol=ColS:int;
      var directed=DirectedS:int;
      var weighted=0:int;
      var timer: Timer;
      if NumCol>2 {
           weighted=1;
      }

      timer.start();
      var src=makeDistArray(Ne,int);
      var dst=makeDistArray(Ne,int);
      //var length=makeDistArray(Nv,int);
      var neighbour=makeDistArray(Nv,int);
      var start_i=makeDistArray(Nv,int);

      var e_weight = makeDistArray(Ne,int);
      var v_weight = makeDistArray(Nv,int);

      var iv=makeDistArray(Ne,int);

      var srcR=makeDistArray(Ne,int);
      var dstR=makeDistArray(Ne,int);
      //var lengthR=makeDistArray(Nv,int);
      var neighbourR=makeDistArray(Nv,int);
      var start_iR=makeDistArray(Nv,int);
      ref  ivR=iv;

      var linenum=0:int;

      var repMsg: string;

      var filesize:int;
      var f = open(FileName, iomode.r);
      var r = f.reader(kind=ionative);
      var dataarray:[0..Ne-1,0..NumCol-1] int;
      r.read(dataarray);
      r.close();
      f.close();
      var startpos, endpos:int;
      var sort:int;
      coforall loc in Locales  {
           on loc {
              var srclocal=src.localSubdomain();
              var dstlocal=dst.localSubdomain();
              var ewlocal=e_weight.localSubdomain();
              /*
              var f = open(FileName, iomode.r);
              var r = f.reader(kind=ionative);
              var line:string;
              var a,b,c:string;
              var curline=0:int;
              while r.readline(line) {
                  if NumCol==2 {
                      (a,b)=  line.splitMsgToTuple(2);
                  } else {
                      (a,b,c)=  line.splitMsgToTuple(3);
                      if ewlocal.contains(curline) {
                          e_weight[curline]=c:int;
                      }
                      //e_weight[curline]=c:int;
                  }
                  if srclocal.contains(curline) {
                       src[curline]=a:int;
                  }
                  if dstlocal.contains(curline) {
                       dst[curline]=b:int;
                  }
                  curline+=1;
              } 

              r.close();
              */
              forall i in srclocal {
                   src[i]=dataarray[i,0];
              }
              forall i in dstlocal {
                   dst[i]=dataarray[i,1];
              }
              if NumCol==3 {
                  forall i in ewlocal {
                      e_weight[i]=dataarray[i,2];
                  }
              }
              forall i in srclocal {
                   src[i]=src[i]+(src[i]==dst[i]);
                   src[i]=src[i]%Nv;
                   dst[i]=dst[i]%Nv;
              }
           }
      }
      timer.stop();
      //writeln("$$$$$$$$$$$$ Reading File takes ", timer.elapsed()," $$$$$$$$$$$$$$$$$$$$$$$");
      timer.start();
      proc twostep_sort() {
        iv = radixSortLSD_ranks(src);
        // permute into sorted order
        var tmpedges=src;
        coforall loc in Locales  {
           on loc {
              forall i in tmpedges.localSubdomain(){
                   tmpedges[i] = src[iv[i]]; //# permute first vertex into sorted order
              }
              forall i in src.localSubdomain(){
                   src[i] = tmpedges[i]; //# permute first vertex into sorted order
              }
              forall i in tmpedges.localSubdomain(){
                   tmpedges[i] = dst[iv[i]]; //# permute first vertex into sorted order
              }
              forall i in dst.localSubdomain(){
                   dst[i] = tmpedges[i]; //# permute first vertex into sorted order
              }
           }
        }
        //tmpedges = src[iv]; //# permute first vertex into sorted order
        //src=tmpedges;
        //tmpedges = dst[iv]; //# permute second vertex into sorted order
        //dst=tmpedges;
        startpos=0;
        sort=0;
        while (startpos < Ne-2) {
           endpos=startpos+1;
           sort=0;
           while (endpos <=Ne-1) {
              if (src[startpos]==src[endpos])  {
                 sort=1;
                 endpos+=1;
                 continue;
              } else {
                 break;
              }
           }//end of while endpos
           if (sort==1) {
              var tmpary:[0..endpos-startpos-1] int;
              tmpary=dst[startpos..endpos-1];
              var ivx=radixSortLSD_ranks(tmpary);
              dst[startpos..endpos-1]=tmpary[ivx];
              sort=0;
           }
           startpos+=1;
        }//end of while startpos
      }// end of twostep_sort()

      proc combine_sort() throws {
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;

             for (bitWidth, ary, neg) in zip(bitWidths, [src,dst], negs) {
                       (bitWidth, neg) = getBitWidth(ary); 
                       totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;
             }
             proc mergedArgsort(param numDigits) throws {
                    //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
                    //             + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
                    var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
                    var curDigit = RSLSD_tupleLow + numDigits - totalDigits;
                    for (ary , nBits, neg) in zip([src,dst], bitWidths, negs) {
                        proc mergeArray(type t) {
                            ref A = ary;
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
                        mergeArray(int); 
                    }
                    var tmpiv = argsortDefault(merged);
                    return tmpiv;
             }

             if totalDigits <=  4 { 
                  iv = mergedArgsort( 4); 
             }

             if totalDigits <=  8 { 
                  iv =  mergedArgsort( 8); 
             }
             if totalDigits <= 16 { 
                  iv = mergedArgsort(16); 
             }
             var tmpedges=src[iv];
             src=tmpedges;
             tmpedges=dst[iv];
             dst=tmpedges;

      }//end combine_sort

      proc set_neighbour(){ 
          for i in 0..Ne-1 do {
             neighbour[src[i]]+=1;
             if (start_i[src[i]] ==-1){
                 start_i[src[i]]=i;
             }

          }
      }

      //twostep_sort();
      combine_sort();
      set_neighbour();

      if (directed==0) { //undirected graph
          proc twostep_sortR(){
             var ivR = radixSortLSD_ranks(srcR);
             var tmpedges=src;
             tmpedges = srcR[ivR]; //# permute first vertex into sorted order
             srcR=tmpedges;
             tmpedges = dstR[ivR]; //# permute second vertex into sorted order
             dstR=tmpedges;
             startpos=0;
             sort=0;
             while (startpos < Ne-2) {
                 endpos=startpos+1;
                 sort=0;
                 while (endpos <=Ne-1) {
                     if (srcR[startpos]==srcR[endpos])  {
                        sort=1;
                        endpos+=1;
                        continue;
                      } else {
                          break;
                      }
                 }//end of while endpos
                 if (sort==1) {
                     var tmparyR:[0..endpos-startpos-1] int;
                     tmparyR=dstR[startpos..endpos-1];
                     var ivxR=radixSortLSD_ranks(tmparyR);
                     dstR[startpos..endpos-1]=tmparyR[ivxR];
                     sort=0;
                 }
                 startpos+=1;
             }//end of while startpos
          }// end of two step R


          proc combine_sortR() throws {
             /* we cannot use the coargsort version because it will break the memory limit */
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;
             for (bitWidth, ary, neg) in zip(bitWidths, [srcR,dstR], negs) {
                 (bitWidth, neg) = getBitWidth(ary); 
                 totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;

             }
             proc mergedArgsort(param numDigits) throws {
               //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
               //          + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
               var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
               var curDigit = RSLSD_tupleLow + numDigits - totalDigits;
               for (ary , nBits, neg) in zip([src,dst], bitWidths, negs) {
                  proc mergeArray(type t) {
                     ref A = ary;
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
                  mergeArray(int); 
               }
               var tmpiv = argsortDefault(merged);
               return tmpiv;
             } 

             if totalDigits <=  4 { 
               ivR = mergedArgsort( 4); 
             }

             if totalDigits <=  8 { 
               ivR =  mergedArgsort( 8); 
             }
             if totalDigits <= 16 { 
               ivR = mergedArgsort(16); 
             }

             var tmpedges = srcR[ivR]; 
             srcR=tmpedges;
             tmpedges = dstR[ivR]; 
             dstR=tmpedges;

          }// end combine_sortR


          proc set_neighbourR(){
             for i in 0..Ne-1 do {
                neighbourR[srcR[i]]+=1;
                if (start_iR[srcR[i]] ==-1){
                    start_iR[srcR[i]]=i;
                }
             }
          }
          //twostep_sortR();
          combine_sortR();
          set_neighbourR();

      }//end of undirected


      var ewName ,vwName:string;
      if (weighted!=0) {
        fillInt(v_weight,1,1000);
        //fillRandom(v_weight,0,100);
        ewName = st.nextName();
        vwName = st.nextName();
        var vwEntry = new shared SymEntry(v_weight);
        var ewEntry = new shared SymEntry(e_weight);
        st.addEntry(vwName, vwEntry);
        st.addEntry(ewName, ewEntry);
      }
      var srcName = st.nextName();
      var dstName = st.nextName();
      var startName = st.nextName();
      var neiName = st.nextName();
      var srcEntry = new shared SymEntry(src);
      var dstEntry = new shared SymEntry(dst);
      var startEntry = new shared SymEntry(start_i);
      var neiEntry = new shared SymEntry(neighbour);
      st.addEntry(srcName, srcEntry);
      st.addEntry(dstName, dstEntry);
      st.addEntry(startName, startEntry);
      st.addEntry(neiName, neiEntry);
      var sNv=Nv:string;
      var sNe=Ne:string;
      var sDirected=directed:string;
      var sWeighted=weighted:string;

      var srcNameR, dstNameR, startNameR, neiNameR:string;
      if (directed!=0) {//for directed graph
          if (weighted!=0) {// for weighted graph
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) +
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) +
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);
          } else {// for unweighted graph
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) +
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) ;

          }
      } else {//for undirected graph

          srcNameR = st.nextName();
          dstNameR = st.nextName();
          startNameR = st.nextName();
          neiNameR = st.nextName();
          var srcEntryR = new shared SymEntry(srcR);
          var dstEntryR = new shared SymEntry(dstR);
          var startEntryR = new shared SymEntry(start_iR);
          var neiEntryR = new shared SymEntry(neighbourR);
          st.addEntry(srcNameR, srcEntryR);
          st.addEntry(dstNameR, dstEntryR);
          st.addEntry(startNameR, startEntryR);
          st.addEntry(neiNameR, neiEntryR);
          if (weighted!=0) {// for weighted graph
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) +
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) +
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) +
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) +
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);
          } else {// for unweighted graph
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) +
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) +
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) +
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) ;
          }

      }
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      timer.stop();
      //writeln("$$$$$$$$$$$$Sorting Edges takes ", timer.elapsed()," $$$$$$$$$$$$$$$$$$$$$$$");
      return repMsg;
  }



  proc segrmatgenMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      //var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (slgNv, sNe_per_v, sp, sdire,swei,rest )
          = payload.decode().splitMsgToTuple(6);
      //writeln(slgNv, sNe_per_v, sp, sdire,swei,rest);
      var lgNv = slgNv: int;
      var Ne_per_v = sNe_per_v: int;
      var p = sp: real;
      var directed=sdire : int;
      var weighted=swei : int;

      var Nv = 2**lgNv:int;
      // number of edges
      var Ne = Ne_per_v * Nv:int;
      // probabilities

      var timer:Timer;
      timer.start();
      var n_vertices=Nv;
      var n_edges=Ne;
      var src=makeDistArray(Ne,int);
      var dst=makeDistArray(Ne,int);
      //var length=makeDistArray(Nv,int);
      var neighbour=makeDistArray(Nv,int);
      var start_i=makeDistArray(Nv,int);

      var iv=makeDistArray(Ne,int);

      //var e_weight=makeDistArray(Nv,int);
      //var v_weight=makeDistArray(Nv,int);


      //length=0;
      coforall loc in Locales  {
          on loc {
              forall i in src.localSubdomain() {
                  src[i]=1;
              }
              forall i in dst.localSubdomain() {
                  dst[i]=0;
              }
              forall i in start_i.localSubdomain() {
                  start_i[i]=-1;
              }
              forall i in neighbour.localSubdomain() {
                  neighbour[i]=0;
              }
          }
      }
      //start_i=-1;
      //neighbour=0;
      //src=1;
      //dst=1;
      var srcName:string ;
      var dstName:string ;
      var startName:string ;
      var neiName:string ;
      var sNv:string;
      var sNe:string;
      var sDirected:string;
      var sWeighted:string;


      proc rmat_gen() {
             var a = p;
             var b = (1.0 - a)/ 3.0:real;
             var c = b;
             var d = b;
             var ab=a+b;
             var c_norm = c / (c + d):real;
             var a_norm = a / (a + b):real;
             // generate edges
             //var src_bit=: [0..Ne-1]int;
             //var dst_bit: [0..Ne-1]int;
             var src_bit=src;
             var dst_bit=dst;

             for ib in 1..lgNv {
                 //var tmpvar: [0..Ne-1] real;
                 var tmpvar=src;
                 fillRandom(tmpvar);
                 coforall loc in Locales  {
                       on loc {
                           forall i in src_bit.localSubdomain() {
                                 src_bit[i]=tmpvar[i]>ab;
                           }       
                       }
                 }
                 //src_bit=tmpvar>ab;
                 fillRandom(tmpvar);
                 coforall loc in Locales  {
                       on loc {
                           forall i in dst_bit.localSubdomain() {
                                 dst_bit[i]=tmpvar[i]>(c_norm * src_bit[i] + a_norm * (~ src_bit[i]));
                           }       
                       }
                 }
                 //dst_bit=tmpvar>(c_norm * src_bit + a_norm * (~ src_bit));
                 coforall loc in Locales  {
                       on loc {
                           forall i in dst.localSubdomain() {
                                 dst[i]=dst[i]+ ((2**(ib-1)) * dst_bit[i]);
                           }       
                           forall i in src.localSubdomain() {
                                 src[i]=src[i]+ ((2**(ib-1)) * src_bit[i]);
                           }       
                       }
                 }
                 //src = src + ((2**(ib-1)) * src_bit);
                 //dst = dst + ((2**(ib-1)) * dst_bit);
             }
             coforall loc in Locales  {
                       on loc {
                           forall i in src_bit.localSubdomain() {
                                 src[i]=src[i]+(src[i]==dst[i]);
                                 src[i]=src[i]%Nv;
                                 dst[i]=dst[i]%Nv;
                           }       
                       }
             }
             //src=src%Nv;
             //dst=dst%Nv;

             //remove self loop
             //src=src+(src==dst);
             //src=src%Nv;
      }//end rmat_gen
      
      proc combine_sort() throws {
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;

             for (bitWidth, ary, neg) in zip(bitWidths, [src,dst], negs) {
                       (bitWidth, neg) = getBitWidth(ary); 
                       totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;
             }
             proc mergedArgsort(param numDigits) throws {
                    //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
                    //             + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
                    var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
                    var curDigit = RSLSD_tupleLow + numDigits - totalDigits;
                    for (ary , nBits, neg) in zip([src,dst], bitWidths, negs) {
                        proc mergeArray(type t) {
                            ref A = ary;
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
                        mergeArray(int); 
                    }
                    var tmpiv = argsortDefault(merged);
                    return tmpiv;
             }

             if totalDigits <=  4 { 
                  iv = mergedArgsort( 4); 
             }

             if totalDigits <=  8 { 
                  iv =  mergedArgsort( 8); 
             }
             if totalDigits <= 16 { 
                  iv = mergedArgsort(16); 
             }
             var tmpedges=src[iv];
             src=tmpedges;
             tmpedges=dst[iv];
             dst=tmpedges;

      }//end combine_sort

      proc twostep_sort(){
             iv = radixSortLSD_ranks(src);
             // permute into sorted order
             var tmpedges=src;
             tmpedges=src[iv];
             src=tmpedges;
             tmpedges=dst[iv];
             dst=tmpedges;
             //# to premute/rename vertices
             var startpos=0, endpos:int;
             var sort=0:int;
             while (startpos < Ne-2) {
                     endpos=startpos+1;
                     sort=0;
                     //writeln("startpos=",startpos,"endpos=",endpos);
                     while (endpos <=Ne-1) {
                         if (src[startpos]==src[endpos])  {
                              sort=1;
                              endpos+=1;
                              continue;
                         } else {
                              break;
                         }
                     }//end of while endpos
                     if (sort==1) {
                         var tmpary:[0..endpos-startpos-1] int;
                         tmpary=dst[startpos..endpos-1];
                         var ivx=radixSortLSD_ranks(tmpary);
                         dst[startpos..endpos-1]=tmpary[ivx];
                         //writeln("src1=",src1,"dst1=",dst1,"ivx=",ivx);
                         sort=0;
                     } 
                     startpos+=1;
             }//end of while startpos
      }// end twostep_sort

      proc set_neighbour(){
             for i in 0..Ne-1 do {
                 neighbour[src[i]]+=1;
                 if (start_i[src[i]] ==-1){
                      start_i[src[i]]=i;
                      //writeln("assign index ",i, " to vertex ",src[i]);
                 }
 
             }
             //neighbour  = length;
      }
      proc set_common_symtable(): string throws {
             srcName = st.nextName();
             dstName = st.nextName();
             startName = st.nextName();
             neiName = st.nextName();
             var srcEntry = new shared SymEntry(src);
             var dstEntry = new shared SymEntry(dst);
             var startEntry = new shared SymEntry(start_i);
             var neiEntry = new shared SymEntry(neighbour);
             st.addEntry(srcName, srcEntry);
             st.addEntry(dstName, dstEntry);
             st.addEntry(startName, startEntry);
             st.addEntry(neiName, neiEntry);
             sNv=Nv:string;
             sNe=Ne:string;
             sDirected=directed:string;
             sWeighted=weighted:string;
             return "success";
      }


      if (directed!=0) {// for directed graph
          if (weighted!=0) { // for weighted graph
             //var e_weight: [0..Ne-1] int;
             //var v_weight: [0..Nv-1] int;
             var e_weight = makeDistArray(Ne,int);
             var v_weight = makeDistArray(Nv,int);
             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$$$$$$ RMAT generate the graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             timer.start();
             //twostep_sort();
             combine_sort();
             set_neighbour();

             var ewName ,vwName:string;
             fillInt(e_weight,1,1000);
             //fillRandom(e_weight,0,100);
             fillInt(v_weight,1,1000);
             //fillRandom(v_weight,0,100);
             ewName = st.nextName();
             vwName = st.nextName();
             var vwEntry = new shared SymEntry(v_weight);
             var ewEntry = new shared SymEntry(e_weight);
             try! st.addEntry(vwName, vwEntry);
             try! st.addEntry(ewName, ewEntry);
      
             set_common_symtable();
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);

          } else {
             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$$$$$$ RMAT generate the graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             timer.start();
             //twostep_sort();
             combine_sort();
             set_neighbour();
             set_common_symtable();
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) ; 
          }
      }// end for directed graph
      else {
          // only for undirected graph, we only declare R variables here
          var srcR=makeDistArray(Ne,int);
          var dstR=makeDistArray(Ne,int);
          //var lengthR=makeDistArray(Nv,int);
          var neighbourR=makeDistArray(Nv,int);
          var start_iR=makeDistArray(Nv,int);
          ref  ivR=iv;

          coforall loc in Locales  {
                       on loc {
                           forall i in start_iR.localSubdomain() {
                                 start_iR[i]=-1;
                           }       
                           forall i in neighbourR.localSubdomain() {
                                 neighbourR=0;
                           }       
                       }
          }
          //start_iR=-1;
          //lengthR=0;
          //neighbourR=0;
          var srcNameR, dstNameR, startNameR, neiNameR:string;
        
          proc combine_sortR() throws {
             /* we cannot use the coargsort version because it will break the memory limit */
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;
             for (bitWidth, ary, neg) in zip(bitWidths, [srcR,dstR], negs) {
                 (bitWidth, neg) = getBitWidth(ary); 
                 totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;

             }
             proc mergedArgsort(param numDigits) throws {
               //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
               //          + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
               var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
               var curDigit = RSLSD_tupleLow + numDigits - totalDigits;
               for (ary , nBits, neg) in zip([src,dst], bitWidths, negs) {
                  proc mergeArray(type t) {
                     ref A = ary;
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
                  mergeArray(int); 
               }
               var tmpiv = argsortDefault(merged);
               return tmpiv;
             } 

             if totalDigits <=  4 { 
               ivR = mergedArgsort( 4); 
             }

             if totalDigits <=  8 { 
               ivR =  mergedArgsort( 8); 
             }
             if totalDigits <= 16 { 
               ivR = mergedArgsort(16); 
             }

             var tmpedges = srcR[ivR]; 
             srcR=tmpedges;
             tmpedges = dstR[ivR]; 
             dstR=tmpedges;

          }// end combine_sortR

          proc   twostep_sortR() {
             ivR = radixSortLSD_ranks(srcR);
             var tmpedges = srcR[ivR]; //# permute first vertex into sorted order
             srcR=tmpedges;
             tmpedges = dstR[ivR]; //# permute second vertex into sorted order
             dstR=tmpedges;
             var startpos=0:int;
             var endpos:int;
             var sort=0;
             while (startpos < Ne-2) {
                endpos=startpos+1;
                sort=0;
                while (endpos <=Ne-1) {
                   if (srcR[startpos]==srcR[endpos])  {
                      sort=1;
                      endpos+=1;
                      continue;
                   } else {
                      break;
                   } 
                }//end of while endpos
                if (sort==1) {
                    var tmparyR:[0..endpos-startpos-1] int;
                    tmparyR=dstR[startpos..endpos-1];
                    var ivxR=radixSortLSD_ranks(tmparyR);
                    dstR[startpos..endpos-1]=tmparyR[ivxR];
                    sort=0;
                } 
                startpos+=1;
             } //end of while startpos
          }// end twostep_sort

          proc    set_neighbourR(){
             for i in 0..Ne-1 do {
                neighbourR[srcR[i]]+=1;
                if (start_iR[srcR[i]] ==-1){
                    start_iR[srcR[i]]=i;
                }
             }
             //neighbourR  = lengthR;
          }

          proc   set_common_symtableR():string throws {
          //proc   set_common_symtableR() {
             srcNameR = st.nextName();
             dstNameR = st.nextName();
             startNameR = st.nextName();
             neiNameR = st.nextName();
             var srcEntryR = new shared SymEntry(srcR);
             var dstEntryR = new shared SymEntry(dstR);
             var startEntryR = new shared SymEntry(start_iR);
             var neiEntryR = new shared SymEntry(neighbourR);
             st.addEntry(srcNameR, srcEntryR);
             st.addEntry(dstNameR, dstEntryR);
             st.addEntry(startNameR, startEntryR);
             st.addEntry(neiNameR, neiEntryR);
             return "success";
          }


          if (weighted!=0) {
             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$$$$$$ RMAT generate the graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             timer.start();
             //twostep_sort();
             combine_sort();
             set_neighbour();
             coforall loc in Locales  {
                       on loc {
                           forall i in srcR.localSubdomain() {
                                 srcR[i]=dst[i];
                                 dstR[i]=src[i];
                           }       
                       }
             }
             //srcR = dst;
             //dstR = src;
             //twostep_sortR(); 
             combine_sortR();
             set_neighbourR();

             //only for weighted  graph
             var ewName ,vwName:string;
             var e_weight = makeDistArray(Ne,int);
             var v_weight = makeDistArray(Nv,int);
             //var e_weight: [0..Ne-1] int;
             //var v_weight: [0..Nv-1] int;

             fillInt(e_weight,1,1000);
             //fillRandom(e_weight,0,100);
             fillInt(v_weight,1,1000);
             //fillRandom(v_weight,0,100);
             ewName = st.nextName();
             vwName = st.nextName();
             var vwEntry = new shared SymEntry(v_weight);
             var ewEntry = new shared SymEntry(e_weight);
             st.addEntry(vwName, vwEntry);
             st.addEntry(ewName, ewEntry);
             // end of weighted!=0
      
             set_common_symtable();
             set_common_symtableR();
 
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) + 
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) + 
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);


          } else {

             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$$$$$$ RMAT generate the graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             timer.start();
             //twostep_sort();
             combine_sort();
             set_neighbour();
             coforall loc in Locales  {
                       on loc {
                           forall i in srcR.localSubdomain() {
                                 srcR[i]=dst[i];
                                 dstR[i]=src[i];
                           }       
                       }
             }
             //srcR = dst;
             //dstR = src;
             //twostep_sortR(); 
             combine_sortR();
             set_neighbourR();
             set_common_symtable();
             set_common_symtableR();
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) + 
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) ; 


          }// end unweighted graph
      }// end undirected graph
      timer.stop();
      //writeln("$$$$$$$$$$$$$$$$$ sorting RMAT graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return repMsg;
  }


  proc segBFSMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      //var pn = Reflection.getRoutineName();
      var repMsg: string;
      //var (n_verticesN,n_edgesN,directedN,weightedN,srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN )
      //    = payload.decode().splitMsgToTuple(10);
      var (n_verticesN,n_edgesN,directedN,weightedN,restpart )
          = payload.decode().splitMsgToTuple(5);
      var Nv=n_verticesN:int;
      var Ne=n_edgesN:int;
      var Directed=directedN:int;
      var Weighted=weightedN:int;
      var depthName:string;
      var timer:Timer;
      timer.start();
      var depth=makeDistArray(Nv,int);
      coforall loc in Locales  {
                  on loc {
                           forall i in depth.localSubdomain() {
                                 depth[i]=-1;
                           }       
                  }
      }
      //var depth=-1: [0..Nv-1] int;
      var root:int;
      var srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN :string;
      var srcRN, dstRN, startRN, neighbourRN:string;

      proc bfs_kernel(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int):string throws{
          var cur_level=0;
          var SetCurF=  new DistBag(int,Locales);
          var SetNextF=  new DistBag(int,Locales);
          //var SetCurF= try! new set(int,parSafe = true);
          //var SetNextF=try!  new set(int,parSafe = true);
          SetCurF.add(root);
          //try! SetCurF.add(root);
          var numCurF=1:int;

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF.these());
                coforall loc in Locales  with (ref SetNextF) {
                   on loc {
                       ref nf=nei;
                       ref sf=start_i;
                       ref df=dst;
                       ref srcf=src;

                       var ld=srcf.localSubdomain();
                       //writeln("the local subdomain is");
                       //writeln(ld);
                       var myele = new set(int,parSafe = true);
                       for i in SetCurF{
                           if ld.contains(i) {
                               myele.add(i);
                           }
                       }
                       //writeln("current locale=",loc, " has elements =", myele);
                       forall i in myele with (ref SetNextF) {
                              var numNF=-1 :int;
                              numNF=nf[i];
                              ref NF=df[sf[i]..sf[i]+numNF-1];
                              if (numNF>0) {
                                   // may be forall j in NF is better?
                                   for j in NF {
                                        if (depth[j]==-1) {
                                           depth[j]=cur_level+1;
                                           SetNextF.add(j);
                                        }
                                   }
                              }
                       }
                       /*
                       var dld=depth.localSubdomain();
                       myele.clear();
                       for i in SetNextF{
                           if dld.contains(i) {
                               myele.add(i);
                           }
                       }
                       forall i in myele {
                           depth[i]=cur_level+1;
                       }
                       */
                   }//end on loc
                }//end forall loc
                cur_level+=1;
               //writeln("SetCurF= ", SetCurF, "SetNextF=", SetNextF, " level ", cur_level+1);
                numCurF=SetNextF.getSize();
                //numCurF=SetNextF.size;
                SetCurF=SetNextF;
                SetNextF.clear();
          }//end while  
          return "success";
      }

      proc return_depth(): string throws{
          var depthName = st.nextName();
          var depthEntry = try! new shared SymEntry(depth);
          try! st.addEntry(depthName, depthEntry);
          //try! st.addEntry(vertexName, vertexEntry);

          var depMsg =  'created ' + st.attrib(depthName);
          //var lrepMsg =  'created ' + st.attrib(levelName) + '+created ' + st.attrib(vertexName) ;
          return depMsg;

      }
      //depthName = st.nextName();
      //var depthEntry = new shared SymEntry(depth);
      //st.addEntry(depthName, depthEntry);
      //repMsg =  'created ' + st.attrib(depthName);
      if (Directed!=0) {
          if (Weighted!=0) {
              //repMsg=BFS_DW(Nv, Ne,Directed,Weighted,restpart,st);
              //var pn = Reflection.getRoutineName();
               (srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN)=
                   restpart.splitMsgToTuple(7);
              root=rootN:int;
              depth[root]=0;
              var ag = new owned SegGraphDW(Nv,Ne,Directed,Weighted,srcN,dstN,
                                 startN,neighbourN,vweightN,eweightN, st);
              bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a);
              repMsg=return_depth();

          } else {
              //repMsg=BFS_D(Nv, Ne,Directed,Weighted,restpart,st);

              (srcN, dstN, startN, neighbourN,rootN )=restpart.splitMsgToTuple(5);
              var ag = new owned SegGraphD(Nv,Ne,Directed,Weighted,srcN,dstN,
                      startN,neighbourN,st);
              root=rootN:int;
              depth[root]=0;
              bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a);
              repMsg=return_depth();

          }
      }
      else {
          if (Weighted!=0) {
              //repMsg=BFS_UDW(Nv, Ne,Directed,Weighted,restpart,st);

               (srcN, dstN, startN, neighbourN,srcRN, dstRN, startRN, neighbourRN,vweightN,eweightN, rootN )=
                   restpart.splitMsgToTuple(11);
               var ag = new owned SegGraphUDW(Nv,Ne,Directed,Weighted,
                      srcN,dstN, startN,neighbourN,
                      srcRN,dstRN, startRN,neighbourRN,
                      vweightN,eweightN, st);
              root=rootN:int;
              depth[root]=0;
              bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a);
              repMsg=return_depth();

          } else {
              //repMsg=BFS_UD(Nv, Ne,Directed,Weighted,restpart,st);

              (srcN, dstN, startN, neighbourN,srcRN, dstRN, startRN, neighbourRN, rootN )=
                   restpart.splitMsgToTuple(9);
              var ag = new owned SegGraphUD(Nv,Ne,Directed,Weighted,
                      srcN,dstN, startN,neighbourN,
                      srcRN,dstRN, startRN,neighbourRN,
                      st);

              root=rootN:int;
              depth[root]=0;
              bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a);
              repMsg=return_depth();
          }
      }
      timer.stop();
      //writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
      return repMsg;

  }










}
