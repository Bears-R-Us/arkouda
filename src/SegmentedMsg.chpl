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
  use RadixSortLSD only radixSortLSD_ranks;


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


  proc segrmatgenMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
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
      var a = p;
      var b = (1.0 - a)/ 3.0:real;
      var c = b;
      var d = b;
      var src,srcR,src1,srcR1: [0..Ne-1] int;
      var dst,dstR,dst1,dstR1: [0..Ne-1] int;
      var e_weight: [0..Ne-1] int;
      var v_weight: [0..Nv-1] int;
      var length: [0..Nv-1] int;
      var lengthR: [0..Nv-1] int;
      var start_i: [0..Nv-1] int;
      var start_iR: [0..Nv-1] int;
      length=0;
      lengthR=0;
      start_i=-1;
      start_iR=-1;
      var n_vertices=Nv;
      var n_edges=Ne;
      src=1;
      dst=1;
      // quantites to use in edge generation loop
      var ab = a+b:real;
      var c_norm = c / (c + d):real;
      var a_norm = a / (a + b):real;
      // generate edges
      var src_bit: [0..Ne-1]int;
      var dst_bit: [0..Ne-1]int;
      for ib in 1..lgNv {
          var tmpvar: [0..Ne-1] real;
          fillRandom(tmpvar);
          src_bit=tmpvar>ab;
          fillRandom(tmpvar);
          dst_bit=tmpvar>(c_norm * src_bit + a_norm * (~ src_bit));
          src = src + ((2**(ib-1)) * src_bit);
          dst = dst + ((2**(ib-1)) * dst_bit);
      }
      src=src%Nv;
      dst=dst%Nv;
      //remove self loop
      src=src+(src==dst);
      src=src%Nv;

      var iv = radixSortLSD_ranks(src);
      // permute into sorted order
      src1 = src[iv]; //# permute first vertex into sorted order
      dst1 = dst[iv]; //# permute second vertex into sorted order
      //# to premute/rename vertices
      var startpos=0, endpos:int;
      var sort=0:int;
      while (startpos < Ne-2) {
         endpos=startpos+1;
         sort=0;
         //writeln("startpos=",startpos,"endpos=",endpos);
         while (endpos <=Ne-1) {
            if (src1[startpos]==src1[endpos])  {
               sort=1;
               endpos+=1;
               continue;
            } else {
               break;
            }
         }//end of while endpos
         if (sort==1) {
            var tmpary:[0..endpos-startpos-1] int;
            tmpary=dst1[startpos..endpos-1];
            var ivx=radixSortLSD_ranks(tmpary);
            dst1[startpos..endpos-1]=tmpary[ivx];
            //writeln("src1=",src1,"dst1=",dst1,"ivx=",ivx);
            sort=0;
         } 
         startpos+=1;
      }//end of while startpos

      for i in 0..Ne-1 do {
        length[src1[i]]+=1;
        if (start_i[src1[i]] ==-1){
           start_i[src1[i]]=i;
           //writeln("assign index ",i, " to vertex ",src1[i]);
        }
 
      }
      //var neighbour  = (+ scan length) - length;
      var neighbour  = length;
      var neighbourR  = neighbour;

      if (directed==0) { //undirected graph

          srcR = dst1;
          dstR = src1;

          var ivR = radixSortLSD_ranks(srcR);
          srcR1 = srcR[ivR]; //# permute first vertex into sorted order
          dstR1 = dstR[ivR]; //# permute second vertex into sorted order
          startpos=0;
          sort=0;
          while (startpos < Ne-2) {
              endpos=startpos+1;
              sort=0;
              while (endpos <=Ne-1) {
                 if (srcR1[startpos]==srcR1[endpos])  {
                    sort=1;
                    endpos+=1;
                    continue;
                 } else {
                    break;
                 } 
              }//end of while endpos
              if (sort==1) {
                  var tmparyR:[0..endpos-startpos-1] int;
                  tmparyR=dstR1[startpos..endpos-1];
                  var ivxR=radixSortLSD_ranks(tmparyR);
                  dstR1[startpos..endpos-1]=tmparyR[ivxR];
                  sort=0;
              } 
              startpos+=1;
          }//end of while startpos


          for i in 0..Ne-1 do {
              lengthR[srcR1[i]]+=1;
              if (start_iR[srcR1[i]] ==-1){
                  start_iR[srcR1[i]]=i;
              }
          }
          //neighbourR  = (+ scan lengthR) - lengthR;
          neighbourR  = lengthR;

      }//end of undirected


      var ewName ,vwName:string;
      if (weighted!=0) {
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
      }
      var srcName = st.nextName();
      var dstName = st.nextName();
      var startName = st.nextName();
      var neiName = st.nextName();
      var srcEntry = new shared SymEntry(src1);
      var dstEntry = new shared SymEntry(dst1);
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
          if (weighted!=0) {
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);
          } else {
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) ; 

          }
      } else {//for undirected graph

          srcNameR = st.nextName();
          dstNameR = st.nextName();
          startNameR = st.nextName();
          neiNameR = st.nextName();
          var srcEntryR = new shared SymEntry(srcR1);
          var dstEntryR = new shared SymEntry(dstR1);
          var startEntryR = new shared SymEntry(start_iR);
          var neiEntryR = new shared SymEntry(neighbourR);
          st.addEntry(srcNameR, srcEntryR);
          st.addEntry(dstNameR, dstEntryR);
          st.addEntry(startNameR, startEntryR);
          st.addEntry(neiNameR, neiEntryR);
          if (weighted!=0) {
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) + 
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) + 
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);
          } else {
              repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) + 
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) ; 
          }

      }
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return repMsg;
  }


  proc segBFSMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      //var (n_verticesN,n_edgesN,directedN,weightedN,srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN )
      //    = payload.decode().splitMsgToTuple(10);
      var (n_verticesN,n_edgesN,directedN,weightedN,restpart )
          = payload.decode().splitMsgToTuple(5);
      var Nv=n_verticesN:int;
      var Ne=n_edgesN:int;
      var Directed=directedN:int;
      var Weighted=weightedN:int;


      if (Directed!=0) {
          if (Weighted!=0) {
              repMsg=BFS_DW(Nv, Ne,Directed,Weighted,restpart,st);
          } else {
              repMsg=BFS_D(Nv, Ne,Directed,Weighted,restpart,st);
          }
      }
      else {
          if (Weighted!=0) {
              repMsg=BFS_UDW(Nv, Ne,Directed,Weighted,restpart,st);
          } else {
              repMsg=BFS_UD(Nv, Ne,Directed,Weighted,restpart,st);
          }
      }
      return repMsg;

  }




  proc BFS_D(Nv:int , Ne:int ,Directed:int ,Weighted:int,restpart:string ,st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;

      var srcN, dstN, startN, neighbourN, rootN :string;

      (srcN, dstN, startN, neighbourN,rootN )=restpart.splitMsgToTuple(5);
      var ag = new owned SegGraphD(Nv,Ne,Directed,Weighted,srcN,dstN,
                      startN,neighbourN,st);


      var root=rootN:int;
      var depth=-1: [0..Nv-1] int;
      depth[root]=0;
      var cur_level=0;
      var SetCurF: domain(int);
      var SetNextF: domain(int);
      SetCurF.add(root);
      var numCurF=1:int;

      while (numCurF>0) {
           SetNextF.clear();
           forall i in SetCurF {
              var numNF=-1 :int;
              ref nf=ag.neighbour.a;
              ref sf=ag.start_i.a;
              ref df=ag.dst.a;
              numNF=nf[i];
              ref NF=df[sf[i]..sf[i]+numNF-1];
              if (numNF>0) {
                forall j in NF {
                //for j in NF {
                   if (depth[j]==-1) {
                      depth[j]=cur_level+1;
                      SetNextF.add(j);
                   }
                }
              }

           }//end forall i
           cur_level+=1;
           //writeln("SetCurF= ", SetCurF, "SetNextF=", SetNextF, " level ", cur_level+1);
           numCurF=SetNextF.size;
           SetCurF=SetNextF;
      }
      var vertexValue = radixSortLSD_ranks(depth);
      var levelValue=depth[vertexValue]; 

      var levelName = st.nextName();
      var vertexName = st.nextName();
      var levelEntry = new shared SymEntry(levelValue);
      var vertexEntry = new shared SymEntry(vertexValue);
      st.addEntry(levelName, levelEntry);
      st.addEntry(vertexName, vertexEntry);
      repMsg =  'created ' + st.attrib(levelName) + '+created ' + st.attrib(vertexName) ;

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return repMsg;
  }




  proc BFS_DW(Nv:int, Ne:int,Directed:int,Weighted:int,restpart:string,st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;

      var srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN :string;
      (srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN)=
                   restpart.splitMsgToTuple(7);

      var ag = new owned SegGraphDW(Nv,Ne,Directed,Weighted,srcN,dstN,
                      startN,neighbourN,vweightN,eweightN, st);
      var root=rootN:int;
      var depth=-1: [0..Nv-1] int;
      depth[root]=0;
      var cur_level=0;
      var SetCurF: domain(int);
      var SetNextF: domain(int);
      SetCurF.add(root);
      var numCurF=1:int;

      while (numCurF>0) {
           SetNextF.clear();
           forall i in SetCurF {
              var numNF=-1 :int;
              ref nf=ag.neighbour.a;
              ref sf=ag.start_i.a;
              ref df=ag.dst.a;
              numNF=nf[i];
              ref NF=df[sf[i]..sf[i]+numNF-1];
              writeln("current node ",i, " has ", numNF, " neighbours and  they are  ",NF);
              if (numNF>0) {
                forall j in NF {
                //for j in NF {
                   writeln("current node ",i, " check neibour ",j, " its depth=",depth[j]);
                   if (depth[j]==-1) {
                      depth[j]=cur_level+1;
                      SetNextF.add(j);
                      writeln("current node ",i, " add ", j, " into level ", cur_level+1, " SetNextF=", SetNextF);
                   }
                }
              }

           }//end forall i
           cur_level+=1;
           writeln("SetCurF= ", SetCurF, "SetNextF=", SetNextF, " level ", cur_level+1);
           numCurF=SetNextF.size;
           SetCurF=SetNextF;
      }
      var vertexValue = radixSortLSD_ranks(depth);
      var levelValue=depth[vertexValue]; 

      var levelName = st.nextName();
      var vertexName = st.nextName();
      var levelEntry = new shared SymEntry(levelValue);
      var vertexEntry = new shared SymEntry(vertexValue);
      st.addEntry(levelName, levelEntry);
      st.addEntry(vertexName, vertexEntry);
      repMsg =  'created ' + st.attrib(levelName) + '+created ' + st.attrib(vertexName) ;

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return repMsg;
  }








  proc BFS_UD(Nv:int , Ne:int ,Directed:int ,Weighted:int,restpart:string ,st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;


      var srcN, dstN, startN, neighbourN, rootN :string;
      var srcRN, dstRN, startRN, neighbourRN:string;

      (srcN, dstN, startN, neighbourN,srcRN, dstRN, startRN, neighbourRN, rootN )=
                   restpart.splitMsgToTuple(9);
      var ag = new owned SegGraphUD(Nv,Ne,Directed,Weighted,
                      srcN,dstN, startN,neighbourN,
                      srcRN,dstRN, startRN,neighbourRN,
                      st);

      var root=rootN:int;
      var depth=-1: [0..Nv-1] int;
      depth[root]=0;
      var cur_level=0;
      var SetCurF: domain(int);
      var SetNextF: domain(int);
      SetCurF.add(root);
      var numCurF=1:int;

      //writeln("========================BSF_UD==================================");
      while (numCurF>0) {
           SetNextF.clear();
           forall i in SetCurF {
              var numNF=-1 :int;
              ref nf=ag.neighbour.a;
              ref sf=ag.start_i.a;
              ref df=ag.dst.a;
              numNF=nf[i];
              ref NF=df[sf[i]..sf[i]+numNF-1];
              if (numNF>0) {
                forall j in NF {
                //writeln("current node ",i, " has neibours ",NF);
                //for j in NF {
                   if (depth[j]==-1) {
                      depth[j]=cur_level+1;
                      SetNextF.add(j);
                      //writeln("current node ",i, " add ", j, 
                      //        " into level ", cur_level+1, " SetNextF=", SetNextF);
                   }
                }
              }
              // reverse direction
              if (Directed!=1) {

                  var numNFR=-1 :int;
                  ref nfR=ag.neighbourR.a;
                  ref sfR=ag.start_iR.a;
                  ref dfR=ag.dstR.a;
                  numNFR=nfR[i];
                  ref NFR=dfR[sfR[i]..sfR[i]+numNFR-1];
                  if (numNFR>0) {
                      //writeln("current node ",i, " has reverse neibours ",NFR);
                      forall j in NFR {
                      //for j in NFR {
                          if (depth[j]==-1) {
                             depth[j]=cur_level+1;
                             SetNextF.add(j);
                             //writeln("current node ",i, " add reverse ", j, 
                             //        " into level ", cur_level+1, " SetNextF=", SetNextF);
                          }
                      } 
                  }
              }


           }//end forall i
           cur_level+=1;
           //writeln("SetCurF= ", SetCurF, "SetNextF=", SetNextF, " level ", cur_level+1);
           numCurF=SetNextF.size;
           SetCurF=SetNextF;
      }
      var vertexValue = radixSortLSD_ranks(depth);
      var levelValue=depth[vertexValue]; 

      var levelName = st.nextName();
      var vertexName = st.nextName();
      var levelEntry = new shared SymEntry(levelValue);
      var vertexEntry = new shared SymEntry(vertexValue);
      st.addEntry(levelName, levelEntry);
      st.addEntry(vertexName, vertexEntry);
      repMsg =  'created ' + st.attrib(levelName) + '+created ' + st.attrib(vertexName) ;

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return repMsg;
  }






  proc BFS_UDW(Nv:int , Ne:int ,Directed:int ,Weighted:int,restpart:string ,st: borrowed SymTab): string throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;


      var srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN :string;
      var srcRN, dstRN, startRN, neighbourRN:string;

      (srcN, dstN, startN, neighbourN,srcRN, dstRN, startRN, neighbourRN,vweightN,eweightN, rootN )=
                   restpart.splitMsgToTuple(11);
      var ag = new owned SegGraphUDW(Nv,Ne,Directed,Weighted,
                      srcN,dstN, startN,neighbourN,
                      srcRN,dstRN, startRN,neighbourRN,
                      vweightN,eweightN, st);

      //writeln("========================BSF_UDW==================================");
      var root=rootN:int;
      var depth=-1: [0..Nv-1] int;
      depth[root]=0;
      var cur_level=0;
      var SetCurF: domain(int);
      var SetNextF: domain(int);
      SetCurF.add(root);
      var numCurF=1:int;

      /*
      writeln("Fisrt Check if the values are correct");
      writeln("src=");
      writeln(ag.src.a);
      writeln("dst=");
      writeln(ag.dst.a);
      writeln("neighbours=");
      writeln(ag.neighbour.a);
      writeln("start=");
      writeln(ag.start_i.a);

      writeln("srcR=");
      writeln(ag.srcR.a);
      writeln("dstR=");
      writeln(ag.dstR.a);
      writeln("neighbours=");
      writeln(ag.neighbourR.a);
      writeln("startR=");
      writeln(ag.start_iR.a);

      for i in 0..ag.n_vertices-1 do {
          writeln("node ",i, " has ", ag.neighbour.a[i], " neighbours", 
                " start=",ag.start_i.a[i], " they are ", 
                ag.dst.a[ag.start_i.a[i]..ag.start_i.a[i]-1+ag.neighbour.a[i]]);
      }
      for i in 0..ag.n_vertices-1 do {
          writeln("reverse node ",i, " has ", ag.neighbourR.a[i], " neighbours", 
                " start=",ag.start_iR.a[i], " they are ", 
                ag.dstR.a[ag.start_iR.a[i]..ag.start_iR.a[i]-1+ag.neighbourR.a[i]]);
      }
      */

      while (numCurF>0) {
           //writeln("start loop SetCurF=", SetCurF);
           SetNextF.clear();
           forall i in SetCurF {
              var numNF=-1 :int;
              ref nf=ag.neighbour.a;
              ref sf=ag.start_i.a;
              ref df=ag.dst.a;
              numNF=nf[i];
              ref NF=df[sf[i]..sf[i]+numNF-1];
              //writeln("current node ",i, " has ", numNF, " neighbours and  they are  ",NF);
              if (numNF>0) {
                forall j in NF {
                //for j in NF {
                   //writeln("current node ",i, " check neibour ",j, " its depth=",depth[j]);
                   if (depth[j]==-1) {
                      depth[j]=cur_level+1;
                      SetNextF.add(j);
                      //writeln("current node ",i, " add ", j, " into level ", cur_level+1, " SetNextF=", SetNextF);
                   }
                }
              }
              // reverse direction
              if (Directed!=1) {
                  var numNFR=-1 :int;
                  ref nfR=ag.neighbourR.a;
                  ref sfR=ag.start_iR.a;
                  ref dfR=ag.dstR.a;
                  numNFR=nfR[i];
                  ref NFR=dfR[sfR[i]..sfR[i]+numNFR-1];
                  //writeln("current node ",i, " has ", numNFR ," reverse neighbours and  they are  ",NFR);
                  if ( numNFR>0) {
                      forall j in NFR {
                      //for j in NFR {
                          //writeln("current node ",i, " check neibour ",j, " its depth=",depth[j]);
                          if (depth[j]==-1) {
                             depth[j]=cur_level+1;
                             SetNextF.add(j);
                             //writeln("current node ",i, " add reverse ", j, 
                             //          " into level ", cur_level+1, " SetNextF=", SetNextF);
                          }
                      } 
                  }
              }


           }//end forall i
           cur_level+=1;
           //writeln("SetCurF= ", SetCurF, "SetNextF=", SetNextF, " level ", cur_level+1);
           numCurF=SetNextF.size;
           SetCurF=SetNextF;
      }
      var vertexValue = radixSortLSD_ranks(depth);
      var levelValue=depth[vertexValue]; 

      var levelName = st.nextName();
      var vertexName = st.nextName();
      var levelEntry = new shared SymEntry(levelValue);
      var vertexEntry = new shared SymEntry(vertexValue);
      st.addEntry(levelName, levelEntry);
      st.addEntry(vertexName, vertexEntry);
      repMsg =  'created ' + st.attrib(levelName) + '+created ' + st.attrib(vertexName) ;

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return repMsg;
  }





}
