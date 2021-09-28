module SegmentedMsg {
  use Reflection;
  use ServerErrors;
  use Logging;
  use Message;
  use SegmentedArray;
  use ServerErrorStrings;
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use RandArray;
  use IO;
  use GenSymIO only jsonToPdArray;

  private config const logLevel = ServerConfig.logLevel;
  const smLogger = new Logger(logLevel);

  proc randomStringsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var (lenStr, dist, charsetStr, arg1str, arg2str, seedStr)
          = payload.splitMsgToTuple(6);
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
              var errorMsg = notImplementedError(pn, dist);      
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
              return new MsgTuple(errorMsg, MsgType.ERROR);    
          }
      }

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentLengthsMsg(cmd: string, payload: string, 
                                          st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var (objtype, segName, valName) = payload.splitMsgToTuple(3);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);
    
    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s objtype: %t segName: %t valName: %t".format(
                   cmd,objtype,segName,valName));

    select objtype {
      when "str" {
        var strings = getSegString(segName, valName, st);
        var lengths = st.addEntry(rname, strings.size, int);
        // Do not include the null terminator in the length
        lengths.a = strings.getLengths() - 1;
      }
      otherwise {
          var errorMsg = notImplementedError(pn, "%s".format(objtype));
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                      
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }

    var repMsg = "created "+st.attrib(rname);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedEfuncMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (subcmd, objtype, segName, valName, valtype, regexStr, valStr) = payload.splitMsgToTuple(7);
      var regex: bool = regexStr.toLower() == "true";

      // check to make sure symbols defined
      st.checkTable(segName);
      st.checkTable(valName);

      var json = jsonToPdArray(valStr, 1);
      var val = json[json.domain.low];
      var rname = st.nextName();
    
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));
    
      select (objtype, valtype) {
          when ("str", "str") {
              var strings = getSegString(segName, valName, st);
              select subcmd {
                  when "contains" {
                      var truth = st.addEntry(rname, strings.size, bool);
                      truth.a = strings.substringSearch(val, SearchMode.contains, regex);
                      repMsg = "created "+st.attrib(rname);
                  }
                  when "startswith" {
                      var truth = st.addEntry(rname, strings.size, bool);
                      truth.a = strings.substringSearch(val, SearchMode.startsWith, regex);
                      repMsg = "created "+st.attrib(rname);
                  }
                  when "endswith" {
                      var truth = st.addEntry(rname, strings.size, bool);
                      truth.a = strings.substringSearch(val, SearchMode.endsWith, regex);
                      repMsg = "created "+st.attrib(rname);
                  }
                  when "match" {
                      var truth = st.addEntry(rname, strings.size, bool);
                      truth.a = strings.substringSearch(val, SearchMode.match, regex);
                      repMsg = "created "+st.attrib(rname);
                  }
                  otherwise {
                      var errorMsg = notImplementedError(pn, "subcmd: %s, (%s, %s)".format(
                                  subcmd, objtype, valtype));
                      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
              }
          }
          otherwise {
            var errorMsg = "(%s, %s)".format(objtype, valtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
          }
      }

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedFindLocMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, segName, valName, patternJson) = payload.splitMsgToTuple(4);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);

    const json = jsonToPdArray(patternJson, 1);
    const pattern: string = json[json.domain.low];

    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %t".format(cmd, objtype));

    select objtype {
      when "str" {
        const rNumMatchesName = st.nextName();
        const rStartsName = st.nextName();
        const rLensName = st.nextName();
        var strings = getSegString(segName, valName, st);
        var (numMatches, matchStarts, matchLens) = strings.findMatchLocations(pattern);
        st.addEntry(rNumMatchesName, new shared SymEntry(numMatches));
        st.addEntry(rStartsName, new shared SymEntry(matchStarts));
        st.addEntry(rLensName, new shared SymEntry(matchLens));
        repMsg = "created %s+created %s+created %s".format(st.attrib(rNumMatchesName),
                                                           st.attrib(rStartsName),
                                                           st.attrib(rLensName));
      }
      otherwise {
        var errorMsg = "%s".format(objtype);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
      }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedFindAllMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, segName, valName, numMatchesName, startsName, lensName, returnMatchOrigStr) = payload.splitMsgToTuple(7);
    const returnMatchOrig: bool = returnMatchOrigStr.toLower() == "true";

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);
    st.checkTable(numMatchesName);
    st.checkTable(startsName);
    st.checkTable(lensName);

    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %t".format(cmd, objtype));

    select objtype {
      when "str" {
        const rSegName = st.nextName();
        const rValName = st.nextName();
        const optName: string = if returnMatchOrig then st.nextName() else "";
        var strings = getSegString(segName, valName, st);
        var numMatches = st.lookup(numMatchesName): borrowed SymEntry(int);
        var starts = st.lookup(startsName): borrowed SymEntry(int);
        var lens = st.lookup(lensName): borrowed SymEntry(int);

        var (off, val, matchOrigins) = strings.findAllMatches(numMatches, starts, lens, returnMatchOrig);
        st.addEntry(rSegName, new shared SymEntry(off));
        st.addEntry(rValName, new shared SymEntry(val));
        repMsg = "created %s+created %s".format(st.attrib(rSegName), st.attrib(rValName));
        if returnMatchOrig {
          st.addEntry(optName, new shared SymEntry(matchOrigins));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      }
      otherwise {
        var errorMsg = "%s".format(objtype);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
      }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st: borrowed SymTab) throws {
    st.addEntry(loname, new shared SymEntry(lo));
    st.addEntry(lvname, new shared SymEntry(lv));
    st.addEntry(roname, new shared SymEntry(ro));
    st.addEntry(rvname, new shared SymEntry(rv));
  }

  proc segmentedPeelMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (subcmd, objtype, segName, valName, valtype, valStr,
         idStr, kpStr, lStr, regexStr, jsonStr) = payload.splitMsgToTuple(11);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));

    select (objtype, valtype) {
    when ("str", "str") {
      var strings = getSegString(segName, valName, st);
      select subcmd {
        when "peel" {
          var times = valStr:int;
          var regex: bool = (regexStr.toLower() == "true");
          var includeDelimiter = (idStr.toLower() == "true");
          var keepPartial = (kpStr.toLower() == "true");
          var left = (lStr.toLower() == "true");
          var json = jsonToPdArray(jsonStr, 1);
          var val = json[json.domain.low];
          var loname = st.nextName();
          var lvname = st.nextName();
          var roname = st.nextName();
          var rvname = st.nextName();

          if regex {
            var (lo, lv, ro, rv) = strings.peelRegex(val, times, includeDelimiter, keepPartial, left);
            createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
          }
          else {
            select (includeDelimiter, keepPartial, left) {
              when (false, false, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, false, false);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
              when (false, false, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, false, true);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
              when (false, true, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, true, false);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
              when (false, true, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, true, true);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
               when (true, false, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, false, false);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
              when (true, false, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, false, true);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
              when (true, true, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, true, false);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
              when (true, true, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, true, true);
                createPeelSymEntries(loname, lo, lvname, lv, roname, ro, rvname, rv, st);
              }
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
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }
    otherwise {
        var errorMsg = notImplementedError(pn, "(%s, %s)".format(objtype, valtype));
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);       
      }
    }
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedHashMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, segName, valName) = payload.splitMsgToTuple(3);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);

    select objtype {
        when "str" {
            var strings = getSegString(segName, valName, st);
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
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        otherwise {
            var errorMsg = notImplementedError(pn, objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);
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
  proc segmentedIndexMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    // 'subcmd' is the type of indexing to perform
    // 'objtype' is the type of segmented array
    var (subcmd, objtype, rest) = payload.splitMsgToTuple(3);
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
                var errorMsg = "Error in %s, unknown subcommand %s".format(pn, subcmd);
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    } catch e: OutOfBoundsError {
        var errorMsg = "index out of bounds";
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: Error {
        var errorMsg = "unknown cause %t".format(e);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }
  }
 
  /*
  Returns the object corresponding to the index
  */ 
  proc segIntIndex(objtype: string, args: [] string, 
                                         st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();

      // check to make sure symbols defined
      st.checkTable(args[1]);
      st.checkTable(args[2]);
      
      select objtype {
          when "str" {
              // Make a temporary strings array
              var strings = getSegString(args[1], args[2], st);
              // Parse the index
              var idx = args[3]:int;
              // TO DO: in the future, we will force the client to handle this
              idx = convertPythonIndexToChapel(idx, strings.size);
              var s = strings[idx];

              var repMsg = "item %s %jt".format("str", s);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
              return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          otherwise { 
              var errorMsg = notImplementedError(pn, objtype); 
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);                          
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

  proc segSliceIndex(objtype: string, args: [] string, 
                                         st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();

    // check to make sure symbols defined
    st.checkTable(args[1]);
    st.checkTable(args[2]);

    select objtype {
        when "str" {
            // Make a temporary string array
            var strings = getSegString(args[1], args[2], st);

            // Parse the slice parameters
            var start = args[3]:int;
            var stop = args[4]:int;
            var stride = args[5]:int;

            // Only stride-1 slices are allowed for now
            if (stride != 1) { 
                var errorMsg = notImplementedError(pn, "stride != 1"); 
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return new MsgTuple(errorMsg, MsgType.ERROR);
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
        
            var repMsg = "created " + st.attrib(newSegName) + " +created " + st.attrib(newValName);
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        otherwise {
            var errorMsg = notImplementedError(pn, objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);          
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

  proc segPdarrayIndex(objtype: string, args: [] string, 
                                 st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();

    // check to make sure symbols defined
    st.checkTable(args[1]);
    st.checkTable(args[2]);

    var newSegName = st.nextName();
    var newValName = st.nextName();
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "objtype:%s".format(objtype));
    
    select objtype {
        when "str" {
            var strings = getSegString(args[1], args[2], st);
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
                        return new MsgTuple(notImplementedError(pn,errorMsg), MsgType.ERROR);
                    }
                }
            } catch e: Error {
                var errorMsg =  e.message();
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        otherwise {
            var errorMsg = "unsupported objtype: %t".format(objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(notImplementedError(pn, objtype), MsgType.ERROR);
        }
    }
    var repMsg = "created " + st.attrib(newSegName) + "+created " + st.attrib(newValName);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segBinopvvMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op,
         // Type and attrib names of left segmented array
         ltype, lsegName, lvalName,
         // Type and attrib names of right segmented array
         rtype, rsegName, rvalName, leftStr, jsonStr)
           = payload.splitMsgToTuple(9);

    // check to make sure symbols defined
    st.checkTable(lsegName);
    st.checkTable(lvalName);
    st.checkTable(rsegName);
    st.checkTable(rvalName);

    select (ltype, rtype) {
        when ("str", "str") {
            var lstrings = getSegString(lsegName, lvalName, st);
            var rstrings = getSegString(rsegName, rvalName, st);

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
                otherwise {
                    var errorMsg = notImplementedError(pn, ltype, op, rtype);
                    smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
              }
           }
       otherwise {
           var errorMsg = unrecognizedTypeError(pn, "("+ltype+", "+rtype+")");
           smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
           return new MsgTuple(errorMsg, MsgType.ERROR);
       } 
    }

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segBinopvsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (op, objtype, segName, valName, valtype, encodedVal)
          = payload.splitMsgToTuple(6);

      // check to make sure symbols defined
      st.checkTable(segName);
      st.checkTable(valName);

      var json = jsonToPdArray(encodedVal, 1);
      var value = json[json.domain.low];
      var rname = st.nextName();

      select (objtype, valtype) {
          when ("str", "str") {
              var strings = getSegString(segName, valName, st);
              select op {
                  when "==" {
                      var e = st.addEntry(rname, strings.size, bool);
                      e.a = (strings == value);
                  }
                  when "!=" {
                      var e = st.addEntry(rname, strings.size, bool);
                      e.a = (strings != value);
                  }
                  otherwise {
                      var errorMsg = notImplementedError(pn, objtype, op, valtype);
                      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
              }
          }
          otherwise {
              var errorMsg = unrecognizedTypeError(pn, "("+objtype+", "+valtype+")");
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
          } 
      }

      repMsg = "created %s".format(st.attrib(rname));
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segIn1dMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (mainObjtype, mainSegName, mainValName, testObjtype, testSegName,
         testValName, invertStr) = payload.splitMsgToTuple(7);

      // check to make sure symbols defined
      st.checkTable(mainSegName);
      st.checkTable(mainValName);
      st.checkTable(testSegName);
      st.checkTable(testValName);

      var invert: bool;
      if invertStr == "True" {invert = true;
      } else if invertStr == "False" {invert = false;
      } else {
          var errorMsg = "Invalid argument in %s: %s (expected True or False)".format(pn, invertStr);
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    
      var rname = st.nextName();
 
      select (mainObjtype, testObjtype) {
          when ("str", "str") {
              var mainStr = getSegString(mainSegName, mainValName, st);
              var testStr = getSegString(testSegName, testValName, st);
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
              return new MsgTuple(errorMsg, MsgType.ERROR);            
          }
      }

      repMsg = "created " + st.attrib(rname);
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segGroupMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var (objtype, segName, valName) = payload.splitMsgToTuple(3);

      // check to make sure symbols defined
      st.checkTable(segName);
      st.checkTable(valName);
      
      var rname = st.nextName();
      select (objtype) {
          when "str" {
              var strings = getSegString(segName, valName, st);
              var iv = st.addEntry(rname, strings.size, int);
              iv.a = strings.argGroup();
          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype+")");
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);            
          }
      }

      var repMsg =  "created " + st.attrib(rname);
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }
}
