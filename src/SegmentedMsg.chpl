module SegmentedMsg {
  use CPtr;
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
  use Map;

  private config const logLevel = ServerConfig.logLevel;
  const smLogger = new Logger(logLevel);

  /**
   * Procedure for assembling disjoint Strings-object / SegString parts
   * This should be a transitional procedure for current client procedure
   * of building and passing the two components separately.  Eventually
   * we'll either encapsulate both parts in a single message or do the
   * parsing and offsets construction on the server.
  */
  proc assembleStringsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var (offsetsName, valuesName) = payload.splitMsgToTuple(2);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s offsetsName: %s valuesName: %s".format(cmd, offsetsName, valuesName));
    st.checkTable(offsetsName);
    st.checkTable(valuesName);
    var offsets = getGenericTypedArrayEntry(offsetsName, st);
    var values = getGenericTypedArrayEntry(valuesName, st);
    var segString = assembleSegStringFromParts(offsets, values, st);

    // NOTE: Clean-up, the client side pieces go out of scope and issue deletions on their own
    // so it is not necessary to manually remove these pieces from the SymTab
    // st.deleteEntry(offsetsName);
    // st.deleteEntry(valuesName);

    // Now return msg binding our newly created SegString object
    var repMsg = "created " + st.attrib(segString.name) + "+created bytes.size %t".format(segString.nBytes);
    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segStrTondarrayMsg(cmd: string, payload: string, st: borrowed SymTab): bytes throws {
      var (name, comp) = payload.splitMsgToTuple(2);
      var entry = getSegString(name, st);
      if comp == "offsets" {
          return _tondarrayMsg(entry.offsets);
      } else if (comp == "values") {
          return _tondarrayMsg(entry.values);
      } else {
          var msg = "Unrecognized component: %s".format(comp);
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(), msg);
          return msg.encode();
      }
  }

  /*
     * Outputs the pdarray as a Numpy ndarray in the form of a 
     * Chapel Bytes object
     */
    proc _tondarrayMsg(entry): bytes throws {
        var arrayBytes: bytes;

        proc distArrToBytes(A: [?D] ?eltType) {
            var ptr = c_malloc(eltType, D.size);
            var localA = makeArrayFromPtr(ptr, D.size:uint);
            localA = A;
            const size = D.size*c_sizeof(eltType):int;
            return createBytesWithOwnedBuffer(ptr:c_ptr(uint(8)), size, size);
        }

        if entry.dtype == DType.Int64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, int).a);
        } else if entry.dtype == DType.Float64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, real).a);
        } else if entry.dtype == DType.Bool {
            arrayBytes = distArrToBytes(toSymEntry(entry, bool).a);
        } else if entry.dtype == DType.UInt8 {
            arrayBytes = distArrToBytes(toSymEntry(entry, uint(8)).a);
        } else {
            var errorMsg = "Error: Unhandled dtype %s".format(entry.dtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return errorMsg.encode(); // return as bytes
        }

       return arrayBytes;
    }

  proc randomStringsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var (lenStr, dist, charsetStr, arg1str, arg2str, seedStr)
          = payload.splitMsgToTuple(6);
      var len = lenStr: int;
      var charset = str2CharSet(charsetStr);
      var repMsg: string;
      select dist.toLower() {
          when "uniform" {
              var minLen = arg1str:int;
              var maxLen = arg2str:int;
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + (maxLen + minLen)*len);
              var (segs, vals) = newRandStringsUniformLength(len, minLen, maxLen, charset, seedStr);
              var strings = getSegString(segs, vals, st);
              repMsg = 'created ' + st.attrib(strings.name) + '+created bytes.size %t'.format(strings.nBytes);
          }
          when "lognormal" {
              var logMean = arg1str:real;
              var logStd = arg2str:real;
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + exp(logMean + (logStd**2)/2):int*len);
              var (segs, vals) = newRandStringsLogNormalLength(len, logMean, logStd, charset, seedStr);
              var strings = getSegString(segs, vals, st);
              repMsg = 'created ' + st.attrib(strings.name) + '+created bytes.size %t'.format(strings.nBytes);
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
    var (objtype, name, legacy_placeholder) = payload.splitMsgToTuple(3);

    // check to make sure symbols defined
    st.checkTable(name);
    
    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s objtype: %t name: %t legacy_placeholder: %t".format(
                   cmd,objtype,name,"legacy_placeholder"));

    select objtype {
      when "str" {
        var strings = getSegString(name, st);
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
      var (subcmd, objtype, name, legacy_placeholder, valtype, regexStr, valStr) = payload.splitMsgToTuple(7);
      var regex: bool = regexStr.toLower() == "true";

      // check to make sure symbols defined
      st.checkTable(name);

      var json = jsonToPdArray(valStr, 1);
      var val = json[json.domain.low];
      var rname = st.nextName();
    
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));
    
      select (objtype, valtype) {
          when ("str", "str") {
              var strings = getSegString(name, st);
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

  proc checkMatchStrings(name: string, st: borrowed SymTab) throws {
    try {
      st.checkTable(name);
    }
    catch {
      throw getErrorWithContext(
          msg="The Strings instance from which this Match is derived has been deleted",
          lineNumber=getLineNumber(),
          routineName=getRoutineName(),
          moduleName=getModuleName(),
          errorClass="MissingStringsError");
    }
  }

  proc segmentedFindLocMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, name, legacy_placeholder, groupNumStr, patternJson) = payload.splitMsgToTuple(5);
    var groupNum: int;
    try {
      groupNum = groupNumStr:int;
    }
    catch {
      var errorMsg = "groupNum could not be interpretted as an int: %s)".format(groupNumStr);
      saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      throw new owned IllegalArgumentError(errorMsg);
    }

    // check to make sure symbols defined
    checkMatchStrings(name, st);

    const json = jsonToPdArray(patternJson, 1);
    const pattern: string = json[json.domain.low];

    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %t".format(cmd, objtype));

    if objtype == "Matcher" || objtype  == "Match" {
      const rNumMatchesName = st.nextName();
      const rStartsName = st.nextName();
      const rLensName = st.nextName();
      const rIndicesName = st.nextName();
      const rSearchBoolName = st.nextName();
      const rSearchScanName = st.nextName();
      const rMatchBoolName = st.nextName();
      const rMatchScanName = st.nextName();
      const rfullMatchBoolName = st.nextName();
      const rfullMatchScanName = st.nextName();
      var strings = getSegString(name, st);

      var (numMatches, matchStarts, matchLens, matchesIndices, searchBools, searchScan, matchBools, matchScan, fullMatchBools, fullMatchScan) = strings.findMatchLocations(pattern, groupNum);
      st.addEntry(rNumMatchesName, new shared SymEntry(numMatches));
      st.addEntry(rStartsName, new shared SymEntry(matchStarts));
      st.addEntry(rLensName, new shared SymEntry(matchLens));
      st.addEntry(rIndicesName, new shared SymEntry(matchesIndices));
      st.addEntry(rSearchBoolName, new shared SymEntry(searchBools));
      st.addEntry(rSearchScanName, new shared SymEntry(searchScan));
      st.addEntry(rMatchBoolName, new shared SymEntry(matchBools));
      st.addEntry(rMatchScanName, new shared SymEntry(matchScan));
      st.addEntry(rfullMatchBoolName, new shared SymEntry(fullMatchBools));
      st.addEntry(rfullMatchScanName, new shared SymEntry(fullMatchScan));

      // Map JSON formatting is broken in Chpl version 1.24, create manually to maintain backwards compatibility
      // var createdMap = new map(keyType=string,valType=string);
      // createdMap.add("NumMatches", "created %s".format(st.attrib(rNumMatchesName)));
      // createdMap.add("Starts", "created %s".format(st.attrib(rStartsName)));
      // createdMap.add("Lens", "created %s".format(st.attrib(rLensName)));
      // createdMap.add("Indices", "created %s".format(st.attrib(rIndicesName)));
      // createdMap.add("SearchBool", "created %s".format(st.attrib(rSearchBoolName)));
      // createdMap.add("SearchInd", "created %s".format(st.attrib(rSearchScanName)));
      // createdMap.add("MatchBool", "created %s".format(st.attrib(rMatchBoolName)));
      // createdMap.add("MatchInd", "created %s".format(st.attrib(rMatchScanName)));
      // createdMap.add("FullMatchBool", "created %s".format(st.attrib(rfullMatchBoolName)));
      // createdMap.add("FullMatchInd", "created %s".format(st.attrib(rfullMatchScanName)));
      // repMsg = "%jt".format(createdMap);
      repMsg = "{";
      repMsg += "%jt: %jt,".format("NumMatches", "created %s".format(st.attrib(rNumMatchesName)));
      repMsg += "%jt: %jt,".format("Starts", "created %s".format(st.attrib(rStartsName)));
      repMsg += "%jt: %jt,".format("Lens", "created %s".format(st.attrib(rLensName)));
      repMsg += "%jt: %jt,".format("Indices", "created %s".format(st.attrib(rIndicesName)));
      repMsg += "%jt: %jt,".format("SearchBool","created %s".format(st.attrib(rSearchBoolName)));
      repMsg += "%jt: %jt,".format("SearchInd", "created %s".format(st.attrib(rSearchScanName)));
      repMsg += "%jt: %jt,".format("MatchBool", "created %s".format(st.attrib(rMatchBoolName)));
      repMsg += "%jt: %jt,".format("MatchInd", "created %s".format(st.attrib(rMatchScanName)));
      repMsg += "%jt: %jt,".format("FullMatchBool", "created %s".format(st.attrib(rfullMatchBoolName)));
      repMsg += "%jt: %jt".format("FullMatchInd", "created %s".format(st.attrib(rfullMatchScanName)));
      repMsg += "}";
    }
    else {
      var errorMsg = "%s".format(objtype);
      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedFindAllMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, name, legacy_placeholder, numMatchesName, startsName, lensName, indicesName, returnMatchOrigStr) = payload.splitMsgToTuple(8);
    const returnMatchOrig: bool = returnMatchOrigStr.toLower() == "true";

    // check to make sure symbols defined
    checkMatchStrings(name, st);
    st.checkTable(numMatchesName);
    st.checkTable(startsName);
    st.checkTable(lensName);
    st.checkTable(indicesName);

    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %t".format(cmd, objtype));

    select objtype {
      when "Matcher" {
        const optName: string = if returnMatchOrig then st.nextName() else "";
        var strings = getSegString(name, st);
        var numMatches = getGenericTypedArrayEntry(numMatchesName, st): borrowed SymEntry(int);
        var starts = getGenericTypedArrayEntry(startsName, st): borrowed SymEntry(int);
        var lens = getGenericTypedArrayEntry(lensName, st): borrowed SymEntry(int);
        var indices = getGenericTypedArrayEntry(indicesName,st): borrowed SymEntry(int);

        var (off, val, matchOrigins) = strings.findAllMatches(numMatches, starts, lens, indices, returnMatchOrig);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
        if returnMatchOrig {
          const optName: string = if returnMatchOrig then st.nextName() else "";
          st.addEntry(optName, new shared SymEntry(matchOrigins));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      }
      when "Match" {
        const optName: string = if returnMatchOrig then st.nextName() else "";
        var strings = getSegString(name, st);
        // numMatches is the matched boolean array for Match objects
        var numMatches = st.lookup(numMatchesName): borrowed SymEntry(bool);
        var starts = st.lookup(startsName): borrowed SymEntry(int);
        var lens = st.lookup(lensName): borrowed SymEntry(int);
        var indices = st.lookup(indicesName): borrowed SymEntry(int);

        var (off, val, matchOrigins) = strings.findAllMatches(numMatches, starts, lens, indices, returnMatchOrig);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
        if returnMatchOrig {
          st.addEntry(optName, new shared SymEntry(matchOrigins));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      }
      // when "Match" do numMatch SymEntry(bool) AND ?t in FindAll declaration in Strings idk if "for k in matchInd..#numMatches[stringInd]" will still work
      otherwise {
        var errorMsg = "%s".format(objtype);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
      }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedSubMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, name, legacy_placeholder, repl, countStr, returnNumSubsStr, patternJson) = payload.splitMsgToTuple(7);
    const returnNumSubs: bool = returnNumSubsStr.toLower() == "true";
    var count: int;
    try {
      count = countStr:int;
    }
    catch {
      var errorMsg = "Count could not be interpretted as an int: %s)".format(countStr);
      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      throw new owned IllegalArgumentError(errorMsg);
    }

    // check to make sure symbols defined
    st.checkTable(name);

    const json = jsonToPdArray(patternJson, 1);
    const pattern: string = json[json.domain.low];

    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %t".format(cmd, objtype));

    select objtype {
      when "Matcher" {
        const optName: string = if returnNumSubs then st.nextName() else "";
        const strings = getSegString(name, st);
        var (off, val, numSubs) = strings.sub(pattern, repl, count, returnNumSubs);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
        if returnNumSubs {
          st.addEntry(optName, new shared SymEntry(numSubs));
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

  proc createPeelSymEntries(lo, lv, ro, rv, st: borrowed SymTab) throws {
    var leftEntry = getSegString(lo, lv, st);
    var rightEntry = getSegString(ro, rv, st);
    return (leftEntry.name, rightEntry.name);
  }

  proc segmentedPeelMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (subcmd, objtype, name, legacy_placeholder, valtype, valStr,
         idStr, kpStr, lStr, regexStr, jsonStr) = payload.splitMsgToTuple(11);

    // check to make sure symbols defined
    st.checkTable(name);

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));

    select (objtype, valtype) {
    when ("str", "str") {
      var strings = getSegString(name, st);
      select subcmd {
        when "peel" {
          var times = valStr:int;
          var regex: bool = (regexStr.toLower() == "true");
          var includeDelimiter = (idStr.toLower() == "true");
          var keepPartial = (kpStr.toLower() == "true");
          var left = (lStr.toLower() == "true");
          var json = jsonToPdArray(jsonStr, 1);
          var val = json[json.domain.low];

          var leftName = "";
          var rightName = "";
          if regex {
            var (lo, lv, ro, rv) = strings.peelRegex(val, times, includeDelimiter, keepPartial, left);
            (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
          }
          else {
            select (includeDelimiter, keepPartial, left) {
              when (false, false, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, false, false);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
              when (false, false, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, false, true);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
              when (false, true, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, true, false);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
              when (false, true, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, false, true, true);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
               when (true, false, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, false, false);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
              when (true, false, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, false, true);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
              when (true, true, false) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, true, false);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
              when (true, true, true) {
                var (lo, lv, ro, rv) = strings.peel(val, times, true, true, true);
                (leftName, rightName) = createPeelSymEntries(lo, lv, ro, rv, st);
              }
            }
          }
          var leftEntry = getSegString(leftName, st);
          var rightEntry = getSegString(rightName, st);
          repMsg = "created %s+created bytes.size %t+created %s+created bytes.size %t".format(
                                                                        st.attrib(leftEntry.name),
                                                                        leftEntry.nBytes,
                                                                        st.attrib(rightEntry.name),
                                                                        rightEntry.nBytes);
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
    var (objtype, name, legacy_placeholder) = payload.splitMsgToTuple(3);

    // check to make sure symbols defined
    st.checkTable(name);

    select objtype {
        when "str" {
            var strings = getSegString(name, st);
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
      var strName = args[1];
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "strName: %s".format(strName));
      st.checkTable(args[1]); // TODO move to single name
      
      select objtype {
          when "str" {
              // Make a temporary strings array
              var strings = getSegString(strName, st);
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
    st.checkTable(args[1]); //TODO remove legacy_placeholder and bump indices below

    select objtype {
        when "str" {
            // Make a temporary string array
            var strings = getSegString(args[1], st);

            // Parse the slice parameters TODO bump this indicies after legacy_placeholder removal
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
            // Compute the slice
            var (newSegs, newVals) = strings[slice];
            // Store the resulting offsets and bytes arrays
            var newStringsObj = getSegString(newSegs, newVals, st);
            var repMsg = "created " + st.attrib(newStringsObj.name) + "+created bytes.size %t".format(newStringsObj.nBytes);
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
    st.checkTable(args[1]);  // TODO update positional args below after removing legacy_placeholder

    var newStringsName = "";
    var nBytes = 0;
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "objtype:%s".format(objtype));
    
    select objtype {
        when "str" {
            var strings = getSegString(args[1], st);
            var iname = args[3];
            var gIV: borrowed GenSymEntry = getGenericTypedArrayEntry(iname, st);
            try {
                select gIV.dtype {
                    when DType.Int64 {
                        var iv = toSymEntry(gIV, int);
                        var (newSegs, newVals) = strings[iv.a];
                        var newStringsObj = getSegString(newSegs, newVals, st);
                        newStringsName = newStringsObj.name;
                        nBytes = newStringsObj.nBytes;
                    }
                    when DType.Bool {
                        var iv = toSymEntry(gIV, bool);
                        var (newSegs, newVals) = strings[iv.a];
                        var newStringsObj = getSegString(newSegs, newVals, st);
                        newStringsName = newStringsObj.name;
                        nBytes = newStringsObj.nBytes;
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
    var repMsg = "created " + st.attrib(newStringsName) + "+created bytes.size %t".format(nBytes);

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segBinopvvMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op,
         // Type and attrib names of left segmented array
         ltype, leftName, left_legacy_placeholder,
         // Type and attrib names of right segmented array
         rtype, rightName, right_legacy_placeholder, leftStr, jsonStr)
           = payload.splitMsgToTuple(9);

    // check to make sure symbols defined
    st.checkTable(leftName);
    st.checkTable(rightName);

    select (ltype, rtype) {
        when ("str", "str") {
            var lstrings = getSegString(leftName, st);
            var rstrings = getSegString(rightName, st);

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
                    var strings:SegString;
                    if left {
                        var (newOffsets, newVals) = lstrings.stick(rstrings, delim, false);
                        strings = getSegString(newOffsets, newVals, st);
                    } else {
                        var (newOffsets, newVals) = lstrings.stick(rstrings, delim, true);
                        strings = getSegString(newOffsets, newVals, st);
                    }
                    // TODO remove second created entry after legacy_placeholder removal
                    repMsg = "created %s+created bytes.size %t".format(st.attrib(strings.name), strings.nBytes);
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
      var (op, objtype, name, legacy_placeholder, valtype, encodedVal)
          = payload.splitMsgToTuple(6);

      // check to make sure symbols defined
      st.checkTable(name);

      var json = jsonToPdArray(encodedVal, 1);
      var value = json[json.domain.low];
      var rname = st.nextName();

      select (objtype, valtype) {
          when ("str", "str") {
              var strings = getSegString(name, st);
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
      var (mainObjtype, mainName, main_legacy_placeholder, testObjtype, testName,
         test_legacy_placeholder, invertStr) = payload.splitMsgToTuple(7);

      // check to make sure symbols defined
      st.checkTable(mainName);
      st.checkTable(testName);

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
              var mainStr = getSegString(mainName, st);
              var testStr = getSegString(testName, st);
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
      var (objtype, name, legacy_placeholder) = payload.splitMsgToTuple(3);

      // check to make sure symbols defined
      st.checkTable(name);
      
      var rname = st.nextName();
      select (objtype) {
          when "str" {
              var strings = getSegString(name, st);
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