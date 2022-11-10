module SegmentedMsg {
  use CTypes;
  use Reflection;
  use ServerErrors;
  use Logging;
  use Message;
  use SegmentedArray;
  use SegmentedString;
  use ServerErrorStrings;
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use RandArray;
  use IO;
  use Map;
  use GenSymIO;

  private config const logLevel = ServerConfig.logLevel;
  const smLogger = new Logger(logLevel);

  /**
  * Build a Segmented Array object based on the segments/values specified.
  **/
  proc assembleSegArrayMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var segName = msgArgs.getValueOf("segments");
    var valName = msgArgs.getValueOf("values"); 
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s segmentsname: %s valuesName: %s".format(cmd, segName, valName));

    var segments = getGenericTypedArrayEntry(segName, st);
    var segs = toSymEntry(segments, int);

    var rtnmap: map(string, string) = new map(string, string);

    var valEntry = st.tab.getBorrowed(valName);
    if valEntry.isAssignableTo(SymbolEntryType.SegStringSymEntry){ //SegString
      var vals = getSegString(valName, st);
      var segArray = getSegArray(segs.a, vals.values.a, st);
      segArray.fillReturnMap(rtnmap, st);
    } 
    else { // pdarray
      var values = getGenericTypedArrayEntry(valName, st);
      select values.dtype {
        when (DType.Int64) {
          var vals = toSymEntry(values, int);
          var segArray = getSegArray(segs.a, vals.a, st);
          segArray.fillReturnMap(rtnmap, st);
        }
        when (DType.UInt64) {
          var vals = toSymEntry(values, uint);
          var segArray = getSegArray(segs.a, vals.a, st);
          segArray.fillReturnMap(rtnmap, st);
        }
        when (DType.Float64) {
          var vals = toSymEntry(values, real);
          var segArray = getSegArray(segs.a, vals.a, st);
          segArray.fillReturnMap(rtnmap, st);
        }
        when (DType.Bool) {
          var vals = toSymEntry(values, bool);
          var segArray = getSegArray(segs.a, vals.a, st);
          segArray.fillReturnMap(rtnmap, st);
        }
        otherwise {
            throw new owned ErrorWithContext("Values array has unsupported dtype %s".format(values.dtype:string),
                                        getLineNumber(),
                                        getRoutineName(),
                                        getModuleName(),
                                        "TypeError");
        }
      }
    }
    var repMsg: string = "%jt".format(rtnmap);
    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc getSANonEmptyMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var repMsg: string = "";
    var name = msgArgs.getValueOf("name"): string;
    var entry = st.tab.getBorrowed(name);
    var genEntry: GenSymEntry = toGenSymEntry(entry);
    var neName = st.nextName();
    select genEntry.dtype {
      when (DType.Int64) {
        var segArr = getSegArray(name, st, int);
        st.addEntry(neName, new shared SymEntry(segArr.getNonEmpty()));
        repMsg = "created " + st.attrib(neName) + "+" + segArr.getNonEmptyCount():string;
      }
      when (DType.UInt64) {
        var segArr = getSegArray(name, st, uint);
        st.addEntry(neName, new shared SymEntry(segArr.getNonEmpty()));
        repMsg = "created " + st.attrib(neName) + "+" + segArr.getNonEmptyCount():string;
      }
      when (DType.Float64) {
        var segArr = getSegArray(name, st, real);
        st.addEntry(neName, new shared SymEntry(segArr.getNonEmpty()));
        repMsg = "created " + st.attrib(neName) + "+" + segArr.getNonEmptyCount():string;
      }
      when (DType.Bool) {
        var segArr = getSegArray(name, st, bool);
        st.addEntry(neName, new shared SymEntry(segArr.getNonEmpty()));
        repMsg = "created " + st.attrib(neName) + "+" + segArr.getNonEmptyCount():string;
      }
      when (DType.UInt8){
        var segArr = getSegArray(name, st, uint(8));
        st.addEntry(neName, new shared SymEntry(segArr.getNonEmpty()));
        repMsg = "created " + st.attrib(neName) + "+" + segArr.getNonEmptyCount():string;
      }
      otherwise {
        throw new owned ErrorWithContext("Values array has unsupported dtype %s".format(genEntry.dtype:string),
                                      getLineNumber(),
                                      getRoutineName(),
                                      getModuleName(),
                                      "TypeError");
      }
    }
    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }


  /**
   * Procedure for assembling disjoint Strings-object / SegString parts
   * This should be a transitional procedure for current client procedure
   * of building and passing the two components separately.  Eventually
   * we'll either encapsulate both parts in a single message or do the
   * parsing and offsets construction on the server.
  */
  proc assembleStringsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    const offsetsName = msgArgs.getValueOf("offsets");
    const valuesName = msgArgs.getValueOf("values");
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

  proc segStrTondarrayMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): bytes throws {
      var entry = getSegString(msgArgs.getValueOf("obj"), st);
      const comp = msgArgs.getValueOf("comp");
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

  proc randomStringsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      const dist = msgArgs.getValueOf("dist");
      const len = msgArgs.get("size").getIntValue();
      const charset = str2CharSet(msgArgs.getValueOf("chars"));
      const seedStr = msgArgs.getValueOf("seed");
      var repMsg: string;
      select dist.toLower() {
          when "uniform" {
              var minLen = msgArgs.get("arg1").getIntValue();
              var maxLen = msgArgs.get("arg2").getIntValue();
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + (maxLen + minLen)*len);
              var (segs, vals) = newRandStringsUniformLength(len, minLen, maxLen, charset, seedStr);
              var strings = getSegString(segs, vals, st);
              repMsg = 'created ' + st.attrib(strings.name) + '+created bytes.size %t'.format(strings.nBytes);
          }
          when "lognormal" {
              var logMean = msgArgs.get("arg1").getRealValue();
              var logStd = msgArgs.get("arg2").getRealValue();
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

  proc segmentLengthsMsg(cmd: string, msgArgs: borrowed MessageArgs,
                                          st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("obj");

    // check to make sure symbols defined
    st.checkTable(name);
    
    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s objtype: %t name: %t".format(
                   cmd,objtype,name));

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

  proc caseChangeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const subcmd = msgArgs.getValueOf("subcmd");
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("obj");


    // check to make sure symbols defined
    st.checkTable(name);

    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s objtype: %t name: %t".format(cmd,objtype,name));

    select objtype {
      when "str" {
        var strings = getSegString(name, st);
        select subcmd {
          when "toLower" {
            var (off, val) = strings.lower();
            var retString = getSegString(off, val, st);
            repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
          }
          when "toUpper" {
            var (off, val) = strings.upper();
            var retString = getSegString(off, val, st);
            repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
          }
          when "toTitle" {
            var (off, val) = strings.title();
            var retString = getSegString(off, val, st);
            repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
          }
          otherwise {
            var errorMsg = notImplementedError(pn, "%s".format(subcmd));
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      }
      otherwise {
          var errorMsg = notImplementedError(pn, "%s".format(objtype));
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc checkCharsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const subcmd = msgArgs.getValueOf("subcmd");
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("obj");

    // check to make sure symbols defined
    st.checkTable(name);

    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s objtype: %t name: %t".format(cmd,objtype,name));

    select objtype {
      when "str" {
        var strings = getSegString(name, st);
        var truth = st.addEntry(rname, strings.size, bool);
        select subcmd {
          when "isLower" {
            truth.a = strings.isLower();
            repMsg = "created "+st.attrib(rname);
          }
          when "isUpper" {
            truth.a = strings.isUpper();
            repMsg = "created "+st.attrib(rname);
          }
          when "isTitle" {
            truth.a = strings.isTitle();
            repMsg = "created "+st.attrib(rname);
          }
          otherwise {
            var errorMsg = notImplementedError(pn, "%s".format(subcmd));
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      }
      otherwise {
          var errorMsg = notImplementedError(pn, "%s".format(objtype));
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedSearchMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      const objtype = msgArgs.getValueOf("objType");
      const name = msgArgs.getValueOf("obj");
      const valtype = msgArgs.getValueOf("valType");
      const val = msgArgs.getValueOf("val");

      // check to make sure symbols defined
      st.checkTable(name);
      var rname = st.nextName();
    
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s objtype: %t valtype: %t".format(
                          cmd,objtype,valtype));
    
      select (objtype, valtype) {
          when ("str", "str") {
              var strings = getSegString(name, st);
              var truth = st.addEntry(rname, strings.size, bool);
              truth.a = strings.substringSearch(val);
              repMsg = "created "+st.attrib(rname);
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

  proc segmentedFindLocMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("parent_name");
    const groupNum = msgArgs.get("groupNum").getIntValue();
    const pattern = msgArgs.getValueOf("pattern");

    // check to make sure symbols defined
    checkMatchStrings(name, st);

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

      var createdMap = new map(keyType=string,valType=string);
      createdMap.add("NumMatches", "created %s".format(st.attrib(rNumMatchesName)));
      createdMap.add("Starts", "created %s".format(st.attrib(rStartsName)));
      createdMap.add("Lens", "created %s".format(st.attrib(rLensName)));
      createdMap.add("Indices", "created %s".format(st.attrib(rIndicesName)));
      createdMap.add("SearchBool", "created %s".format(st.attrib(rSearchBoolName)));
      createdMap.add("SearchInd", "created %s".format(st.attrib(rSearchScanName)));
      createdMap.add("MatchBool", "created %s".format(st.attrib(rMatchBoolName)));
      createdMap.add("MatchInd", "created %s".format(st.attrib(rMatchScanName)));
      createdMap.add("FullMatchBool", "created %s".format(st.attrib(rfullMatchBoolName)));
      createdMap.add("FullMatchInd", "created %s".format(st.attrib(rfullMatchScanName)));
      repMsg = "%jt".format(createdMap);
    }
    else {
      var errorMsg = "%s".format(objtype);
      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedFindAllMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("parent_name");
    const numMatchesName = msgArgs.getValueOf("num_matches");
    const startsName = msgArgs.getValueOf("starts");
    const lensName = msgArgs.getValueOf("lengths");
    const indicesName = msgArgs.getValueOf("indices");
    const returnMatchOrig = msgArgs.get("rtn_origins").getBoolValue();

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

  proc segmentedSubMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("obj");
    const repl = msgArgs.getValueOf("repl");
    const returnNumSubs: bool = msgArgs.get("rtn_num_subs").getBoolValue();
    const count = msgArgs.get("count").getIntValue();
    const pattern = msgArgs.getValueOf("pattern");

    // check to make sure symbols defined
    st.checkTable(name);

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

  proc segmentedStripMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    var objtype = msgArgs.getValueOf("objType");
    var name = msgArgs.getValueOf("name");

    // check to make sure symbols defined
    st.checkTable(name);

    select (objtype) {
      when ("str") {
        var strings = getSegString(name, st);
        var (off, val) = strings.strip(msgArgs.getValueOf("chars"));
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
      otherwise {
          var errorMsg = notImplementedError(pn, "%s".format(objtype));
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  proc createPeelSymEntries(lo, lv, ro, rv, st: borrowed SymTab) throws {
    var leftEntry = getSegString(lo, lv, st);
    var rightEntry = getSegString(ro, rv, st);
    return (leftEntry.name, rightEntry.name);
  }

  proc segmentedPeelMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const subcmd = msgArgs.getValueOf("subcmd");
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("obj");
    const valtype = msgArgs.getValueOf("valType");
    const times = msgArgs.get("times").getIntValue();
    const includeDelimiter = msgArgs.get("id").getBoolValue();
    const keepPartial = msgArgs.get("keepPartial").getBoolValue();
    const left = msgArgs.get("lStr").getBoolValue();
    const regex = msgArgs.get("regex").getBoolValue();
    const val = msgArgs.getValueOf("delim");


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

  proc segmentedHashMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("obj");

    // check to make sure symbols defined
    st.checkTable(name);

    select objtype {
        when "str" {
            var strings = getSegString(name, st);
            var hashes = strings.siphash();
            var name1 = st.nextName();
            var hash1 = st.addEntry(name1, hashes.size, uint);
            var name2 = st.nextName();
            var hash2 = st.addEntry(name2, hashes.size, uint);
            forall (h, h1, h2) in zip(hashes, hash1.a, hash2.a) {
                (h1,h2) = h:(uint,uint);
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
  proc segmentedIndexMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    // 'subcmd' is the type of indexing to perform
    // 'objtype' is the type of segmented array
    const subcmd = msgArgs.getValueOf("subcmd");
    const objtype = msgArgs.getValueOf("objType");
    const name = msgArgs.getValueOf("obj");
    const dtype = str2dtype(msgArgs.getValueOf("dtype"));
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "subcmd: %s objtype: %s".format(subcmd,objtype));
    try {
        select subcmd {
            when "intIndex" {
                return segIntIndex(objtype, name, msgArgs.getValueOf("key"), dtype, st);
            }
            when "sliceIndex" {
                var slice = msgArgs.get("key").getList(3);
                return segSliceIndex(objtype, name, slice, dtype, st);
            }
            when "pdarrayIndex" {
                return segPdarrayIndex(objtype, name, msgArgs.getValueOf("key"), dtype, st);
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
  proc segIntIndex(objtype: string, objName: string, key: string, dtype: DType,
                                         st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();

      // check to make sure symbols defined
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "strName: %s".format(objName));
      st.checkTable(objName);
      
      select objtype {
          when "str" {
              // Make a temporary strings array
              var strings = getSegString(objName, st);
              // Parse the index
              var idx = key:int;
              // TO DO: in the future, we will force the client to handle this
              idx = convertPythonIndexToChapel(idx, strings.size);
              var s = strings[idx];

              var repMsg = "item %s %jt".format("str", s);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
              return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when "SegArray" {
            var rname = st.nextName();
            const idx = key: int;  // negative indexes already handled by the client
            select (dtype) {
              when (DType.Int64) { 
                var segarr = getSegArray(objName, st, int);
                var s = segarr[idx];
                st.addEntry(rname, new shared SymEntry(s));
              }
              when (DType.UInt64) { 
                var segarr = getSegArray(objName, st, uint);
                var s = segarr[idx];
                st.addEntry(rname, new shared SymEntry(s));
              }
              when (DType.Float64) { 
                var segarr = getSegArray(objName, st, real);
                var s = segarr[idx];
                st.addEntry(rname, new shared SymEntry(s));
              }
              when (DType.Bool) { 
                var segarr = getSegArray(objName, st, bool);
                var s = segarr[idx];
                st.addEntry(rname, new shared SymEntry(s));
              }
              otherwise {
                var errorMsg = "Unsupported SegArray DType %s".format(dtype2str(dtype));
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return new MsgTuple(errorMsg, MsgType.ERROR);
              }
            }
            var repMsg = "created " + st.attrib(rname);
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

  proc segSliceIndex(objtype: string, objName: string, key: [] string, dtype: DType,
                                         st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    // check to make sure symbols defined
    st.checkTable(objName);

    // Parse the slice parameters
    var start = key[0]:int;
    var stop = key[1]:int;
    var stride = key[2]:int;

    // Only stride-1 slices are allowed for now
    if (stride != 1) { 
        var errorMsg = notImplementedError(pn, "stride != 1"); 
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    // TO DO: in the future, we will force the client to handle this
    var slice = convertPythonSliceToChapel(start, stop);

    select objtype {
        when "str" {
            // Make a temporary string array
            var strings = getSegString(objName, st);

            // Compute the slice
            var (newSegs, newVals) = strings[slice];
            // Store the resulting offsets and bytes arrays
            var newStringsObj = getSegString(newSegs, newVals, st);
            repMsg = "created " + st.attrib(newStringsObj.name) + "+created bytes.size %t".format(newStringsObj.nBytes);
        }
        when "SegArray" {
          var rtnmap: map(string, string);
          select dtype {
            when (DType.Int64) { 
              var segarr = getSegArray(objName, st, int);
              // Compute the slice
              var (newSegs, newVals) = segarr[slice];
              var newSegArr = getSegArray(newSegs, newVals, st);
              newSegArr.fillReturnMap(rtnmap, st);
            }
            when (DType.UInt64) { 
              var segarr = getSegArray(objName, st, uint);
              // Compute the slice
              var (newSegs, newVals) = segarr[slice];
              var newSegArr = getSegArray(newSegs, newVals, st);
              newSegArr.fillReturnMap(rtnmap, st);
            }
            when (DType.Float64) { 
              var segarr = getSegArray(objName, st, real);
              // Compute the slice
              var (newSegs, newVals) = segarr[slice];
              var newSegArr = getSegArray(newSegs, newVals, st);
              newSegArr.fillReturnMap(rtnmap, st);
            }
            when (DType.Bool) { 
              var segarr = getSegArray(objName, st, bool);
              // Compute the slice
              var (newSegs, newVals) = segarr[slice];
              var newSegArr = getSegArray(newSegs, newVals, st);
              newSegArr.fillReturnMap(rtnmap, st);
            }
            otherwise {
              var errorMsg = "Invalid segarray type";
                      smLogger.error(getModuleName(),getRoutineName(),
                                                    getLineNumber(),errorMsg); 
                      return new MsgTuple(notImplementedError(pn,errorMsg), MsgType.ERROR);
            }
          }
          repMsg = "%jt".format(rtnmap);
        }
        otherwise {
            var errorMsg = notImplementedError(pn, objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);          
        }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc convertPythonSliceToChapel(start:int, stop:int): range(stridable=false) {
    if (start <= stop) {
      return start..(stop-1);
    } else {
      return 1..0;
    }
  }

  proc segPdarrayIndex(objtype: string, objName: string, iname: string, dtype: DType,
                       st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    // check to make sure symbols defined
    st.checkTable(objName);
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "objtype:%s".format(objtype));

    var gIV: borrowed GenSymEntry = getGenericTypedArrayEntry(iname, st);
    
    select objtype {
        when "str" {
            var newStringsName = "";
            var nBytes = 0;
            var strings = getSegString(objName, st);
            try {
                select gIV.dtype {
                    when DType.Int64 {
                        var iv = toSymEntry(gIV, int);
                        var (newSegs, newVals) = strings[iv.a];
                        var newStringsObj = getSegString(newSegs, newVals, st);
                        newStringsName = newStringsObj.name;
                        nBytes = newStringsObj.nBytes;
                        repMsg = "created " + st.attrib(newStringsName) + "+created bytes.size %t".format(nBytes);
                    }
                    when DType.UInt64 {
                        var iv = toSymEntry(gIV, uint);
                        var (newSegs, newVals) = strings[iv.a];
                        var newStringsObj = getSegString(newSegs, newVals, st);
                        newStringsName = newStringsObj.name;
                        nBytes = newStringsObj.nBytes;
                        repMsg = "created " + st.attrib(newStringsName) + "+created bytes.size %t".format(nBytes);
                    } 
                    when DType.Bool {
                        var iv = toSymEntry(gIV, bool);
                        var (newSegs, newVals) = strings[iv.a];
                        var newStringsObj = getSegString(newSegs, newVals, st);
                        newStringsName = newStringsObj.name;
                        nBytes = newStringsObj.nBytes;
                        repMsg = "created " + st.attrib(newStringsName) + "+created bytes.size %t".format(nBytes);
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
        when "SegArray" {
          var rtnmap: map(string, string);
          select dtype {
            when DType.Int64 {
              var segArr = getSegArray(objName, st, int);
              select gIV.dtype {
                when DType.Int64 {
                  var iv = toSymEntry(gIV, int);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.UInt64 {
                  var iv = toSymEntry(gIV, uint);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.Bool {
                  var iv = toSymEntry(gIV, bool);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                otherwise {
                    var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                    smLogger.error(getModuleName(),getRoutineName(),
                                                  getLineNumber(),errorMsg); 
                    return new MsgTuple(notImplementedError(pn,errorMsg), MsgType.ERROR);
                }
              }
            }
            when DType.UInt64 {
              var segArr = getSegArray(objName, st, uint);
              select gIV.dtype {
                when DType.Int64 {
                  var iv = toSymEntry(gIV, int);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.UInt64 {
                  var iv = toSymEntry(gIV, uint);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.Bool {
                  var iv = toSymEntry(gIV, bool);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                otherwise {
                    var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                    smLogger.error(getModuleName(),getRoutineName(),
                                                  getLineNumber(),errorMsg); 
                    return new MsgTuple(notImplementedError(pn,errorMsg), MsgType.ERROR);
                }
              }
            }
            when DType.Float64 {
              var segArr = getSegArray(objName, st, real);
              select gIV.dtype {
                when DType.Int64 {
                  var iv = toSymEntry(gIV, int);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.UInt64 {
                  var iv = toSymEntry(gIV, uint);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.Bool {
                  var iv = toSymEntry(gIV, bool);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                otherwise {
                    var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                    smLogger.error(getModuleName(),getRoutineName(),
                                                  getLineNumber(),errorMsg); 
                    return new MsgTuple(notImplementedError(pn,errorMsg), MsgType.ERROR);
                }
              }
            }
            when DType.Bool {
              var segArr = getSegArray(objName, st, bool);
              select gIV.dtype {
                when DType.Int64 {
                  var iv = toSymEntry(gIV, int);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.UInt64 {
                  var iv = toSymEntry(gIV, uint);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                when DType.Bool {
                  var iv = toSymEntry(gIV, bool);
                  var (newSegs, newVals) = segArr[iv.a];
                  var newSegArr = getSegArray(newSegs, newVals, st);
                  newSegArr.fillReturnMap(rtnmap, st);
                }
                otherwise {
                    var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                    smLogger.error(getModuleName(),getRoutineName(),
                                                  getLineNumber(),errorMsg); 
                    return new MsgTuple(notImplementedError(pn,errorMsg), MsgType.ERROR);
                }
              }
            }
            otherwise {
                var errorMsg = notImplementedError(pn, objtype);
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return new MsgTuple(errorMsg, MsgType.ERROR);          
            }
          }
          repMsg = "%jt".format(rtnmap);
        }
        otherwise {
            var errorMsg = "unsupported objtype: %t".format(objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(notImplementedError(pn, objtype), MsgType.ERROR);
        }
    }

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segBinopvvMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    const op = msgArgs.getValueOf("op");
    const ltype = msgArgs.getValueOf("objType");
    const leftName = msgArgs.getValueOf("obj");
    const rtype = msgArgs.getValueOf("otherType");
    const rightName = msgArgs.getValueOf("other");

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
                    const left = msgArgs.get("left").getBoolValue();
                    const delim = msgArgs.getValueOf("delim");
                    var strings:SegString;
                    if left {
                        var (newOffsets, newVals) = lstrings.stick(rstrings, delim, false);
                        strings = getSegString(newOffsets, newVals, st);
                    } else {
                        var (newOffsets, newVals) = lstrings.stick(rstrings, delim, true);
                        strings = getSegString(newOffsets, newVals, st);
                    }
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

  proc segBinopvsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      const op = msgArgs.getValueOf("op");
      const objtype = msgArgs.getValueOf("objType");
      const name = msgArgs.getValueOf("obj");
      const valtype = msgArgs.getValueOf("otherType");
      const value = msgArgs.getValueOf("other");

      // check to make sure symbols defined
      st.checkTable(name);
      
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

  proc segIn1dMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      const mainObjtype = msgArgs.getValueOf("objType");
      const mainName = msgArgs.getValueOf("obj");
      const testObjtype = msgArgs.getValueOf("otherType");
      const testName = msgArgs.getValueOf("other");
      const invert = msgArgs.get("invert").getBoolValue();

      // check to make sure symbols defined
      st.checkTable(mainName);
      st.checkTable(testName);
    
      var rname = st.nextName();
 
      select (mainObjtype, testObjtype) {
          when ("str", "str") {
              var mainStr = getSegString(mainName, st);
              var testStr = getSegString(testName, st);
              var e = st.addEntry(rname, mainStr.size, bool);
              e.a = in1d(mainStr, testStr, invert);
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

  proc segGroupMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      const objtype = msgArgs.getValueOf("objType");
      const name = msgArgs.getValueOf("obj");

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

  proc stringsToJSONMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var strings = getSegString(msgArgs.getValueOf("name"), st);
    var size = strings.size;
    var rtn: [0..#size] string;

    forall i in 0..#size {
      rtn[i] = strings[i];
    }

    var repMsg = "%jt".format(rtn);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedSubstringMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    var objtype = msgArgs.getValueOf("objType");
    var name = msgArgs.getValueOf("name");
    
    // check to make sure symbols defined
    st.checkTable(name);

    select (objtype) {
      when ("str") {
        var strings = getSegString(name, st);
        var returnOrigins = msgArgs.get("returnOrigins").getBoolValue();
        var (off, byt, longEnough) = strings.getFixes(msgArgs.get("nChars").getIntValue(),
                                                      msgArgs.getValueOf("kind"): Fixes,
                                                      msgArgs.get("proper").getBoolValue());
        var retString = getSegString(off, byt, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
        if returnOrigins {
          var leName = st.nextName();
          st.addEntry(leName, new shared SymEntry(longEnough));
          repMsg += "+created " + st.attrib(leName);
        } 
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }
      otherwise {
          var errorMsg = notImplementedError(pn, "%s".format(objtype));
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }
  
  use CommandMap;
  registerFunction("segmentLengths", segmentLengthsMsg, getModuleName());
  registerFunction("caseChange", caseChangeMsg, getModuleName());
  registerFunction("checkChars", checkCharsMsg, getModuleName());
  registerFunction("segmentedHash", segmentedHashMsg, getModuleName());
  registerFunction("segmentedSearch", segmentedSearchMsg, getModuleName());
  registerFunction("segmentedFindLoc", segmentedFindLocMsg, getModuleName());
  registerFunction("segmentedFindAll", segmentedFindAllMsg, getModuleName());
  registerFunction("segmentedPeel", segmentedPeelMsg, getModuleName());
  registerFunction("segmentedSub", segmentedSubMsg, getModuleName());
  registerFunction("segmentedStrip", segmentedStripMsg, getModuleName());
  registerFunction("segmentedIndex", segmentedIndexMsg, getModuleName());
  registerFunction("segmentedBinopvv", segBinopvvMsg, getModuleName());
  registerFunction("segmentedBinopvs", segBinopvsMsg, getModuleName());
  registerFunction("segmentedGroup", segGroupMsg, getModuleName());
  registerFunction("segmentedIn1d", segIn1dMsg, getModuleName());
  registerFunction("randomStrings", randomStringsMsg, getModuleName());
  registerFunction("segArr-assemble", assembleSegArrayMsg, getModuleName());
  registerFunction("segArr-getNonEmpty", getSANonEmptyMsg, getModuleName());
  registerFunction("segStr-assemble", assembleStringsMsg, getModuleName());
  registerFunction("stringsToJSON", stringsToJSONMsg, getModuleName());
  registerBinaryFunction("segStr-tondarray", segStrTondarrayMsg, getModuleName());
  registerFunction("segmentedSubstring", segmentedSubstringMsg, getModuleName());
}
