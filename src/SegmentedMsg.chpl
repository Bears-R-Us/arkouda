module SegmentedMsg {
  use Reflection;
  use ServerErrors;
  use Logging;
  use Message;
  use SegmentedString;
  use ServerErrorStrings;
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use RandArray;
  use IO;
  use GenSymIO;
  use BigInteger;
  use Math;
  use HashUtils;
  use Map;
  use CTypes;
  use IOUtils;
  use CommAggregation;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const smLogger = new Logger(logLevel, logChannel);

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
    var repMsg = "".join("created ", st.attrib(segString.name), "+created bytes.size ", segString.nBytes:string);
    smLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segStrTondarrayMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var entry = getSegString(msgArgs.getValueOf("obj"), st);
      const comp = msgArgs.getValueOf("comp");
      if comp == "offsets" {
          return MsgTuple.payload(tondarrayMsg(entry.offsets));
      } else if (comp == "values") {
          return MsgTuple.payload(tondarrayMsg(entry.values));
      } else {
          const msg = "Unrecognized component: %s".format(comp);
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(), msg);
          return MsgTuple.error(msg);
      }
  }

  /*
     * Outputs the pdarray as a Numpy ndarray in the form of a 
     * Chapel Bytes object
     */
    proc tondarrayMsg(entry): bytes throws {
        var arrayBytes: bytes;

        proc distArrToBytes(A: [?D] ?eltType) {
            var ptr = allocate(eltType, D.size);
            var localA = makeArrayFromPtr(ptr, D.size:uint);
            localA = A;
            const size = D.size*c_sizeof(eltType):int;
            return bytes.createAdoptingBuffer(ptr:c_ptr(uint(8)), size, size);
        }

        if entry.dtype == DType.Int64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, int).a);
        } else if entry.dtype == DType.Float64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, real).a);
        } else if entry.dtype == DType.Bool {
            arrayBytes = distArrToBytes(toSymEntry(entry, bool).a);
        } else if entry.dtype == DType.UInt8 {
            arrayBytes = distArrToBytes(toSymEntry(entry, uint(8)).a);
        } else if entry.dtype == DType.BigInt {
            arrayBytes = distArrToBytes(toSymEntry(entry, bigint).a);
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
              repMsg = 'created ' + st.attrib(strings.name) + '+created bytes.size %?'.format(strings.nBytes);
          }
          when "lognormal" {
              var logMean = msgArgs.get("arg1").getRealValue();
              var logStd = msgArgs.get("arg2").getRealValue();
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + exp(logMean + (logStd**2)/2):int*len);
              var (segs, vals) = newRandStringsLogNormalLength(len, logMean, logStd, charset, seedStr);
              var strings = getSegString(segs, vals, st);
              repMsg = 'created ' + st.attrib(strings.name) + '+created bytes.size %?'.format(strings.nBytes);
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
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    const name = msgArgs.getValueOf("obj");

    // check to make sure symbols defined
    st.checkTable(name);
    
    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s objtype: %? name: %?".format(
                   cmd,objtype,name));

    select objtype {
      when ObjType.STRINGS {
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

  proc getSegStringPropertyMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    const property = msgArgs.getValueOf("property");
    const name = msgArgs.getValueOf("obj");

    var genSym = toGenSymEntry(st[name]);

    if genSym.dtype != DType.Strings{
      var errorMsg = notImplementedError(pn, "%s".format(dtype2str(genSym.dtype)));
      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                      
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }
    
    var ssentry = toSegStringSymEntry(genSym);
    var rname = st.nextName();
    select property{
      when "get_bytes" {
        st.addEntry(rname, ssentry.bytesEntry);
      }
      when "get_offsets" {
        st.addEntry(rname, ssentry.offsetsEntry);
      }
      otherwise {
        var errorMsg = notImplementedError(pn,property);
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
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    const name = msgArgs.getValueOf("obj");


    // check to make sure symbols defined
    st.checkTable(name);

    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s objtype: %? name: %?".format(cmd,objtype,name));

    select objtype {
      when ObjType.STRINGS {
        var strings = getSegString(name, st);
        select subcmd {
          when "toLower" {
            var (off, val) = strings.lower();
            var retString = getSegString(off, val, st);
            repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
          }
          when "toUpper" {
            var (off, val) = strings.upper();
            var retString = getSegString(off, val, st);
            repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
          }
          when "toTitle" {
            var (off, val) = strings.title();
            var retString = getSegString(off, val, st);
            repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
          }
          when "capitalize" {
            var (off, val) = strings.capitalize();
            var retString = getSegString(off, val, st);
            repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
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
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    const name = msgArgs.getValueOf("obj");

    // check to make sure symbols defined
    st.checkTable(name);

    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s objtype: %? name: %?".format(cmd,objtype,name));

    select objtype {
      when ObjType.STRINGS {
        var strings = getSegString(name, st);
        var truth = st.addEntry(rname, strings.size, bool);
        select subcmd {
          when "isDecimal" {
            truth.a = strings.isDecimal();
            repMsg = "created "+st.attrib(rname);
          }
          when "isNumeric" {
            truth.a = strings.isNumeric();
            repMsg = "created "+st.attrib(rname);
          }
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
          when "isalnum" {
            truth.a = strings.isalnum();
            repMsg = "created "+st.attrib(rname);
          }
          when "isalpha" {
            truth.a = strings.isalpha();
            repMsg = "created "+st.attrib(rname);
          }
          when "isdigit" {
            truth.a = strings.isdigit();
            repMsg = "created "+st.attrib(rname);
          }
          when "isempty" {
            truth.a = strings.isempty();
            repMsg = "created "+st.attrib(rname);
          }
          when "isspace" {
            truth.a = strings.isspace();
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
      const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
      const name = msgArgs.getValueOf("obj");
      const valtype = msgArgs.getValueOf("valType");
      const val = msgArgs.getValueOf("val");

      // check to make sure symbols defined
      st.checkTable(name);
      var rname = st.nextName();
    
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s objtype: %? valtype: %?".format(
                          cmd,objtype,valtype));
    
      select (objtype, valtype) {
          when (ObjType.STRINGS, "str") {
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
                   "cmd: %s objtype: %?".format(cmd, objtype));

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
      st.addEntry(rNumMatchesName, createSymEntry(numMatches));
      st.addEntry(rStartsName, createSymEntry(matchStarts));
      st.addEntry(rLensName, createSymEntry(matchLens));
      st.addEntry(rIndicesName, createSymEntry(matchesIndices));
      st.addEntry(rSearchBoolName, createSymEntry(searchBools));
      st.addEntry(rSearchScanName, createSymEntry(searchScan));
      st.addEntry(rMatchBoolName, createSymEntry(matchBools));
      st.addEntry(rMatchScanName, createSymEntry(matchScan));
      st.addEntry(rfullMatchBoolName, createSymEntry(fullMatchBools));
      st.addEntry(rfullMatchScanName, createSymEntry(fullMatchScan));

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
      repMsg = formatJson(createdMap);
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
                   "cmd: %s objtype: %?".format(cmd, objtype));

    select objtype {
      when "Matcher" {
        const optName: string = if returnMatchOrig then st.nextName() else "";
        var strings = getSegString(name, st);
        var numMatches = getGenericTypedArrayEntry(numMatchesName, st): borrowed SymEntry(int, 1);
        var starts = getGenericTypedArrayEntry(startsName, st): borrowed SymEntry(int, 1);
        var lens = getGenericTypedArrayEntry(lensName, st): borrowed SymEntry(int, 1);
        var indices = getGenericTypedArrayEntry(indicesName,st): borrowed SymEntry(int, 1);

        var (off, val, matchOrigins) = strings.findAllMatches(numMatches, starts, lens, indices, returnMatchOrig);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
        if returnMatchOrig {
          const optName: string = if returnMatchOrig then st.nextName() else "";
          st.addEntry(optName, createSymEntry(matchOrigins));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      }
      when "Match" {
        const optName: string = if returnMatchOrig then st.nextName() else "";
        var strings = getSegString(name, st);
        // numMatches is the matched boolean array for Match objects
        var numMatches = st[numMatchesName]: borrowed SymEntry(bool, 1);
        var starts = st[startsName]: borrowed SymEntry(int, 1);
        var lens = st[lensName]: borrowed SymEntry(int, 1);
        var indices = st[indicesName]: borrowed SymEntry(int, 1);

        var (off, val, matchOrigins) = strings.findAllMatches(numMatches, starts, lens, indices, returnMatchOrig);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
        if returnMatchOrig {
          st.addEntry(optName, createSymEntry(matchOrigins));
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
                   "cmd: %s objtype: %?".format(cmd, objtype));

    select objtype {
      when "Matcher" {
        const optName: string = if returnNumSubs then st.nextName() else "";
        const strings = getSegString(name, st);
        var (off, val, numSubs) = strings.sub(pattern, repl, count, returnNumSubs);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
        if returnNumSubs {
          st.addEntry(optName, createSymEntry(numSubs));
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

    var objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    var name = msgArgs.getValueOf("name");

    // check to make sure symbols defined
    st.checkTable(name);

    select (objtype) {
      when (ObjType.STRINGS) {
        var strings = getSegString(name, st);
        var (off, val) = strings.strip(msgArgs.getValueOf("chars"));
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
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
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
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
                         "cmd: %s subcmd: %s objtype: %? valtype: %?".format(
                          cmd,subcmd,objtype,valtype));

    select (objtype, valtype) {
    when (ObjType.STRINGS, "str") {
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
          repMsg = "created %s+created bytes.size %?+created %s+created bytes.size %?".format(
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
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    select objtype {
        when ObjType.STRINGS {
            const name = msgArgs.getValueOf("obj");
            // check to make sure symbols defined
            st.checkTable(name);
            var strings = getSegString(name, st);
            var hashes = strings.siphash();
            var name1 = st.nextName();
            var hash1 = st.addEntry(name1, hashes.size, uint);
            var name2 = st.nextName();
            var hash2 = st.addEntry(name2, hashes.size, uint);
            forall (h, h1, h2) in zip(hashes, hash1.a, hash2.a) {
                (h1,h2) = h:(uint,uint);
            }
            repMsg = "created " + st.attrib(name1) + "+created " + st.attrib(name2);
        }
        when ObjType.SEGARRAY {
            // check to make sure symbols defined
            const segName = msgArgs.getValueOf("segments");
            const valName = msgArgs.getValueOf("values");
            const valObjType = msgArgs.getValueOf("valObjType");
            st.checkTable(segName);
            st.checkTable(valName);
            var (upper, lower) = segarrayHash(segName, valName, valObjType, st);
            var upperName = st.nextName();
            st.addEntry(upperName, createSymEntry(upper));
            var lowerName = st.nextName();
            st.addEntry(lowerName, createSymEntry(lower));
            repMsg = "created %s+created %s".format(st.attrib(upperName), st.attrib(lowerName));
        }
        otherwise {
            var errorMsg = notImplementedError(pn, objtype: string);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
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
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
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
        var errorMsg = "unknown cause %?".format(e);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }
  }
 
  /*
  Returns the object corresponding to the index
  */ 
  proc segIntIndex(objtype: ObjType, objName: string, key: string, dtype: DType,
                                         st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();

      // check to make sure symbols defined
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "strName: %s".format(objName));
      st.checkTable(objName);
      
      select objtype {
          when ObjType.STRINGS {
              // Make a temporary strings array
              var strings = getSegString(objName, st);
              // Parse the index
              var idx = key:int;
              // TO DO: in the future, we will force the client to handle this
              idx = convertPythonIndexToChapel(idx, strings.size);
              var s = strings[idx];

              var repMsg = "item str "+formatJson(s);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
              return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          otherwise { 
              var errorMsg = notImplementedError(pn, objtype: string); 
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

  proc segSliceIndex(objtype: ObjType, objName: string, key: [] string, dtype: DType,
                                         st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    // check to make sure symbols defined
    st.checkTable(objName);

    // Parse the slice parameters
    var start = key[0]:int;
    var stop = key[1]:int;
    var stride = key[2]:int;

    select objtype {
        when ObjType.STRINGS {
            // Make a temporary string array
            var strings = getSegString(objName, st);

            // Compute the slice
            var (newSegs, newVals) = if stride == 1 then strings[convertPythonSliceToChapel(start, stop)] else strings[convertSliceToStridableRange(start, stop, stride)];
            // Store the resulting offsets and bytes arrays
            var newStringsObj = getSegString(newSegs, newVals, st);
            repMsg = "created " + st.attrib(newStringsObj.name) + "+created bytes.size %?".format(newStringsObj.nBytes);
        }
        otherwise {
            var errorMsg = notImplementedError(pn, objtype: string);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);          
        }
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc convertPythonSliceToChapel(start:int, stop:int): range() {
    if (start <= stop) {
      return start..(stop-1);
    } else {
      return 1..0;
    }
  }


  proc convertSliceToStridableRange(start: int, stop: int, stride: int): range(strides=strideKind.any) {
    var slice: range(strides=strideKind.any);
    // backwards iteration with negative stride
    if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
    // forward iteration with positive stride
    else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
    // BAD FORM start < stop and stride is negative
    else {slice = 1..0;}
    return slice;
  }

  proc segPdarrayIndex(objtype: ObjType, objName: string, iname: string, dtype: DType,
                       st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    // check to make sure symbols defined
    st.checkTable(objName);
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "objtype:%s".format(objtype));

    var gIV: borrowed GenSymEntry = getGenericTypedArrayEntry(iname, st);
    
    select objtype {
        when ObjType.STRINGS {
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
                        repMsg = "created " + st.attrib(newStringsName) + "+created bytes.size %?".format(nBytes);
                    }
                    when DType.UInt64 {
                        var iv = toSymEntry(gIV, uint);
                        var (newSegs, newVals) = strings[iv.a];
                        var newStringsObj = getSegString(newSegs, newVals, st);
                        newStringsName = newStringsObj.name;
                        nBytes = newStringsObj.nBytes;
                        repMsg = "created " + st.attrib(newStringsName) + "+created bytes.size %?".format(nBytes);
                    } 
                    when DType.Bool {
                        var iv = toSymEntry(gIV, bool);
                        var (newSegs, newVals) = strings[iv.a];
                        var newStringsObj = getSegString(newSegs, newVals, st);
                        newStringsName = newStringsObj.name;
                        nBytes = newStringsObj.nBytes;
                        repMsg = "created " + st.attrib(newStringsName) + "+created bytes.size %?".format(nBytes);
                    }
                    otherwise {
                        var errorMsg = "("+objtype: string+","+dtype2str(gIV.dtype)+")";
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
            var errorMsg = "unsupported objtype: %?".format(objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(notImplementedError(pn, objtype: string), MsgType.ERROR);
        }
    }

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segBinopvvMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    const op = msgArgs.getValueOf("op");
    const ltype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    const leftName = msgArgs.getValueOf("obj");
    const rtype = msgArgs.getValueOf("otherType").toUpper(): ObjType;
    const rightName = msgArgs.getValueOf("other");

    // check to make sure symbols defined
    st.checkTable(leftName);
    st.checkTable(rightName);

    select (ltype, rtype) {
        when (ObjType.STRINGS, ObjType.STRINGS) {
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
                    repMsg = "created %s+created bytes.size %?".format(st.attrib(strings.name), strings.nBytes);
                    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                }
                otherwise {
                    var errorMsg = notImplementedError(pn, ltype: string, op, rtype: string);
                    smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
              }
           }
       otherwise {
           var errorMsg = unrecognizedTypeError(pn, "("+ltype: string+", "+rtype: string+")");
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
      const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
      const name = msgArgs.getValueOf("obj");
      const valtype = msgArgs.getValueOf("otherType");
      const value = msgArgs.getValueOf("other");

      // check to make sure symbols defined
      st.checkTable(name);
      
      var rname = st.nextName();

      select (objtype, valtype) {
          when (ObjType.STRINGS, "str") {
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
                      var errorMsg = notImplementedError(pn, objtype: string, op, valtype);
                      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
              }
          }
          otherwise {
              var errorMsg = unrecognizedTypeError(pn, "("+objtype: string+", "+valtype+")");
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
      const mainObjtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
      const mainName = msgArgs.getValueOf("obj");
      const testObjtype = msgArgs.getValueOf("otherType").toUpper(): ObjType;
      const testName = msgArgs.getValueOf("other");
      const invert = msgArgs.get("invert").getBoolValue();

      // check to make sure symbols defined
      st.checkTable(mainName);
      st.checkTable(testName);
    
      var rname = st.nextName();
 
      select (mainObjtype, testObjtype) {
          when (ObjType.STRINGS, ObjType.STRINGS) {
              var mainStr = getSegString(mainName, st);
              var testStr = getSegString(testName, st);
              var e = st.addEntry(rname, mainStr.size, bool);
              e.a = in1d(mainStr, testStr, invert);
          }
          otherwise {
              var errorMsg = unrecognizedTypeError(pn, "("+mainObjtype: string+", "+testObjtype: string+")");
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
      const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
      const name = msgArgs.getValueOf("obj");

      // check to make sure symbols defined
      st.checkTable(name);
      
      var rname = st.nextName();
      select (objtype) {
          when ObjType.STRINGS {
              var strings = getSegString(name, st);
              var iv = st.addEntry(rname, strings.size, int);
              iv.a = strings.argGroup();
          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype: string+")");
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

    var repMsg = formatJson(rtn);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedSubstringMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    var objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    var name = msgArgs.getValueOf("name");
    
    // check to make sure symbols defined
    st.checkTable(name);

    select (objtype) {
      when (ObjType.STRINGS) {
        var strings = getSegString(name, st);
        var returnOrigins = msgArgs.get("returnOrigins").getBoolValue();
        var (off, byt, longEnough) = strings.getFixes(msgArgs.get("nChars").getIntValue(),
                                                      msgArgs.getValueOf("kind"): Fixes,
                                                      msgArgs.get("proper").getBoolValue());
        var retString = getSegString(off, byt, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
        if returnOrigins {
          var leName = st.nextName();
          st.addEntry(leName, createSymEntry(longEnough));
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

  proc segmentedWhereMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var repMsg: string;
    const segStrName = msgArgs.getValueOf("seg_str");
    const other = msgArgs.getValueOf("other");
    const conditionName = msgArgs.getValueOf("condition");
    const newLensName = msgArgs.getValueOf("new_lens");
    const isStrLiteral = msgArgs.get("is_str_literal").getBoolValue();
    // check to make sure symbols defined
    st.checkTable(segStrName);
    st.checkTable(conditionName);
    st.checkTable(newLensName);
    if !isStrLiteral {
      st.checkTable(other);
    }

    const strings = getSegString(segStrName, st);
    ref condition = toSymEntry(getGenericTypedArrayEntry(conditionName, st), bool).a;
    ref newLens = toSymEntry(getGenericTypedArrayEntry(newLensName, st), int).a;

    var (off, val) = if isStrLiteral then strings.segStrWhere(other, condition, newLens) else strings.segStrWhere(getSegString(other, st), condition, newLens);
    var retString = getSegString(off, val, st);
    repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedFullMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var repMsg: string;
    const segStrSize = msgArgs.getValueOf("size"): int;
    const segStrFillValue = msgArgs.getValueOf("fill_value");

    var (off, val) = segStrFull(segStrSize, segStrFillValue);
    var retString = getSegString(off, val, st);
    repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc flipStringMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    const name = msgArgs.getValueOf("obj");

    // check to make sure symbols defined
    st.checkTable(name);
    var strings = getSegString(name, st);
    ref origVals = strings.values.a;
    ref offs = strings.offsets.a;
    const lengths = strings.getLengths();

    ref retOffs = makeDistArray(lengths.domain, int);
    forall i in lengths.domain with (var valAgg = newDstAggregator(int)) {
      valAgg.copy(retOffs[lengths.domain.high - i], lengths[i]);
    }
    retOffs = (+ scan retOffs) - retOffs;

    var flippedVals = makeDistArray(strings.values.a.domain, uint(8));
    forall (off, len, j) in zip(offs, lengths, 0..) with (var valAgg = newDstAggregator(uint(8))) {
      var i = 0;
      for b in interpretAsBytes(origVals, off..#len, borrow=true) {
        valAgg.copy(flippedVals[retOffs[lengths.domain.high - j] + i], b:uint(8));
        i += 1;
      }
    }
    var retString = getSegString(retOffs, flippedVals, st);
    var repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
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
  registerFunction("segStr-assemble", assembleStringsMsg, getModuleName());
  registerFunction("stringsToJSON", stringsToJSONMsg, getModuleName());
  registerFunction("segStr-tondarray", segStrTondarrayMsg, getModuleName());
  registerFunction("segmentedSubstring", segmentedSubstringMsg, getModuleName());
  registerFunction("segmentedWhere", segmentedWhereMsg, getModuleName());
  registerFunction("segmentedFull", segmentedFullMsg, getModuleName());
  registerFunction("getSegStringProperty", getSegStringPropertyMsg, getModuleName());
  registerFunction("flipString", flipStringMsg, getModuleName());
}
