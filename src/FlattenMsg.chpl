module FlattenMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerErrors;
  use Reflection;
  use Flatten;
  use ServerConfig;
  use SegmentedArray;
  use GenSymIO;
  use Logging;
  use Message;
  
  private config const logLevel = ServerConfig.logLevel;
  const fmLogger = new Logger(logLevel);

  proc segFlattenMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var (name, objtype, returnSegsStr, regexStr, delimJson) = payload.splitMsgToTuple(5);
    const returnSegs: bool = returnSegsStr.toLower() == "true";
    const regex: bool = regexStr.toLower() == "true";
    const arr = jsonToPdArray(delimJson, 1);
    const delim: string = arr[arr.domain.low];
    var repMsg: string;
    select objtype {
      when "str" {
        const rSegName = st.nextName();
        const rValName = st.nextName();
        var (stringsName, legacy_placeholder) = name.splitMsgToTuple('+', 2);
        const strings = getSegString(stringsName, st);
        var (off, val, segs) = strings.flatten(delim, returnSegs, regex);
        var stringsObj = getSegString(off, val, st);
        repMsg = "created %s+created bytes.size %t".format(st.attrib(stringsObj.name), stringsObj.nBytes);
        if returnSegs {
          const optName: string = st.nextName();
          st.addEntry(optName, new shared SymEntry(segs));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      } otherwise {
        throw new owned ErrorWithContext("Not implemented for objtype %s".format(objtype),
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "TypeError");
      }
    }

    fmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);         
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedSplitMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, name, legacy_placeholder, maxsplitStr, returnSegsStr, patternJson) = payload.splitMsgToTuple(6);
    const returnSegs: bool = returnSegsStr.toLower() == "true";
    var maxsplit: int;
    try {
      maxsplit = maxsplitStr:int;
    }
    catch {
      var errorMsg = "maxsplit could not be interpretted as an int: %s)".format(maxsplitStr);
      fmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      throw new owned IllegalArgumentError(errorMsg);
    }

    // check to make sure symbols defined
    st.checkTable(name);

    const json = jsonToPdArray(patternJson, 1);
    const pattern: string = json[json.domain.low];

    fmLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %t".format(cmd, objtype));

    select objtype {
      when "Matcher" {
        const optName: string = if returnSegs then st.nextName() else "";
        const strings = getSegString(name, st);
        var (off, val, segs) = strings.split(pattern, maxsplit, returnSegs);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
        if returnSegs {
          st.addEntry(optName, new shared SymEntry(segs));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      }
      otherwise {
        throw new owned ErrorWithContext("Not implemented for objtype %s".format(objtype),
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "TypeError");
      }
    }
    fmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }
}
