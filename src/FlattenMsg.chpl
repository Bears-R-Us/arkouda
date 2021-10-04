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
        const optName: string = if returnSegs then st.nextName() else "";
        var (segName, valName) = name.splitMsgToTuple('+', 2);
        const strings = getSegString(segName, valName, st);
        var (off, val, segs) = strings.flatten(delim, returnSegs, regex);
        st.addEntry(rSegName, new shared SymEntry(off));
        st.addEntry(rValName, new shared SymEntry(val));
        repMsg = "created %s+created %s".format(st.attrib(rSegName), st.attrib(rValName));
        if returnSegs {
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
    var (objtype, segName, valName, maxsplitStr, returnSegsStr, patternJson) = payload.splitMsgToTuple(6);
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
    st.checkTable(segName);
    st.checkTable(valName);

    const json = jsonToPdArray(patternJson, 1);
    const pattern: string = json[json.domain.low];

    fmLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %t".format(cmd, objtype));

    select objtype {
      when "Matcher" {
        const rSegName = st.nextName();
        const rValName = st.nextName();
        const optName: string = if returnSegs then st.nextName() else "";
        const strings = getSegString(segName, valName, st);
        var (off, val, segs) = strings.split(pattern, maxsplit, returnSegs);
        st.addEntry(rSegName, new shared SymEntry(off));
        st.addEntry(rValName, new shared SymEntry(val));
        repMsg = "created %s+created %s".format(st.attrib(rSegName), st.attrib(rValName));
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
