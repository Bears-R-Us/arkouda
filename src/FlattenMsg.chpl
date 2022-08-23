module FlattenMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerErrors;
  use Reflection;
  use Flatten;
  use ServerConfig;
  use SegmentedString;
  use Logging;
  use Message;
  
  private config const logLevel = ServerConfig.logLevel;
  const fmLogger = new Logger(logLevel);

  proc segFlattenMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var msgArgs = parseMessageArgs(payload, 5);
    const objtype = msgArgs.getValueOf("objtype");
    const returnSegs: bool = msgArgs.get("return_segs").getBoolValue();
    const regex: bool = msgArgs.get("regex").getBoolValue();
    const arr = msgArgs.get("delim").getList(1);
    const delim: string = arr[arr.domain.low];
    var repMsg: string;
    select objtype {
      when "str" {
        const rSegName = st.nextName();
        const rValName = st.nextName();
        const strings = getSegString(msgArgs.getValueOf("values"), st);
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
    var msgArgs = parseMessageArgs(payload, 5);

    const objtype = msgArgs.getValueOf("objtype");
    const name = msgArgs.getValueOf("parent_name");
    const returnSegs = msgArgs.get("return_segs").getBoolValue();
    var maxsplit: int = msgArgs.get("max").getIntValue();

    // check to make sure symbols defined
    st.checkTable(name);

    const json = msgArgs.get("pattern").getList(1);
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

  use CommandMap;
  registerFunction("segmentedFlatten", segFlattenMsg, getModuleName());
  registerFunction("segmentedSplit", segmentedSplitMsg, getModuleName());
}
