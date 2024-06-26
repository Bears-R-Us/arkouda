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
  private config const logChannel = ServerConfig.logChannel;
  const fmLogger = new Logger(logLevel, logChannel);

  proc segFlattenMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    const objtype = msgArgs.getValueOf("objtype").toUpper(): ObjType;
    const returnSegs: bool = msgArgs.get("return_segs").getBoolValue();
    const regex: bool = msgArgs.get("regex").getBoolValue();
    const delim: string = msgArgs.getValueOf("delim");
    var repMsg: string;
    select objtype {
      when ObjType.STRINGS {
        const rSegName = st.nextName();
        const rValName = st.nextName();
        const strings = getSegString(msgArgs.getValueOf("values"), st);
        var (off, val, segs) = strings.flatten(delim, returnSegs, regex);
        var stringsObj = getSegString(off, val, st);
        repMsg = "created %s+created bytes.size %?".format(st.attrib(stringsObj.name), stringsObj.nBytes);
        if returnSegs {
          const optName: string = st.nextName();
          st.addEntry(optName, createSymEntry(segs));
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

  proc segmentedSplitMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;

    const objtype = msgArgs.getValueOf("objtype");
    const name = msgArgs.getValueOf("parent_name");
    const returnSegs = msgArgs.get("return_segs").getBoolValue();
    var maxsplit: int = msgArgs.get("max").getIntValue();

    // check to make sure symbols defined
    st.checkTable(name);

    const pattern: string = msgArgs.getValueOf("pattern");

    fmLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                   "cmd: %s objtype: %?".format(cmd, objtype));

    select objtype {
      when "Matcher" {
        const optName: string = if returnSegs then st.nextName() else "";
        const strings = getSegString(name, st);
        var (off, val, segs) = strings.split(pattern, maxsplit, returnSegs);
        var retString = getSegString(off, val, st);
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
        if returnSegs {
          st.addEntry(optName, createSymEntry(segs));
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
