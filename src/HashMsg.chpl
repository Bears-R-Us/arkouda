module HashMsg {
  use Reflection;
  use ServerErrors;
  use ServerErrorStrings;
  use Logging;
  use Message;

  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use CommAggregation;
  use ServerConfig;
  use SegmentedString;
  use AryUtil;
  use UniqueMsg;
  use Map;
  use IOUtils;
  use HashUtils;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const hmLogger = new Logger(logLevel, logChannel);

  proc categoricalHashMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    if objtype != ObjType.CATEGORICAL {
      var errorMsg = notImplementedError(pn, objtype: string);
      hmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }
    const categoriesName = msgArgs.getValueOf("categories");
    const codesName = msgArgs.getValueOf("codes");
    st.checkTable(categoriesName);
    st.checkTable(codesName);
    var (upper, lower) = categoricalHash(categoriesName, codesName, st);
    var upperName = st.nextName();
    st.addEntry(upperName, createSymEntry(upper));
    var lowerName = st.nextName();
    st.addEntry(lowerName, createSymEntry(lower));
    var createdMap = new map(keyType=string,valType=string);
    createdMap.add("upperHash", "created %s".format(st.attrib(upperName)));
    createdMap.add("lowerHash", "created %s".format(st.attrib(lowerName)));
    repMsg = formatJson(createdMap);
    hmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc hashArraysMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var n = msgArgs.get("length").getIntValue();
    var s = msgArgs.get("size").getIntValue();
    var namesList = msgArgs.get("nameslist").getList(n);
    var typesList = msgArgs.get("typeslist").getList(n);
    var (size, hasStr, allSmallStrs, extraArraysNeeded, numStrings, names, types) = validateArraysSameLength(n, namesList, typesList, st);

    // Call hashArrays on list of given array names
    var hashes = hashArrays(size, names, types, st);
    var upper = makeDistArray(s, uint);
    var lower = makeDistArray(s, uint);

    // Assign upper and lower bit values to their respective entries
    forall (up, low, h) in zip(upper, lower, hashes) {
      (up, low) = h;
    }

    var upperName = st.nextName();
    st.addEntry(upperName, createSymEntry(upper));
    var lowerName = st.nextName();
    st.addEntry(lowerName, createSymEntry(lower));

    var createdMap = new map(keyType=string,valType=string);
    createdMap.add("upperHash", "created %s".format(st.attrib(upperName)));
    createdMap.add("lowerHash", "created %s".format(st.attrib(lowerName)));
    var repMsg = formatJson(createdMap);
    hmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  use CommandMap;
  registerFunction("hashList", hashArraysMsg, getModuleName());
  registerFunction("categoricalHash", categoricalHashMsg, getModuleName());
}
