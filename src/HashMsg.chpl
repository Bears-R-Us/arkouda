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

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const hmLogger = new Logger(logLevel, logChannel);

  proc categoricalHash(categoriesName: string, codesName: string, st: borrowed SymTab) throws {
    var categories = getSegString(categoriesName, st);
    var codes = toSymEntry(getGenericTypedArrayEntry(codesName, st), int);
    // hash categories first
    var hashes = categories.siphash();
    // then do expansion indexing at codes
    ref ca = codes.a;
    var expandedHashes: [ca.domain] (uint, uint);
    forall (eh, c) in zip(expandedHashes, ca) with (var agg = newSrcAggregator((uint, uint))) {
      agg.copy(eh, hashes[c]);
    }
    var hash1 = makeDistArray(ca.size, uint);
    var hash2 = makeDistArray(ca.size, uint);
    forall (h, h1, h2) in zip(expandedHashes, hash1, hash2) {
      (h1,h2) = h:(uint,uint);
    }
    return (hash1, hash2);
  }

  proc categoricalHashMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    const objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    select objtype {
      when ObjType.CATEGORICAL {
        // check to make sure symbols defined
        const categoriesName = msgArgs.getValueOf("categories");
        const codesName = msgArgs.getValueOf("codes");
        st.checkTable(categoriesName);
        st.checkTable(codesName);
        var (upper, lower) = categoricalHash(categoriesName, codesName, st);
        var upperName = st.nextName();
        st.addEntry(upperName, new shared SymEntry(upper));
        var lowerName = st.nextName();
        st.addEntry(lowerName, new shared SymEntry(lower));
        repMsg = "created " + st.attrib(upperName) + "+created " + st.attrib(lowerName);
      }
      otherwise {
        var errorMsg = notImplementedError(pn, objtype: string);
        hmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
    hmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc hashArraysMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var n = msgArgs.get("length").getIntValue();
    var s = msgArgs.get("size").getIntValue();
    var namesList = msgArgs.get("nameslist").getList(n);
    var typesList = msgArgs.get("typeslist").getList(n);
    var (size, hasStr, names, types) = validateArraysSameLength(n, namesList, typesList, st);

    // Call hashArrays on list of given array names
    var hashes = hashArrays(size, names, types, st);
    var upper = makeDistArray(s, uint);
    var lower = makeDistArray(s, uint);

    // Assign upper and lower bit values to their respective entries
    forall (up, low, h) in zip(upper, lower, hashes) {
      (up, low) = h;
    }

    var upperName = st.nextName();
    st.addEntry(upperName, new shared SymEntry(upper));
    var lowerName = st.nextName();
    st.addEntry(lowerName, new shared SymEntry(lower));

    var repMsg = "created %s+created %s".format(st.attrib(upperName), st.attrib(lowerName));
    hmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  use CommandMap;
  registerFunction("hashList", hashArraysMsg, getModuleName());
  registerFunction("categoricalHash", categoricalHashMsg, getModuleName());
}
