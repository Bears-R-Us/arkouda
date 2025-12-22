module CastMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Reflection;
  use SegmentedString;
  use ServerErrors;
  use Logging;
  use Message;
  use ServerErrorStrings;
  use ServerConfig;
  use Cast;
  use BigInteger;
  use UInt128;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const castLogger = new Logger(logLevel, logChannel);

  @arkouda.instantiateAndRegister(prefix="cast")
  proc castArray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
    type array_dtype_from,
    type array_dtype_to,
    param array_nd: int
  ): MsgTuple throws
    where !((isRealType(array_dtype_from) || isImagType(array_dtype_from) || isComplexType(array_dtype_from)) && array_dtype_to == bigint) &&
          !(array_dtype_from == bigint && array_dtype_to == bool)
  {
    const a = st[msgArgs["name"]]: SymEntry(array_dtype_from, array_nd);
    try {
      if array_dtype_from == UInt128 || array_dtype_to == UInt128 {
        const b: [a.a.domain] array_dtype_to =
          [i in a.a.domain] a.a[i]: array_dtype_to;
        return st.insert(new shared SymEntry(b));
      } else {
        const b = a.a: array_dtype_to;
        return st.insert(new shared SymEntry(b));
      }
    } catch {
      return MsgTuple.error("bad value in cast from %s to %s".format(
        type2str(array_dtype_from),
        type2str(array_dtype_to)
      ));
    }
  }

  @arkouda.instantiateAndRegister(prefix="castToStrings")
  proc castArrayToStrings(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws {
    const name = msgArgs["name"].toScalar(string);
    var gse: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
    return castGenSymEntryToString(gse, st, array_dtype);
  }

  @arkouda.instantiateAndRegister(prefix="castStringsTo")
  proc castStringsToArray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws {
    const name = msgArgs["name"].toScalar(string),
          errors = msgArgs["opt"].toScalar(string).toLower() : ErrorMode;

    const strings = getSegString(name, st);

    if array_dtype == bigint {
      return MsgTuple.success(castStringToBigInt(strings, st, errors));
    } else {
      return MsgTuple.success(castStringToSymEntry(strings, st, array_dtype, errors));
    }
  }

  proc transmuteFloatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    var name = msgArgs.getValueOf("name");
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"name: %s".format(name));
    var e = toSymEntry(getGenericTypedArrayEntry(name, st), real);
    var transmuted = makeDistArray(e.a.domain, uint);
    transmuted = [ei in e.a] ei.transmute(uint(64));
    var transmuteName = st.nextName();
    st.addEntry(transmuteName, createSymEntry(transmuted));
    var repMsg = "created " + st.attrib(transmuteName);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  use CommandMap;
  registerFunction("transmuteFloat", transmuteFloatMsg, getModuleName());
}
