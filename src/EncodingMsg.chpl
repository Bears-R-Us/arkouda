module EncodingMsg {
    use Subprocess;
    use Reflection;
    use Logging;
    use ServerConfig;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use CommAggregation;
    use ServerErrors;
    use ServerErrorStrings;
    use Codecs;
    use CTypes;

    use AryUtil;

    use SegmentedString;

    private config const logLevel = ServerConfig.logLevel;
    const emLogger = new Logger(logLevel);

    proc encodeDecodeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var repMsg: string;

      var toEncoding = msgArgs.getValueOf("toEncoding").toUpper();
      var fromEncoding = msgArgs.getValueOf("fromEncoding").toUpper();
      
      var stringsObj = getSegString(msgArgs.getValueOf("obj"), st);

      var (offsets, vals) = encodeDecode(stringsObj, toEncoding, fromEncoding);
      var encodedStrings = getSegString(offsets, vals, st);
      repMsg = "created " + st.attrib(encodedStrings.name) + "+created bytes.size %t".format(encodedStrings.nBytes);

      emLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc encodeDecode(stringsObj, toEncoding: string, fromEncoding: string) throws {
      ref origVals = stringsObj.values.a;
      ref offs = stringsObj.offsets.a;
      // TODO: is this inefficient?
      const lengths = stringsObj.getLengths();

      // TODO: We only need one of these, they have the same info
      var encodeLengths: [stringsObj.offsets.a.domain] int;
      var encodeOffsets: [stringsObj.offsets.a.domain] int;
      
      // First get lengths
      forall (i, off, len) in zip(0..#stringsObj.size, offs, lengths) {
        var slice = new lowLevelLocalizingSlice(origVals, off..#len);
        encodeLengths[i] = getBufLength(slice.ptr: c_ptr(uint(8)), len, toEncoding, fromEncoding);
      }

      // TODO: Is this efficient?
      encodeOffsets = (+ scan encodeLengths) - encodeLengths;
      var finalValues = makeDistArray(+ reduce encodeLengths, uint(8));
      forall (i, off, len) in zip(0..#stringsObj.size, offs, lengths) {
        var slice = new lowLevelLocalizingSlice(origVals, off..#len);
        var encodedStr = encodeStr(slice.ptr: c_ptr(uint(8)), len, encodeLengths[i], toEncoding, fromEncoding);
        finalValues[encodeOffsets[i]..#encodeLengths[i]] = encodedStr;
      }

      return (encodeOffsets, finalValues);
    }

    use CommandMap;
    registerFunction("encode", encodeDecodeMsg, getModuleName());
    registerFunction("decode", encodeDecodeMsg, getModuleName());
}
