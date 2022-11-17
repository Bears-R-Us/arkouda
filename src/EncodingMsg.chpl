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
    use SegmentedComputation;

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

      // add 1 for null terminator
      var encodeLengths = getBufLengths(offs, origVals, toEncoding, fromEncoding) + 1;
      var encodeOffsets = (+ scan encodeLengths);
      encodeOffsets -= encodeLengths;
      
      var encodedValues = encodeSegments(offs, origVals, encodeLengths, toEncoding, fromEncoding);

      return (encodeOffsets, encodedValues);
    }

    proc getBufLengths(segments: [?D] int, values: [?vD] ?t, toEncoding: string, fromEncoding: string) throws {
      var res: [D] int;
      if (D.size == 0) {
        return res;
      }

      const (startSegInds, numSegs, lengths) = computeSegmentOwnership(segments, vD);
    
      // Start task parallelism
      coforall loc in Locales {
        on loc {
          const myFirstSegIdx = startSegInds[loc.id];
          const myNumSegs = max(0, numSegs[loc.id]);
          const mySegInds = {myFirstSegIdx..#myNumSegs};
          // Segment offsets whose bytes are owned by loc
          // Lengths of segments whose bytes are owned by loc
          var mySegs, myLens: [mySegInds] int;
          forall i in mySegInds with (var agg = new SrcAggregator(int)) {
            agg.copy(mySegs[i], segments[i]);
            agg.copy(myLens[i], lengths[i]);
          }
          try {
            forall (start, len, i) in zip(mySegs, myLens, mySegInds) with (var agg = newDstAggregator(int)) {
              var slice = new lowLevelLocalizingSlice(values, start..#len);
              agg.copy(res[i], getBufLength(slice.ptr: c_ptr(uint(8)), len, toEncoding, fromEncoding));
            }
          } catch {
            throw new owned ErrorWithContext("Error",
                                             getLineNumber(),
                                             getRoutineName(),
                                             getModuleName(),
                                             "IllegalArgumentError");
          }
        }
      }
      return res;
    }

    proc encodeSegments(segments: [?D] int, values: [?vD] uint(8), encodeLengths: [D] int, toEncoding: string, fromEncoding: string) throws {
      var res = makeDistArray(+ reduce encodeLengths, uint(8));
      if (D.size == 0) {
        return res;
      }

      const (startSegInds, numSegs, lengths) = computeSegmentOwnership(segments, vD);
    
      // Start task parallelism
      coforall loc in Locales {
        on loc {
          const myFirstSegIdx = startSegInds[loc.id];
          const myNumSegs = max(0, numSegs[loc.id]);
          const mySegInds = {myFirstSegIdx..#myNumSegs};
          // Segment offsets whose bytes are owned by loc
          // Lengths of segments whose bytes are owned by loc
          var mySegs, myLens: [mySegInds] int;
          forall i in mySegInds with (var agg = new SrcAggregator(int)) {
            agg.copy(mySegs[i], segments[i]);
            agg.copy(myLens[i], lengths[i]);
          }
          try {
            forall (start, len, i, encodeLen) in zip(mySegs, myLens, mySegInds, encodeLengths) {
              var slice = new lowLevelLocalizingSlice(values, start..#len);
              var encodedStr = encodeStr(slice.ptr: c_ptr(uint(8)), len, encodeLen, toEncoding, fromEncoding);
              var agg = newDstAggregator(uint(8));
              for j in start..#len {
                agg.copy(res[j], encodedStr[j-start]);
              }
            }
          } catch {
            throw new owned ErrorWithContext("Error",
                                             getLineNumber(),
                                             getRoutineName(),
                                             getModuleName(),
                                             "IllegalArgumentError");
          }
        }
      }
      return res;
    }


    use CommandMap;
    registerFunction("encode", encodeDecodeMsg, getModuleName());
    registerFunction("decode", encodeDecodeMsg, getModuleName());
}
