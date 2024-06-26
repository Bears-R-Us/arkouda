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
    private config const logChannel = ServerConfig.logChannel;
    const emLogger = new Logger(logLevel, logChannel);

    proc encodeDecodeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var repMsg: string;

      var toEncoding = msgArgs.getValueOf("toEncoding").toUpper();
      var fromEncoding = msgArgs.getValueOf("fromEncoding").toUpper();
      
      var stringsObj = getSegString(msgArgs.getValueOf("obj"), st);

      var (offsets, vals) = encodeDecode(stringsObj, toEncoding, fromEncoding);
      var encodedStrings = getSegString(offsets, vals, st);
      repMsg = "created " + st.attrib(encodedStrings.name) + "+created bytes.size %?".format(encodedStrings.nBytes);

      emLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc encodeDecode(stringsObj, toEncoding: string, fromEncoding: string) throws {
      ref origVals = stringsObj.values.a;
      ref offs = stringsObj.offsets.a;
      // TODO: is this inefficient?
      const lengths = stringsObj.getLengths();

      var encodeLengths = getBufLengths(offs, origVals, toEncoding, fromEncoding);
      // add 1 for null terminator
      if toEncoding != "IDNA" && fromEncoding != "IDNA" then
        encodeLengths += 1;
      var encodeOffsets = (+ scan encodeLengths);
      encodeOffsets -= encodeLengths;
      
      var encodedValues = encodeSegments(offs, origVals, encodeOffsets, encodeLengths, toEncoding, fromEncoding);

      return (encodeOffsets, encodedValues);
    }

    proc getBufLengths(segments: [?D] int, ref values: [?vD] ?t, toEncoding: string, fromEncoding: string) throws {
      var res = makeDistArray(D, int);
      if (D.size == 0) {
        return res;
      }

      const (startSegInds, numSegs, lengths) = computeSegmentOwnership(segments, vD);
    
      // Start task parallelism
      coforall loc in Locales with (ref values, ref res) {
        on loc {
          const locTo = toEncoding;
          const locFrom = fromEncoding;
          
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
              agg.copy(res[i], getBufLength(slice.ptr: c_ptr(uint(8)), len, locTo, locFrom));
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

    proc encodeSegments(segments: [?D] int, ref values: [?vD] uint(8), encodeOffsets: [D] int, encodeLengths: [D] int, toEncoding: string, fromEncoding: string) throws {
      var res = makeDistArray(+ reduce encodeLengths, uint(8));
      if (D.size == 0) {
        return res;
      }

      const (startSegInds, numSegs, lengths) = computeSegmentOwnership(segments, vD);
      const (eStartSegInds, eNumSegs, eLengths) = computeSegmentOwnership(encodeOffsets, res.domain);
    
      // Start task parallelism
      coforall loc in Locales with (ref values, ref res) {
        on loc {
          const locTo = toEncoding;
          const locFrom = fromEncoding;
          
          const myFirstSegIdx = startSegInds[loc.id];
          const myNumSegs = max(0, numSegs[loc.id]);
          const mySegInds = {myFirstSegIdx..#myNumSegs};

          const myFirstESegIdx = eStartSegInds[loc.id];
          const myNumESegs = max(0, eNumSegs[loc.id]);
          const myESegInds = {myFirstESegIdx..#myNumESegs};
          // Segment offsets whose bytes are owned by loc
          // Lengths of segments whose bytes are owned by loc
          var mySegs, myLens, myELens, myESegs: [mySegInds] int;
          forall i in mySegInds with (var agg = new SrcAggregator(int)) {
            agg.copy(mySegs[i], segments[i]);
            agg.copy(myLens[i], lengths[i]);
            agg.copy(myELens[i], eLengths[i]);
            agg.copy(myESegs[i], encodeOffsets[i]);
          }
          try {
            forall (start, len, i, eStart, eLen) in zip(mySegs, myLens, mySegInds, myESegs, myELens) {
              var slice = new lowLevelLocalizingSlice(values, start..#len);
              var encodedStr = encodeStr(slice.ptr: c_ptr(uint(8)), len, eLen, locTo, locFrom);
              var agg = newDstAggregator(uint(8));
              for j in eStart..#eLen {
                agg.copy(res[j], encodedStr[j-eStart]);
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
