module SegmentedComputation {
  use CommAggregation;
  use SipHash;
  use ServerErrors;
  private use Cast;
  use Reflection;

  proc computeSegmentOwnership(segments: [?D] int, vD) {
    const low = vD.low;
    const size = vD.size;
    // Locale that owns each segment's bytes
    var startLocales: [D] int = [seg in segments] ((seg - low)*numLocales) / size;
    // Index of first segment each locale owns bytes for
    var startSegInds: [LocaleSpace] int;
    // Mark true where owning locale changes
    const change = [(i, sl) in zip(D, startLocales)] if (i == D.low) then true else (sl != startLocales[i-1]);
    // Wherever the owning locale increments, record that as first segment for the locale
    forall (c, sl, i) in zip(change, startLocales, D) {
      if c {
        startSegInds[sl] = i;
      }
    }

    // Number of segments each locale owns bytes for
    var numSegs: [LocaleSpace] int;
    // small number of iterations, no comms
    for l in Locales {
      if (l.id == numLocales - 1) {
        numSegs[l.id] = D.size - startSegInds[l.id];
      } else {
        numSegs[l.id] = startSegInds[l.id + 1] - startSegInds[l.id];
      }
    }

    const lengths = [(s, i) in zip(segments, D)] if (i == D.high) then (vD.size - s) else (segments[i+1] - s);

    return (startSegInds, numSegs, lengths);
  }

  enum SegFunction {
    SipHash128,
    StringToNumericStrict,
    StringToNumericIgnore,
    StringToNumericReturnValidity,
    StringCompareLiteralEq,
    StringCompareLiteralNeq,
    StringSearch,
    StringIsLower,
    StringIsUpper,
    StringIsTitle,
  }
  
  proc computeOnSegments(segments: [?D] int, values: [?vD] ?t, param function: SegFunction, type retType, const strArg: string = "") throws {
    // type retType = if (function == SegFunction.StringToNumericReturnValidity) then (outType, bool) else outType;
    var res: [D] retType;
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
          // Apply function to bytes of each owned segment, aggregating return value to res
          if function == SegFunction.StringSearch {
            forall (start, len, i) in zip(mySegs, myLens, mySegInds) with (var agg = newDstAggregator(retType), var myRegex = _unsafeCompileRegex(strArg)) {
              agg.copy(res[i], stringSearch(values, start..#len, myRegex));
            }
          } else {
            forall (start, len, i) in zip(mySegs, myLens, mySegInds) with (var agg = newDstAggregator(retType)) {
              select function {
                when SegFunction.SipHash128 {
                  agg.copy(res[i], sipHash128(values, start..#len));
                }
                when SegFunction.StringToNumericStrict {
                  agg.copy(res[i], stringToNumericStrict(values, start..#len, retType));
                }
                when SegFunction.StringToNumericIgnore {
                  agg.copy(res[i], stringToNumericIgnore(values, start..#len, retType));
                }
                when SegFunction.StringToNumericReturnValidity {
                  agg.copy(res[i], stringToNumericReturnValidity(values, start..#len, retType[0]));
                }
                when SegFunction.StringCompareLiteralEq {
                  agg.copy(res[i], stringCompareLiteralEq(values, start..#len, strArg));
                }
                when SegFunction.StringCompareLiteralNeq {
                  agg.copy(res[i], stringCompareLiteralNeq(values, start..#len, strArg));
                }
                when SegFunction.StringIsLower {
                  agg.copy(res[i], stringIsLower(values, start..#len));
                }
                when SegFunction.StringIsUpper {
                  agg.copy(res[i], stringIsUpper(values, start..#len));
                }
                when SegFunction.StringIsTitle {
                  agg.copy(res[i], stringIsTitle(values, start..#len));
                }
                otherwise {
                  compilerError("Unrecognized segmented function");
                }
              }
            }
          }
        } catch {
          throw new owned ErrorWithContext("Error computing %s on string or segment".format(function:string),
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "IllegalArgumentError");
        }
      }
    }
    return res;
  }
}
