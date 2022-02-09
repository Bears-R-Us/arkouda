module SegmentedComputation {
  use CommAggregation;
  use SipHash;

  proc computeSegmentOwnership(segments: [?D] int, vD) {
    // Locale that owns each segment's bytes
    var startLocales: [D] int;
    const low = vD.low;
    const size = vD.size;
    // Vector op on segments, should be fast
    forall (sl, seg) in zip(startLocales, segments) {
      // which locale do my bytes start on?
      sl = ((seg - low)*numLocales) / size;
    }

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

    var lengths: [D] int;
    forall (l, s, i) in zip(lengths, segments, D) {
      if (i == D.high) {
        l = vD.size - s;
      } else {
        l = segments[i+1] - s;
      }
    }

    return (startSegInds, numSegs, lengths);
  }

  enum SegFunction {
    SipHash128,
  }
  
  proc computeOnSegments(segments: [?D] int, values: [?vD] ?t, param function: SegFunction, type retType) throws {
    var res: [D] retType;
    if (D.size == 0) {
      return res;
    }

    const (startSegInds, numSegs, lengths) = computeSegmentOwnership(segments, vD);
    
    // Start task parallelism
    coforall loc in Locales {
      on loc {
        const myFirstSegIdx = startSegInds[loc.id];
        const myNumSegs = numSegs[loc.id];
        const mySegInds = {myFirstSegIdx..#myNumSegs};
        // Segment offsets whose bytes are owned by loc
        // Lengths of segments whose bytes are owned by loc
        var mySegs, myLens: [mySegInds] int;
        forall i in mySegInds with (var agg = new SrcAggregator(int)) {
          agg.copy(mySegs[i], segments[i]);
          agg.copy(myLens[i], lengths[i]);
        }
        // Apply function to bytes of each owned segment, aggregating return value to res
        forall (start, len, i) in zip(mySegs, myLens, mySegInds) with (var agg = newDstAggregator(retType)) {
          select function {
            when SegFunction.SipHash128 {
              agg.copy(res[i], sipHash128(values, start..#len));
            }
            otherwise {
              compilerError("Unrecognized segmented function");
            }
          }
        }
      }
    }
    return res;
  }
}
