module Flatten {
  use SegmentedArray;
  use Errors;
  use SymArrayDmap;
  use CommAggregation;
  use Reflection;
  
  proc SegString.flatten(delim: string, returnSegs: bool) throws {
    if delim.numBytes == 0 {
      throw new owned ErrorWithContext("Cannot flatten with empty delimiter",
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "ValueError");
    }
    if this.size == 0 {
      return (this.offsets.a, this.values.a, this.offsets.a);
    }
    // delimHits is true immediately following instances of delim, i.e.
    // at the starts of newly created segments
    var delimHits: [this.values.aD] bool;
    const hitD = this.values.aD.interior(this.values.size - delim.numBytes);
    delimHits[hitD] = this.findSubstringInBytes(delim)[hitD.translate(-delim.numBytes)];
    // Hits could be overlapping, if delim is palindromic and > 1 byte
    // Convert to greedy, erasing overlapping hits
    for i in 1..(delim.numBytes-1) {
      delimHits[hitD] &= !delimHits[hitD.translate(-i)];
    }
    // Allocate values. Each delim will be replaced by a null byte
    const nHits = + reduce delimHits;
    const valSize = this.values.size - (nHits * (delim.numBytes - 1));
    const valDom = makeDistDom(valSize);
    var val: [valDom] uint(8);
    // Allocate offsets. Each instance of delim creates an additional segment.
    const offDom = makeDistDom(this.offsets.size + nHits);
    var off: [offDom] int;
    // Need to merge original offsets with new offsets from delims
    // offTruth is true at start of every segment (new or old)
    var offTruth: [this.values.aD] bool = delimHits;
    forall o in this.offsets.a with (var agg = newDstAggregator(bool)) {
      agg.copy(offTruth[o], true);
    }
    // Running segment number
    var scanOff = (+ scan offTruth) - offTruth;
    // Copy over the offsets; later we will shrink them if delim is > 1 byte
    forall (i, o, t) in zip(this.values.aD, scanOff, offTruth) with (var agg = newDstAggregator(int)) {
      if t {
        agg.copy(off[o], i);
      }
    }
    // If user requests mapping of original strings to new ones,
    // Then we need to build segments based on how many delims were
    // present in each string, plus the original string boundaries.
    // Fortunately, scanOff already has this information.
    const segD = if returnSegs then this.offsets.aD else makeDistDom(0);
    var seg: [segD] int;
    if returnSegs {
      forall (s, o) in zip(seg, this.offsets.a) with (var agg = newSrcAggregator(int)) {
        agg.copy(s, scanOff[o]);
      }
    }
    // Now copy the values
    if delim.numBytes == 1 {
      // Can simply overwrite delim with null byte
      val = this.values.a;
      forall (vi, t) in zip(delimHits.domain, delimHits) {
        if t {
          // Previous index is where delim occurred
          val[vi-1] = 0:uint(8);
        }
      }
    } else {
      // Need to do a gather, much like this.this([] int) but with modified offsets
      // Strategy: generate a dest-local array of indices to gather from src, so
      // we can use srcAggregator to copy remote values to local dest.

      // Form the index array by initializing as a derivative and then integrating.
      // Within a substring, values are consecutive, so derivative is one.
      // Initialize all derivatives to one, then overwrite substring boundaries.
      var srcIdx: [valDom] int = 1;
      // For substring boundaries, derivative = 
      //     if (previous string terminated by delim) then delim.numBytes else 1;
      var followsDelim: [offDom] bool;
      forall (d, o) in zip(followsDelim, off) with (var agg = newSrcAggregator(bool)) {
        agg.copy(d, delimHits[o]);
      }
      const boundaryDeriv = (followsDelim:int * (delim.numBytes - 1)) + 1;
      // Next step requires offsets to be translated to new domain, i.e. with
      // delims removed.
      // Number of delims preceding current string
      const delimsBefore = + scan followsDelim;
      // Each delim gets replaced by null byte (length 1)
      off -= delimsBefore * (delim.numBytes - 1);
      // Use dest offsets to overwrite the derivative at the substring boundaries
      forall (o, d) in zip(off, boundaryDeriv) with (var agg = newDstAggregator(int)) {
        agg.copy(srcIdx[o], d);
      }
      // Force first derivative to match start of src domain
      srcIdx[valDom.low] = this.values.aD.low;
      // Perform integration to compute gather indices
      srcIdx = (+ scan srcIdx);
      // Now we have dest-local copy of src indices, so gather with a src aggregator
      ref va = this.values.a;
      forall (v, si) in zip(val, srcIdx) with (var agg = newSrcAggregator(uint(8))) {
        agg.copy(v, va[si]);
      }
      // Finally, fill in null terminators
      forall o in off[offDom.interior(off.size-1)] with (var agg = newDstAggregator(uint(8))) {
        agg.copy(val[o-1], 0:uint(8));
      }
    }
    return (off, val, seg);
  }
}