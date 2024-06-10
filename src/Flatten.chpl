module Flatten {
  use ServerConfig;

  use AryUtil;
  use SegmentedString;
  use ServerErrors;
  use SymArrayDmap;
  use CommAggregation;
  use Reflection;
  use Regex;
  use CTypes;

  config const NULL_STRINGS_VALUE = 0:uint(8);

  /*
    Given a SegString where each string encodes a variable-length sequence delimited by a regex,
    flattenRegex unpacks the sequences into a flat array of individual elements.
    If returnSegs is set to True, a mapping between the original strings and new array elements will be returned

    Note: the regular expression engine used, re2, does not support lookahead/lookbehind

    :arg delim: regex delimter used to split strings into substrings
    :type delim: string

    :arg returnSegs: If True, also return mapping of original strings to first substring in return array
    :type returnSegs: bool

    :returns: Strings – Flattened substrings with delimiters removed and (optional) int64 pdarray – For each original string, the index of first corresponding substring in the return array
  */
  // DEPRECATED - All regex flatten calls now redirect to SegString.split
  // TODO: Remove flattenRegex
  proc SegString.flattenRegex(delim: string, returnSegs: bool) throws {
    checkCompile(delim);
    ref origOffsets = this.offsets.a;
    ref origVals = this.values.a;
    const lengths = this.getLengths();

    overMemLimit((this.offsets.size * numBytes(int)) + (2 * this.values.size * numBytes(int)));
    var numMatches = makeDistArray(this.offsets.a.domain, int);
    var writeToVal = makeDistArray(this.values.a.domain, true);
    var nullByteLocations = makeDistArray(this.values.a.domain, false);

    // since the delim matches are variable length, we don't know what the size of flattenedVals should be until we've found the matches
    forall (i, off, len) in zip(this.offsets.a.domain, origOffsets, lengths) with (var myRegex = unsafeCompileRegex(delim.encode()),
                                                                             var writeAgg = newDstAggregator(bool),
                                                                             var nbAgg = newDstAggregator(bool),
                                                                             var matchAgg = newDstAggregator(int)) {
      var matchessize = 0;
      // for each string, find delim matches and set the positions of matches in writeToVal to false (non-matches will be copied to flattenedVals)
      // mark the locations of null bytes (the positions before original offsets and the last character of matches)
      for m in myRegex.matches(interpretAsBytes(origVals, off..#len, borrow=true)) {
        var match = m[0];
        // set writeToVal to false for matches (except the last character of the match because we will write a null byte)
        for k in (off + match.byteOffset:int)..#(match.numBytes - 1) {
          writeAgg.copy(writeToVal[k], false);
        }
        // is writeToVal[(off + match.offset:int)..#(match.size - 1)] = false more efficient or for loop with aggregator?
        nbAgg.copy(nullByteLocations[off + match.byteOffset:int + (match.numBytes - 1)], true);
        matchessize += 1;
      }
      if off != 0 {
        // the position before an offset is a null byte (except for off == 0)
        nbAgg.copy(nullByteLocations[off - 1], true);
      }
      matchAgg.copy(numMatches[i], matchessize);
    }

    // writeToVal is true for positions to copy origVals (non-matches) and positions to write a null byte
    var flattenedVals = makeDistArray(+ reduce writeToVal, uint(8));
    // Each match is replaced with a null byte, so new offsets.size = totalNumMatches + old offsets.size
    var flattenedOffsets = makeDistArray((+ reduce numMatches) + this.offsets.size, int);

    // check there's enough room to create copies for the IndexTransform scans and throw if creating a copy would go over memory limit
    overMemLimit(2 * numBytes(int) * writeToVal.size);
    // the IndexTransforms start at 0 and increment after hitting a writeToVal/nullByteLocation condition
    // we do this because the indexing set for origVals is different from flattenedVals/Offsets (they are different lengths)
    // when looping over the origVals domain, the IndexTransforms act as a function: origVals.domain -> flattenedVals/Offsets.domain
    var valsIndexTransform = (+ scan writeToVal) - writeToVal;
    var offsIndexTransform = (+ scan nullByteLocations) - nullByteLocations + 1;  // the plus one is to leave space for the offset 0 edge case

    forall (origInd, flatValInd, offInd) in zip(this.values.a.domain, valsIndexTransform, offsIndexTransform) with (var valAgg = newDstAggregator(uint(8)),
                                                                                                              var offAgg = newDstAggregator(int)) {
      // writeToVal is true for positions to copy origVals (non-matches) and positions to write a null byte
      if writeToVal[origInd] {
        if origInd == 0 {
          // offset 0 edge case
          offAgg.copy(flattenedOffsets[0], 0);
        }
        if nullByteLocations[origInd] {
          // nullbyte location, copy nullbyte into flattenedVals
          valAgg.copy(flattenedVals[flatValInd], NULL_STRINGS_VALUE);
          if origInd != this.values.a.domain.high {
            // offset points to position after null byte
            offAgg.copy(flattenedOffsets[offInd], flatValInd + 1);
          }
        }
        else {
          // non-match location, copy origVal into flattenedVals
          valAgg.copy(flattenedVals[flatValInd], origVals[origInd]);
        }
      }
    }

    // build segments mapping from original Strings to flattenedStrings
    const segmentsDom = if returnSegs then this.offsets.a.domain else makeDistDom(0);
    var segments = makeDistArray(segmentsDom, int);
    if returnSegs {
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * numMatches.size);
      // each match results in a new element (it is replaced with a null byte)
      // so the mapping is a running sum of all previous replacements plus the curent index
      segments = (+ scan numMatches) - numMatches + segmentsDom;
    }
    return (flattenedOffsets, flattenedVals, segments);
  }

  /*
    Split string by the occurrences of pattern. If maxsplit is nonzero, at most maxsplit splits occur
    If returnSegs is set to True, a mapping between the original strings and new array elements will be returned

    :arg pattern: regex pattern used to split strings into substrings
    :type pattern: string

    :arg initMaxSplit: If maxsplit is nonzero, at most maxsplit splits occur. If zero, split on all occurences of pattern
    :type initMaxSplit: int

    :arg returnSegs: If True, also return mapping of original strings to first substring in return array
    :type returnSegs: bool

    :returns: Strings – Substrings with pattern matches removed and (optional) int64 pdarray – For each original string, the index of first corresponding substring in the return array
  */
  proc SegString.split(pattern: string, initMaxSplit: int, returnSegs: bool) throws {
    // This function is extremely similar to regexFlatten but with a maxSplit cap
    checkCompile(pattern);
    ref origOffsets = this.offsets.a;
    ref origVals = this.values.a;
    const lengths = this.getLengths();

    overMemLimit((this.offsets.size * numBytes(int)) + (2 * this.values.size * numBytes(int)));
    var numMatches = makeDistArray(this.offsets.a.domain, int);
    var writeToVal = makeDistArray(this.values.a.domain, true);
    var nullByteLocations = makeDistArray(this.values.a.domain, false);

    // maxSplit = 0 means replace all occurances, so we set maxsplit equal to 10**9
    var maxsplit = if initMaxSplit == 0 then 10**9:int else initMaxSplit;
    // since the pattern matches are variable length, we don't know what the size of splitVals should be until we've found the matches
    forall (i, off, len) in zip(this.offsets.a.domain, origOffsets, lengths) with (var myRegex = unsafeCompileRegex(pattern.encode()),
                                                                             var writeAgg = newDstAggregator(bool),
                                                                             var nbAgg = newDstAggregator(bool),
                                                                             var matchAgg = newDstAggregator(int)) {
      var matchessize = 0;
      // for each string, find pattern matches and set the positions of matches in writeToVal to false (non-matches will be copied to splitVals)
      // mark the locations of null bytes (the positions before original offsets and the last character of matches)
      for m in myRegex.matches(interpretAsBytes(origVals, off..#len, borrow=true)) {
        var match = m[0];
        // set writeToVal to false for matches (except the last character of the match because we will write a null byte)
        for k in (off + match.byteOffset:int)..#(match.numBytes - 1) {
          writeAgg.copy(writeToVal[k], false);
        }
        nbAgg.copy(nullByteLocations[off + match.byteOffset:int + (match.numBytes - 1)], true);
        matchessize += 1;
        if matchessize == maxsplit { break; }
      }
      if off != 0 {
        // the position before an offset is a null byte (except for off == 0)
        nbAgg.copy(nullByteLocations[off - 1], true);
      }
      matchAgg.copy(numMatches[i], matchessize);
    }
    // writeToVal is true for positions to copy origVals (non-matches) and positions to write a null byte
    var splitVals = makeDistArray(+ reduce writeToVal, uint(8));
    // Each match is replaced with a null byte, so new offsets.size = totalNumMatches + old offsets.size
    var splitOffsets = makeDistArray((+ reduce numMatches) + this.offsets.size, int);

    var valsIndexTransform = (+ scan writeToVal) - writeToVal;
    var offsIndexTransform = (+ scan nullByteLocations) - nullByteLocations + 1;

    forall (origInd, origVal, splitValInd, offInd) in zip(this.values.a.domain, origVals, valsIndexTransform, offsIndexTransform) with (var valAgg = newDstAggregator(uint(8)),
                                                                                                               var offAgg = newDstAggregator(int)) {
      // writeToVal is true for positions to copy origVals (non-matches) and positions to write a null byte
      if writeToVal[origInd] {
        if origInd == 0 {
          // offset 0 edge case
          offAgg.copy(splitOffsets[0], 0);
        }
        if nullByteLocations[origInd] {
          // nullbyte location, copy nullbyte into splitVals
          valAgg.copy(splitVals[splitValInd], NULL_STRINGS_VALUE);
          if origInd != this.values.a.domain.high {
            // offset points to position after null byte
            offAgg.copy(splitOffsets[offInd], splitValInd + 1);
          }
        }
        else {
          // non-match location, copy origVal into splitVals
          valAgg.copy(splitVals[splitValInd], origVal);
        }
      }
    }

     // build segments mapping from original Strings to flattenedStrings
    const segmentsDom = if returnSegs then this.offsets.a.domain else makeDistDom(0);
    var segments = makeDistArray(segmentsDom, int);
    if returnSegs {
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * numMatches.size);
      // each match results in a new element (it is replaced with a null byte)
      // so the mapping is a running sum of all previous replacements plus the curent index
      segments = (+ scan numMatches) - numMatches + segmentsDom;
    }
    return (splitOffsets, splitVals, segments);
  }

  proc SegString.flatten(delim: string, returnSegs: bool, regex: bool = false) throws {
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
    if regex {
      // we still perform the empty delimeter check above
      return flattenRegex(delim, returnSegs);
    }
    // delimHits is true immediately following instances of delim, i.e.
    // at the starts of newly created segments
    var delimHits = makeDistArray(this.values.a.domain, bool);
    const hitD = this.values.a.domain.interior(this.values.size - delim.numBytes);
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
    var val = makeDistArray(valDom, uint(8));
    // Allocate offsets. Each instance of delim creates an additional segment.
    const offDom = makeDistDom(this.offsets.size + nHits);
    var off = makeDistArray(offDom, int);
    // Need to merge original offsets with new offsets from delims
    // offTruth is true at start of every segment (new or old)
    var offTruth = makeDistArray(delimHits);
    forall o in this.offsets.a with (var agg = newDstAggregator(bool)) {
      agg.copy(offTruth[o], true);
    }
    // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
    overMemLimit(numBytes(int) * offTruth.size);
    // Running segment number
    var scanOff = (+ scan offTruth) - offTruth;
    // Copy over the offsets; later we will shrink them if delim is > 1 byte
    forall (i, o, t) in zip(this.values.a.domain, scanOff, offTruth) with (var agg = newDstAggregator(int)) {
      if t {
        agg.copy(off[o], i);
      }
    }
    // If user requests mapping of original strings to new ones,
    // Then we need to build segments based on how many delims were
    // present in each string, plus the original string boundaries.
    // Fortunately, scanOff already has this information.
    const segD = if returnSegs then this.offsets.a.domain else makeDistDom(0);
    var seg = makeDistArray(segD, int);
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
      var srcIdx = makeDistArray(valDom, 1);
      // For substring boundaries, derivative = 
      //     if (previous string terminated by delim) then delim.numBytes else 1;
      var followsDelim = makeDistArray(offDom, bool);
      forall (d, o) in zip(followsDelim, off) with (var agg = newSrcAggregator(bool)) {
        agg.copy(d, delimHits[o]);
      }
      const boundaryDeriv = (followsDelim:int * (delim.numBytes - 1)) + 1;
      // Next step requires offsets to be translated to new domain, i.e. with
      // delims removed.
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * followsDelim.size);
      // Number of delims preceding current string
      const delimsBefore = + scan followsDelim;
      // Each delim gets replaced by null byte (length 1)
      off -= delimsBefore * (delim.numBytes - 1);
      // Use dest offsets to overwrite the derivative at the substring boundaries
      forall (o, d) in zip(off, boundaryDeriv) with (var agg = newDstAggregator(int)) {
        agg.copy(srcIdx[o], d);
      }
      // Force first derivative to match start of src domain
      srcIdx[valDom.low] = this.values.a.domain.low;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * srcIdx.size);
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
