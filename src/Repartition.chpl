module Repartition
{

  // The goal of this module is to provide helper functions to facilitate redistributing data between
  // locales.

  use PrivateDist;
  use SegmentedString;
  use List;
  use BigInteger;
  use Search;
  use Math;
  use MultiTypeSymbolTable;

  record innerArray {
    type t;
    var Dom: domain(1);
    var Arr: [Dom] t;

    proc init(in Dom: domain(1), type t) {
      this.t = t;
      this.Dom = Dom;
    }

    proc init(type t) {
      this.t = t;
    }
  }

  // This function attempts to grab as much as possible from the local slice of the bytes.
  // Given a SegString (global offsets + global byte buffer), it figures out for each
  // locale which *parts* of which strings intersect that locale's local chunk of bytes,
  // and collects bookkeeping needed to rebuild those pieces into local strings.
  proc balancedSegStringRetrieval(const ref strings: SegString):
    [PrivateSpace] innerArray(string)
  {
    // References to the global segment offsets and underlying byte array
    ref offsets  = strings.offsets.a;
    ref strBytes = strings.values.a;

    // Per-locale first and last index (global) of local chunk; -1 means empty.
    // These are global byte indices into strBytes for each locale’s localSubdomain.
    var firstIdxPerLoc, lastIdxPerLoc: [LocaleSpace] int = -1;

    // Per-locale end of string corresponding to last offset (global) of local chunk; -1 means empty.
    // This will hold, for each locale, the byte index where the *last* string that begins
    // on this locale ends (exclusive). Used as an upper bound when slicing.
    var lastOffsetEndPerLoc: [LocaleSpace] int = -1;

    // Last byte index in the global byte buffer
    const endOfBytes = strBytes.domain.last;

    // For each locale, the first string-start offset (global) that resides on that locale.
    // -1 means that locale has no offsets.
    var firstOffsetPerLoc: [LocaleSpace] int = -1;

    // First pass: for each locale, record its local ranges in the global bytes and offsets.
    forall loc in 0..#numLocales {

      // Local chunk of the bytes that resides on locale `loc`
      const currSubdomain = strBytes.localSubdomain(Locales[loc]);
      if !currSubdomain.isEmpty() {
        // Global first/last indices of this locale's byte chunk
        firstIdxPerLoc[loc] = currSubdomain.first;
        lastIdxPerLoc[loc] = currSubdomain.last;
      }

      // Local chunk of the offsets that resides on locale `loc`
      const offsetSubdomain = offsets.localSubdomain(Locales[loc]);
      if !offsetSubdomain.isEmpty() {
        // Record the first string-start offset (global) for this locale
        firstOffsetPerLoc[loc] = offsets[offsetSubdomain.first];
      }

    }

    // Second pass: for each locale, determine where the strings that *start* on this
    // locale end in the global byte buffer.
    forall loc in 0..#numLocales {

      if firstOffsetPerLoc[loc] != -1 {
        // Find the next locale that has at least one string-start offset.
        // The first offset of the next such locale is the end bound for
        // the string that starts at firstOffsetPerLoc[loc] (if we're
        // looking across locales).
        for i in (loc + 1)..(numLocales - 1) {
          if firstOffsetPerLoc[i] != -1 {
            lastOffsetEndPerLoc[loc] = firstOffsetPerLoc[i];
            break;
          }
        }
        // If there is no later locale with offsets, then this locale's
        // last string ends at the end of the byte buffer (exclusive).
        if lastOffsetEndPerLoc[loc] == -1 {
          lastOffsetEndPerLoc[loc] = endOfBytes + 1;
        }
      }

    }

    // For each locale, these track the index of the first/last string (in the
    // global offsets array) that intersects this locale's byte chunk.
    var firstStringIndices, lastStringIndices: [LocaleSpace] int = -1;

    // Global start byte offset of the first/last string for each locale
    var firstStringStartOffsets, lastStringStartOffsets: [LocaleSpace] int = -1;

    // Total size (in bytes) of the first/last string for each locale (full string size,
    // not just the portion on that locale).
    var firstStringSizes, lastStringSizes: [LocaleSpace] int = -1;

    // Number of bytes of the first/last string that actually reside on each locale.
    // This may be less than the full string size if the string is split across locales.
    var firstStringNumBytesEachLocale, lastStringNumBytesEachLocale: [LocaleSpace] int = -1;

    // Now, on each locale, we examine the portion of the offsets that live here to
    // figure out which strings intersect each locale's local byte chunk.
    coforall loc in Locales do on loc {
      // Upper bound (exclusive) for strings that start on this locale
      const myLastOffsetEnd = lastOffsetEndPerLoc[here.id];

      // Local slice of the offsets that actually reside on this locale
      ref myOffsets = offsets.localSlice[offsets.localSubdomain()];

      // Local copies of global first/last byte indices per locale
      const myFirstIdxPerLoc = firstIdxPerLoc;
      const myLastIdxPerLoc = lastIdxPerLoc;

      if !myOffsets.isEmpty() {

        // For *every* locale i, check whether that locale's byte range overlaps
        // any string whose starting offset is in myOffsets (i.e., whose segment
        // begins on this locale).
        forall i in 0..#numLocales {
          // ----- FIRST STRING FOR LOCALE i -----
          // If locale i has bytes, and its first byte index falls within the
          // range of strings that start on this locale, then find which string
          // contains that starting byte.
          if myFirstIdxPerLoc[i] >= myOffsets.first && myFirstIdxPerLoc[i] < myLastOffsetEnd {
            // Find the greatest offset in myOffsets that is <= myFirstIdxPerLoc[i]
            // binarySearch returns (found?, index)
            var result = binarySearch(myOffsets, myFirstIdxPerLoc[i]);
            const idx = if result[0] then result[1] else result[1] - 1;

            // Record the global string index for the first string on locale i
            firstStringIndices[i] = idx;
            firstStringStartOffsets[i] = myOffsets[idx];

            // Determine the end of that string: next offset if available,
            // otherwise the lastOffsetEnd for this locale.
            const endStr = if idx < myOffsets.domain.last
                           then myOffsets[idx + 1]
                           else myLastOffsetEnd;

            // Full size of the string in bytes
            const size = endStr - myOffsets[idx];
            firstStringSizes[i] = size;

            // Portion of this string that resides on locale i's byte range
            // (truncate end to not go past that locale’s last byte + 1)
            const endStrThisLoc = min(endStr, myLastIdxPerLoc[i] + 1);
            firstStringNumBytesEachLocale[i] = endStrThisLoc - myFirstIdxPerLoc[i];
          }

          // ----- LAST STRING FOR LOCALE i -----
          // Similarly, if locale i has bytes and its last byte index falls in
          // this locale’s string-start range, determine the last string that
          // intersects locale i.
          if myLastIdxPerLoc[i] >= myOffsets.first && myLastIdxPerLoc[i] < myLastOffsetEnd {
            var result = binarySearch(myOffsets, myLastIdxPerLoc[i]);
            const idx = if result[0] then result[1] else result[1] - 1;

            // Record the global string index for the last string on locale i
            lastStringIndices[i] = idx;
            lastStringStartOffsets[i] = myOffsets[idx];

            // End of that string, similar logic as above
            const endStr = if idx < myOffsets.domain.last
                           then myOffsets[idx + 1]
                           else myLastOffsetEnd;

            // Full size of the string in bytes
            const size = endStr - myOffsets[idx];
            lastStringSizes[i] = size;

            // Portion of this string that resides on locale i:
            // start at max of the string start and the locale’s first byte
            const beginStrThisLoc = max(myOffsets[idx], myFirstIdxPerLoc[i]);
            lastStringNumBytesEachLocale[i] = myLastIdxPerLoc[i] + 1 - beginStrThisLoc;
          }
        }

      }
    }

    // Flatten per-locale first/last string info into a single array so we can
    // iterate over all "boundary strings" (first/last per locale) uniformly.
    // Even indices (2*i) hold first-string info for locale i,
    // odd  indices (2*i+1) hold last-string info for locale i.
    var stringIndices: [0..#(2*numLocales)] int = -1;
    var stringNumBytesEachLocale: [0..#(2*numLocales)] int = -1;

    // Pack first/last string indices and their per-locale byte counts into
    // the flattened arrays above.
    for i in 0..#numLocales {
      stringIndices[2*i] = firstStringIndices[i];
      stringIndices[2*i + 1] = lastStringIndices[i];
      stringNumBytesEachLocale[2*i] = firstStringNumBytesEachLocale[i];
      stringNumBytesEachLocale[2*i + 1] = lastStringNumBytesEachLocale[i];
    }

    // For each locale, flags indicating whether it should "own" the lower
    // (first) or upper (last) boundary string for that locale.
    // Only one locale will own each boundary string globally, to avoid
    // reconstructing shared boundary strings multiple times.
    var takeLowerBoundaryString: [0..#numLocales] bool = false;
    var takeUpperBoundaryString: [0..#numLocales] bool = false;

    // Initialize the current string index (global string ID) we are tracking
    // among all boundary strings.
    var currStringIdx = stringIndices[0];
    // Track, for the current string, which locale has the largest number of
    // bytes of that string locally.
    var bestBytesPerLoc = stringNumBytesEachLocale[0];
    var bestLoc = 0;

    // Sweep over all boundary entries (first+last per locale) to choose,
    // for each global string index, a single owning locale (the one with
    // the largest local byte contribution of that string).
    for i in 1..<(2 * numLocales) {
      if currStringIdx == -1 {
        // We haven't started tracking a string yet: find the first valid one.
        if stringIndices[i] != -1 {
          currStringIdx = stringIndices[i];
          bestBytesPerLoc = stringNumBytesEachLocale[i];
          // Recover locale index by dividing i by 2 (bit-shift).
          bestLoc = i >> 1;
        } else {
          continue;
        }
      }
      // If we encounter a *different* string index (and it's valid), then we
      // finalize ownership for the previous string and start tracking this new one.
      if stringIndices[i] != currStringIdx && stringIndices[i] != -1 {
        // If this locale's best entry for currStringIdx was recorded at its
        // "lower boundary" slot (2*bestLoc), mark that lower boundary as taken.
        // Otherwise, it must have been the upper boundary for that locale.
        if stringIndices[bestLoc << 1] == currStringIdx then
          takeLowerBoundaryString[bestLoc] = true;
        else
          takeUpperBoundaryString[bestLoc] = true;

        // Start tracking the new string index
        currStringIdx = stringIndices[i];
        bestBytesPerLoc = stringNumBytesEachLocale[i];
        bestLoc = i >> 1;
      } else if stringNumBytesEachLocale[i] > bestBytesPerLoc {
        // Same string index as currStringIdx: update the best locale if this
        // boundary entry has more bytes for that string.
        bestBytesPerLoc = stringNumBytesEachLocale[i];
        bestLoc = i >> 1;
      }
    }

    // Finalize ownership for the last string we were tracking (if any).
    if currStringIdx != -1 {
      if stringIndices[bestLoc << 1] == currStringIdx then
        takeLowerBoundaryString[bestLoc] = true;
      else
        takeUpperBoundaryString[bestLoc] = true;
    }

    // Number of strings to reconstruct per locale (including chosen boundary
    // strings, plus any fully-contained interior strings).
    var numStringsPerLocale: [0..#numLocales] int = -1;

    for i in 0..#numLocales {
      // Count the number of *interior* strings fully contained on this locale:
      // strings whose indices are strictly between firstStringIndices[i]
      // and lastStringIndices[i]. If first == last, there are no interior strings.
      if lastStringIndices[i] != firstStringIndices[i] then
        numStringsPerLocale[i] = lastStringIndices[i] - firstStringIndices[i] - 1;
      else
        numStringsPerLocale[i] = 0;

      // Add 1 if this locale was chosen to own the lower boundary string,
      // and/or 1 if it owns the upper boundary string.
      numStringsPerLocale[i] += takeLowerBoundaryString[i] + takeUpperBoundaryString[i];
    }

    // Result: for each locale, we store an innerArray(string) holding the
    // subset of strings that are "owned" by that locale (based on the
    // boundary-ownership and interior-string logic computed above).
    var decodedStrings: [PrivateSpace] innerArray(string);

    // Reconstruct the actual string values on each locale.
    coforall loc in Locales do on loc {

      // Number of strings this locale is responsible for.
      const numStringsThisLocale = numStringsPerLocale[here.id];

      // Only do work if this locale has at least one string to reconstruct.
      if numStringsThisLocale > 0 {

        // Allocate the innerArray to hold the decoded strings for this locale.
        decodedStrings[here.id] = new innerArray({0..#numStringsThisLocale}, string);

        // Convenience reference to the backing array of strings.
        ref myDecodedStringsArr = decodedStrings[here.id].Arr;

        // Global indices (in offsets) of the first and last strings that intersect
        // this locale’s byte chunk.
        const lowerBoundary = firstStringIndices[here.id];
        const upperBoundary = lastStringIndices[here.id];

        // Current write position in myDecodedStringsArr.
        var i = 0;

        // ----- LOWER BOUNDARY STRING (if this locale owns it) -----
        if takeLowerBoundaryString[here.id] {

          // Global starting byte offset of the lower boundary string.
          const currOffset = offsets[lowerBoundary];

          // Full size (bytes) of the lower boundary string.
          const size = firstStringSizes[here.id];

          // Extract the full string bytes into a local byte array.
          // The slice strBytes[currOffset..#size] has 'size' elements.
          var byteArray: [0..#size] uint(8) = strBytes[currOffset..#size];

          // Interpret the first 'size' bytes as a string (excluding any terminator
          // at the end, if it’s included in size).
          myDecodedStringsArr[i] = interpretAsString(byteArray, 0..<size);

          // Move on to the next slot.
          i += 1;

        }

        // ----- INTERIOR STRINGS FULLY CONTAINED ON THIS LOCALE -----
        // If there are any strings whose indices lie strictly between
        // lowerBoundary and upperBoundary, they are fully contained on this
        // locale’s byte range. We can reconstruct them directly from local bytes.
        if upperBoundary - lowerBoundary > 1 {

          // Offsets for the interior strings: these are the offsets for strings
          // with indices (lowerBoundary+1) .. upperBoundary.
          // The domain is sized to match the number of interior strings, but
          // starts at 'i' so we can directly index with 'idx' below.
          var relevantOffsets: [i..#(upperBoundary - lowerBoundary)] int =
            offsets[(lowerBoundary + 1)..upperBoundary];

          // Local view of the bytes on this locale.
          ref myBytes = strBytes.localSlice[strBytes.localSubdomain()];

          // For each interior string, slice the local bytes between consecutive
          // offsets and convert that region into a string.
          forall idx in i..#(upperBoundary - lowerBoundary - 1) {

            myDecodedStringsArr[idx] =
              interpretAsString(myBytes, relevantOffsets[idx]..<relevantOffsets[idx + 1]);

          }

          // Advance 'i' past all interior strings just written.
          i += upperBoundary - lowerBoundary - 1;

        }

        // ----- UPPER BOUNDARY STRING (if this locale owns it) -----
        if takeUpperBoundaryString[here.id] {

          // Global starting byte offset of the upper boundary string.
          const currOffset = offsets[upperBoundary];

          // Full size (bytes) of the upper boundary string.
          const size = lastStringSizes[here.id];

          // Extract the full string bytes into a local byte array.
          var byteArray: [0..#size] uint(8) = strBytes[currOffset..#size];

          // Interpret those bytes as a string.
          myDecodedStringsArr[i] = interpretAsString(byteArray, 0..<size);

        }

      }

    }

    // Return the per-locale collection of decoded strings.
    return decodedStrings;

  }

  // Build a SegString from a distributed array-of-arrays of strings.
  // Each locale owns one innerArray(string), and we pack all those
  // strings into a single SegString representation.
  proc segStringFromInnerArray(const ref stringsArrays: [] innerArray(string), st: borrowed SymTab):
    SegString throws
  {

    // Per-locale counts: number of strings on each locale
    var numStringsPerLocale: [0..#numLocales] int;
    // Per-locale counts: total number of bytes needed to store all of that
    // locale's strings (including null-terminators) in a flat byte buffer
    var numBytesPerLocale: [0..#numLocales] int;

    // First pass: for each locale, count how many strings and how many bytes
    coforall loc in Locales do on loc {
      // Local reference to this locale's innerArray of strings
      ref myArr = stringsArrays[here.id].Arr;

      // Store number of strings on this locale
      numStringsPerLocale[here.id] = myArr.size;

      // Compute total bytes for all strings on this locale
      var allStringsSize = 0;

      // Sum up the length of each string plus 1 byte for null terminator
      forall s in myArr with (+ reduce allStringsSize) {
        // + 1 for null-terminated strings in bytes
        allStringsSize += s.size + 1;
      }

      // Store total byte count for this locale
      numBytesPerLocale[here.id] = allStringsSize;
    }

    // Exclusive-style prefix sums to compute offsets by locale.
    // stringOffsetByLocale[i] will be the starting index in the global
    // segment array for locale i's strings.
    var stringOffsetByLocale = + scan numStringsPerLocale;
    // Similarly, bytesOffsetByLocale[i] is the starting byte index for
    // locale i's string data in the global byte array.
    var bytesOffsetByLocale = + scan numBytesPerLocale;

    // Allocate symbol-table entries for the global segment array (esegs)
    // and the underlying byte array (evals).
    // The "last" value of the scan is the total number of elements.
    var esegs = createTypedSymEntry(stringOffsetByLocale.last, int);
    var evals = createTypedSymEntry(bytesOffsetByLocale.last, uint(8));

    // Shorthand references to the underlying arrays
    ref esa = esegs.a;
    ref eva = evals.a;

    // Convert inclusive scans into per-locale starting offsets by subtracting
    // the local counts.
    stringOffsetByLocale -= numStringsPerLocale;
    bytesOffsetByLocale -= numBytesPerLocale;

    // Allocate and wire up a new SegString that uses the segment and
    // byte arrays we just created.
    var retString = assembleSegStringFromParts(esegs, evals, st);

    // Second pass: for each locale, fill in its portion of the global
    // segment offsets (esa) and bytes (eva).
    coforall loc in Locales do on loc {

      // Local array of strings on this locale
      ref strArray = stringsArrays[here.id].Arr;

      // Local scratch buffer to hold all bytes for this locale's strings
      var myBytes: [0..#numBytesPerLocale[here.id]] uint(8);

      // Per-string offsets (start index in myBytes) for this locale
      var strOffsets: [0..#strArray.size] int;

      // Per-string sizes (including null-terminator) for this locale
      var strSizes: [0..#strArray.size] int;

      // Compute size of each string in bytes, including the null terminator
      forall (i, str) in zip(0..#strArray.size, strArray) {
        strSizes[i] = str.size + 1;
      }

      // Compute prefix sum of sizes to get offsets
      strOffsets = (+ scan strSizes) - strSizes;

      // Copy each string's raw bytes into the local buffer myBytes at the
      // appropriate offset. We only copy "size - 1" bytes and leave the
      // last byte (null terminator) to be implicitly zero or set elsewhere.
      forall (i, size, str) in zip(0..#strArray.size, strSizes, strArray) {

        const strBytes = str.bytes();

        // Copy the string bytes into the slice of myBytes that corresponds
        // to this string's segment.
        myBytes[strOffsets[i]..#(size-1)] = strBytes[0..#(size-1)];

      }

      // Write this locale's segment offsets into the global segment array.
      // We shift each local offset by the global byte offset for this locale,
      // and place them in the appropriate slice of esa.
      esa[stringOffsetByLocale[here.id]..#strArray.size] =
        strOffsets[0..#strArray.size] + bytesOffsetByLocale[here.id];

      // Write this locale's byte data into the global byte array in its
      // assigned slice.
      eva[bytesOffsetByLocale[here.id]..#numBytesPerLocale[here.id]] =
        myBytes[0..#numBytesPerLocale[here.id]];

    }

    // Return the fully assembled SegString
    return retString;
  }

  // Note, the arrays passed here must have PrivateSpace domains.
  proc repartitionByHashString(const ref strOffsets: [] list(int),
                                 const ref strBytes: [] list(uint(8))):
    ([PrivateSpace] list(int), [PrivateSpace] list(uint(8)))
  {
    var destLocales: [PrivateSpace] list(int);

    coforall loc in Locales do on loc {

      var myStrOffsets = strOffsets[here.id].toArray();
      var myStrBytes = strBytes[here.id].toArray();
      var myDestLocales: [0..#myStrOffsets.size] int;

      forall i in myStrOffsets.domain {
        var start = myStrOffsets[i];
        var end = if i == myStrOffsets.size - 1 then myStrBytes.size else myStrOffsets[i + 1];
        var str = interpretAsString(myStrBytes, start..<end);
        myDestLocales[i] = (str.hash() % numLocales): int;
      }

      destLocales[here.id] = new list(myDestLocales);
    }

    return repartitionByLocaleString(destLocales, strOffsets, strBytes);
  }

  // Note, the arrays passed here must have PrivateSpace domains.
  proc repartitionByHashStringArray(const ref strOffsets: [] innerArray(int),
                                 const ref strBytes: [] innerArray(uint(8))):
    ([PrivateSpace] innerArray(int), [PrivateSpace] innerArray(uint(8)))
  {
    var destLocales: [PrivateSpace] innerArray(int);

    coforall loc in Locales do on loc {

      ref myStrOffsets = strOffsets[here.id].Arr;
      ref myStrBytes = strBytes[here.id].Arr;
      destLocales[here.id] = new innerArray({0..#myStrOffsets.size}, int);
      ref myDestLocales = destLocales[here.id].Arr;

      forall i in myStrOffsets.domain {
        const start = myStrOffsets[i];
        const end = if i == myStrOffsets.size - 1 then myStrBytes.size else myStrOffsets[i + 1];
        const str = interpretAsString(myStrBytes, start..<end);
        myDestLocales[i] = (str.hash() % numLocales): int;
      }
    }

    return repartitionByLocaleStringArray(destLocales, strOffsets, strBytes);
  }

  // Note, the arrays passed here must have PrivateSpace domains.
  proc repartitionByHashArray(type t,
                              const ref vals: [] innerArray(t))
  {
    var destLocales: [PrivateSpace] innerArray(int);

    coforall loc in Locales do on loc {

      ref myVals = vals[here.id].Arr;
      destLocales[here.id] = new innerArray({myVals.domain.low..myVals.domain.high}, int);
      ref myDestLocales = destLocales[here.id].Arr;

      forall (val, idx) in zip(myVals, myVals.domain) {
        myDestLocales[idx] = (val.hash() % numLocales): int;
      }
    }

    return repartitionByLocaleArray(t, destLocales, vals);
  }

  // Note, the arrays passed here must have PrivateSpace domains. With Chapel
  // 2.5, distribution equality with PrivateSpace doesn't work.
  // https://github.com/chapel-lang/chapel/pull/27397 is the upstream PR to
  // fix it.
  proc repartitionByLocaleString(const ref destLocales: [] list(int),
                                 const ref strOffsets: [] list(int),
                                 const ref strBytes: [] list(uint(8))):
    ([PrivateSpace] list(int), [PrivateSpace] list(uint(8)))
  {
    var maxBytesPerLocale: int;
    var maxStringsPerLocale: int;
    var numBytesReceivingByLocale: [PrivateSpace] [0..#numLocales] int;
    var numStringsReceivingByLocale: [PrivateSpace] [0..#numLocales] int;
    var allStrSizes: [PrivateSpace] list(int);

    // First we need to figure out how many bytes and strings are getting transferred.
    // Also calculating the sizes of each string so that indexing is easier down the road.

    coforall loc in Locales 
      with (max reduce maxBytesPerLocale, max reduce maxStringsPerLocale) 
      do on loc
    {
      const myDestLocales = destLocales[here.id].toArray();
      const myStrOffsets = strOffsets[here.id].toArray();
      const myStrBytesSize = strBytes[here.id].size;
      var bytesPerLocale: [0..#numLocales] int = 0;
      var stringsPerLocale: [0..#numLocales] int = 0;
      var sizes: [0..#myDestLocales.size] int = 0;

      forall idx in myDestLocales.domain with (+ reduce bytesPerLocale, + reduce stringsPerLocale) {
        var destLoc = myDestLocales[idx];
        const start = myStrOffsets[idx];
        const end = if idx == myDestLocales.size - 1 then myStrBytesSize else myStrOffsets[idx + 1];
        const size = end - start;

        sizes[idx] = size;
        bytesPerLocale[destLoc] += size;
        stringsPerLocale[destLoc] += 1;
      }

      maxBytesPerLocale = max reduce bytesPerLocale;
      maxStringsPerLocale = max reduce stringsPerLocale;

      forall i in 0..#numLocales {
        numBytesReceivingByLocale[i][here.id] = bytesPerLocale[i];
        numStringsReceivingByLocale[i][here.id] = stringsPerLocale[i];
      }

      allStrSizes[here.id] = new list(sizes);
    }

    var recvOffsets: [PrivateSpace] [0..#numLocales] [0..#maxStringsPerLocale] int;
    var recvBytes: [PrivateSpace] [0..#numLocales] [0..#maxBytesPerLocale] uint(8);

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      const myDestLocales = destLocales[here.id].toArray();
      const myStrOffsets = strOffsets[here.id].toArray();
      const myStrBytes = strBytes[here.id].toArray();
      const mySizes = allStrSizes[here.id].toArray();
      var idxInDestLoc: [0..#myDestLocales.size] int = 0;
      var offsetInDestLoc: [0..#myDestLocales.size] int = 0;
      var bytesPerLocale: [0..#numLocales] int = 0;
      var numStringsPerLocale: [0..#numLocales] int = 0;

      // First we need to figure out what the destination index will be
      // for each offset and for the bytes

      for i in 0..#numLocales {
        var onCurrLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then 1 else 0;
        var currLocSizes = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then mySizes[j] else 0;
        
        var idxInCurrLoc = (+ scan onCurrLoc) - onCurrLoc;
        idxInDestLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then idxInCurrLoc[j] 
                                                     else idxInDestLoc[j];

        var offsetInCurrLoc = (+ scan currLocSizes) - currLocSizes;
        offsetInDestLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then offsetInCurrLoc[j]
                                                        else offsetInDestLoc[j];

        bytesPerLocale[i] = + reduce currLocSizes;
        numStringsPerLocale[i] = + reduce onCurrLoc;
      }

      var sendOffsets: [0..#numLocales] [0..#maxStringsPerLocale] int;
      var sendBytes: [0..#numLocales] [0..#maxBytesPerLocale] uint(8);

      forall idx in 0..#myDestLocales.size {

        var destLoc = myDestLocales[idx];
        var idxInOffsetArr = idxInDestLoc[idx];
        var offset = offsetInDestLoc[idx];
        var size = mySizes[idx];
        
        sendOffsets[destLoc][idxInOffsetArr] = offset;
        sendBytes[destLoc][offset..#size] = myStrBytes[myStrOffsets[idx]..#size];

      }

      // Maybe could be a forall but I don't know how that plays with the bulk transfer.
      for i in 0..#numLocales {
        recvOffsets[i][here.id][0..#numStringsPerLocale[i]] = sendOffsets[i][0..#numStringsPerLocale[i]];
        recvBytes[i][here.id][0..#bytesPerLocale[i]] = sendBytes[i][0..#bytesPerLocale[i]];
      }

    }

    var returnedOffsets: [PrivateSpace] list(int);
    var returnedBytes: [PrivateSpace] list(uint(8));

    // Now that the buffers have been filled, we're going to group them together into a single list.
    // Strictly speaking, this probably isn't necessary, but it does make it more friendly to work with

    coforall loc in Locales do on loc {
      const ref numBytesReceivedByLoc = numBytesReceivingByLocale[here.id];
      const ref numStringsReceivedByLoc = numStringsReceivingByLocale[here.id];
      const ref myRecvOffsets = recvOffsets[here.id];
      const ref myRecvBytes = recvBytes[here.id];
      var numBytesReceived = + reduce numBytesReceivedByLoc;
      var numStringsReceived = + reduce numStringsReceivedByLoc;
      var myOffsets: [0..#numStringsReceived] int = 0;
      var myBytes: [0..#numBytesReceived] uint(8) = 0;
      var byteOffsetAdjuster = (+ scan numBytesReceivedByLoc) - numBytesReceivedByLoc;
      var idxOffsetAdjuster = (+ scan numStringsReceivedByLoc) - numStringsReceivedByLoc;

      for i in 0..#numLocales {

        var byteOffsetThisLoc = byteOffsetAdjuster[i];
        var idxOffsetThisLoc = idxOffsetAdjuster[i];
        myOffsets[idxOffsetThisLoc..#numStringsReceivedByLoc[i]]
          = myRecvOffsets[i][0..#numStringsReceivedByLoc[i]] + byteOffsetThisLoc;
        myBytes[byteOffsetThisLoc..#numBytesReceivedByLoc[i]]
          = myRecvBytes[i][0..#numBytesReceivedByLoc[i]];

      }

      returnedOffsets[here.id] = new list(myOffsets);
      returnedBytes[here.id] = new list(myBytes);

    }

    return (returnedOffsets, returnedBytes);

  }

  proc repartitionByLocaleStringArray(const ref destLocales: [] innerArray(int),
                                 const ref strOffsets: [] innerArray(int),
                                 const ref strBytes: [] innerArray(uint(8))):
    ([PrivateSpace] innerArray(int), [PrivateSpace] innerArray(uint(8)))
  {
    var numBytesSendingByLocale: [PrivateSpace] [0..#numLocales] int;
    var numStringsSendingByLocale: [PrivateSpace] [0..#numLocales] int;
    var allStrSizes: [PrivateSpace] innerArray(int);
    var sendOffsets: [PrivateSpace] [0..#numLocales] innerArray(int);
    var sendBytes: [PrivateSpace] [0..#numLocales] innerArray(uint(8));

    // First we need to figure out how many bytes and strings are getting transferred.
    // Also calculating the sizes of each string so that indexing is easier down the road.

    coforall loc in Locales do on loc
    {
      const ref myDestLocales = destLocales[here.id].Arr;
      const ref myStrOffsets = strOffsets[here.id].Arr;
      const ref myStrBytes = strBytes[here.id].Arr;
      const myStrBytesSize = myStrBytes.size;
      var bytesPerLocale: [0..#numLocales] int = 0;
      var stringsPerLocale: [0..#numLocales] int = 0;
      allStrSizes[here.id] = new innerArray(myDestLocales.domain, int);
      ref sizes = allStrSizes[here.id].Arr;
      const topEnd = myDestLocales.domain.high;

      forall idx in myDestLocales.domain with (+ reduce bytesPerLocale, + reduce stringsPerLocale) {
        var destLoc = myDestLocales[idx];
        const start = myStrOffsets[idx];
        const end = if idx == topEnd then myStrBytesSize else myStrOffsets[idx + 1];
        const size = end - start;

        sizes[idx] = size;
        bytesPerLocale[destLoc] += size;
        stringsPerLocale[destLoc] += 1;
      }

      numBytesSendingByLocale[here.id] = bytesPerLocale;
      numStringsSendingByLocale[here.id] = stringsPerLocale;

      var currLocIndAllLocales: [myDestLocales.domain] int;
      var currLocOffsetAllLocales: [myDestLocales.domain] int;

      /*
      // It would be very cool if we could do things this way:

      for i in 0..#numLocales {
        sendOffsets[here.id][i] = new innerArray({0..#stringsPerLocale[i]}, int);
        sendBytes[here.id][i] = new innerArray({0..#bytesPerLocale[i]}, uint(8));

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        const currLocSizes = doCurrLoc * sizes;
        const currLocOffsets = (+ scan currLocSizes) - currLocSizes;

        currLocIndAllLocales += doCurrLoc * currLocInd;
        currLocOffsetAllLocales += doCurrLoc * currLocOffsets;
      }

      ref currSendOffsets = [i in 0..#numLocales] sendOffsets[here.id][i].Arr;
      ref currSendBytes = [i in 0..#numLocales] sendBytes[here.id][i].Arr;

      forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
        const currSize = sizes[j];
        currSendOffsets[dl][currLocIndAllLocales[j]] = currLocOffsetAllLocales[j];
        currSendBytes[dl][currLocOffsetAllLocales[j]..#currSize]
                      = myStrBytes[myStrOffsets[j]..#currSize];
      }

      // Notice that there's only one forall at the end. This would potentially be faster.
      // But because we can't do an array of refs (currently, at least), there's going to be a forall
      // inside the for, and we do a forall for each of the locales. With all the vectorized stuff
      // we're kind of already doing that so I'm not convinced what follows is significantly slower.
      */

      for i in 0..#numLocales {
        sendOffsets[here.id][i] = new innerArray({0..#stringsPerLocale[i]}, int);
        sendBytes[here.id][i] = new innerArray({0..#bytesPerLocale[i]}, uint(8));

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        const currLocSizes = doCurrLoc * sizes;
        const currLocOffsets = (+ scan currLocSizes) - currLocSizes;

        currLocIndAllLocales += doCurrLoc * currLocInd;
        currLocOffsetAllLocales += doCurrLoc * currLocOffsets;

        ref currSendOffsets = sendOffsets[here.id][i].Arr;
        ref currSendBytes = sendBytes[here.id][i].Arr;

        forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
          if dl == i {
            const currSize = sizes[j];
            currSendOffsets[currLocIndAllLocales[j]] = currLocOffsetAllLocales[j];
            currSendBytes[currLocOffsetAllLocales[j]..#currSize]
                        = myStrBytes[myStrOffsets[j]..#currSize];
          }
        }
      }
      
    }

    var recvOffsets: [PrivateSpace] innerArray(int);
    var recvBytes: [PrivateSpace] innerArray(uint(8));

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      
      const numStringsReceivingByLocale = [i in 0..#numLocales] numStringsSendingByLocale[i][here.id];
      const numBytesReceivingByLocale = [i in 0..#numLocales] numBytesSendingByLocale[i][here.id];
      const stringOffsetByLocale = (+ scan numStringsReceivingByLocale) - numStringsReceivingByLocale;
      const byteOffsetByLocale = (+ scan numBytesReceivingByLocale) - numBytesReceivingByLocale;

      recvBytes[here.id] = new innerArray({0..#(+ reduce numBytesReceivingByLocale)}, uint(8));
      recvOffsets[here.id] = new innerArray({0..#(+ reduce numStringsReceivingByLocale)}, int);
      ref myRecvBytes = recvBytes[here.id].Arr;
      ref myRecvOffsets = recvOffsets[here.id].Arr;

      for i in 0..#numLocales {

        // myRecvOffsets[stringOffsetByLocale[i]..#numStringsReceivingByLocale[i]]
        //              = sendOffsets[i][here.id].Arr + byteOffsetByLocale[i];
        myRecvOffsets[stringOffsetByLocale[i]..#numStringsReceivingByLocale[i]]
                    = sendOffsets[i][here.id].Arr;
        myRecvOffsets[stringOffsetByLocale[i]..#numStringsReceivingByLocale[i]] += byteOffsetByLocale[i];
        myRecvBytes[byteOffsetByLocale[i]..#numBytesReceivingByLocale[i]] = sendBytes[i][here.id].Arr;

      }

    }

    return (recvOffsets, recvBytes);

  }

  // Note, the arrays passed here must have PrivateSpace domains.
  proc repartitionByLocale(type t,
                           const ref destLocales: [] list(int),
                           const ref vals: [] list(t))
  {
    type eltType = vals.eltType.eltType;

    var maxValsPerLocale: int;
    var numValsReceivingByLocale: [PrivateSpace] [0..#numLocales] int;

    coforall loc in Locales 
      with (max reduce maxValsPerLocale) 
      do on loc
    {
      const ref myDestLocales = destLocales[here.id];
      const ref myVals = vals[here.id];
      var valsPerLocale: [0..#numLocales] int = 0;

      forall idx in 0..#myDestLocales.size with (+ reduce valsPerLocale) {
        var destLoc = myDestLocales[idx];
        valsPerLocale[destLoc] += 1;
      }

      maxValsPerLocale = max reduce valsPerLocale;

      forall i in 0..#numLocales {
        numValsReceivingByLocale[i][here.id] = valsPerLocale[i];
      }

    }

    var recvVals: [PrivateSpace] [0..#numLocales] [0..#maxValsPerLocale] eltType;

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      const ref myDestLocales = destLocales[here.id];
      const ref myVals = vals[here.id];
      var idxInDestLoc: [0..#myDestLocales.size] int = 0;
      var numValsPerLocale: [0..#numLocales] int = 0;

      // First we need to figure out what the destination index will be for each value

      for i in 0..#numLocales {
        var onCurrLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then 1 else 0;
        
        var idxInCurrLoc = (+ scan onCurrLoc) - onCurrLoc;
        idxInDestLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then idxInCurrLoc[j] 
                                                     else idxInDestLoc[j];

        numValsPerLocale[i] = + reduce onCurrLoc;
      }

      var sendVals: [0..#numLocales] [0..#maxValsPerLocale] eltType;

      forall idx in 0..#myDestLocales.size {

        var destLoc = myDestLocales[idx];
        var idxInValArr = idxInDestLoc[idx];
        
        sendVals[destLoc][idxInValArr] = myVals[idx];

      }

      // Maybe could be a forall but I don't know how that plays with the bulk transfer.
      for i in 0..#numLocales {
        recvVals[i][here.id][0..#numValsPerLocale[i]] = sendVals[i][0..#numValsPerLocale[i]];
      }

    }

    var returnedVals: [PrivateSpace] list(eltType);

    // Now that the buffers have been filled, we're going to group them together into a single list.
    // Strictly speaking, this probably isn't necessary, but it does make it more friendly to work with

    coforall loc in Locales do on loc {
      const ref numValsReceivedByLoc = numValsReceivingByLocale[here.id];
      const ref myRecvVals = recvVals[here.id];
      var numValsReceived = + reduce numValsReceivedByLoc;
      var myVals: [0..#numValsReceived] eltType;
      var idxOffsetAdjuster = (+ scan numValsReceivedByLoc) - numValsReceivedByLoc;

      for i in 0..#numLocales {

        var idxOffsetThisLoc = idxOffsetAdjuster[i];
        myVals[idxOffsetThisLoc..#numValsReceivedByLoc[i]] = myRecvVals[i][0..#numValsReceivedByLoc[i]];

      }

      returnedVals[here.id] = new list(myVals);

    }

    return returnedVals;

  }

  proc repartitionByLocaleArray(type t,
                                const ref destLocales: [] innerArray(int),
                                const ref vals: [] innerArray(t))
  {
    type eltType = vals.eltType.t;

    var numValsSendingByLocale: [PrivateSpace] [0..#numLocales] int;
    var sendVals: [PrivateSpace] [0..#numLocales] innerArray(t);

    coforall loc in Locales do on loc
    {

      const ref myDestLocales = destLocales[here.id].Arr;
      const ref myVals = vals[here.id].Arr;
      var valsPerLocale: [0..#numLocales] int = 0;

      forall idx in myDestLocales.domain with (+ reduce valsPerLocale) {
        var destLoc = myDestLocales[idx];
        valsPerLocale[destLoc] += 1;
      }

      numValsSendingByLocale[here.id] = valsPerLocale;
      
      var currLocIndAllLocales: [myDestLocales.domain] int;

      /*
      // It would be very cool if we could do things this way:

      for i in 0..#numLocales {
        sendVals[here.id][i] = new innerArray({0..#valsPerLocale[i]}, eltType);

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        
        currLocIndAllLocales += doCurrLoc * currLocInd;
      }

      ref currSendVals = [i in 0..#numLocales] sendVals[here.id][i].Arr;

      forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
        currSendVals[dl][currLocIndAllLocales[j]] = myVals[j];
      }

      // Notice that there's only one forall at the end. This would potentially be faster.
      // But because we can't do an array of refs (currently, at least), there's going to be a forall
      // inside the for, and we do a forall for each of the locales. With all the vectorized stuff
      // we're kind of already doing that so I'm not convinced what follows is significantly slower.
      */

      for i in 0..#numLocales {
        sendVals[here.id][i] = new innerArray({0..#valsPerLocale[i]}, eltType);

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        
        currLocIndAllLocales += doCurrLoc * currLocInd;

        ref currSendVals = sendVals[here.id][i].Arr;

        forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
          if dl == i {
            currSendVals[currLocIndAllLocales[j]] = myVals[j];
          }
        }
      }

    }

    var recvVals: [PrivateSpace] innerArray(eltType);

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      
      const numValsReceivingByLocale = [i in 0..#numLocales] numValsSendingByLocale[i][here.id];
      const valOffsetByLocale = (+ scan numValsReceivingByLocale) - numValsReceivingByLocale;

      recvVals[here.id] = new innerArray({0..#(+ reduce numValsReceivingByLocale)}, eltType);

      ref myRecvVals = recvVals[here.id].Arr;

      for i in 0..#numLocales {

        myRecvVals[valOffsetByLocale[i]..#numValsReceivingByLocale[i]] = sendVals[i][here.id].Arr;
        
      }

    }

    return recvVals;

  }

  proc repartitionByLocaleMultiArray(type t,
                                    const ref destLocales: [] innerArray(int),
                                    const ref vals: [] [] innerArray(t))
  {
    // Notes:
    // - We assume `vals` is indexed as [field][PrivateSpace], i.e., vals[k][here.id]
    // - `innerArray(t)` is your existing wrapper with `.Arr` and
    //   a ctor like new innerArray(dom, eltType)

    type eltType = t;

    const Fields = vals.domain.dim(0);        // range over the "multi" dimension
    const LocaleRange = 0..#numLocales;       // 0..numLocales-1

    var numValsSendingByLocale: [PrivateSpace] [LocaleRange] int;
    var sendVals: [Fields] [PrivateSpace] [LocaleRange] innerArray(eltType);

    // 1) Build per-locale send buffers on each locale (for all fields)
    coforall loc in Locales do on loc {

      const ref myDestLocales = destLocales[here.id].Arr;

      // Count how many elements this locale will send to each destination locale.
      var valsPerLocale: [LocaleRange] int = 0;

      forall idx in myDestLocales.domain with (+ reduce valsPerLocale) {
        const destLoc = myDestLocales[idx];
        valsPerLocale[destLoc] += 1;
      }

      numValsSendingByLocale[here.id] = valsPerLocale;

      // We'll reuse the computed positions for each destination locale.
      var currLocIndAllLocales: [myDestLocales.domain] int = 0;

      // For each destination locale, compute its local indices and fill all fields.
      for i in LocaleRange {
        // Allocate one send buffer per (field, destLocale)
        for k in Fields {
          sendVals[k][here.id][i] = new innerArray({0..#valsPerLocale[i]}, eltType);
        }

        // Boolean mask of entries headed to locale i; then 0-based indices within that subset.
        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;

        // Accumulate the per-position index (same for all fields)
        currLocIndAllLocales += doCurrLoc * currLocInd;

        // Fill every field's send buffer for this dest locale
        for k in Fields {
          const ref myValsK = vals[k][here.id].Arr;
          ref currSendValsK = sendVals[k][here.id][i].Arr;

          forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
            if dl == i {
              currSendValsK[currLocIndAllLocales[j]] = myValsK[j];
            }
          }
        }
      }
    }

    // 2) Build per-locale receive buffers and splice in segments from every source locale
    var recvVals: [Fields] [PrivateSpace] innerArray(eltType);

    coforall loc in Locales do on loc {
      // For this receiving locale, how many values will arrive from each source locale?
      const numValsReceivingByLocale = [i in LocaleRange] numValsSendingByLocale[i][here.id];
      const valOffsetByLocale = (+ scan numValsReceivingByLocale) - numValsReceivingByLocale;
      const totalIncoming = + reduce numValsReceivingByLocale;

      // Allocate one receive buffer per field
      for k in Fields {
        recvVals[k][here.id] = new innerArray({0..#totalIncoming}, eltType);
      }

      // Splice in, locale by locale, for every field
      for i in LocaleRange {
        const count = numValsReceivingByLocale[i];
        const off   = valOffsetByLocale[i];

        for k in Fields {
          ref myRecvValsK = recvVals[k][here.id].Arr;
          myRecvValsK[off..#count] = sendVals[k][i][here.id].Arr;
        }
      }
    }

    return recvVals;
  }

}
