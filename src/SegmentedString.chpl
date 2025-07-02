module SegmentedString {
  use AryUtil;
  use CTypes;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use CommAggregation;
  use UnorderedCopy;
  use SipHash;
  use SegStringSort;
  use RadixSortLSD;
  use PrivateDist;
  use ServerConfig;
  use Unique;
  use Time;
  use Reflection;
  use Logging;
  use ServerErrors;
  use SegmentedComputation;
  use Regex;

  use Subprocess;
  use Path;
  use FileSystem;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const ssLogger = new Logger(logLevel, logChannel);

  private config param useHash = true;
  param SegmentedStringUseHash = useHash;

  enum Fixes {
    prefixes,
    suffixes,
  };  

  private config param regexMaxCaptures = ServerConfig.regexMaxCaptures;

  config const NULL_STRINGS_VALUE = 0:uint(8);

  proc getSegString(name: string, st: borrowed SymTab): owned SegString throws {
      var abstractEntry = st[name];
      if !abstractEntry.isAssignableTo(SymbolEntryType.SegStringSymEntry) {
          var errorMsg = "Error: Unhandled SymbolEntryType %s".format(abstractEntry.entryType);
          ssLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          throw new Error(errorMsg);
      }
      var entry:SegStringSymEntry = abstractEntry: borrowed SegStringSymEntry;
      return new owned SegString(name, entry);
  }

  /*
   * This version of the getSegString method takes segments and values arrays as
   * inputs, generates the SymEntry objects for each and passes the
   * offset and value SymTab lookup names to the alternate init method
   */
  proc getSegString(segments: [] int, values: [] uint(8), st: borrowed SymTab): owned SegString throws {
      var offsetsEntry = createSymEntry(segments);
      var valuesEntry = createSymEntry(values);
      var stringsEntry = new shared SegStringSymEntry(offsetsEntry, valuesEntry, string);
      var name = st.nextName();
      st.addEntry(name, stringsEntry);
      return getSegString(name, st);
  }

  proc assembleSegStringFromParts(offsets:GenSymEntry, values:GenSymEntry, st:borrowed SymTab): owned SegString throws {
      var offs = toSymEntry(offsets, int);
      var vals = toSymEntry(values, uint(8));
      // This probably invokes a copy, but it's a temporary legacy bridge until we can pass
      // array components as a single message.
      return getSegString(offs.a, vals.a, st);
  }

  proc assembleSegStringFromParts(offsets:SymEntry(int), values:SymEntry(uint(8)), st:borrowed SymTab): owned SegString throws {
      var stringsEntry = new shared SegStringSymEntry(offsets, values, string);
      var name = st.nextName();
      st.addEntry(name, stringsEntry);
      return getSegString(name, st);
  }

  /**
   * Represents an array of strings, implemented as a segmented array of bytes.
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining string-relevant
   * operations.
   */
  class SegString {
 
    var name: string;

    var composite: borrowed SegStringSymEntry;

    /**
     * The pdarray containing the offsets, which are the start indices of
     * the bytearrays, each of which corresponds to an individual string.
     */ 
    var offsets: shared SymEntry(int, 1);

    /**
     * The pdarray containing the complete byte array composed of bytes
     * corresponding to each string, joined by nulls. Note: the null byte
     * is uint(8) value of zero.
     */ 
    var values: shared SymEntry(uint(8), 1);
    
    /**
     * The number of strings in the segmented array
     */
    var size: int;
    
    /**
     * The total number of bytes in the entire segmented array including
     * the bytes corresonding to the strings as well as the nulls
     * separating the string bytes.
     */ 
    var nBytes: int;

    /* 
     * This method should not be called directly. Instead, call one of the
     * getSegString factory methods.
     */
    proc init(entryName:string, entry:borrowed SegStringSymEntry) {
        name = entryName;
        composite = entry;
        offsets = composite.offsetsEntry: shared SymEntry(int, 1);
        values = composite.bytesEntry: shared SymEntry(uint(8), 1);
        size = offsets.size;
        nBytes = values.size;
    }

    proc show(n: int = 3) throws {
      if (size >= 2*n) {
        for i in 0..#n {
            ssLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
        for i in size-n..#n {
            ssLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
      } else {
        for i in 0..#size {
            ssLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
      }
    }

    /* Retrieve one string from the array */
    proc this(idx: ?t): string throws where t == int || t == uint {
      if (idx < offsets.a.domain.low) || (idx > offsets.a.domain.high) {
        throw new owned OutOfBoundsError();
      }
      // Start index of the string
      var start = offsets.a[idx:int];
      // Index of last (null) byte in string
      var end: int;
      if (idx == size - 1) {
        end = nBytes - 1;
      } else {
        end = offsets.a[idx:int+1] - 1;
      }
      // Take the slice of the bytearray and "cast" it to a chpl string
      var s = interpretAsString(values.a, start..end);
      return s;
    }

    /* Take a slice of strings from the array. The slice must be a 
       Chapel range, i.e. low..high by stride, not a Python slice.
       Returns arrays for the segment offsets and bytes of the slice.*/
    proc this(const slice: range()) throws {
      if (slice.low < offsets.a.domain.low) || (slice.high > offsets.a.domain.high) {
          ssLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
          "Array is out of bounds");
          throw new owned OutOfBoundsError();
      }
      // Early return for zero-length result
      if (size == 0) || (slice.size == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      // Start of bytearray slice
      var start = offsets.a[slice.low];
      // End of bytearray slice
      var end: int;
      if (slice.high == offsets.a.domain.high) {
        // if slice includes the last string, go to the end of values
        end = values.a.domain.high;
      } else {
        end = offsets.a[slice.high+1] - 1;
      }
      // Segment offsets of the new slice
      var newSegs = makeDistArray(slice.size, int);
      ref oa = offsets.a;
      // newSegs = offsets.a[slice] - start;
      forall (i, ns) in zip(newSegs.domain, newSegs) with (var agg = newSrcAggregator(int)) {
        agg.copy(ns, oa[slice.low + i]);
      }
      // Offsets need to be re-zeroed
      newSegs -= start;
      // Bytearray of the new slice
      var newVals = makeDistArray(end - start + 1, uint(8));
      ref va = values.a;
      // newVals = values.a[start..end];
      forall (i, nv) in zip(newVals.domain, newVals) with (var agg = newSrcAggregator(uint(8))) {
        agg.copy(nv, va[start + i]);
      }
      return (newSegs, newVals);
    }

    proc this(const slice: range(strides=strideKind.any)) throws {
      var aa = makeDistArray(slice.size, int);
      aa = slice;
      return this[aa];
    }

    /* Gather strings by index. Returns arrays for the segment offsets
       and bytes of the gathered strings.*/
    proc this(iv: [?D] ?t) throws where t == int || t == uint {
      use ChplConfig;
      
      // Early return for zero-length result
      if (D.size == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      // Check all indices within bounds
      var ivMin = min reduce iv;
      var ivMax = max reduce iv;
      if (ivMin < 0) || (ivMax >= offsets.size) {
          ssLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "Array out of bounds");
          throw new owned OutOfBoundsError();
      }
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                              "Computing lengths and offsets");
      var t1 = timeSinceEpoch().totalSeconds();
      ref oa = offsets.a;
      const low = offsets.a.domain.low, high = offsets.a.domain.high;
      // Gather the right and left boundaries of the indexed strings
      // NOTE: cannot compute lengths inside forall because agg.copy will
      // experience race condition with loop-private variable
      var right = makeDistArray(D, int);
      var left = makeDistArray(D, int);
      forall (r, l, idx) in zip(right, left, iv) with (var agg = newSrcAggregator(int)) {
        if (idx == high) {
          agg.copy(r, values.size);
        } else {
          agg.copy(r, oa[idx:int+1]);
        }
        agg.copy(l, oa[idx:int]);
      }
      // Lengths of segments including null bytes
      var gatheredLengths: [D] int = right - left;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * gatheredLengths.size);
      // The returned offsets are the 0-up cumulative lengths
      var gatheredOffsets = (+ scan gatheredLengths);
      // The total number of bytes in the gathered strings
      var retBytes = gatheredOffsets[D.high];
      gatheredOffsets -= gatheredLengths;
      
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                "aggregation in %i seconds".format(timeSinceEpoch().totalSeconds() - t1));
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Copying values");
      if logLevel == LogLevel.DEBUG {
          t1 = timeSinceEpoch().totalSeconds();
      }
      var gatheredVals = makeDistArray(retBytes, uint(8));
      if CHPL_COMM != 'none' {
        // Compute the src index for each byte in gatheredVals
        /* For performance, we will do this with a scan, so first we need an array
           with the difference in index between the current and previous byte. For
           the interior of a segment, this is just one, but at the segment boundary,
           it is the difference between the src offset of the current segment ("left")
           and the src index of the last byte in the previous segment (right - 1).
        */
        var srcIdx = makeDistArray(retBytes, int);
        srcIdx = 1;
        var diffs = makeDistArray(D, int);
        diffs[D.low] = left[D.low]; // first offset is not affected by scan

        forall idx in D {
          if idx!=0 {
            diffs[idx] = left[idx] - (right[idx-1]-1);
          }
        }
        // Set srcIdx to diffs at segment boundaries
        forall (go, d) in zip(gatheredOffsets, diffs) with (var agg = newDstAggregator(int)) {
          agg.copy(srcIdx[go], d);
        }
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * srcIdx.size);
        srcIdx = + scan srcIdx;
        // Now srcIdx has a dst-local copy of the source index and vals can be efficiently gathered
        ref va = values.a;
        forall (v, si) in zip(gatheredVals, srcIdx) with (var agg = newSrcAggregator(uint(8))) {
          agg.copy(v, va[si]);
        }
      } else {
        ref va = values.a;
        // Copy string data to gathered result
        forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, iv) {
          for pos in 0..#gl {
            gatheredVals[go+pos] = va[oa[idx:int]+pos];
          }
        }
      }
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "Gathered offsets and vals in %i seconds".format(
                                           timeSinceEpoch().totalSeconds() -t1));
      return (gatheredOffsets, gatheredVals);
    }

    /* Logical indexing (compress) of strings. */
    proc this(iv: [?D] bool) throws {
      // Index vector must be same domain as array
      if (D != offsets.a.domain) {
          ssLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "Array out of bounds");
          throw new owned OutOfBoundsError();
      }
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                 "Computing lengths and offsets");
      var t1 = timeSinceEpoch().totalSeconds();
      ref oa = offsets.a;
      const low = offsets.a.domain.low, high = offsets.a.domain.high;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * iv.size);
      // Calculate the destination indices
      var steps = + scan iv;
      var newSize = steps[high];
      steps -= iv;
      // Early return for zero-length result
      if (newSize == 0) {
        return (makeDistArray(0, int), makeDistArray(0, uint(8)));
      }
      var segInds = makeDistArray(newSize, int);
      forall (t, dst, idx) in zip(iv, steps, D) with (var agg = newDstAggregator(int)) {
        if t {
          agg.copy(segInds[dst], idx);
        }
      }
      return this[segInds];
    }

    /* Apply a hash function to all strings. This is useful for grouping
       and set membership. The hash used is SipHash128.*/
    proc siphash() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.SipHash128, 2*uint(64));
    }

    /* Return a permutation that groups the strings. Because hashing is used,
       this permutation will not sort the strings, but all equivalent strings
       will fall in one contiguous block. */
    proc argGroup() throws {
      if useHash {
        // Hash all strings
        ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Hashing strings"); 
        var t1: real;
        if logLevel == LogLevel.DEBUG { t1 = timeSinceEpoch().totalSeconds(); }
        var hashes = this.siphash();

        if logLevel == LogLevel.DEBUG { 
            ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "hashing took %? seconds\nSorting hashes".format(timeSinceEpoch().totalSeconds() - t1));
            t1 = timeSinceEpoch().totalSeconds();
        }

        // Return the permutation that sorts the hashes
        var iv = radixSortLSD_ranks(hashes);
        if logLevel == LogLevel.DEBUG { 
            ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "sorting took %? seconds".format(timeSinceEpoch().totalSeconds() - t1));
        }
        if logLevel == LogLevel.DEBUG {
          var sortedHashes = [i in iv] hashes[i];
          var diffs = sortedHashes[(iv.domain.low+1)..#(iv.size-1)] - 
                                                 sortedHashes[(iv.domain.low)..#(iv.size-1)];
          printAry("diffs = ", diffs);
          var nonDecreasing = [(d0,d1) in diffs] ((d0 > 0) || ((d0 == 0) && (d1 >= 0)));
          ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                    "Are hashes sorted? %i".format(&& reduce nonDecreasing));
        }
        return iv;
      } else {
        var iv = argsort();
        return iv;
      }
    }

    /* Return lengths of all strings, including null terminator. */
    proc getLengths() throws {
      var lengths = makeDistArray(offsets.a.domain, int);
      if (size == 0) {
        return lengths;
      }
      ref oa = offsets.a;
      const low = offsets.a.domain.low;
      const high = offsets.a.domain.high;
      forall (i, o, l) in zip(offsets.a.domain, oa, lengths) {
        if (i == high) {
          l = values.size - o;
        } else {
          l = oa[i+1] - o;
        }
      }
      /* lengths[low..high-1] = (oa[low+1..high] - oa[low..high-1]); */
      /* lengths[high] = values.size - oa[high]; */
      return lengths;
    }

    /*
      Given a SegString, return a new SegString with all uppercase characters from the original replaced with their lowercase equivalent
      :returns: Strings – Substrings with uppercase characters replaced with lowercase equivalent
    */
    proc lower() throws {
      ref origVals = this.values.a;
      ref offs = this.offsets.a;
      var lowerVals = makeDistArray(this.values.a.domain, uint(8));
      const lengths = this.getLengths();
      forall (off, len) in zip(offs, lengths) with (var valAgg = newDstAggregator(uint(8))) {
        var i = 0;
        for b in interpretAsBytes(origVals, off..#len, borrow=true).toLower() {
          valAgg.copy(lowerVals[off+i], b:uint(8));
          i += 1;
        }
      }
      return (offs, lowerVals);
    }

    /*
      Given a SegString, return a new SegString with all lowercase characters from the original replaced with their uppercase equivalent
      :returns: Strings – Substrings with lowercase characters replaced with uppercase equivalent
    */
    proc upper() throws {
      ref origVals = this.values.a;
      ref offs = this.offsets.a;
      var upperVals = makeDistArray(this.values.a.domain, uint(8));
      const lengths = this.getLengths();
      forall (off, len) in zip(offs, lengths) with (var valAgg = newDstAggregator(uint(8))) {
        var i = 0;
        for b in interpretAsBytes(origVals, off..#len, borrow=true).toUpper() {
          valAgg.copy(upperVals[off+i], b:uint(8));
          i += 1;
        }
      }
      return (offs, upperVals);
    }

    /*
      Given a SegString, return a new SegString with first character of each original element replaced with its uppercase equivalent
      and the remaining characters replaced with their lowercase equivalent.  The first character following a space character will be uppercase.
      :returns: Strings – Substrings with first characters replaced with uppercase equivalent and remaining characters replaced with
      their lowercase equivalent.  The first character following a space character will be uppercase.
    */
    proc title() throws {
      ref origVals = this.values.a;
      ref offs = this.offsets.a;
      var titleVals: [this.values.a.domain] uint(8);
      const lengths = this.getLengths();
      forall (off, len) in zip(offs, lengths) with (var valAgg = newDstAggregator(uint(8))) {
        var i = 0;
        for b in interpretAsBytes(origVals, off..#len, borrow=true).toTitle() {
          valAgg.copy(titleVals[off+i], b:uint(8));
          i += 1;
        }
      }
      return (offs, titleVals);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is a decimal
      :returns: [domain] bool where index i indicates whether the string i of the SegString is a decimal
    */
    proc isDecimal() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsDecimal, bool);
    }
    
    /*
      Returns list of bools where index i indicates whether the string i of the SegString is numeric 
      :returns: [domain] bool where index i indicates whether the string i of the SegString is numeric
    */
    proc isNumeric() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsNumeric, bool);
    }

    /*
      Given a SegString, return a new SegString with first character of each original element replaced with its uppercase equivalent
      and the remaining characters replaced with their lowercase equivalent
      :returns: Strings – Substrings with first characters replaced with uppercase equivalent and remaining characters replaced with
      their lowercase equivalent
    */
    proc capitalize() throws {
      ref origVals = this.values.a;
      ref offs = this.offsets.a;
      var capitalizedVals: [this.values.a.domain] uint(8);
      const lengths = this.getLengths();
      forall (off, len) in zip(offs, lengths) with (var valAgg = newDstAggregator(uint(8))) {
        var i = 0;
        var first = true;
        for char in interpretAsString(origVals, off..#len, borrow=true).items(){
          if(first){
            for b in char.toUpper().bytes(){
              valAgg.copy(capitalizedVals[off+i], b:uint(8));
              i += 1;
            }
            first = false;
          }else{
            for b in char.toLower().bytes(){
              valAgg.copy(capitalizedVals[off+i], b:uint(8));
              i += 1;
            }
          }
        }
      }
      return (offs, capitalizedVals);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is entirely lowercase
      :returns: [domain] bool where index i indicates whether the string i of the SegString is entirely lowercase
    */
    proc isLower() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsLower, bool);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is entirely uppercase
      :returns: [domain] bool where index i indicates whether the string i of the SegString is entirely uppercase
    */
    proc isUpper() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsUpper, bool);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is titlecase
      :returns: [domain] bool where index i indicates whether the string i of the SegString is titlecase
    */
    proc isTitle() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsTitle, bool);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is alphanumeric
      :returns: [domain] bool where index i indicates whether the string i of the SegString is alphanumeric
    */
    proc isalnum() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsAlphaNumeric, bool);
    }

     /*
      Returns list of bools where index i indicates whether the string i of the SegString is alphabetic
      :returns: [domain] bool where index i indicates whether the string i of the SegString is alphabetic
    */
    proc isalpha() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsAlphabetic, bool);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is digits
      :returns: [domain] bool where index i indicates whether the string i of the SegString is digits
    */
    proc isdigit() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsDigit, bool);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is empty
      :returns: [domain] bool where index i indicates whether the string i of the SegString is empty
    */
    proc isempty() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsEmpty, bool);
    }

    /*
      Returns list of bools where index i indicates whether the string i of the SegString is whitespace
      :returns: [domain] bool where index i indicates whether the string i of the SegString is whitespace
    */
    proc isspace() throws {
      return computeOnSegments(offsets.a, values.a, SegFunction.StringIsSpace, bool);
    }

    proc bytesToUintArr(const max_bytes:int, lens: [?D] ?t, st) throws {
      // bytes contained in strings < 128 bits, so concatenating is better than the hash
      ref off = offsets.a;
      ref vals = values.a;
      if max_bytes <= 8 {
        // we only need one uint array
        var numeric = makeDistArray(offsets.a.domain, uint);
        forall (o, l, n) in zip(off, lens, numeric) {
          n = stringBytesToUintArr(vals, o..#l);
        }
        const concatName = st.nextName();
        st.addEntry(concatName, createSymEntry(numeric));
        return concatName;
      }
      else {
        // we need two uint arrays
        var numeric1, numeric2 = makeDistArray(offsets.a.domain, uint);
        forall (o, l, n1, n2) in zip(off, lens, numeric1, numeric2) {
          const half = (l/2):int;
          n1 = stringBytesToUintArr(vals, o..#half);
          n2 = stringBytesToUintArr(vals, (o+half)..<(o+l));
        }
        const concat1Name = st.nextName();
        const concat2Name = st.nextName();
        st.addEntry(concat1Name, createSymEntry(numeric1));
        st.addEntry(concat2Name, createSymEntry(numeric2));
        return "%s+%s".format(concat1Name, concat2Name);
      }
    }

    proc findSubstringInBytes(const substr: string) throws {
      // Find the start position of every occurence of substr in the flat bytes array
      // Start by making a right-truncated subdomain representing all valid starting positions for substr of given length
      var D: subdomain(values.a.domain) = values.a.domain[values.a.domain.low..#(values.size - substr.numBytes + 1)];
      // Every start position is valid until proven otherwise
      var truth = makeDistArray(D, true);
      // Shift the flat values one byte at a time and check against corresponding byte of substr
      for (b, i) in zip(substr.chpl_bytes(), 0..) {
        truth &= (values.a[D.translate(i)] == b);
      }
      return truth;
    }

    /*
      Given a SegString, finds pattern matches and returns pdarrays containing the number, start postitions, and lengths of matches
      :arg pattern: The regex pattern used to find matches
      :type pattern: string
      :arg groupNum: The number of the capture group to be returned
      :type groupNum: int
      :returns: int64 pdarray – For each original string, the number of pattern matches and int64 pdarray – The start positons of pattern matches and int64 pdarray – The lengths of pattern matches
    */
    proc findMatchLocations(const pattern: string, groupNum: int) throws {
      checkCompile(pattern);
      ref origOffsets = this.offsets.a;
      ref origVals = this.values.a;
      const lengths = this.getLengths();

      overMemLimit((4 * this.offsets.size * numBytes(int)) + (3 * this.values.size * numBytes(int)));
      var numMatches = makeDistArray(this.offsets.a.domain, int);
      var matchStartBool = makeDistArray(this.values.a.domain, false);
      var sparseLens = makeDistArray(this.values.a.domain, int);
      var sparseStarts = makeDistArray(this.values.a.domain, int);
      var searchBools = makeDistArray(this.offsets.a.domain, false);
      var matchBools = makeDistArray(this.offsets.a.domain, false);
      var fullMatchBools = makeDistArray(this.offsets.a.domain, false);
      forall (i, off, len) in zip(this.offsets.a.domain, origOffsets, lengths) with (var myRegex = unsafeCompileRegex(pattern),
                                                                               var lenAgg = newDstAggregator(int),
                                                                               var startPosAgg = newDstAggregator(int),
                                                                               var startBoolAgg = newDstAggregator(bool),
                                                                               var searchBoolAgg = newDstAggregator(bool),
                                                                               var matchBoolAgg = newDstAggregator(bool),
                                                                               var fullMatchBoolAgg = newDstAggregator(bool),
                                                                               var numMatchAgg = newDstAggregator(int)) {
        var matchessize = 0;
        for m in myRegex.matches(interpretAsString(origVals, off..#len, borrow=true), regexMaxCaptures) {
          var match = m[0];
          var group = m[groupNum];
          if group.byteOffset != -1 {
            lenAgg.copy(sparseLens[off + group.byteOffset:int], group.numBytes);
            startPosAgg.copy(sparseStarts[off + group.byteOffset:int], group.byteOffset:int);
            startBoolAgg.copy(matchStartBool[off + group.byteOffset:int], true);
            matchessize += 1;
            searchBoolAgg.copy(searchBools[i], true);
            if match.byteOffset == 0 {
              matchBoolAgg.copy(matchBools[i], true);
            }
            if match.numBytes == len-1 {
              fullMatchBoolAgg.copy(fullMatchBools[i], true);
            }
          }
        }
        numMatchAgg.copy(numMatches[i], matchessize);
      }
      var totalMatches = + reduce numMatches;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit((2 * numBytes(int) * matchStartBool.size) + (3 * numBytes(int) * searchBools.size));

      // the matchTransform starts at 0 and increment after hitting a matchStart
      // when looping over the origVals domain, matchTransform acts as a function: origVals.domain -> makeDistDom(totalMatches)
      var matchTransform = + scan matchStartBool - matchStartBool;
      var matchesIndicies = (+ scan numMatches) - numMatches;
      var matchStarts = makeDistArray(totalMatches, int);
      var matchLens = makeDistArray(totalMatches, int);
      [i in this.values.a.domain] if (matchStartBool[i] == true) {
        matchStarts[matchTransform[i]] = sparseStarts[i];
        matchLens[matchTransform[i]] = sparseLens[i];
      }
      var searchScan = (+ scan searchBools) - searchBools;
      var matchScan = (+ scan matchBools) - matchBools;
      var fullMatchScan = (+ scan fullMatchBools) - fullMatchBools;
      return (numMatches, matchStarts, matchLens, matchesIndicies, searchBools, searchScan, matchBools, matchScan, fullMatchBools, fullMatchScan);
    }

    /*
      Given a SegString, return a new SegString only containing matches of the regex pattern,
      If returnMatchOrig is set to True, return a pdarray containing the index of the original string each pattern match is from
      :arg numMatchesEntry: For each string in SegString, the number of pattern matches
      :type numMatchesEntry: borrowed SymEntry(int) or borrowed SysmEntry(bool)
      :arg startsEntry: The starting postions of pattern matches
      :type startsEntry: borrowed SymEntry(int)
      :arg lensEntry: The lengths of pattern matches
      :type lensEntry: borrowed SymEntry(int)
      :arg returnMatchOrig: If True, return a pdarray containing the index of the original string each pattern match is from
      :type returnMatchOrig: bool
      :returns: Strings – Only the portions of Strings which match pattern and (optional) int64 pdarray – For each pattern match, the index of the original string it was in
    */
    proc findAllMatches(const numMatchesEntry: ?t, const startsEntry: borrowed SymEntry(int,1), const lensEntry: borrowed SymEntry(int,1), const indicesEntry: borrowed SymEntry(int,1), const returnMatchOrig: bool) throws where t == borrowed SymEntry(int,1) || t == borrowed SymEntry(bool,1) {
      ref origVals = this.values.a;
      ref origOffsets = this.offsets.a;
      ref numMatches = numMatchesEntry.a;
      ref matchStarts = startsEntry.a;
      ref matchLens = lensEntry.a;
      ref indices = indicesEntry.a;

      overMemLimit(matchLens.size * numBytes(int));
      var absoluteStarts = makeDistArray(matchLens.size, int);
      forall (off, numMatch, matchInd) in zip(origOffsets, numMatches, indices) with (var absAgg = newDstAggregator(int)) {
        var localizedStarts = new lowLevelLocalizingSlice(matchStarts, matchInd..#numMatch);
        for k in 0..#numMatch {
          // Each string has numMatches[stringInd] number of pattern matches, so matchOrigins needs to repeat stringInd for numMatches[stringInd] times
          absAgg.copy(absoluteStarts[matchInd+k], localizedStarts.ptr[k] + off);
        }
      }

      // matchesValsSize is the total length of all matches + the number of matches (to account for null bytes)
      var matchesValsSize = (+ reduce matchLens) + matchLens.size;
      // check there's enough room to create a copy for scan and to allocate matchesVals/Offsets
      overMemLimit((matchesValsSize * numBytes(uint(8))) + (2 * matchLens.size * numBytes(int)));
      var matchesVals = makeDistArray(matchesValsSize, uint(8));
      var matchesOffsets = makeDistArray(matchLens.size, int);
      // + current index to account for null bytes
      var matchesIndicies = + scan matchLens - matchLens + lensEntry.a.domain;
      forall (i, start, len, matchesInd) in zip(lensEntry.a.domain, absoluteStarts, matchLens, matchesIndicies) with (var valAgg = newDstAggregator(uint(8)), var offAgg = newDstAggregator(int)) {
        for j in 0..#len {
          // copy in match
          valAgg.copy(matchesVals[matchesInd + j], origVals[start + j]);
        }
        // write null byte after each match
        valAgg.copy(matchesVals[matchesInd + len], 0:uint(8));
        if i == 0 {
          offAgg.copy(matchesOffsets[i], 0);
        }
        if i != lensEntry.a.domain.high {
          offAgg.copy(matchesOffsets[i+1], matchesInd + len + 1);
        }
      }

      // build matchOrigins mapping from matchesStrings (pattern matches) to the original Strings they were found in
      const matchOriginsDom = if returnMatchOrig then makeDistDom(matchesOffsets.size) else makeDistDom(0);
      var matchOrigins = makeDistArray(matchOriginsDom, int);
      if returnMatchOrig {
        forall (stringInd, matchInd) in zip(this.offsets.a.domain, indices) with (var originAgg = newDstAggregator(int)) {
          for k in matchInd..#numMatches[stringInd] {
            // Each string has numMatches[stringInd] number of pattern matches, so matchOrigins needs to repeat stringInd for numMatches[stringInd] times
            originAgg.copy(matchOrigins[k], stringInd);
          }
        }
      }
      return (matchesOffsets, matchesVals, matchOrigins);
    }

    /*
      Substitute pattern matches with repl. If count is nonzero, at most count substitutions occur
      If returnNumSubs is set to True, the number of substitutions per string will be returned

      :arg pattern: regex pattern used to find matches
      :type pattern: string

      :arg replStr: the string to replace pattern matches with
      :type replStr: string

      :arg initCount: If count is nonzero, at most count splits occur. If zero, substitute all occurences of pattern
      :type initCount: int

      :arg returnNumSubs: If True, also return the number of substitutions per string
      :type returnNumSubs: bool

      :returns: Strings – Substrings with pattern matches substituted and (optional) int64 pdarray – For each original string, the number of susbstitutions
    */
    proc sub(pattern: string, replStr: string, initCount: int, returnNumSubs: bool) throws {
      checkCompile(pattern);
      ref origOffsets = this.offsets.a;
      ref origVals = this.values.a;
      const lengths = this.getLengths();

      overMemLimit((2 * this.offsets.size * numBytes(int)) + (2 * this.values.size * numBytes(int)));
      var numReplacements = makeDistArray(this.offsets.a.domain, int);
      var replacedLens = makeDistArray(this.offsets.a.domain, int);
      var nonMatch = makeDistArray(this.values.a.domain, true);
      var matchStartBool = makeDistArray(this.values.a.domain, false);

      var repl = replStr:bytes;
      // count = 0 means substitute all occurances, so we set count equal to 10**9
      var count = if initCount == 0 then 10**9:int else initCount;
      // since the pattern matches are variable length, we don't know what the size of subbedVals should be until we've found the matches
      forall (i, off, len) in zip(this.offsets.a.domain, origOffsets, lengths) with (var myRegex = unsafeCompileRegex(pattern),
                                                                               var numReplAgg = newDstAggregator(int),
                                                                               var LenAgg = newDstAggregator(int),
                                                                               var nonMatchAgg = newDstAggregator(bool),
                                                                               var startAgg = newDstAggregator(bool)) {
        var replacementCounter = 0;
        var replLen = 0;
        for m in myRegex.matches(interpretAsString(origVals, off..#len, borrow=true)) {
          var match = m[0];
          for k in (off + match.byteOffset:int)..#match.numBytes {
            nonMatchAgg.copy(nonMatch[k], false);
          }
          startAgg.copy(matchStartBool[off + match.byteOffset:int], true);
          replacementCounter += 1;
          replLen += match.numBytes;
          if replacementCounter == count { break; }
        }
        numReplAgg.copy(numReplacements[i], replacementCounter);
        LenAgg.copy(replacedLens[i], replLen);
      }
      // new val size is the original - (total length of replacements) + (repl.size * total numReplacements)
      const valSize = this.values.size - (+ reduce replacedLens) + (repl.size * (+ reduce numReplacements));
      var subbedVals = makeDistArray(valSize, uint(8));
      overMemLimit(2 * this.offsets.size * numBytes(int));
      // new offsets can be directly calculated
      // new offsets is the original - (running sum of replaced lens) + (running sum of replacements * repl.size)
      var subbedOffsets = origOffsets - ((+ scan replacedLens) - replacedLens) + (repl.size * ((+ scan numReplacements) - numReplacements));
      forall (subOff, origOff, origLen) in zip(subbedOffsets, origOffsets, lengths) with (var valAgg = newDstAggregator(uint(8))) {
        var j = 0;
        var localizedVals = new lowLevelLocalizingSlice(origVals, origOff..#origLen);
        for i in 0..#origLen {
          if matchStartBool[origOff + i] {
            for k in repl {
              valAgg.copy(subbedVals[subOff+j], k:uint(8));
              j += 1;
            }
          }
          if nonMatch[origOff + i] {
            valAgg.copy(subbedVals[subOff+j], localizedVals.ptr[i]);
            j += 1;
          }
        }
      }
      return (subbedOffsets, subbedVals, numReplacements);
    }

    proc segStrWhere(otherStr: ?t, condition: [] bool, ref newLens: [] int) throws where t == string {
      // add one to account for null bytes
      newLens += 1;
      ref origOffs = this.offsets.a;
      ref origVals = this.values.a;
      const other = otherStr:bytes;

      overMemLimit(newLens.size * numBytes(int));
      var whereOffs = (+ scan newLens) - newLens;
      const valSize = (+ reduce newLens);
      var whereVals = makeDistArray(valSize, uint(8));

      forall (whereOff, origOff, len, cond) in zip(whereOffs, origOffs, newLens, condition) with (var valAgg = newDstAggregator(uint(8))) {
        if cond {
          var localizedVals = new lowLevelLocalizingSlice(origVals, origOff..#len);
          for i in 0..#len {
            valAgg.copy(whereVals[whereOff+i], localizedVals.ptr[i]);
          }
        }
        else {
          for i in 0..#(len-1) {
            valAgg.copy(whereVals[whereOff+i], other[i]:uint(8));
          }
          // write null byte
          valAgg.copy(whereVals[whereOff+(len-1)], 0:uint(8));
        }
      }
      return (whereOffs, whereVals);
    }

    proc segStrWhere(other: ?t, condition: [] bool, ref newLens: [] int) throws where t == owned SegString {
      // add one to account for null bytes
      newLens += 1;
      ref origOffs = this.offsets.a;
      ref origVals = this.values.a;
      ref otherOffs = other.offsets.a;
      ref otherVals = other.values.a;

      overMemLimit(newLens.size * numBytes(int));
      var whereOffs = (+ scan newLens) - newLens;
      const valSize = (+ reduce newLens);
      var whereVals = makeDistArray(valSize, uint(8));

      forall (whereOff, origOff, otherOff, len, cond) in zip(whereOffs, origOffs, otherOffs, newLens, condition) with (var valAgg = newDstAggregator(uint(8))) {
        const localizedVals = if cond then new lowLevelLocalizingSlice(origVals, origOff..#len) else new lowLevelLocalizingSlice(otherVals, otherOff..#len);
        for i in 0..#len {
          valAgg.copy(whereVals[whereOff+i], localizedVals.ptr[i]);
        }
      }
      return (whereOffs, whereVals);
    }


    /*
      Strip out all of the leading and trailing characters of each element of a segstring that are
      called out in the "chars" argument.

      :arg chars: the set of characters to be removed
      :type chars: string

      :returns: Strings – substrings with stripped characters from the original string and the offsets into those substrings
    */
    
    proc strip(chars: string) throws {
      ref origOffsets = this.offsets.a;
      ref origVals = this.values.a;
      const lengths = this.getLengths();

      var replacedLens = makeDistArray(this.offsets.a.domain, int);

      forall (off, len, rlen) in zip(origOffsets, lengths, replacedLens) {
        if chars.isEmpty() {
          rlen = interpretAsBytes(origVals, off..#len).strip().size + 1;
        } else {
          rlen = interpretAsBytes(origVals, off..#len).strip(chars:bytes).size + 1;
        }
      }
      var retVals: [makeDistDom(+ reduce replacedLens)] uint(8);
      var retOffs = (+ scan replacedLens) - replacedLens;

      forall (off, len, roff) in zip(origOffsets, lengths, retOffs) with (var valAgg = newDstAggregator(uint(8))) {
        var i = 0;
        if chars.isEmpty() {
          for b in interpretAsBytes(origVals, off..#len).strip() {
            valAgg.copy(retVals[roff+i], b:uint(8));
            i += 1;
          }
        } else {
          for b in interpretAsBytes(origVals, off..#len).strip(chars:bytes) {
            valAgg.copy(retVals[roff+i], b:uint(8));
            i += 1;
          }
        }
      }
      return (retOffs, retVals);
    }

    /*
      Returns list of bools where index i indicates whether the regular expression, pattern, matched string i of the SegString

      Note: the regular expression engine used, re2, does not support lookahead/lookbehind

      :arg pattern: regex pattern to be applied to strings in SegString
      :type pattern: string

      :returns: [domain] bool where index i indicates whether the regular expression, pattern, matched string i of the SegString
    */
    proc substringSearch(const pattern: string) throws {
      // We need to check that pattern compiles once to avoid server crash from !try in _unsafecompile
      checkCompile(pattern);
      return computeOnSegments(offsets.a, values.a, SegFunction.StringSearch, bool, pattern);
    }

    /*
      Peel off one or more fields matching the regular expression, delimiter, from each string (similar
      to string.partition), returning two new arrays of strings.
      *Warning*: This function is experimental and not guaranteed to work.

      Note: the regular expression engine used, re2, does not support lookahead/lookbehind

      :arg delimter: regex delimter where the split in SegString will occur
      :type delimter: string

      :arg times: The number of times the delimiter is sought, i.e. skip over the first (times-1) delimiters
      :type times: int

      :arg includeDelimiter: If true, append the delimiter to the end of the first return array
                              By default, it is prepended to the beginning of the second return array.
      :type includeDelimiter: bool

      :arg keepPartial: If true, a string that does not contain <times> instances of
                        the delimiter will be returned in the first array. By default,
                        such strings are returned in the second array.
      :type keepPartial: bool

      :arg left: If true, peel from the left
      :type left: bool

      :returns: Components to build 2 SegStrings (leftOffsets, leftVals, rightOffsets, rightVals)
    */
    proc peelRegex(const delimiter: string, const times: int, const includeDelimiter: bool, const keepPartial: bool, const left: bool) throws {
      checkCompile(delimiter);
      // should we do len check here? re2.compile('') is valid regex and matches everything
      ref oa = offsets.a;
      ref va = values.a;
      const lengths = getLengths() - 1;
      var leftEnd = makeDistArray(offsets.a.domain, int);
      var rightStart = makeDistArray(offsets.a.domain, int);

      forall (o, len, i) in zip(oa, lengths, offsets.a.domain) with (var myRegex = unsafeCompileRegex(delimiter)) {
        var matches = myRegex.matches(interpretAsString(va, o..#len, borrow=true));
        if matches.size < times {
          // not enough occurances of delim, the entire string stays together, and the param args
          // determine whether it ends up on the left or right
          if left == keepPartial {  // they counteract each other
            // if both true or both false
            // Goes on the left
            leftEnd[i] = o + len - 1;
            rightStart[i] = o + len;
          }
          else {
            // if one is true but not the other
            // Goes on the right
            leftEnd[i] = o - 1;
            rightStart[i] = o;
          }
        }
        else {
          // The string can be peeled; figure out where to split
          var match_index: int = if left then (times - 1) else (matches.size - times);
          var match = matches[match_index][0];
          var j: int = o + match.byteOffset: int;
          // j is now the start of the correct delimiter
          // tweak leftEnd and rightStart based on includeDelimiter
          if includeDelimiter {
            if left {
              leftEnd[i] = j + match.numBytes - 1;
              rightStart[i] = j + match.numBytes;
            }
            else {
              leftEnd[i] = j - 1;
              rightStart[i] = j;
            }
          }
          else {
            leftEnd[i] = j - 1;
            rightStart[i] = j + match.numBytes;
          }
        }
      }
      // this section is the same as `peel`
      // Compute lengths and offsets for left and right return arrays
      const leftLengths = leftEnd - oa + 2;
      const rightLengths = lengths - (rightStart - oa) + 1;
      // check there's enough room to create copies for the scans and throw if creating copies would go over memory limit
      overMemLimit(numBytes(int) * (leftLengths.size + rightLengths.size));
      const leftOffsets = (+ scan leftLengths) - leftLengths;
      const rightOffsets = (+ scan rightLengths) - rightLengths;
      // Allocate values and fill
      var leftVals = makeDistArray((+ reduce leftLengths), uint(8));
      var rightVals = makeDistArray((+ reduce rightLengths), uint(8));
      // Fill left values
      forall (srcStart, dstStart, len) in zip(oa, leftOffsets, leftLengths) with (var agg = newDstAggregator(uint(8))) {
        var localIdx = new lowLevelLocalizingSlice(va, srcStart..#(len-1));
        for i in 0..#(len-1) {
          agg.copy(leftVals[dstStart+i], localIdx.ptr[i]);
        }
      }
      // Fill right values
      forall (srcStart, dstStart, len) in zip(rightStart, rightOffsets, rightLengths) with (var agg = newDstAggregator(uint(8))) {
        var localIdx = new lowLevelLocalizingSlice(va, srcStart..#(len-1));
        for i in 0..#(len-1) {
          agg.copy(rightVals[dstStart+i], localIdx.ptr[i]);
        }
      }
      return (leftOffsets, leftVals, rightOffsets, rightVals);
    }

    proc peel(const delimiter: string, const times: int, param includeDelimiter: bool, param keepPartial: bool, param left: bool) throws {
      param stride = if left then 1 else -1;
      const dBytes = delimiter.numBytes;
      const lengths = getLengths() - 1;
      var leftEnd = makeDistArray(offsets.a.domain, int);
      var rightStart = makeDistArray(offsets.a.domain, int);
      const truth = findSubstringInBytes(delimiter);
      const D = truth.domain;
      ref oa = offsets.a;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * truth.size);
      var numHits = (+ scan truth) - truth;
      const high = offsets.a.domain.high;
      forall i in offsets.a.domain {
        // First, check whether string contains enough instances of delimiter to peel
        var hasEnough: bool;
        if oa[i] > D.high {
          // When the last string(s) is/are shorter than the substr
          hasEnough = false;
        } else if ((i == high) || (oa[i+1] > D.high)) {
          hasEnough = ((+ reduce truth) - numHits[oa[i]]) >= times;
        } else {
          hasEnough = (numHits[oa[i+1]] - numHits[oa[i]]) >= times;
        }
        if !hasEnough {
          // If not, then the entire string stays together, and the param args
          // determine whether it ends up on the left or right
          if left {
            if keepPartial {
              // Goes on the left
              leftEnd[i] = oa[i] + lengths[i] - 1;
              rightStart[i] = oa[i] + lengths[i];
            } else {
              // Goes on the right
              leftEnd[i] = oa[i] - 1;
              rightStart[i] = oa[i];
            }
          } else {
            if keepPartial {
              // Goes on the right
              leftEnd[i] = oa[i] - 1;
              rightStart[i] = oa[i];
            } else {
              // Goes on the left
              leftEnd[i] = oa[i] + lengths[i] - 1;
              rightStart[i] = oa[i] + lengths[i];
            }
          }
        } else {
          // The string can be peeled; figure out where to split
          var nDelim = 0;
          var j: int;
          if left {
            j = oa[i];
          } else {
            // If coming from the right, need to handle edge case of last string
            if i == high {
              j = values.a.domain.high - 1;
            } else {
              j = oa[i+1] - 2;
            }
          }
          // Step until the delimiter is encountered the exact number of times
          while true {
            if (j <= D.high) && truth[j] {
              nDelim += 1;
            }
            if nDelim == times {
              break;
            }
            j += stride;
          }
          // j is now the start of the correct delimiter
          // tweak leftEnd and rightStart based on includeDelimiter
          if left {
            if includeDelimiter {
              leftEnd[i] = j + dBytes - 1;
              rightStart[i] = j + dBytes;
            } else {
              leftEnd[i] = j - 1;
              rightStart[i] = j + dBytes;
            }
          } else {
            if includeDelimiter {
              leftEnd[i] = j - 1;
              rightStart[i] = j;
            } else {
              leftEnd[i] = j - 1;
              rightStart[i] = j + dBytes;
            }
          }
        }
      }
      // Compute lengths and offsets for left and right return arrays
      const leftLengths = leftEnd - oa + 2;
      const rightLengths = lengths - (rightStart - oa) + 1;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * leftLengths.size);
      const leftOffsets = (+ scan leftLengths) - leftLengths;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * rightLengths.size);
      const rightOffsets = (+ scan rightLengths) - rightLengths;
      // Allocate values and fill
      var leftVals = makeDistArray((+ reduce leftLengths), uint(8));
      var rightVals = makeDistArray((+ reduce rightLengths), uint(8));
      ref va = values.a;
      // Fill left values
      forall (srcStart, dstStart, len) in zip(oa, leftOffsets, leftLengths) with (var agg = newDstAggregator(uint(8))) {
        var localIdx = new lowLevelLocalizingSlice(va, srcStart..#(len-1));
        for i in 0..#(len-1) {
          agg.copy(leftVals[dstStart+i], localIdx.ptr[i]);
        }
      }
      // Fill right values
      forall (srcStart, dstStart, len) in zip(rightStart, rightOffsets, rightLengths) with (var agg = newDstAggregator(uint(8))) {
        var localIdx = new lowLevelLocalizingSlice(va, srcStart..#(len-1));
        for i in 0..#(len-1) {
          agg.copy(rightVals[dstStart+i], localIdx.ptr[i]);
        }
      }
      return (leftOffsets, leftVals, rightOffsets, rightVals);
    }

    proc stick(other: SegString, delim: string, param right: bool) throws {
        if (offsets.a.domain != other.offsets.a.domain) {
            throw getErrorWithContext(
                           msg="The SegString offsets to not match",
                           lineNumber = getLineNumber(),
                           routineName = getRoutineName(),
                           moduleName = getModuleName(),
                           errorClass="ArgumentError");
        }
      // Combine lengths and compute new offsets
      var leftLen = getLengths() - 1;
      var rightLen = other.getLengths() - 1;
      const newLengths = leftLen + rightLen + delim.numBytes + 1;
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int) * newLengths.size);
      var newOffsets = (+ scan newLengths);
      const newBytes = newOffsets[offsets.a.domain.high];
      newOffsets -= newLengths;
      // Allocate new values array
      var newVals = makeDistArray(newBytes, uint(8));
      // Copy in the left and right-hand values, separated by the delimiter
      ref va1 = values.a;
      ref va2 = other.values.a;
      forall (o1, o2, no, l1, l2) in zip(offsets.a, other.offsets.a, newOffsets, leftLen, rightLen) with (var agg = newDstAggregator(uint(8))) {
        var pos = no;
        // Left side
        if right {
          var localIdx = new lowLevelLocalizingSlice(va1, o1..#l1);
          for i in 0..#l1 {
            agg.copy(newVals[pos+i], localIdx.ptr[i]);
          }
          pos += l1;
        } else {
          var localIdx = new lowLevelLocalizingSlice(va2, o2..#l2);
          for i in 0..#l2 {
            agg.copy(newVals[pos+i], localIdx.ptr[i]);
          }
          pos += l2;
        }
        // Delimiter
        for (i, b) in zip(0..#delim.numBytes, delim.chpl_bytes()) {
          agg.copy(newVals[pos+i], b);
        }
        pos += delim.numBytes;
        // Right side
        if right {
          var localIdx = new lowLevelLocalizingSlice(va2, o2..#l2);
          for i in 0..#l2 {
            agg.copy(newVals[pos+i], localIdx.ptr[i]);
          }
        } else {
          var localIdx = new lowLevelLocalizingSlice(va1, o1..#l1);
          for i in 0..#l1 {
            agg.copy(newVals[pos+i], localIdx.ptr[i]);
          }
        }
      }
      return (newOffsets, newVals);
    }

    proc ediff():[offsets.a.domain] int throws {
      var diff = makeDistArray(offsets.a.domain, int);
      if (size < 2) {
        return diff;
      }
      ref oa = offsets.a;
      ref va = values.a;
      const high = offsets.a.domain.high;
      forall (i, a) in zip(offsets.a.domain, diff) {
        if (i < high) {
          var asc: bool;
          const left = oa[i]..oa[i+1]-1;
          if (i < high - 1) {
            const right = oa[i+1]..oa[i+2]-1;
            a = -memcmp(va, left, va, right);
          } else { // i == high - 1
            const right = oa[i+1]..values.a.domain.high;
            a = -memcmp(va, left, va, right);
          }
        } else { // i == high
          a = 0;
        } 
      }
      return diff;
    }

    proc isSorted():bool throws {
      if (size < 2) {
        return true;
      }
      return (&& reduce (ediff() >= 0));
    }

    proc argsort(checkSorted:bool=false): [offsets.a.domain] int throws {
      const ref D = offsets.a.domain;
      const ref va = values.a;
      if checkSorted && isSorted() {
          ssLogger.warn(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "argsort called on already sorted array");
          var ranks = makeDistArray(D, int);
          ranks = [i in D] i;
          return ranks;
      }
      var ranks = twoPhaseStringSort(this);
      return ranks;
    }

    proc getFixes(n: int, kind: Fixes, proper: bool) throws {
      const lengths = getLengths() - 1;
      var longEnough = makeDistArray(size, bool);
      if proper {
        longEnough = (lengths > n);
      } else {
        longEnough = (lengths >= n);
      }
      var nFound = + reduce longEnough;
      var retOffsets = makeDistArray(nFound, int);
      forall (i, o) in zip(retOffsets.domain, retOffsets) {
        o = i * (n + 1);
      }
      const retDom = makeDistDom(nFound * (n + 1));
      var retBytes = makeDistArray(retDom, uint(8));
      var srcInds = makeDistArray(retDom, int);
      var dstInds = (+ scan longEnough) - longEnough;
      ref oa = offsets.a;
      if kind == Fixes.prefixes {
        forall (d, o, le) in zip(dstInds, oa, longEnough) with (var agg = newDstAggregator(int)) {
          if le {
            const srcIndStart = d * (n + 1);
            for j in 0..#(n+1) {
              agg.copy(srcInds[srcIndStart + j], o + j);
            }
          }
        }
      } else if kind == Fixes.suffixes {
        forall (d, o, l, le) in zip(dstInds, oa, lengths, longEnough) with (var agg = newDstAggregator(int)) {
          if le {
            const srcIndStart = d * (n + 1);
            const byteStart = o + l - n;
            for j in 0..#(n+1) {
              agg.copy(srcInds[srcIndStart + j], byteStart + j);
            }
          }
        }
      }
      ref va = values.a;
      forall (b, i) in zip(retBytes, srcInds) with (var agg = newSrcAggregator(uint(8))) {
        agg.copy(b, va[i]);
      }
      return (retOffsets, retBytes, longEnough);
    }

  } // class SegString


  inline proc memcmp(const ref x: [] uint(8), const xinds, const ref y: [] uint(8), const yinds): int {
    const l = min(xinds.size, yinds.size);
    var ret: int = 0;
    for (i, j) in zip(xinds.low..#l, yinds.low..#l) {
      ret = x[i]:int - y[j]:int;
      if (ret != 0) {
        break;
      }
    }
    if (ret == 0) {
      ret = xinds.size - yinds.size;
    }
    return ret;
  }


  /* Test for equality between two same-length arrays of strings. Returns
     a boolean vector of the same length. */
  operator ==(lss:SegString, rss:SegString) throws {
    return compare(lss, rss, true);
  }

  /* Test for inequality between two same-length arrays of strings. Returns
     a boolean vector of the same length. */
  operator !=(lss:SegString, rss:SegString) throws {
    return compare(lss, rss, false);
  }

  /* Element-wise comparison of two same-length arrays of strings. The
     polarity parameter determines whether the comparison checks for 
     equality (polarity=true, result is true where elements are equal) 
     or inequality (polarity=false, result is true where elements differ). */
  private proc compare(lss:SegString, rss:SegString, param polarity: bool) throws {
    // String arrays must be same size
    if (lss.size != rss.size) {
        throw getErrorWithContext(
                           msg="The String arrays must be the same size",
                           lineNumber = getLineNumber(),
                           routineName = getRoutineName(),
                           moduleName = getModuleName(),
                           errorClass="ArgumentError");
    }
    if useHash {
      const lh = lss.siphash();
      const rh = rss.siphash();
      return if polarity then (lh == rh) else (lh != rh);
    }
    const ref oD = lss.offsets.a.domain;
    // Start by assuming all elements differ, then correct for those that are equal
    // This translates to an initial value of false for == and true for !=
    var truth = makeDistArray(oD, !polarity);
    // Early exit for zero-length result
    if (lss.size == 0) {
      return truth;
    }
    ref lvalues = lss.values.a;
    ref loffsets = lss.offsets.a;
    ref rvalues = rss.values.a;
    ref roffsets = rss.offsets.a;
    // Compare segments in parallel
    // Segments are guaranteed to be on same locale, but bytes are not
    forall (t, lo, ro, idx) in zip(truth, loffsets, roffsets, oD) 
      with (var agg = newDstAggregator(bool)) {
      var llen: int;
      var rlen: int;
      if (idx == oD.high) {
        llen = lvalues.size - lo - 1;
        rlen = rvalues.size - ro - 1;
      } else {
        llen = loffsets[idx+1] - lo - 1;
        rlen = roffsets[idx+1] - ro - 1;
      }
      // Only compare bytes if lengths are equal
      if (llen == rlen) {
        var allEqual = true;
        // TO DO: consider an on clause here to ensure at least one access is local
        for pos in 0..#llen {
          if (lvalues[lo+pos] != rvalues[ro+pos]) {
            allEqual = false;
            break;
          }
        }
        // Only if lengths and all bytes are equal, override the default value
        if allEqual {
          // For ==, the output should be true; for !=, false
          agg.copy(t, polarity);
        }
      }
    }
    return truth;
  }

  /* Test an array of strings for equality against a constant string. Return a boolean
     vector the same size as the array. */
  operator ==(ss:SegString, testStr: string) throws {
    return compare(ss, testStr, SegFunction.StringCompareLiteralEq);
  }
  
  /* Test an array of strings for inequality against a constant string. Return a boolean
     vector the same size as the array. */
  operator !=(ss:SegString, testStr: string) throws {
    return compare(ss, testStr, SegFunction.StringCompareLiteralNeq);
  }

  inline proc stringCompareLiteralEq(ref values, rng, testStr) {
    if rng.size == (testStr.numBytes + 1) {
      const s = interpretAsString(values, rng);
      return (s == testStr);
    } else {
      return false;
    }
  }

  inline proc stringCompareLiteralNeq(ref values, rng, testStr) {
    if rng.size == (testStr.numBytes + 1) {
      const s = interpretAsString(values, rng);
      return (s != testStr);
    } else {
      return true;
    }
  }
  
  /* Element-wise comparison of an arrays of string against a target string. 
     The polarity parameter determines whether the comparison checks for 
     equality (polarity=true, result is true where elements equal target) 
     or inequality (polarity=false, result is true where elements differ from 
     target). */
  proc compare(ss:SegString, const testStr: string, param function: SegFunction) throws {
    if testStr.numBytes == 0 {
      // Comparing against the empty string is a quick check for zero length
      const lengths = ss.getLengths() - 1;
      return if function == SegFunction.StringCompareLiteralEq then (lengths == 0) else (lengths != 0);
    }
    return computeOnSegments(ss.offsets.a, ss.values.a, function, bool, testStr);
  }

  /*
    Returns Regexp.compile if pattern can be compiled without an error
  */
  proc checkCompile(const pattern: ?t) throws where t == bytes || t == string {
    try {
      return new regex(pattern);
    }
    catch {
      var errorMsg = "re2 could not compile pattern: %s".format(pattern);
      ssLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      throw new owned IllegalArgumentError(errorMsg);
    }
  }

  proc unsafeCompileRegex(const pattern: ?t) where t == bytes || t == string {
    // This is a private function and should not be called to compile pattern. Use checkCompile instead

    // This proc is a workaound to allow declaring regexps using a with clause in forall loops
    // since using declarations with throws are illegal
    // It is only called after checkCompile so the try! will not result in a server crash
    return try! new regex(pattern);
  }

  inline proc stringSearch(ref values, rng, myRegex) throws {
    return myRegex.search(interpretAsString(values, rng, borrow=true)).matched;
  }

  /*
    The SegFunction called by computeOnSegments for isLower
  */
  inline proc stringIsLower(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isLower();
  }

  /*
    The SegFunction called by computeOnSegments for isUpper
  */
  inline proc stringIsUpper(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isUpper();
  }

  /*
    The SegFunction called by computeOnSegments for isTitle
  */
  inline proc stringIsTitle(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isTitle();
  }

  /*
    The SegFunction called by computeOnSegments for isalnum
  */
  inline proc stringIsAlphaNumeric(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isAlnum();
  }

  /*
    The SegFunction called by computeOnSegments for isalpha
  */
  inline proc stringIsAlphabetic(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isAlpha();
  }

  /*
    The SegFunction called by computeOnSegments for isdecimal, using isDigit
  */
  inline proc stringIsDecimal(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isDigit();
  }

  /*
    The SegFunction called by computeOnSegments for isnumeric
   */

  inline proc stringIsNumeric(ref values, rng) throws {
    const myString = interpretAsString(values, rng, borrow=true);
    return isNumericString(myString);
  }

  /*
    The SegFunction called by computeOnSegments for isdigit
  */
  inline proc stringIsDigit(ref values, rng) throws {
    use In1d;
    const specialDigits = "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉";
    const myString = interpretAsString(values, rng, borrow=true);
    // test if string is all regular digits for early out
    if myString.isDigit(){
      return true;
    }
    // test if string is all alphanumeric characters for early out
    else if myString.isAlpha() || myString.isEmpty() {
      return false;
    }
    // string contains at least one special digit character, full test
    else {
      // this function converts bytes to a uint64 to keep all bytes together for the comparison
      // by shifting by 8 to combine into a single uint64 to use In1d on integer values
      proc toUint64(c) {
        var ret:uint = 0;
        var i = 0;
        for b in c.bytes() {
          var tmp = b:uint;
          ret |= (tmp << (i*8));
          i += 1;
        }
        return ret;
      }
      var specialArray = [s in specialDigits] toUint64(s);
      var myStringArray = [b in myString] toUint64(b);
      var digitTruth = [b in myString] b.isDigit();
      const specialTruth = In1d.in1d[myStringArray, specialArray];
      return & reduce (specialTruth | digitTruth);
    }
  }

  /*
    The SegFunction called by computeOnSegments for isempty
  */
  inline proc stringIsEmpty(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isEmpty();
  }

  /*
    The SegFunction called by computeOnSegments for isspace
  */
  inline proc stringIsSpace(ref values, rng) throws {
    return interpretAsString(values, rng, borrow=true).isSpace();
  }

  inline proc stringBytesToUintArr(ref values, rng) throws {
      var localSlice = new lowLevelLocalizingSlice(values, rng);
      return | reduce [i in 0..#rng.size] (localSlice.ptr(i):uint)<<(8*(rng.size-1-i));
  }

  /* Test array of strings for membership in another array (set) of strings. Returns
     a boolean vector the same size as the first array. */
  proc in1d(mainStr: SegString, testStr: SegString, invert=false) throws where useHash {
    use In1d;
    // Early exit for zero-length result
    if (mainStr.size == 0) {
      var truth = makeDistArray(mainStr.offsets.a.domain, bool);
      return truth;
    }
    var a = mainStr.siphash();
    var b = testStr.siphash();
    return in1d(a, b, invert);
  }

  proc concat(s1: [] int, v1: [] uint(8), s2: [] int, v2: [] uint(8)) throws {
    // TO DO: extend to axis == 1
    var segs = makeDistArray(s1.size + s2.size, int);
    var vals = makeDistArray(v1.size + v2.size, uint(8));
    ref sD = segs.domain;
    segs[sD.interior(-s1.size)] = s1;
    segs[sD.interior(s2.size)] = s2 + v1.size;
    ref vD = vals.domain;
    vals[vD.interior(-v1.size)] = v1;
    vals[vD.interior(v2.size)] = v2;
    return (segs, vals);
  }

  private config const in1dSortThreshold = 64;
  
  proc in1d(mainStr: SegString, testStr: SegString, invert=false) throws where !useHash {
    var truth = makeDistArray(mainStr.offsets.a.domain, bool);
    // Early exit for zero-length result
    if (mainStr.size == 0) {
      return truth;
    }
    if (testStr.size <= in1dSortThreshold) {
      for i in 0..#testStr.size {
        truth |= (mainStr == testStr[i]);
      }
      if invert { truth = !truth; }
      return truth;
    } else {
      // This is inspired by numpy in1d
      const (uoMain, uvMain, cMain, revIdx) = uniqueGroup(mainStr, returnInverse=true);
      const (uoTest, uvTest, cTest, revTest) = uniqueGroup(testStr);
      const (segs, vals) = concat(uoMain, uvMain, uoTest, uvTest);
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
           "Unique strings in first array: %?\nUnique strings in second array: %?\nConcat length: %?".format(
                                             uoMain.size, uoTest.size, segs.size));
      var st = new owned SymTab();
      const ar = new owned SegString(segs, vals, st);
      const order = ar.argsort();
      const (sortedSegs, sortedVals) = ar[order];
      const sar = new owned SegString(sortedSegs, sortedVals, st);
      if logLevel == LogLevel.DEBUG { 
          ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "Sorted concatenated unique strings:"); 
          sar.show(10); 
          stdout.flush(); 
      }
      const D = sortedSegs.domain;
      // First compare lengths and only check pairs whose lengths are equal (because gathering them is expensive)
      var flag = makeDistArray(D, bool);
      const lengths = sar.getLengths();
      const ref saro = sar.offsets.a;
      const ref sarv = sar.values.a;
      const high = D.high;
      forall (i, f, o, l) in zip(D, flag, saro, lengths) {
        if (i < high) && (l == lengths[i+1]) {
          const left = o..saro[i+1]-1;
          var eq: bool;
          if (i < high - 1) {
            const right = saro[i+1]..saro[i+2]-1;
            eq = (memcmp(sarv, left, sarv, right) == 0);
          } else {
            const ref right = saro[i+1]..sar.values.a.domain.high;
            eq = (memcmp(sarv, left, sarv, right) == 0);
          }
          if eq {
            f = true;
            flag[i+1] = true;
          }
        }
      }
      
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "Flag pop: %?".format(+ reduce flag));

      // Now flag contains true for both elements of duplicate pairs
      if invert {flag = !flag;}
      // Permute back to unique order
      var ret = makeDistArray(D, bool);
      forall (o, f) in zip(order, flag) with (var agg = newDstAggregator(bool)) {
        agg.copy(ret[o], f);
      }
      if logLevel == LogLevel.DEBUG {
          ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "Ret pop: %?".format(+ reduce ret));
      }
      // Broadcast back to original (pre-unique) order
      var truth = makeDistArray(mainStr.offsets.a.domain, bool);
      forall (t, i) in zip(truth, revIdx) with (var agg = newSrcAggregator(bool)) {
        agg.copy(t, ret[i]);
      }
      return truth;
    }
  }

  proc segStrFull(arrSize: int, fillValue: string) throws {

    var offsets = makeDistArray(arrSize, int);
    const nullTermString = (fillValue:bytes + NULL_STRINGS_VALUE:bytes);
    const strSize = nullTermString.size;
    var values = makeDistArray(arrSize*strSize, uint(8));

    forall (i, off) in zip(offsets.domain, offsets) with (var agg = newDstAggregator(uint(8))) {
      off = i * strSize;
      for j in 0..#strSize {
        agg.copy(values[off + j], nullTermString[j]:uint(8));
      }
    }
    return (offsets, values);
  }

  /*
     Interpret a region of a byte array as a Chapel string. If `borrow=false` a
     new string is returned, otherwise the string borrows memory from the array
     (reduces memory allocations if the string isn't needed after array)
   */
  proc interpretAsString(ref bytearray: [?D] uint(8), region: range(?), borrow=false): string {
    var localSlice = new lowLevelLocalizingSlice(bytearray, region);
    // Byte buffer is null-terminated, so length is region.size - 1
    try {
      if localSlice.isOwned {
        localSlice.isOwned = false;
        return string.createAdoptingBuffer(localSlice.ptr, region.size-1, region.size);
      } else if borrow {
        return string.createBorrowingBuffer(localSlice.ptr, region.size-1, region.size);
      } else {
        return string.createCopyingBuffer(localSlice.ptr, region.size-1, region.size);
      }
    } catch {
      return "<error interpreting bytes as string>";
    }
  }

  /*
     Interpret a region of a byte array as bytes. Modeled after interpretAsString
   */
  proc interpretAsBytes(ref bytearray: [?D] uint(8), region: range(?), borrow=false): bytes {
    var localSlice = new lowLevelLocalizingSlice(bytearray, region);
    // Byte buffer is null-terminated, so length is region.size - 1
    try {
      if localSlice.isOwned {
        localSlice.isOwned = false;
        return bytes.createAdoptingBuffer(localSlice.ptr, region.size-1, region.size);
      } else if borrow {
        return string.createBorrowingBuffer(localSlice.ptr, region.size-1, region.size);
      } else {
        return bytes.createCopyingBuffer(localSlice.ptr, region.size-1, region.size);
      }
    } catch {
      return b"<error interpreting uint(8) as bytes>";
    }
  }

  inline proc isNumericString (s: string) {
      var good = true;
      for letter in s {
          if !isNumericChar(letter) {
              good = false;
              break;
          }
      }
      return good;
  }

  inline proc isNumericChar (c: string) {
      var code : int(32) = c.toCodepoint();
      return isNumericInt(code);
  }

  inline proc arraySearch(a : [], item) : bool {
      var found : bool = false;
      forall entry in a with (| reduce found) {
          if item == entry {
              found = true;
          }
      }
      return found;
  }

  inline proc isNumericInt (a : int ) : bool {
  
  //  The array below contains all the values that np.char.isnumeric considers
  //  numeric.  This includes the ascii characters 0 through 9, and a host of
  //  unicode characters representing numeric superscript, subscripts, encircled
  //  numerals, numerals in parentheses, vulgar fractions, non-English versions
  //  of numerals, etc.
  
    var allNumericUnicodes : [0..1922] int = [
        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39 ,
        0xb2, 0xb3, 0xb9, 0xbc, 0xbd, 0xbe, 0x660, 0x661, 0x662, 0x663 ,
        0x664, 0x665, 0x666, 0x667, 0x668, 0x669, 0x6f0, 0x6f1, 0x6f2, 0x6f3 ,
        0x6f4, 0x6f5, 0x6f6, 0x6f7, 0x6f8, 0x6f9, 0x7c0, 0x7c1, 0x7c2, 0x7c3 ,
        0x7c4, 0x7c5, 0x7c6, 0x7c7, 0x7c8, 0x7c9, 0x966, 0x967, 0x968, 0x969 ,
        0x96a, 0x96b, 0x96c, 0x96d, 0x96e, 0x96f, 0x9e6, 0x9e7, 0x9e8, 0x9e9 ,
        0x9ea, 0x9eb, 0x9ec, 0x9ed, 0x9ee, 0x9ef, 0x9f4, 0x9f5, 0x9f6, 0x9f7 ,
        0x9f8, 0x9f9, 0xa66, 0xa67, 0xa68, 0xa69, 0xa6a, 0xa6b, 0xa6c, 0xa6d ,
        0xa6e, 0xa6f, 0xae6, 0xae7, 0xae8, 0xae9, 0xaea, 0xaeb, 0xaec, 0xaed ,
        0xaee, 0xaef, 0xb66, 0xb67, 0xb68, 0xb69, 0xb6a, 0xb6b, 0xb6c, 0xb6d ,
        0xb6e, 0xb6f, 0xb72, 0xb73, 0xb74, 0xb75, 0xb76, 0xb77, 0xbe6, 0xbe7 ,
        0xbe8, 0xbe9, 0xbea, 0xbeb, 0xbec, 0xbed, 0xbee, 0xbef, 0xbf0, 0xbf1 ,
        0xbf2, 0xc66, 0xc67, 0xc68, 0xc69, 0xc6a, 0xc6b, 0xc6c, 0xc6d, 0xc6e ,
        0xc6f, 0xc78, 0xc79, 0xc7a, 0xc7b, 0xc7c, 0xc7d, 0xc7e, 0xce6, 0xce7 ,
        0xce8, 0xce9, 0xcea, 0xceb, 0xcec, 0xced, 0xcee, 0xcef, 0xd58, 0xd59 ,
        0xd5a, 0xd5b, 0xd5c, 0xd5d, 0xd5e, 0xd66, 0xd67, 0xd68, 0xd69, 0xd6a ,
        0xd6b, 0xd6c, 0xd6d, 0xd6e, 0xd6f, 0xd70, 0xd71, 0xd72, 0xd73, 0xd74 ,
        0xd75, 0xd76, 0xd77, 0xd78, 0xde6, 0xde7, 0xde8, 0xde9, 0xdea, 0xdeb ,
        0xdec, 0xded, 0xdee, 0xdef, 0xe50, 0xe51, 0xe52, 0xe53, 0xe54, 0xe55 ,
        0xe56, 0xe57, 0xe58, 0xe59, 0xed0, 0xed1, 0xed2, 0xed3, 0xed4, 0xed5 ,
        0xed6, 0xed7, 0xed8, 0xed9, 0xf20, 0xf21, 0xf22, 0xf23, 0xf24, 0xf25 ,
        0xf26, 0xf27, 0xf28, 0xf29, 0xf2a, 0xf2b, 0xf2c, 0xf2d, 0xf2e, 0xf2f ,
        0xf30, 0xf31, 0xf32, 0xf33, 0x1040, 0x1041, 0x1042, 0x1043, 0x1044, 0x1045 ,
        0x1046, 0x1047, 0x1048, 0x1049, 0x1090, 0x1091, 0x1092, 0x1093, 0x1094, 0x1095 ,
        0x1096, 0x1097, 0x1098, 0x1099, 0x1369, 0x136a, 0x136b, 0x136c, 0x136d, 0x136e ,
        0x136f, 0x1370, 0x1371, 0x1372, 0x1373, 0x1374, 0x1375, 0x1376, 0x1377, 0x1378 ,
        0x1379, 0x137a, 0x137b, 0x137c, 0x16ee, 0x16ef, 0x16f0, 0x17e0, 0x17e1, 0x17e2 ,
        0x17e3, 0x17e4, 0x17e5, 0x17e6, 0x17e7, 0x17e8, 0x17e9, 0x17f0, 0x17f1, 0x17f2 ,
        0x17f3, 0x17f4, 0x17f5, 0x17f6, 0x17f7, 0x17f8, 0x17f9, 0x1810, 0x1811, 0x1812 ,
        0x1813, 0x1814, 0x1815, 0x1816, 0x1817, 0x1818, 0x1819, 0x1946, 0x1947, 0x1948 ,
        0x1949, 0x194a, 0x194b, 0x194c, 0x194d, 0x194e, 0x194f, 0x19d0, 0x19d1, 0x19d2 ,
        0x19d3, 0x19d4, 0x19d5, 0x19d6, 0x19d7, 0x19d8, 0x19d9, 0x19da, 0x1a80, 0x1a81 ,
        0x1a82, 0x1a83, 0x1a84, 0x1a85, 0x1a86, 0x1a87, 0x1a88, 0x1a89, 0x1a90, 0x1a91 ,
        0x1a92, 0x1a93, 0x1a94, 0x1a95, 0x1a96, 0x1a97, 0x1a98, 0x1a99, 0x1b50, 0x1b51 ,
        0x1b52, 0x1b53, 0x1b54, 0x1b55, 0x1b56, 0x1b57, 0x1b58, 0x1b59, 0x1bb0, 0x1bb1 ,
        0x1bb2, 0x1bb3, 0x1bb4, 0x1bb5, 0x1bb6, 0x1bb7, 0x1bb8, 0x1bb9, 0x1c40, 0x1c41 ,
        0x1c42, 0x1c43, 0x1c44, 0x1c45, 0x1c46, 0x1c47, 0x1c48, 0x1c49, 0x1c50, 0x1c51 ,
        0x1c52, 0x1c53, 0x1c54, 0x1c55, 0x1c56, 0x1c57, 0x1c58, 0x1c59, 0x2070, 0x2074 ,
        0x2075, 0x2076, 0x2077, 0x2078, 0x2079, 0x2080, 0x2081, 0x2082, 0x2083, 0x2084 ,
        0x2085, 0x2086, 0x2087, 0x2088, 0x2089, 0x2150, 0x2151, 0x2152, 0x2153, 0x2154 ,
        0x2155, 0x2156, 0x2157, 0x2158, 0x2159, 0x215a, 0x215b, 0x215c, 0x215d, 0x215e ,
        0x215f, 0x2160, 0x2161, 0x2162, 0x2163, 0x2164, 0x2165, 0x2166, 0x2167, 0x2168 ,
        0x2169, 0x216a, 0x216b, 0x216c, 0x216d, 0x216e, 0x216f, 0x2170, 0x2171, 0x2172 ,
        0x2173, 0x2174, 0x2175, 0x2176, 0x2177, 0x2178, 0x2179, 0x217a, 0x217b, 0x217c ,
        0x217d, 0x217e, 0x217f, 0x2180, 0x2181, 0x2182, 0x2185, 0x2186, 0x2187, 0x2188 ,
        0x2189, 0x2460, 0x2461, 0x2462, 0x2463, 0x2464, 0x2465, 0x2466, 0x2467, 0x2468 ,
        0x2469, 0x246a, 0x246b, 0x246c, 0x246d, 0x246e, 0x246f, 0x2470, 0x2471, 0x2472 ,
        0x2473, 0x2474, 0x2475, 0x2476, 0x2477, 0x2478, 0x2479, 0x247a, 0x247b, 0x247c ,
        0x247d, 0x247e, 0x247f, 0x2480, 0x2481, 0x2482, 0x2483, 0x2484, 0x2485, 0x2486 ,
        0x2487, 0x2488, 0x2489, 0x248a, 0x248b, 0x248c, 0x248d, 0x248e, 0x248f, 0x2490 ,
        0x2491, 0x2492, 0x2493, 0x2494, 0x2495, 0x2496, 0x2497, 0x2498, 0x2499, 0x249a ,
        0x249b, 0x24ea, 0x24eb, 0x24ec, 0x24ed, 0x24ee, 0x24ef, 0x24f0, 0x24f1, 0x24f2 ,
        0x24f3, 0x24f4, 0x24f5, 0x24f6, 0x24f7, 0x24f8, 0x24f9, 0x24fa, 0x24fb, 0x24fc ,
        0x24fd, 0x24fe, 0x24ff, 0x2776, 0x2777, 0x2778, 0x2779, 0x277a, 0x277b, 0x277c ,
        0x277d, 0x277e, 0x277f, 0x2780, 0x2781, 0x2782, 0x2783, 0x2784, 0x2785, 0x2786 ,
        0x2787, 0x2788, 0x2789, 0x278a, 0x278b, 0x278c, 0x278d, 0x278e, 0x278f, 0x2790 ,
        0x2791, 0x2792, 0x2793, 0x2cfd, 0x3007, 0x3021, 0x3022, 0x3023, 0x3024, 0x3025 ,
        0x3026, 0x3027, 0x3028, 0x3029, 0x3038, 0x3039, 0x303a, 0x3192, 0x3193, 0x3194 ,
        0x3195, 0x3220, 0x3221, 0x3222, 0x3223, 0x3224, 0x3225, 0x3226, 0x3227, 0x3228 ,
        0x3229, 0x3248, 0x3249, 0x324a, 0x324b, 0x324c, 0x324d, 0x324e, 0x324f, 0x3251 ,
        0x3252, 0x3253, 0x3254, 0x3255, 0x3256, 0x3257, 0x3258, 0x3259, 0x325a, 0x325b ,
        0x325c, 0x325d, 0x325e, 0x325f, 0x3280, 0x3281, 0x3282, 0x3283, 0x3284, 0x3285 ,
        0x3286, 0x3287, 0x3288, 0x3289, 0x32b1, 0x32b2, 0x32b3, 0x32b4, 0x32b5, 0x32b6 ,
        0x32b7, 0x32b8, 0x32b9, 0x32ba, 0x32bb, 0x32bc, 0x32bd, 0x32be, 0x32bf, 0x3405 ,
        0x3483, 0x382a, 0x3b4d, 0x4e00, 0x4e03, 0x4e07, 0x4e09, 0x4e24, 0x4e5d, 0x4e8c ,
        0x4e94, 0x4e96, 0x4eac, 0x4ebf, 0x4ec0, 0x4edf, 0x4ee8, 0x4f0d, 0x4f70, 0x4fe9 ,
        0x5006, 0x5104, 0x5146, 0x5169, 0x516b, 0x516d, 0x5341, 0x5343, 0x5344, 0x5345 ,
        0x534c, 0x53c1, 0x53c2, 0x53c3, 0x53c4, 0x56db, 0x58f1, 0x58f9, 0x5e7a, 0x5efe ,
        0x5eff, 0x5f0c, 0x5f0d, 0x5f0e, 0x5f10, 0x62d0, 0x62fe, 0x634c, 0x67d2, 0x6d1e ,
        0x6f06, 0x7396, 0x767e, 0x7695, 0x79ed, 0x8086, 0x842c, 0x8cae, 0x8cb3, 0x8d30 ,
        0x920e, 0x94a9, 0x9621, 0x9646, 0x964c, 0x9678, 0x96f6, 0xa620, 0xa621, 0xa622 ,
        0xa623, 0xa624, 0xa625, 0xa626, 0xa627, 0xa628, 0xa629, 0xa6e6, 0xa6e7, 0xa6e8 ,
        0xa6e9, 0xa6ea, 0xa6eb, 0xa6ec, 0xa6ed, 0xa6ee, 0xa6ef, 0xa830, 0xa831, 0xa832 ,
        0xa833, 0xa834, 0xa835, 0xa8d0, 0xa8d1, 0xa8d2, 0xa8d3, 0xa8d4, 0xa8d5, 0xa8d6 ,
        0xa8d7, 0xa8d8, 0xa8d9, 0xa900, 0xa901, 0xa902, 0xa903, 0xa904, 0xa905, 0xa906 ,
        0xa907, 0xa908, 0xa909, 0xa9d0, 0xa9d1, 0xa9d2, 0xa9d3, 0xa9d4, 0xa9d5, 0xa9d6 ,
        0xa9d7, 0xa9d8, 0xa9d9, 0xa9f0, 0xa9f1, 0xa9f2, 0xa9f3, 0xa9f4, 0xa9f5, 0xa9f6 ,
        0xa9f7, 0xa9f8, 0xa9f9, 0xaa50, 0xaa51, 0xaa52, 0xaa53, 0xaa54, 0xaa55, 0xaa56 ,
        0xaa57, 0xaa58, 0xaa59, 0xabf0, 0xabf1, 0xabf2, 0xabf3, 0xabf4, 0xabf5, 0xabf6 ,
        0xabf7, 0xabf8, 0xabf9, 0xf96b, 0xf973, 0xf978, 0xf9b2, 0xf9d1, 0xf9d3, 0xf9fd ,
        0xff10, 0xff11, 0xff12, 0xff13, 0xff14, 0xff15, 0xff16, 0xff17, 0xff18, 0xff19 ,
        0x10107, 0x10108, 0x10109, 0x1010a, 0x1010b, 0x1010c, 0x1010d, 0x1010e, 0x1010f, 0x10110 ,
        0x10111, 0x10112, 0x10113, 0x10114, 0x10115, 0x10116, 0x10117, 0x10118, 0x10119, 0x1011a ,
        0x1011b, 0x1011c, 0x1011d, 0x1011e, 0x1011f, 0x10120, 0x10121, 0x10122, 0x10123, 0x10124 ,
        0x10125, 0x10126, 0x10127, 0x10128, 0x10129, 0x1012a, 0x1012b, 0x1012c, 0x1012d, 0x1012e ,
        0x1012f, 0x10130, 0x10131, 0x10132, 0x10133, 0x10140, 0x10141, 0x10142, 0x10143, 0x10144 ,
        0x10145, 0x10146, 0x10147, 0x10148, 0x10149, 0x1014a, 0x1014b, 0x1014c, 0x1014d, 0x1014e ,
        0x1014f, 0x10150, 0x10151, 0x10152, 0x10153, 0x10154, 0x10155, 0x10156, 0x10157, 0x10158 ,
        0x10159, 0x1015a, 0x1015b, 0x1015c, 0x1015d, 0x1015e, 0x1015f, 0x10160, 0x10161, 0x10162 ,
        0x10163, 0x10164, 0x10165, 0x10166, 0x10167, 0x10168, 0x10169, 0x1016a, 0x1016b, 0x1016c ,
        0x1016d, 0x1016e, 0x1016f, 0x10170, 0x10171, 0x10172, 0x10173, 0x10174, 0x10175, 0x10176 ,
        0x10177, 0x10178, 0x1018a, 0x1018b, 0x102e1, 0x102e2, 0x102e3, 0x102e4, 0x102e5, 0x102e6 ,
        0x102e7, 0x102e8, 0x102e9, 0x102ea, 0x102eb, 0x102ec, 0x102ed, 0x102ee, 0x102ef, 0x102f0 ,
        0x102f1, 0x102f2, 0x102f3, 0x102f4, 0x102f5, 0x102f6, 0x102f7, 0x102f8, 0x102f9, 0x102fa ,
        0x102fb, 0x10320, 0x10321, 0x10322, 0x10323, 0x10341, 0x1034a, 0x103d1, 0x103d2, 0x103d3 ,
        0x103d4, 0x103d5, 0x104a0, 0x104a1, 0x104a2, 0x104a3, 0x104a4, 0x104a5, 0x104a6, 0x104a7 ,
        0x104a8, 0x104a9, 0x10858, 0x10859, 0x1085a, 0x1085b, 0x1085c, 0x1085d, 0x1085e, 0x1085f ,
        0x10879, 0x1087a, 0x1087b, 0x1087c, 0x1087d, 0x1087e, 0x1087f, 0x108a7, 0x108a8, 0x108a9 ,
        0x108aa, 0x108ab, 0x108ac, 0x108ad, 0x108ae, 0x108af, 0x108fb, 0x108fc, 0x108fd, 0x108fe ,
        0x108ff, 0x10916, 0x10917, 0x10918, 0x10919, 0x1091a, 0x1091b, 0x109bc, 0x109bd, 0x109c0 ,
        0x109c1, 0x109c2, 0x109c3, 0x109c4, 0x109c5, 0x109c6, 0x109c7, 0x109c8, 0x109c9, 0x109ca ,
        0x109cb, 0x109cc, 0x109cd, 0x109ce, 0x109cf, 0x109d2, 0x109d3, 0x109d4, 0x109d5, 0x109d6 ,
        0x109d7, 0x109d8, 0x109d9, 0x109da, 0x109db, 0x109dc, 0x109dd, 0x109de, 0x109df, 0x109e0 ,
        0x109e1, 0x109e2, 0x109e3, 0x109e4, 0x109e5, 0x109e6, 0x109e7, 0x109e8, 0x109e9, 0x109ea ,
        0x109eb, 0x109ec, 0x109ed, 0x109ee, 0x109ef, 0x109f0, 0x109f1, 0x109f2, 0x109f3, 0x109f4 ,
        0x109f5, 0x109f6, 0x109f7, 0x109f8, 0x109f9, 0x109fa, 0x109fb, 0x109fc, 0x109fd, 0x109fe ,
        0x109ff, 0x10a40, 0x10a41, 0x10a42, 0x10a43, 0x10a44, 0x10a45, 0x10a46, 0x10a47, 0x10a48 ,
        0x10a7d, 0x10a7e, 0x10a9d, 0x10a9e, 0x10a9f, 0x10aeb, 0x10aec, 0x10aed, 0x10aee, 0x10aef ,
        0x10b58, 0x10b59, 0x10b5a, 0x10b5b, 0x10b5c, 0x10b5d, 0x10b5e, 0x10b5f, 0x10b78, 0x10b79 ,
        0x10b7a, 0x10b7b, 0x10b7c, 0x10b7d, 0x10b7e, 0x10b7f, 0x10ba9, 0x10baa, 0x10bab, 0x10bac ,
        0x10bad, 0x10bae, 0x10baf, 0x10cfa, 0x10cfb, 0x10cfc, 0x10cfd, 0x10cfe, 0x10cff, 0x10d30 ,
        0x10d31, 0x10d32, 0x10d33, 0x10d34, 0x10d35, 0x10d36, 0x10d37, 0x10d38, 0x10d39, 0x10e60 ,
        0x10e61, 0x10e62, 0x10e63, 0x10e64, 0x10e65, 0x10e66, 0x10e67, 0x10e68, 0x10e69, 0x10e6a ,
        0x10e6b, 0x10e6c, 0x10e6d, 0x10e6e, 0x10e6f, 0x10e70, 0x10e71, 0x10e72, 0x10e73, 0x10e74 ,
        0x10e75, 0x10e76, 0x10e77, 0x10e78, 0x10e79, 0x10e7a, 0x10e7b, 0x10e7c, 0x10e7d, 0x10e7e ,
        0x10f1d, 0x10f1e, 0x10f1f, 0x10f20, 0x10f21, 0x10f22, 0x10f23, 0x10f24, 0x10f25, 0x10f26 ,
        0x10f51, 0x10f52, 0x10f53, 0x10f54, 0x10fc5, 0x10fc6, 0x10fc7, 0x10fc8, 0x10fc9, 0x10fca ,
        0x10fcb, 0x11052, 0x11053, 0x11054, 0x11055, 0x11056, 0x11057, 0x11058, 0x11059, 0x1105a ,
        0x1105b, 0x1105c, 0x1105d, 0x1105e, 0x1105f, 0x11060, 0x11061, 0x11062, 0x11063, 0x11064 ,
        0x11065, 0x11066, 0x11067, 0x11068, 0x11069, 0x1106a, 0x1106b, 0x1106c, 0x1106d, 0x1106e ,
        0x1106f, 0x110f0, 0x110f1, 0x110f2, 0x110f3, 0x110f4, 0x110f5, 0x110f6, 0x110f7, 0x110f8 ,
        0x110f9, 0x11136, 0x11137, 0x11138, 0x11139, 0x1113a, 0x1113b, 0x1113c, 0x1113d, 0x1113e ,
        0x1113f, 0x111d0, 0x111d1, 0x111d2, 0x111d3, 0x111d4, 0x111d5, 0x111d6, 0x111d7, 0x111d8 ,
        0x111d9, 0x111e1, 0x111e2, 0x111e3, 0x111e4, 0x111e5, 0x111e6, 0x111e7, 0x111e8, 0x111e9 ,
        0x111ea, 0x111eb, 0x111ec, 0x111ed, 0x111ee, 0x111ef, 0x111f0, 0x111f1, 0x111f2, 0x111f3 ,
        0x111f4, 0x112f0, 0x112f1, 0x112f2, 0x112f3, 0x112f4, 0x112f5, 0x112f6, 0x112f7, 0x112f8 ,
        0x112f9, 0x11450, 0x11451, 0x11452, 0x11453, 0x11454, 0x11455, 0x11456, 0x11457, 0x11458 ,
        0x11459, 0x114d0, 0x114d1, 0x114d2, 0x114d3, 0x114d4, 0x114d5, 0x114d6, 0x114d7, 0x114d8 ,
        0x114d9, 0x11650, 0x11651, 0x11652, 0x11653, 0x11654, 0x11655, 0x11656, 0x11657, 0x11658 ,
        0x11659, 0x116c0, 0x116c1, 0x116c2, 0x116c3, 0x116c4, 0x116c5, 0x116c6, 0x116c7, 0x116c8 ,
        0x116c9, 0x11730, 0x11731, 0x11732, 0x11733, 0x11734, 0x11735, 0x11736, 0x11737, 0x11738 ,
        0x11739, 0x1173a, 0x1173b, 0x118e0, 0x118e1, 0x118e2, 0x118e3, 0x118e4, 0x118e5, 0x118e6 ,
        0x118e7, 0x118e8, 0x118e9, 0x118ea, 0x118eb, 0x118ec, 0x118ed, 0x118ee, 0x118ef, 0x118f0 ,
        0x118f1, 0x118f2, 0x11950, 0x11951, 0x11952, 0x11953, 0x11954, 0x11955, 0x11956, 0x11957 ,
        0x11958, 0x11959, 0x11c50, 0x11c51, 0x11c52, 0x11c53, 0x11c54, 0x11c55, 0x11c56, 0x11c57 ,
        0x11c58, 0x11c59, 0x11c5a, 0x11c5b, 0x11c5c, 0x11c5d, 0x11c5e, 0x11c5f, 0x11c60, 0x11c61 ,
        0x11c62, 0x11c63, 0x11c64, 0x11c65, 0x11c66, 0x11c67, 0x11c68, 0x11c69, 0x11c6a, 0x11c6b ,
        0x11c6c, 0x11d50, 0x11d51, 0x11d52, 0x11d53, 0x11d54, 0x11d55, 0x11d56, 0x11d57, 0x11d58 ,
        0x11d59, 0x11da0, 0x11da1, 0x11da2, 0x11da3, 0x11da4, 0x11da5, 0x11da6, 0x11da7, 0x11da8 ,
        0x11da9, 0x11f50, 0x11f51, 0x11f52, 0x11f53, 0x11f54, 0x11f55, 0x11f56, 0x11f57, 0x11f58 ,
        0x11f59, 0x11fc0, 0x11fc1, 0x11fc2, 0x11fc3, 0x11fc4, 0x11fc5, 0x11fc6, 0x11fc7, 0x11fc8 ,
        0x11fc9, 0x11fca, 0x11fcb, 0x11fcc, 0x11fcd, 0x11fce, 0x11fcf, 0x11fd0, 0x11fd1, 0x11fd2 ,
        0x11fd3, 0x11fd4, 0x12400, 0x12401, 0x12402, 0x12403, 0x12404, 0x12405, 0x12406, 0x12407 ,
        0x12408, 0x12409, 0x1240a, 0x1240b, 0x1240c, 0x1240d, 0x1240e, 0x1240f, 0x12410, 0x12411 ,
        0x12412, 0x12413, 0x12414, 0x12415, 0x12416, 0x12417, 0x12418, 0x12419, 0x1241a, 0x1241b ,
        0x1241c, 0x1241d, 0x1241e, 0x1241f, 0x12420, 0x12421, 0x12422, 0x12423, 0x12424, 0x12425 ,
        0x12426, 0x12427, 0x12428, 0x12429, 0x1242a, 0x1242b, 0x1242c, 0x1242d, 0x1242e, 0x1242f ,
        0x12430, 0x12431, 0x12432, 0x12433, 0x12434, 0x12435, 0x12436, 0x12437, 0x12438, 0x12439 ,
        0x1243a, 0x1243b, 0x1243c, 0x1243d, 0x1243e, 0x1243f, 0x12440, 0x12441, 0x12442, 0x12443 ,
        0x12444, 0x12445, 0x12446, 0x12447, 0x12448, 0x12449, 0x1244a, 0x1244b, 0x1244c, 0x1244d ,
        0x1244e, 0x1244f, 0x12450, 0x12451, 0x12452, 0x12453, 0x12454, 0x12455, 0x12456, 0x12457 ,
        0x12458, 0x12459, 0x1245a, 0x1245b, 0x1245c, 0x1245d, 0x1245e, 0x1245f, 0x12460, 0x12461 ,
        0x12462, 0x12463, 0x12464, 0x12465, 0x12466, 0x12467, 0x12468, 0x12469, 0x1246a, 0x1246b ,
        0x1246c, 0x1246d, 0x1246e, 0x16a60, 0x16a61, 0x16a62, 0x16a63, 0x16a64, 0x16a65, 0x16a66 ,
        0x16a67, 0x16a68, 0x16a69, 0x16ac0, 0x16ac1, 0x16ac2, 0x16ac3, 0x16ac4, 0x16ac5, 0x16ac6 ,
        0x16ac7, 0x16ac8, 0x16ac9, 0x16b50, 0x16b51, 0x16b52, 0x16b53, 0x16b54, 0x16b55, 0x16b56 ,
        0x16b57, 0x16b58, 0x16b59, 0x16b5b, 0x16b5c, 0x16b5d, 0x16b5e, 0x16b5f, 0x16b60, 0x16b61 ,
        0x16e80, 0x16e81, 0x16e82, 0x16e83, 0x16e84, 0x16e85, 0x16e86, 0x16e87, 0x16e88, 0x16e89 ,
        0x16e8a, 0x16e8b, 0x16e8c, 0x16e8d, 0x16e8e, 0x16e8f, 0x16e90, 0x16e91, 0x16e92, 0x16e93 ,
        0x16e94, 0x16e95, 0x16e96, 0x1d2c0, 0x1d2c1, 0x1d2c2, 0x1d2c3, 0x1d2c4, 0x1d2c5, 0x1d2c6 ,
        0x1d2c7, 0x1d2c8, 0x1d2c9, 0x1d2ca, 0x1d2cb, 0x1d2cc, 0x1d2cd, 0x1d2ce, 0x1d2cf, 0x1d2d0 ,
        0x1d2d1, 0x1d2d2, 0x1d2d3, 0x1d2e0, 0x1d2e1, 0x1d2e2, 0x1d2e3, 0x1d2e4, 0x1d2e5, 0x1d2e6 ,
        0x1d2e7, 0x1d2e8, 0x1d2e9, 0x1d2ea, 0x1d2eb, 0x1d2ec, 0x1d2ed, 0x1d2ee, 0x1d2ef, 0x1d2f0 ,
        0x1d2f1, 0x1d2f2, 0x1d2f3, 0x1d360, 0x1d361, 0x1d362, 0x1d363, 0x1d364, 0x1d365, 0x1d366 ,
        0x1d367, 0x1d368, 0x1d369, 0x1d36a, 0x1d36b, 0x1d36c, 0x1d36d, 0x1d36e, 0x1d36f, 0x1d370 ,
        0x1d371, 0x1d372, 0x1d373, 0x1d374, 0x1d375, 0x1d376, 0x1d377, 0x1d378, 0x1d7ce, 0x1d7cf ,
        0x1d7d0, 0x1d7d1, 0x1d7d2, 0x1d7d3, 0x1d7d4, 0x1d7d5, 0x1d7d6, 0x1d7d7, 0x1d7d8, 0x1d7d9 ,
        0x1d7da, 0x1d7db, 0x1d7dc, 0x1d7dd, 0x1d7de, 0x1d7df, 0x1d7e0, 0x1d7e1, 0x1d7e2, 0x1d7e3 ,
        0x1d7e4, 0x1d7e5, 0x1d7e6, 0x1d7e7, 0x1d7e8, 0x1d7e9, 0x1d7ea, 0x1d7eb, 0x1d7ec, 0x1d7ed ,
        0x1d7ee, 0x1d7ef, 0x1d7f0, 0x1d7f1, 0x1d7f2, 0x1d7f3, 0x1d7f4, 0x1d7f5, 0x1d7f6, 0x1d7f7 ,
        0x1d7f8, 0x1d7f9, 0x1d7fa, 0x1d7fb, 0x1d7fc, 0x1d7fd, 0x1d7fe, 0x1d7ff, 0x1e140, 0x1e141 ,
        0x1e142, 0x1e143, 0x1e144, 0x1e145, 0x1e146, 0x1e147, 0x1e148, 0x1e149, 0x1e2f0, 0x1e2f1 ,
        0x1e2f2, 0x1e2f3, 0x1e2f4, 0x1e2f5, 0x1e2f6, 0x1e2f7, 0x1e2f8, 0x1e2f9, 0x1e4f0, 0x1e4f1 ,
        0x1e4f2, 0x1e4f3, 0x1e4f4, 0x1e4f5, 0x1e4f6, 0x1e4f7, 0x1e4f8, 0x1e4f9, 0x1e8c7, 0x1e8c8 ,
        0x1e8c9, 0x1e8ca, 0x1e8cb, 0x1e8cc, 0x1e8cd, 0x1e8ce, 0x1e8cf, 0x1e950, 0x1e951, 0x1e952 ,
        0x1e953, 0x1e954, 0x1e955, 0x1e956, 0x1e957, 0x1e958, 0x1e959, 0x1ec71, 0x1ec72, 0x1ec73 ,
        0x1ec74, 0x1ec75, 0x1ec76, 0x1ec77, 0x1ec78, 0x1ec79, 0x1ec7a, 0x1ec7b, 0x1ec7c, 0x1ec7d ,
        0x1ec7e, 0x1ec7f, 0x1ec80, 0x1ec81, 0x1ec82, 0x1ec83, 0x1ec84, 0x1ec85, 0x1ec86, 0x1ec87 ,
        0x1ec88, 0x1ec89, 0x1ec8a, 0x1ec8b, 0x1ec8c, 0x1ec8d, 0x1ec8e, 0x1ec8f, 0x1ec90, 0x1ec91 ,
        0x1ec92, 0x1ec93, 0x1ec94, 0x1ec95, 0x1ec96, 0x1ec97, 0x1ec98, 0x1ec99, 0x1ec9a, 0x1ec9b ,
        0x1ec9c, 0x1ec9d, 0x1ec9e, 0x1ec9f, 0x1eca0, 0x1eca1, 0x1eca2, 0x1eca3, 0x1eca4, 0x1eca5 ,
        0x1eca6, 0x1eca7, 0x1eca8, 0x1eca9, 0x1ecaa, 0x1ecab, 0x1ecad, 0x1ecae, 0x1ecaf, 0x1ecb1 ,
        0x1ecb2, 0x1ecb3, 0x1ecb4, 0x1ed01, 0x1ed02, 0x1ed03, 0x1ed04, 0x1ed05, 0x1ed06, 0x1ed07 ,
        0x1ed08, 0x1ed09, 0x1ed0a, 0x1ed0b, 0x1ed0c, 0x1ed0d, 0x1ed0e, 0x1ed0f, 0x1ed10, 0x1ed11 ,
        0x1ed12, 0x1ed13, 0x1ed14, 0x1ed15, 0x1ed16, 0x1ed17, 0x1ed18, 0x1ed19, 0x1ed1a, 0x1ed1b ,
        0x1ed1c, 0x1ed1d, 0x1ed1e, 0x1ed1f, 0x1ed20, 0x1ed21, 0x1ed22, 0x1ed23, 0x1ed24, 0x1ed25 ,
        0x1ed26, 0x1ed27, 0x1ed28, 0x1ed29, 0x1ed2a, 0x1ed2b, 0x1ed2c, 0x1ed2d, 0x1ed2f, 0x1ed30 ,
        0x1ed31, 0x1ed32, 0x1ed33, 0x1ed34, 0x1ed35, 0x1ed36, 0x1ed37, 0x1ed38, 0x1ed39, 0x1ed3a ,
        0x1ed3b, 0x1ed3c, 0x1ed3d, 0x1f100, 0x1f101, 0x1f102, 0x1f103, 0x1f104, 0x1f105, 0x1f106 ,
        0x1f107, 0x1f108, 0x1f109, 0x1f10a, 0x1f10b, 0x1f10c, 0x1fbf0, 0x1fbf1, 0x1fbf2, 0x1fbf3 ,
        0x1fbf4, 0x1fbf5, 0x1fbf6, 0x1fbf7, 0x1fbf8, 0x1fbf9, 0x20001, 0x20064, 0x200e2, 0x20121 ,
        0x2092a, 0x20983, 0x2098c, 0x2099c, 0x20aea, 0x20afd, 0x20b19, 0x22390, 0x22998, 0x23b1b ,
        0x2626d, 0x2f890 ,
      ];
      return arraySearch (allNumericUnicodes, a);
  }
}
