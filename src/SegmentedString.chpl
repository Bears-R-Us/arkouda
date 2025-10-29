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
  use NumericUnicodes;

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
      st.insert(offsetsEntry);  // insert() is needed for checkpointing
      st.insert(valuesEntry);
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
          rlen = interpretAsBytes(origVals, off..#len, true).strip().size + 1;
        } else {
          rlen = interpretAsBytes(origVals, off..#len, true).strip(chars:bytes).size + 1;
        }
      }
      var retVals: [makeDistDom(+ reduce replacedLens)] uint(8);
      var retOffs = (+ scan replacedLens) - replacedLens;

      forall (off, len, roff) in zip(origOffsets, lengths, retOffs) with (var valAgg = newDstAggregator(uint(8))) {
        var i = 0;
        if chars.isEmpty() {
          for b in interpretAsBytes(origVals, off..#len, true).strip() {
            valAgg.copy(retVals[roff+i], b:uint(8));
            i += 1;
          }
        } else {
          for b in interpretAsBytes(origVals, off..#len, true).strip(chars:bytes) {
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
        return bytes.createBorrowingBuffer(localSlice.ptr, region.size-1, region.size);
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
        good = isNumericChar(letter);
        if !good then break;
      }
      return good;
  }

  inline proc isNumericChar (c: string) {
      var code : int(32) = c.toCodepoint();
      return allNumericUnicodes.contains(code) ;
  }

}
