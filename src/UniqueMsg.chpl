/* unique finding and counting algorithms
 these are all based on dense histograms and sparse histograms(assoc domains/arrays)

 you could also use a sort if you got into a real bind with really
 large dense ranges of values and large arrays...

 *** need to factor in sparsity estimation somehow ***
 for example if (a.max-a.min > a.size) means that a's are sparse

 */
module UniqueMsg
{
    use ServerConfig;
    use AryUtil;
    
    use Time only;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedArray;
    use ServerErrorStrings;

    use RadixSortLSD;
    use Unique;
    use SipHash;
    use CommAggregation;
    
    private config const logLevel = ServerConfig.logLevel;
    const umLogger = new Logger(logLevel);

    proc assumeSortedShortcut(n, fields, st) throws {
      // very similar to uniqueAndCount but skips sort
      var (size, hasStr, names, types) = validateArraysSameLength(n, fields, st);
      if (size == 0) {
        return (new shared SymEntry(0, int), new shared SymEntry(0, int));
      }
      proc skipSortHelper(type t, keys: [?D] t) throws {
        // skip sorting, set permutation to 0..#size and go directly to finding segment boundaries.
        var permutation = new shared SymEntry(keys.size, int);
        permutation.a = permutation.aD;
        var (uniqueKeys, counts) = uniqueFromSorted(keys);
        var segments = new shared SymEntry(counts.size, int);
        segments.a = (+ scan counts) - counts;
        return (permutation, segments);
      }

      // If no strings are present and row values can fit in 128 bits (8 digits),
      // then pack into tuples of uint(16) for sorting keys.
      if !hasStr {
        var (totalDigits, bitWidths, negs) = getNumDigitsNumericArrays(names, st);
        if totalDigits <= 2 { return skipSortHelper(2*uint(bitsPerDigit), mergeNumericArrays(2, size, totalDigits, bitWidths, negs, names, st)); }
        if totalDigits <= 4 { return skipSortHelper(4*uint(bitsPerDigit), mergeNumericArrays(4, size, totalDigits, bitWidths, negs, names, st)); }
        if totalDigits <= 6 { return skipSortHelper(6*uint(bitsPerDigit), mergeNumericArrays(6, size, totalDigits, bitWidths, negs, names, st)); }
        if totalDigits <= 8 { return skipSortHelper(8*uint(bitsPerDigit), mergeNumericArrays(8, size, totalDigits, bitWidths, negs, names, st)); }
      }

      // If here, either the row values are too large to fit in 128 bits, or
      // strings are present and must be hashed anyway, so hash all arrays
      // and combine hashes of row values into sorting keys.
      return skipSortHelper(2*uint(64), hashArrays(size, names, types, st));
    }

    proc uniqueMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        var (returnGroupsStr, assumeSortedStr, nstr, rest) = payload.splitMsgToTuple(4);
        // flag to return segments and permutation for GroupBy
        const returnGroups = if (returnGroupsStr == "True") then true else false;
        const assumeSorted = if (assumeSortedStr == "True") then true else false;
        var repMsg: string = "";
        // number of arrays
        var n = nstr:int;
        var fields = rest.split();
        var (permutation, segments) = if assumeSorted then assumeSortedShortcut(n, fields, st) else uniqueAndCount(n, fields, st);
        
        // If returning grouping info, add to SymTab and prepend to repMsg
        if returnGroups {
          var pname = st.nextName();
          st.addEntry(pname, permutation);
          repMsg += "created " + st.attrib(pname);
          var sname = st.nextName();
          st.addEntry(sname, segments);
          repMsg += "+created " + st.attrib(sname) + "+";
        }
        // Indices of first unique key in original array
        // These are the value of the permutation at the start of each group
        var uniqueKeyInds = new shared SymEntry(segments.size, int);
        if (segments.size > 0) {
          // Avoid initializing aggregators if empty array
          ref perm = permutation.a;
          ref segs = segments.a;
          ref inds = uniqueKeyInds.a;
          forall (i, s) in zip(inds, segs) with (var agg = newSrcAggregator(int)) {
            agg.copy(i, perm[s]);
          }
        }
        var iname = st.nextName();
        st.addEntry(iname, uniqueKeyInds);
        repMsg += "created " + st.attrib(iname);
        /* // Gather unique values, store in SymTab, and build repMsg */
        /* repMsg += storeUniqueKeys(n, fields, gatherInds, st); */
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc storeUniqueKeys(n, fields, gatherInds, st): string throws {
      // Number of unique keys
      const size = gatherInds.size;
      // An underestimate for strings, unfortunately
      overMemLimit(n*size*numBytes(int));
      var repMsg: string;
      const low = fields.domain.low;
      var names = fields[low..#n];
      var types = fields[low+n..#n];
      // For each input array, gather unique values
      for (name, objtype, i) in zip(names, types, 0..) {
        var newName = st.nextName();
        select objtype {
          when "pdarray", "category" {
            var g = getGenericTypedArrayEntry(name, st);
            // Gathers unique values, stores in SymTab, and returns repMsg chunk
            proc gatherHelper(type t) throws {
              var e = toSymEntry(g, t);
              ref ea = e.a;
              var unique = st.addEntry(newName, size, t);
              forall (u, i) in zip(unique.a, gatherInds) with (var agg = newSrcAggregator(t)) {
                agg.copy(u, ea[i]);
              }
              if repMsg.size > 0 {
                repMsg += " +";
              }
              repMsg += "created " + st.attrib(newName);
            }
            select g.dtype {
              when DType.Int64 {
                gatherHelper(int);
              }
              when DType.UInt64 {
                gatherHelper(uint);
              }
              when DType.Float64 {
                gatherHelper(real);
              }
              when DType.Bool {
                gatherHelper(bool);
              }
            }
          }
          when "str" {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = getSegString(myNames, st);
            var (uSegs, uVals) = g[gatherInds];
            var newStringsObj = getSegString(uSegs, uVals, st);
            repMsg += "created " + st.attrib(newStringsObj.name) + "+created bytes.size %t".format(newStringsObj.nBytes);
          }
        }
      }
      return repMsg;
    }

    proc uniqueAndCount(n, fields, st) throws {
      if (n > 128) {
        throw new owned ErrorWithContext("Cannot hash more than 128 arrays",
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "ArgumentError");
      }
      var (size, hasStr, names, types) = validateArraysSameLength(n, fields, st);
      if (size == 0) {
        return (new shared SymEntry(0, int), new shared SymEntry(0, int));
      }
      proc helper(itemsize, type t, keys: [?D] t) throws {
        // Sort the keys
        var sortMem = radixSortLSD_memEst(keys.size, itemsize);
        overMemLimit(sortMem);
        var kr = radixSortLSD(keys);
        // Unpack the permutation and sorted keys
        // var perm: [kr.domain] int;
        var permutation = new shared SymEntry(kr.size, int);
        ref perm = permutation.a;
        var sortedKeys: [D] t;
        forall (sh, p, val) in zip(sortedKeys, perm, kr) {
          (sh, p) = val;
        }
        // Get the unique keys and the count of each
        var (uniqueKeys, counts) = uniqueFromSorted(sortedKeys);
        // Compute offset of each group in sorted array
        var segments = new shared SymEntry(counts.size, int);
        segments.a = (+ scan counts) - counts;
        return (permutation, segments);
      }
      
      // If no strings are present and row values can fit in 128 bits (8 digits),
      // then pack into tuples of uint(16) for sorting keys.
      if !hasStr {
        var (totalDigits, bitWidths, negs) = getNumDigitsNumericArrays(names, st);
        if totalDigits <= 2 { return helper(2 * bitsPerDigit / 8, 2*uint(bitsPerDigit), mergeNumericArrays(2, size, totalDigits, bitWidths, negs, names, st)); }
        if totalDigits <= 4 { return helper(4 * bitsPerDigit / 8, 4*uint(bitsPerDigit), mergeNumericArrays(4, size, totalDigits, bitWidths, negs, names, st)); }
        if totalDigits <= 6 { return helper(6 * bitsPerDigit / 8, 6*uint(bitsPerDigit), mergeNumericArrays(6, size, totalDigits, bitWidths, negs, names, st)); }
        if totalDigits <= 8 { return helper(8 * bitsPerDigit / 8, 8*uint(bitsPerDigit), mergeNumericArrays(8, size, totalDigits, bitWidths, negs, names, st)); }
      }

      // If here, either the row values are too large to fit in 128 bits, or
      // strings are present and must be hashed anyway, so hash all arrays
      // and combine hashes of row values into sorting keys.
      return helper(16, 2*uint(64), hashArrays(size, names, types, st));
    }

    proc hashArrays(size, names, types, st): [] 2*uint throws {
      overMemLimit(numBytes(uint) * size * 2);
      var dom = makeDistDom(size);
      var hashes: [dom] 2*uint(64);
      /* Hashes of subsequent arrays cannot be simply XORed
       * because equivalent values will cancel each other out.
       * Thus, a non-linear function must be applied to each array,
       * hence we do a rotation by the ordinal of the array. This
       * will only handle up to 128 arrays before rolling back around.
       */
      proc rotl(h:2*uint(64), n:int):2*uint(64) {
        use BitOps;
        // no rotation
        if (n == 0) { return h; }
        // Rotate each 64-bit word independently, then swap tails
        const (h1, h2) = h;
        // Mask for tail (right-hand portion)
        const rmask = (1 << n) - 1;
        // Mask for head (left-hand portion)
        const lmask = 2**64 - 1 - rmask;
        // Rotate each word
        var r1 = rotl(h1, n);
        var r2 = rotl(h2, n);
        // Swap tails
        r1 = (r1 & lmask) | (r2 & rmask);
        r2 = (r2 & lmask) | (r1 & rmask);
        return (r1, r2);
      }
      for (name, objtype, i) in zip(names, types, 0..) {
        select objtype {
          when "pdarray", "category" {
            var g = getGenericTypedArrayEntry(name, st);
            select g.dtype {
              when DType.Int64 {
                var e = toSymEntry(g, int);
                forall (h, x) in zip(hashes, e.a) {
                  h ^= rotl(sipHash128(x), i);
                }
              }
              when DType.UInt64 {
                var e = toSymEntry(g, uint);
                forall (h, x) in zip(hashes, e.a) {
                  h ^= rotl(sipHash128(x), i);
                }
              }
              when DType.Float64 {
                var e = toSymEntry(g, real);
                forall (h, x) in zip(hashes, e.a) {
                  h ^= rotl(sipHash128(x), i);
                }
              }
              when DType.Bool {
                var e = toSymEntry(g, bool);
                forall (h, x) in zip(hashes, e.a) {
                  h ^= rotl((0:uint, x:uint), i);
                }
              }
            }
          }
          when "str" {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = getSegString(myNames, st);
            hashes ^= rotl(g.hash(), i);
          }
        }
      }
      return hashes;
    }

    proc registerMe() {
      use CommandMap;
      registerFunction("unique", uniqueMsg, getModuleName());
    }
}
