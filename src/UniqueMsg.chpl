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
    
    use ArkoudaTimeCompat as Time;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use GenSymIO;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;

    use RadixSortLSD;
    use Unique;
    use SipHash;
    use CommAggregation;
    use SegmentedArray;
    use HashMsg;
    
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const umLogger = new Logger(logLevel, logChannel);

    proc uniqueMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        // flag to return segments and permutation for GroupBy
        const returnGroups = msgArgs.get("returnGroupStr").getBoolValue();
        const assumeSorted = msgArgs.get("assumeSortedStr").getBoolValue();
        var repMsg: string = "";
        // number of arrays
        var n = msgArgs.get("nstr").getIntValue();
        var keynames = msgArgs.get("keynames").getList(n);
        var keytypes = msgArgs.get("keytypes").getList(n);
        var (permutation, segments) = uniqueAndCount(n, keynames, keytypes, assumeSorted, st);
        
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
        var uniqueKeyInds = createSymEntry(segments.size, int);
        if segments.size > 0 {
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

    proc storeUniqueKeys(n, names: [] string, types: [] string, gatherInds, st): string throws {
      // Number of unique keys
      const size = gatherInds.size;
      // An underestimate for strings, unfortunately
      overMemLimit(n*size*numBytes(int));
      var repMsg: string;
      // For each input array, gather unique values
      for (name, objtype, i) in zip(names, types, 0..) {
        var newName = st.nextName();
        select objtype.toUpper(): ObjType {
          when ObjType.PDARRAY, ObjType.CATEGORICAL {
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
          when ObjType.STRINGS {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = getSegString(myNames, st);
            var (uSegs, uVals) = g[gatherInds];
            var newStringsObj = getSegString(uSegs, uVals, st);
            repMsg += "created " + st.attrib(newStringsObj.name) + "+created bytes.size %?".doFormat(newStringsObj.nBytes);
          }
        }
      }
      return repMsg;
    }

    proc uniqueAndCount(n, namesList: [] string, typesList: [] string, assumeSorted: bool, st) throws {
      if (n > 128) {
        throw new owned ErrorWithContext("Cannot hash more than 128 arrays",
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "ArgumentError");
      }
      var (size, hasStr, allSmallStrs, extraArraysNeeded, numStrings, names, types) = validateArraysSameLength(n, namesList, typesList, st);
      if size == 0 {
        return (createSymEntry(0, int), createSymEntry(0, int));
      }
      proc helper(itemsize, type t, keys: [?D] t) throws {
        var permutation = createSymEntry(keys.size, int);
        var sortedKeys = makeDistArray(keys);

        if assumeSorted {
          // set permutation to 0..#size and go directly to finding segment boundaries.
          permutation.a = permutation.a.domain;
        }
        else {
          // Sort the keys
          overMemLimit(radixSortLSD_memEst(keys.size, itemsize));
          var kr = radixSortLSD(keys);
          // Unpack the permutation and sorted keys
          ref perm = permutation.a;
          forall (sh, p, val) in zip(sortedKeys, perm, kr) {
            (sh, p) = val;
          }
        }
        // Get the unique keys and the count of each
        var (uniqueKeys, counts) = uniqueFromSorted(sortedKeys);
        // Compute offset of each group in sorted array
        var segments = createSymEntry(counts.size, int);
        segments.a = (+ scan counts) - counts;
        return (permutation, segments);
      }

      inline proc cleanup(strNames: [] string, st) throws {
        for name in strNames {
          st.deleteEntry(name);
        }
      }

      var strNames: [0..#(numStrings + extraArraysNeeded)] string;
      var newNames: [0..#(n+extraArraysNeeded)] string;
      var newTypes: [0..#(n+extraArraysNeeded)] string;
      var replaceStrings = hasStr && allSmallStrs;
      if replaceStrings {
        var strIdx = 0;
        var nameIdx = 0;
        for (name, objtype) in zip(namesList, types) {
          if objtype.toUpper(): ObjType == ObjType.STRINGS {
            var (myNames, _) = name.splitMsgToTuple("+", 2);
            var strings = getSegString(myNames, st);
            const lengths = strings.getLengths() - 1;
            const max_bytes = (max reduce lengths);
            for arrName in strings.bytesToUintArr(max_bytes, lengths, st).split("+") {
              strNames[strIdx] = arrName;
              strIdx += 1;
              newNames[nameIdx] = arrName;
              newTypes[nameIdx] = "PDARRAY";
              nameIdx += 1;
            }
          }
          else {
            newNames[nameIdx] = name;
            newTypes[nameIdx] = objtype;
            nameIdx += 1;
          }
        }
      }

      proc digitHelper(helperNames = names, helperTypes = types) throws {
        // If row values can fit in 128 bits (8 digits) and all strings are small,
        // then pack into tuples of uint(16) for sorting keys.
        if !hasStr || allSmallStrs {
          var (totalDigits, bitWidths, negs) = getNumDigitsNumericArrays(helperNames, st);
          if totalDigits <= 2 { return helper(2 * bitsPerDigit / 8, 2*uint(bitsPerDigit), mergeNumericArrays(2, size, totalDigits, bitWidths, negs, helperNames, st)); }
          else if totalDigits <= 4 { return helper(4 * bitsPerDigit / 8, 4*uint(bitsPerDigit), mergeNumericArrays(4, size, totalDigits, bitWidths, negs, helperNames, st)); }
          else if totalDigits <= 6 { return helper(6 * bitsPerDigit / 8, 6*uint(bitsPerDigit), mergeNumericArrays(6, size, totalDigits, bitWidths, negs, helperNames, st)); }
          else if totalDigits <= 8 { return helper(8 * bitsPerDigit / 8, 8*uint(bitsPerDigit), mergeNumericArrays(8, size, totalDigits, bitWidths, negs, helperNames, st)); }
        }
        // If here, either the row values are too large to fit in 128 bits, or
        // large strings are present and must be hashed anyway, so hash all arrays
        // and combine hashes of row values into sorting keys.
        return helper(16, 2*uint(64), hashArrays(size, helperNames, helperTypes, st));
      }

      var (perm, segments) = if replaceStrings then digitHelper(newNames, newTypes) else digitHelper();
      cleanup(strNames, st);
      return (perm, segments);
    }

    proc hashArrays(size, names, types, st): [] 2*uint throws {
      overMemLimit(numBytes(uint) * size * 2);
      var dom = makeDistDom(size);
      var hashes = makeDistArray(dom, 2*uint);
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
        select objtype.toUpper(): ObjType {
          when ObjType.PDARRAY {
            var g = getGenericTypedArrayEntry(name, st);
            select g.dtype {
              when DType.Int64 {
                var e = toSymEntry(g, int);
                ref ea = e.a;
                forall (h, x) in zip(hashes, ea) {
                  h ^= rotl(sipHash128(x), i);
                }
              }
              when DType.UInt64 {
                var e = toSymEntry(g, uint);
                ref ea = e.a;
                forall (h, x) in zip(hashes, ea) {
                  h ^= rotl(sipHash128(x), i);
                }
              }
              when DType.Float64 {
                var e = toSymEntry(g, real);
                ref ea = e.a;
                forall (h, x) in zip(hashes, ea) {
                  h ^= rotl(sipHash128(x), i);
                }
              }
              when DType.Bool {
                var e = toSymEntry(g, bool);
                ref ea = e.a;
                forall (h, x) in zip(hashes, ea) {
                  h ^= rotl((0:uint, x:uint), i);
                }
              }
            }
          }
          when ObjType.STRINGS {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = getSegString(myNames, st);
            hashes ^= rotl(g.siphash(), i);
          }
          when ObjType.SEGARRAY {
            var segComps = jsonToMap(name);
            var (upper, lower) = segarrayHash(segComps["segments"], segComps["values"], segComps["valObjType"], st);
            forall (h, u, l) in zip(hashes, upper, lower) {
              h ^= rotl((u,l), i);
            }
          }
          when ObjType.CATEGORICAL {
            var catComps = jsonToMap(name);
            var (upper, lower) = categoricalHash(catComps["categories"], catComps["codes"], st);
            forall (h, u, l) in zip(hashes, upper, lower) {
              h ^= rotl((u,l), i);
            }
          }
        }
      }
      return hashes;
    }

    use CommandMap;
    registerFunction("unique", uniqueMsg, getModuleName());
}
