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

    proc uniqueMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var (returnGroupsStr, nstr, rest) = payload.splitMsgToTuple(3);
        // flag to return segments and permutation for GroupBy
        const returnGroups = if (returnGroupsStr == "True") then true else false;
        var repMsg: string = "";
        // number of arrays
        var n = nstr:int;
        var fields = rest.split();
        // hash each "row", or set of parallel array elements
        // this function will validate that all arrays are same length
        var hashes = hashArrays(n, fields, st);
        // Sort the hashes
        var kr = radixSortLSD(hashes);
        // Unpack the permutation and sorted hashes
        // var perm: [kr.domain] int;
        var permutation = new shared SymEntry(kr.size, int);
        ref perm = permutation.a;
        var sortedHashes: [kr.domain] 2*uint;
        forall (sh, p, val) in zip(sortedHashes, perm, kr) {
          (sh, p) = val;
        }
        // Get the unique hashes and the count of each
        var (uniqueHashes, counts) = uniqueFromSorted(sortedHashes);
        // Compute offset of each group in sorted array
        var segments = new shared SymEntry(counts.size, int);
        segments.a = (+ scan counts) - counts;
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
        var gatherInds: [segments.aD] int;
        ref segs = segments.a;
        forall (g, s) in zip(gatherInds, segs) with (var agg = newSrcAggregator(int)) {
          agg.copy(g, perm[s]);
        }
        // Gather unique values, store in SymTab, and build repMsg
        repMsg += storeUniqueKeys(n, fields, gatherInds, st);
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

    proc hashArrays(n, fields, st) throws {
      if (n > 128) {
        throw new owned ErrorWithContext("Cannot hash more than 128 arrays",
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "ArgumentError");
      }
      var (size, hasStr, names, types) = validateArraysSameLength(n, fields, st);
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
