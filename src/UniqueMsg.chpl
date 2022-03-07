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
    
    private config const logLevel = ServerConfig.logLevel;
    const umLogger = new Logger(logLevel);
    
    /* unique take a pdarray and returns a pdarray with the unique values */
    proc oldUniqueMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (objtype, name, returnCountsStr, returnPermStr) = payload.splitMsgToTuple(4);
        // flag to return counts of each unique value
        // same size as unique array
        var returnCounts = if (returnCountsStr == "True") then true else false;
        var returnPerm = if (returnPermStr == "True") then true else false;
        select objtype {
            when "pdarray" {
                // get next symbol name for unique
                var vname = st.nextName();
                // get next symbol anme for counts
                var cname = st.nextName();
                var pname = st.nextName();
                umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                          "cmd: %s name: %s returnCounts: %t returnPerm: %t vname: %s cname: %s".format(
                          cmd,name,returnCounts,vname,cname));
        
                var gEnt: borrowed GenSymEntry;
                
                try {  
                    gEnt = getGenericTypedArrayEntry(name, st);
                } catch e: Error {
                    throw new owned ErrorWithContext("lookup for %s failed".format(name),
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "UnknownSymbolError");                
                }

                // the upper limit here is the same as argsort/radixSortLSD_keys
                // check and throw if over memory limit
                overMemLimit(radixSortLSD_memEst(gEnt.size, gEnt.itemsize));
        
                select (gEnt.dtype) {
                  when (DType.Int64) {
                    var e = toSymEntry(gEnt,int);

                    var (aV,aC,aP) = uniqueSort(e.a);
                    st.addEntry(vname, new shared SymEntry(aV));
                    if returnCounts {
                        st.addEntry(cname, new shared SymEntry(aC));
                    }
                    if returnPerm {
                        st.addEntry(pname, new shared SymEntry(aP));
                    }
                  } when (DType.UInt64) {
                    var e = toSymEntry(gEnt,uint);

                    var (aV,aC,aP) = uniqueSort(e.a);
                    st.addEntry(vname, new shared SymEntry(aV));
                    if returnCounts {
                        st.addEntry(cname, new shared SymEntry(aC));
                    }
                    if returnPerm {
                        st.addEntry(pname, new shared SymEntry(aP));
                    }
                  }
                  otherwise {
                      var errorMsg = notImplementedError("unique",gEnt.dtype);
                      umLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
        
            repMsg = "created " + st.attrib(vname);

            if returnCounts {
                repMsg += " +created " + st.attrib(cname);
            }
            umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);  
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when "str" {
              var str = getSegString(name, st);

              /*
               * The upper limit here is the similar to argsort/radixSortLSD_keys, but with 
               * a few more scratch arrays check and throw if over memory limit.
               */
              overMemLimit((8 * str.size * 8)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
              var (uo, uv, c, inv) = uniqueGroup(str);
              var myStr = getSegString(uo, uv, st);
              // TODO remove second, legacy call to st.attrib(myStr.name)
              repMsg = "created %s+created bytes.size %t".format(st.attrib(myStr.name), myStr.nBytes);

              if returnCounts {
                  var countName = st.nextName();
                  st.addEntry(countName, new shared SymEntry(c));
                  repMsg += " +created " + st.attrib(countName);
              }

              umLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
              return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          otherwise { 
             var errorMsg = notImplementedError(Reflection.getRoutineName(), objtype);
             umLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
             return new MsgTuple(errorMsg, MsgType.ERROR);              
           }
        }
    }
    
    /* value_counts takes a pdarray and returns two pdarrays unique values and counts for each value */
    proc value_countsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.splitMsgToTuple(1);

        // get next symbol name
        var vname = st.nextName();
        var cname = st.nextName();
        umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "cmd: %s name: %s vname: %s cname: %s".format(cmd, name, vname, cname));

        var gEnt: borrowed GenSymEntry;
        
        try {  
            gEnt = getGenericTypedArrayEntry(name, st);
        } catch e: Error {
            throw new owned ErrorWithContext("lookup for %s failed".format(name),
                               getLineNumber(),
                               getRoutineName(),
                               getModuleName(),
                               "UnknownSymbolError");    
        }

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                /* var eMin:int = min reduce e.a; */
                /* var eMax:int = max reduce e.a; */

                /* // how many bins in histogram */
                /* var bins = eMax-eMin+1; */
                /* umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins = %t".format(bins));*/

                /* if (bins <= mBins) { */
                /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins <= %t".format(mBins));*/
                /*     var (aV,aC) = uniquePerLocHistGlobHist(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */
                /* else if (bins <= lBins) { */
                /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins <= %t".format(lBins));*/
                /*     var (aV,aC) = uniquePerLocAssocGlobHist(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */
                /* else { */
                /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins = %t".format(bins));*/
                /*     var (aV,aC) = uniquePerLocAssocGlobAssoc(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */

                var (aV,aC) = uniqueSort(e.a);
                st.addEntry(vname, new shared SymEntry(aV));
                st.addEntry(cname, new shared SymEntry(aC));
            }
            otherwise {
                var errorMsg = notImplementedError(pn,gEnt.dtype);
                umLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);                 
            }
        }
        repMsg = "created " + st.attrib(vname) + " +created " + st.attrib(cname);
        umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc uniqueMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var (returnGroupsStr, nstr, rest) = payload.splitMsgToTuple(3);
        // flag to return segments and permutation for GroupBy
        const returnGroups = if (returnCountsStr == "True") then true else false;
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
        var perm: [kr.domain] int;
        var sortedHashes: [kr.domain] 2*(uint64);
        forall (sh, p, val) in zip(sortedHashes, perm, kr) {
          (sh, p) = val;
        }
        // Get the unique hashes and the count of each
        var (uniqueHashes, counts) = uniqueFromSorted(sortedHashes);
        // Compute offset of each group in sorted array
        var segments = (+ scan counts) - counts;
        // If returning grouping info, add to SymTab and prepend to repMsg
        if returnGroups {
          var pname = st.nextName();
          st.addEntry(pname, perm);
          repMsg += "created " + st.attrib(pname);
          var sname = st.nextName();
          st.addEntry(sname, segments);
          repMsg += "+created " + st.attrib(sname);
        }
        // Indices of first unique key in original array
        // These are the value of the permutation at the start of each group
        var gatherInds: [segments.domain] int;
        forall (g, s) in zip(gatherInds, segments) with (var agg = newSrcAggregator(int)) {
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
          when ("pdarray", "category") {
            var g = getGenericTypedArrayEntry(name, st);
            // Gathers unique values, stores in SymTab, and returns repMsg chunk
            proc gatherHelper(type t) {
              var e = toSymEntry(g, t);
              ref ea = e.a;
              var unique = st.addEntry(newName, size, t);
              forall (u, i) in zip(unique.a, gatherInds) with (var agg = newSrcAggregator(t)) {
                agg.copy(u, ea[i]);
              }
              return "+created " + st.attrib(newName);
            }
            select g.dtype {
              when DType.Int64 {
                repMsg += gatherHelper(int);
              }
              when DType.UInt64 {
                repMsg += gatherHelper(uint);
              }
              when DType.Float64 {
                repMsg += gatherHelper(real);
              }
              when DType.Bool {
                repMsg += gatherHelper(bool);
              }
            }
          }
          when "str" {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = getSegStringEntry(myNames, st);
            var (uSegs, uVals) = g[gatherInds];
            var newStringsObj = getSegString(uSegs, uVals, st);
            repMsg += "+created " + st.attrib(newStringsObj.name) + "+created bytes.size %t".format(newStringsObj.nBytes);
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
      var (size, hasStr) = validateArraysSameLength(n, fields);
      overMemLimit(numBytes(uint) * g.size * 2);
      var dom = makeDistDom(size);
      var hashes: [dom] 2*uint(64);
      /* Hashes of subsequent arrays cannot be simply XORed
       * because equivalent values will cancel each other out.
       * Thus, a non-linear function must be applied to each array,
       * hence we do a rotation by the ordinal of the array. This
       * will only handle up to 128 arrays before rolling back around.
       */
      proc rotl(h:2*uint(64), n:int):2*uint(64) {
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
      const low = fields.domain.low;
      var names = fields[low..#n];
      var types = fields[low+n..#n];
      for (name, objtype, i) in zip(names, types, 0..) {
        select objtype {
          when ("pdarray", "category") {
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
            var g = getSegStringEntry(myNames, st);
            hashes ^= rotl(g.hash(), i);
          }
        }
      }
      return hashes;
    }

    proc registerMe() {
      use CommandMap;
      registerFunction("unique", uniqueMsg, getModuleName());
      registerFunction("value_counts", value_countsMsg, getModuleName());
    }
}
