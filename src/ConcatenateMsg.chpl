module ConcatenateMsg
{
    use ServerConfig;

    use Time;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use BigInteger;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;
    use CommAggregation;
    use PrivateDist;
    
    use AryUtil;
    
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const cmLogger = new Logger(logLevel, logChannel);

    /* Concatenate a list of arrays together
       to form one array
     */
    proc concatenateMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab) : MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
        var mode = msgArgs.getValueOf("mode");
        var n = msgArgs.get("nstr").getIntValue(); // number of arrays to sort
        var names = msgArgs.get("names").getList(n);
        
        cmLogger.debug(getModuleName(),getRoutineName(), getLineNumber(), 
              "number of arrays: %i names: %?".format(n,names));

        // Check that fields contains the stated number of arrays
        if (n != names.size) { 
            var errorMsg = incompatibleArgumentsError(pn, 
                             "Expected %i arrays but got %i".format(n, names.size)); 
            cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                               
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        /* var arrays: [0..#n] borrowed GenSymEntry; */
        var size: int = 0;
        var blocksizes: [PrivateSpace] int;
        var blockValSizes: [PrivateSpace] int;
        var nbytes: int = 0;          
        var dtype: DType;
        // Check that all arrays exist in the symbol table and have the same size
        for (name, i) in zip(names, 1..) {
            var valSize: int;
            select objtype {
                when ObjType.STRINGS {
                    try {
                        // get the values/bytes portion of strings
                        var segString = getSegString(name, st);
                        nbytes += segString.nBytes;
                        valSize = segString.nBytes;
                    } catch e: Error {
                        throw getErrorWithContext(
                           msg="lookup for %s failed".format(name),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="UnknownSymbolError");                    
                    }
                    cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "name: %s".format(name));
                }
                when ObjType.PDARRAY {
                    cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                 "pdarray name %s".format(name));
                }
                otherwise { 
                    var errorMsg = notImplementedError(pn, objtype: string); 
                    cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
                    return new MsgTuple(errorMsg,MsgType.ERROR);                  
                }
            }

            st.checkTable(name, "concatenateMsg");
            var abstractEntry = st.lookup(name);
            var (entryDtype, entrySize, entryItemSize) = getArraySpecFromEntry(abstractEntry);
            
            if (i == 1) {
              dtype = entryDtype;
            } else { // Check that all dtype's are the same across the list of arrays to concat
                if (dtype != entryDtype) {
                    var errorMsg = incompatibleArgumentsError(pn, 
                             "Expected %s dtype but got %s dtype".format(dtype2str(dtype), 
                                    dtype2str(entryDtype)));
                    cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg,MsgType.ERROR);
                }
            }
            // accumulate size from each array size
            size += entrySize;
            if mode == "interleave" {
              const dummyDomain = makeDistDom(entrySize);
              coforall loc in Locales with (ref blocksizes, ref blockValSizes) {
                on loc {
                  const mynumsegs = dummyDomain.localSubdomain().size;
                  blocksizes[here.id] += mynumsegs;
                  /* If the size of the array is less than the number of locales,
                   * some locales will have no segments. For those locales, skip
                   * the byte size computation because, not only is it unnecessary,
                   * but it also causes an out of bounds array index due to the 
                   * way the low and high of the empty domain are computed. 
                   */
                  if (objtype == ObjType.STRINGS) && (mynumsegs > 0) {
                    const stringEntry = toSegStringSymEntry(abstractEntry);
                    const e = stringEntry.offsetsEntry;
                    const firstSeg = e.a[e.a.domain.localSubdomain().low];
                    var mybytes: int;
                    /* If this locale contains the last segment, we cannot use the
                     * next segment offset to calculate the number of bytes for this
                     * locale, and we must instead use the total size of the values
                     * array.
                     */
                    if (e.a.domain.localSubdomain().high >= e.a.domain.high) {
                      mybytes = valSize - firstSeg;
                    } else {
                      mybytes = e.a[e.a.domain.localSubdomain().high + 1] - firstSeg;
                    }
                    blockValSizes[here.id] += mybytes;
                  }
                }
              }
            }
        }
        var blockstarts: [PrivateSpace] int;
        var blockValStarts: [PrivateSpace] int;
        if mode == "interleave" {
          // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
          overMemLimit(numBytes(int) * blocksizes.size);
          blockstarts = (+ scan blocksizes) - blocksizes;

          // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
          overMemLimit(numBytes(int) * blockValSizes.size);
          blockValStarts = (+ scan blockValSizes) - blockValSizes;
        }

        // allocate a new array in the symboltable
        // and copy in arrays
        select objtype {
            when ObjType.STRINGS {
                // var segName = st.nextName();
                // var esegs = st.addEntry(segName, size, int);
                // var valName = st.nextName();
                // var evals = st.addEntry(valName, nbytes, uint(8));
                // Allocate the two components of a Segmented
                var esegs = createTypedSymEntry(size, int);
                var evals = createTypedSymEntry(nbytes, uint(8));
                ref esa = esegs.a;
                ref eva = evals.a;
                var segStart = 0;
                var valStart = 0;

                // Let's allocate a new SegString for the return object
                var retString = assembleSegStringFromParts(esegs, evals, st);
                for (rawName, i) in zip(names, 1..) {
                    var (strName, legacy_placerholder) = rawName.splitMsgToTuple('+', 2);
                    var segString = getSegString(strName, st);
                    var thisSegs = segString.offsets;
                    var newSegs = thisSegs.a + valStart;
                    var thisVals = segString.values;
                    if mode == "interleave" {
                      coforall loc in Locales with (ref blockstarts, ref blockValStarts) {
                        on loc {
                          // Number of strings on this locale for this input array
                          const mynsegs = thisSegs.a.domain.localSubdomain().size;
                          // If no strings on this locale, skip to avoid out of bounds array
                          // accesses
                          if mynsegs > 0 {
                            ref mysegs = thisSegs.a.localSlice[thisSegs.a.domain.localSubdomain()];
                            // Segments must be rebased to start from blockValStart,
                            // which is the current pointer to this locale's chunk of
                            // the values array
                            esegs.a[{blockstarts[here.id]..#mynsegs}] = mysegs - mysegs[thisSegs.a.domain.localSubdomain().low] + blockValStarts[here.id];
                            blockstarts[here.id] += mynsegs;
                            const firstSeg = thisSegs.a[thisSegs.a.domain.localSubdomain().low];
                            var mybytes: int;
                            // If locale contains last string, must use overall number of bytes
                            // to compute size, instead of start of next string
                            if (thisSegs.a.domain.localSubdomain().high >= thisSegs.a.domain.high) {
                              mybytes = thisVals.size - firstSeg;
                            } else {
                              mybytes = thisSegs.a[thisSegs.a.domain.localSubdomain().high + 1] - firstSeg;
                            }
                            evals.a[{blockValStarts[here.id]..#mybytes}] = thisVals.a[firstSeg..#mybytes];
                            blockValStarts[here.id] += mybytes;
                          }
                        }
                      }
                    } else {
                      forall (i, s) in zip(newSegs.domain, newSegs) with (var agg = newDstAggregator(int)) {
                        agg.copy(esa[i+segStart], s);
                      }
                      forall (i, v) in zip(thisVals.a.domain, thisVals.a) with (var agg = newDstAggregator(uint(8))) {
                        agg.copy(eva[i+valStart], v);
                      }
                      segStart += thisSegs.size;
                      valStart += thisVals.size;
                    }
                }
                var repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);
                // var repMsg = "created " + st.attrib(retString.name) + "+created " + st.attrib(retString.name);
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "created concatenated pdarray %s".format(st.attrib(retString.name)));
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when ObjType.PDARRAY {
                var rname = st.nextName();
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                             "creating pdarray %s of type %?".format(rname,dtype));
                select (dtype) {
                    when DType.Int64 {
                        // create array to copy into
                        var e = st.addEntry(rname, size, int);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(getGenericTypedArrayEntry(name, st), int);
                            if mode == "interleave" {
                              coforall loc in Locales with (ref blockstarts) {
                                on loc {
                                  const size = o.a.domain.localSubdomain().size;
                                  e.a[{blockstarts[here.id]..#size}] = o.a.localSlice[o.a.domain.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = e.a;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.a.domain, o.a) with (var agg = newDstAggregator(int)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.Float64 {
                        // create array to copy into
                        var e = st.addEntry(rname, size, real);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(getGenericTypedArrayEntry(name, st), real);
                            if mode == "interleave" {
                              coforall loc in Locales with (ref blockstarts) {
                                on loc {
                                  const size = o.a.domain.localSubdomain().size;
                                  e.a[{blockstarts[here.id]..#size}] = o.a.localSlice[o.a.domain.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = e.a;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.a.domain, o.a) with (var agg = newDstAggregator(real)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.Bool {
                        // create array to copy into
                        var e = st.addEntry(rname, size, bool);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(getGenericTypedArrayEntry(name, st), bool);
                            if mode == "interleave" {
                              coforall loc in Locales with (ref blockstarts) {
                                on loc {
                                  const size = o.a.domain.localSubdomain().size;
                                  e.a[{blockstarts[here.id]..#size}] = o.a.localSlice[o.a.domain.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = e.a;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.a.domain, o.a) with (var agg = newDstAggregator(bool)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.UInt64 {
                        // create array to copy into
                        var e = st.addEntry(rname, size, uint);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(getGenericTypedArrayEntry(name, st), uint);
                            if mode == "interleave" {
                              coforall loc in Locales with (ref blockstarts) {
                                on loc {
                                  const size = o.a.domain.localSubdomain().size;
                                  e.a[{blockstarts[here.id]..#size}] = o.a.localSlice[o.a.domain.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = e.a;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.a.domain, o.a) with (var agg = newDstAggregator(uint)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.BigInt {
                        // create array to copy into
                        var tmp = makeDistArray(size, bigint);
                        var start: int = 0;
                        var max_bits = -1;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(getGenericTypedArrayEntry(name, st), bigint);
                            max_bits = max(max_bits, o.max_bits);
                            if mode == "interleave" {
                              coforall loc in Locales with (ref blockstarts) {
                                on loc {
                                  const size = o.a.domain.localSubdomain().size;
                                  tmp[{blockstarts[here.id]..#size}] = o.a.localSlice[o.a.domain.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = tmp;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.a.domain, o.a) with (var agg = newDstAggregator(bigint)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        if max_bits != -1 {
                          var max_size = 1:bigint;
                          max_size <<= max_bits;
                          max_size -= 1;
                          forall t in tmp with (var local_max_size = max_size) {
                            t &= local_max_size;
                          }
                        }
                        st.addEntry(rname, createSymEntry(tmp, max_bits));
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    otherwise {
                        var errorMsg = notImplementedError("concatenate",dtype);
                        cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg,MsgType.ERROR);                      
                    }
                }

                repMsg = "created " + st.attrib(rname);
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            otherwise { 
                var errorMsg = notImplementedError(pn, objtype: string); 
                cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    use CommandMap;
    registerFunction("concatenate", concatenateMsg, getModuleName());
}
