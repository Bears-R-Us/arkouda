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
    use CTypes;
    use Set;
    use List;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const cmLogger = new Logger(logLevel, logChannel);

    use CommandMap;

    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html#array_api.concat
    /* Concatenate a list of arrays together
       to form one array
    */
    @arkouda.instantiateAndRegister(prefix='concatenate')
    proc concatenateMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const names = msgArgs["names"].toScalarList(string),
              nArrays = names.size,
              axis = msgArgs["axis"].getPositiveIntValue(array_nd),
              offsets = msgArgs["offsets"].toScalarArray(int, nArrays);

        const arrNames = names.toArray();

        var dtype: DType;
        dtype = getGenericTypedArrayEntry(arrNames[0], st).dtype;
        var same: bool = true;
        for n in arrNames {
          if getGenericTypedArrayEntry(n, st).dtype != dtype {
            same = false;
          }
        }

        if !same {
          const errMsg = "All arrays must have the same data type.";
          cmLogger.error(getModuleName(), pn, getLineNumber(), errMsg);
          return MsgTuple.error(errMsg);
        }

        // Retrieve the arrays from the symbol table
        const eIns = for n in arrNames do st[n]: borrowed SymEntry(array_dtype, array_nd),
              shapes = [i in 0..<nArrays] eIns[i].tupShape,
              (valid, shapeOut) = concatenatedShape(shapes, axis, array_nd);
        var eOut = createSymEntry((...shapeOut), array_dtype);

        if !valid {
            const errMsg = "All arrays must have the same shape except in the concatenation axis.";
            cmLogger.error(getModuleName(), pn, getLineNumber(), errMsg);
            return MsgTuple.error(errMsg);
        } else {
            for (arrIdx, arr) in zip(eIns.domain, eIns) {
                const offset = offsets[arrIdx];
                const arrDomain = arr.a.domain;
                var translation: array_nd * int;
                translation[axis] = offset;
                const domainTranslated = arrDomain.translate(translation);
                eOut.a[domainTranslated] = arr.a;
            }
            return st.insert(eOut);
        }
    }

    // Function to validate shapes and determine output shape
    private proc concatenatedShape(shapes: [?d] ?t, axis: int, param N: int): (bool, N*int)
        where isHomogeneousTuple(t)
    {
        var shapeOut: N*int,
            firstShape = shapes[0];

        // Ensure all shapes match except for the concatenation axis
        for i in 1..d.last {
            for param j in 0..<N {
                if j != axis && shapes[i][j] != firstShape[j] {
                    return (false, shapeOut);
                }
            }
        }

        // Compute output shape
        shapeOut = firstShape;
        shapeOut[axis] = + reduce [i in 0..<d.size] shapes[i][axis]; // Sum sizes along axis

        return (true, shapeOut);
    }

    /* Concatenate a list of arrays together
       to form one array
     */
    proc concatenateStrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab) : MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
        var mode = msgArgs.getValueOf("mode");
        var names = msgArgs.get("names").toScalarList(string);
        var n = names.size;
        
        cmLogger.debug(getModuleName(),getRoutineName(), getLineNumber(), 
              "number of arrays: %i names: %?".format(n,names));

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

            st.checkTable(name, "concatenateStrMsg");
            var abstractEntry = st[name];
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
    registerFunction("concatenateStr", concatenateStrMsg, getModuleName());

    proc concatenateUniqueStrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        var repMsg: string;
        var names = msgArgs.get("names").toScalarList(string);
        var n = names.size;


        cmLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                    "concatenate unique strings from %i arrays: %?".format(n, names));

        // Each locale gets its own set
        var localeSets = makeDistArray(numLocales, set(string));

        // Initialize sets
        coforall loc in Locales do on loc {
            localeSets[here.id] = new set(string);
        }

        // Collect all unique strings from each input SegmentedString
        for rawName in names {
            var (strName, _) = rawName.splitMsgToTuple('+', 2);
            try {
                var segString = getSegString(strName, st);
                cmLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                            "Processing SegString: %s".format(strName));

                // Grab the strings by locale and throw them in the sets.
                coforall loc in Locales do on loc {
                    ref globalOffsets = segString.offsets.a;
                    const offsetsDom = segString.offsets.a.domain.localSubdomain();
                    const offsets = segString.offsets.a.localSlice[offsetsDom];
                    const topEnd = if offsetsDom.high >= segString.offsets.a.domain.high then segString.values.a.size else globalOffsets[offsetsDom.high + 1];
                    var locSet = new set(string);
                    forall idx in offsetsDom with (+ reduce locSet) {
                        const start = offsets[idx];
                        const end = if idx == offsetsDom.high then topEnd else offsets[idx + 1];
                        var str = interpretAsString(segString.values.a, start..<end);
                        locSet.add(str);
                    }
                    localeSets[here.id] |= locSet;
                }

            } catch e: Error {
                throw getErrorWithContext(
                    msg="lookup for %s failed".format(rawName),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="UnknownSymbolError");
            }
        }

        // These variables will help us create the buffers.
        var numStringsReceived = makeDistArray(numLocales, [0..#numLocales] int);
        var numBytesReceived = makeDistArray(numLocales, [0..#numLocales] int);
        var numStringsSending = makeDistArray(numLocales, [0..#numLocales] int);
        var numBytesSending = makeDistArray(numLocales, [0..#numLocales] int);

        var maxStringsPerPeer: int = 0;
        var maxBytesPerPeer: int = 0;

        var destLocaleByStrIdx: [PrivateSpace] list(int);
        var strOffsetInLocale: [PrivateSpace] list(int);
        var strIdxInLocale: [PrivateSpace] list(int);

        // Hash all the strings and figure out how many strings are going to each locale, and how many bytes that takes
        coforall loc in Locales with (max reduce maxStringsPerPeer, max reduce maxBytesPerPeer) do on loc {
            const strArray = localeSets[here.id].toArray();
            var destLoc: [0..#strArray.size] int;
            var strSize: [0..#strArray.size] int;
            var strOffsets: [0..#strArray.size] int;
            var strIndices: [0..#strArray.size] int;

            var sendingStrings: [0..#numLocales] int;
            var sendingBytes: [0..#numLocales] int;
            forall strIdx in strArray.domain with (+ reduce sendingStrings, + reduce sendingBytes) {
                const str = strArray[strIdx];
                const targetID: int = (str.hash() % numLocales): int;
                const strBytes = str.bytes();
                destLoc[strIdx] = targetID;
                strSize[strIdx] = strBytes.size + 1;
                sendingStrings[targetID] += 1;
                sendingBytes[targetID] += strBytes.size + 1;
            }
            destLocaleByStrIdx[here.id] = new list(destLoc);
            forall i in 0..#numLocales {
                numStringsReceived[i][here.id] = sendingStrings[i];
                numBytesReceived[i][here.id] = sendingBytes[i];
                numStringsSending[here.id][i] = sendingStrings[i];
                numBytesSending[here.id][i] = sendingBytes[i];
            }

            for i in 0..#numLocales {
                var currStrSize = forall j in strSize.domain do (if destLoc[j] == i then strSize[j] else 0);
                var offsets = (+ scan currStrSize) - currStrSize;
                var currLocIndices = forall j in strSize.domain do (if destLoc[j] == i then 1 else 0);
                var indices = (+ scan currLocIndices) - currLocIndices;
                forall j in strIndices.domain {
                    if destLoc[j] == i {
                        strOffsets[j] = offsets[j];
                        strIndices[j] = indices[j];
                    }
                }
            }
            strOffsetInLocale[here.id] = new list(strOffsets);
            strIdxInLocale[here.id] = new list(strIndices);

            maxStringsPerPeer = max reduce sendingStrings;
            maxBytesPerPeer = max reduce sendingBytes;
        }

        // This may be a little wasteful with the buffers but I couldn't think of a better option.
        var sendOffsets = makeDistArray(numLocales, [0..#numLocales] [0..#(maxStringsPerPeer+1)] int);
        var recvOffsets = makeDistArray(numLocales, [0..#numLocales] [0..#(maxStringsPerPeer+1)] int);
        var sendBytes = makeDistArray(numLocales, [0..#numLocales] [0..#(maxBytesPerPeer)] uint(8));
        var recvBytes = makeDistArray(numLocales, [0..#numLocales] [0..#(maxBytesPerPeer)] uint(8));

        // In particular, I'm hoping that by setting up these buffers as uint(8) type
        // that the domain mapping will ship off a bunch of data at the same time and cut down on puts and gets.

        // Now we prep the data in the send variables and then place it on the destination locale in the recv buffer.

        coforall loc in Locales do on loc {
            var mySendOffsets = sendOffsets[here.id];
            var mySendBytes = sendBytes[here.id];

            const strArray = localeSets[here.id].toArray();

            const strOffsets = strOffsetInLocale[here.id].toArray();
            const destLoc = destLocaleByStrIdx[here.id].toArray();
            const idxInLocale = strIdxInLocale[here.id].toArray();

            forall i in strArray.domain {
                const str = strArray[i];
                const targetID = destLoc[i];
                const offset = strOffsets[i];
                const locInd = idxInLocale[i];

                const strBytes = str.bytes();
                const strSize = strBytes.size;
                mySendOffsets[targetID][locInd] = offset;
                mySendBytes[targetID][offset..#strSize] = strBytes[0..#strSize];
                mySendBytes[targetID][offset + strSize] = 0;
            }

            forall i in 0..#numLocales {
                mySendOffsets[i][numStringsSending[here.id][i]] = numBytesSending[here.id][i];
                recvOffsets[i][here.id][0..#(numStringsSending[here.id][i]+1)] = mySendOffsets[i][0..#numStringsSending[here.id][i]];
                recvBytes[i][here.id][0..#numBytesSending[here.id][i]] = mySendBytes[i][0..#numBytesSending[here.id][i]];
            }
        }

        // I need variables to store how many strings each locale has
        // how many bytes each locale has
        // It'd be nice to store the set as an array on each Locale so I only have to convert once.
        // Then I can convert the string array to uint(8) on each locale
        // Use some global ref to how many bytes per and do some kinda + scan numBytes thing
        // And I think I'm done. I don't care about rebalancing, the bytes array is already distributed.
        // So ship data via domains and I don't think that can really be beat with the aggregators.
        // Certainly not beat by one byte at a time.
        // This also beats the calculation of which locale to send data to. Who cares? Just send it.
        // offsets by locale does not line up with bytes by locale. So it doesn't matter.

        // Intentionally not distributed because I'm going to need these available to every locale.

        var numStringsPerLocale: [0..#numLocales] int;
        var numBytesPerLocale: [0..#numLocales] int;

        coforall loc in Locales with(ref numStringsPerLocale, ref numBytesPerLocale) do on loc {
            var strSet = new set(string);

            for i in 0..#numLocales {
                forall j in 1..#numStringsReceived[here.id][i] with (+ reduce strSet) {
                    const start = recvOffsets[here.id][i][j - 1];
                    const end = recvOffsets[here.id][i][j];
                    var str = interpretAsString(recvBytes[here.id][i], start..<end);
                    strSet.add(str);
                }
            }

            const strArray = strSet.toArray();
            numStringsPerLocale[here.id] = strArray.size;
            var size = 0;

            forall str in strArray with (+ reduce size) {
                size += str.size + 1;
            }

            numBytesPerLocale[here.id] = size;

            localeSets[here.id] = strSet;
        }

        // I need to set up the segString here.
        // I need to create the running sum of strings per locale
        var stringOffsetByLocale = + scan numStringsPerLocale;
        var bytesOffsetByLocale = + scan numBytesPerLocale;
        // I need to create the running sum of bytes per locale.

        var esegs = createTypedSymEntry(stringOffsetByLocale.last, int);
        var evals = createTypedSymEntry(bytesOffsetByLocale.last, uint(8));
        ref esa = esegs.a;
        ref eva = evals.a;

        stringOffsetByLocale -= numStringsPerLocale;
        bytesOffsetByLocale -= numBytesPerLocale;

        // Let's allocate a new SegString for the return object
        var retString = assembleSegStringFromParts(esegs, evals, st);

        coforall loc in Locales do on loc {

            const strArray = localeSets[here.id].toArray();
            var myBytes: [0..#numBytesPerLocale[here.id]] uint(8);
            var byteIdx = 0;
            var strOffsets: [0..#strArray.size] int;

            // This could probably be turned into a forall, using some sort of + scan on a strBytesSizes array of some kind.

            for (i, str) in zip(0..#strArray.size, strArray) {

                strOffsets[i] = byteIdx;

                const strBytes = str.bytes();
                const size = strBytes.size;

                myBytes[byteIdx..#size] = strBytes[0..#size];
                byteIdx += size + 1;

            }

            esa[stringOffsetByLocale[here.id]..#strArray.size] = strOffsets[0..#strArray.size] + bytesOffsetByLocale[here.id];
            eva[bytesOffsetByLocale[here.id]..#numBytesPerLocale[here.id]] = myBytes[0..#numBytesPerLocale[here.id]];

        }

        // Store the result in the symbol table and return
        repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %?".format(retString.nBytes);

        cmLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                    "Created unique concatenated SegmentedString: %s".format(st.attrib(retString.name)));

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    registerFunction("concatenateUniquely", concatenateUniqueStrMsg, getModuleName());
}
