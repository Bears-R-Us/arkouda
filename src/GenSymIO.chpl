 module GenSymIO {
    use IO;
    use Path;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use FileSystem;
    use FileIO;
    use Sort;
    use NumPyDType;
    use List;
    use Set;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use ServerConfig;
    use SegmentedString;
    use Map;
    use CTypes;
    use CommAggregation;
    use IOUtils;
    use BigInteger;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const gsLogger = new Logger(logLevel, logChannel);
    config const NULL_STRINGS_VALUE = 0:uint(8);


    //  prototype deep copy function

    @arkouda.registerCommand()
    proc deepcopy(x : [?d] ?t) : [d] t throws
        where (t==int || t==real || t==uint || t==bigint || t==bool) {
        var retVal = makeDistArray(d, t);
        if numLocales == 1 {
            forall (x_, r_) in zip (x, retVal) { r_ = x_ ; }
            return retVal;
        } else {
            coforall loc in Locales do on loc {
                var xLD = x.localSubdomain() ;
                retVal[xLD] = x[xLD];
            }
        }
        return retVal;
    }

    /*
     * Creates a pdarray server-side and returns the SymTab name used to
     * retrieve the pdarray from the SymTab.
    */
    @arkouda.instantiateAndRegister
    proc array(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
        where array_dtype != bigint
    {
        const shape = msgArgs["shape"].toScalarTuple(int, array_nd);

        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "dtype: %? shape: %?".format(array_dtype:string,shape));

        return st.insert(new shared SymEntry(makeArrayFromBytes(msgArgs.payload, shape, array_dtype)));
    }

    proc makeArrayFromBytes(ref payload: bytes, shape: ?N*int, type t): [] t throws {
        var size = 1;
        for s in shape do size *= s;
        overMemLimit(2*size*typeSize(t));

        var ret = makeDistArray((...shape), t),
            localA = makeArrayFromPtr(payload.c_str():c_ptr(void):c_ptr(t), num_elts=size:uint);
        if N == 1 {
            ret = localA;
        } else {
            forall (i, a) in zip(localA.domain, localA) with (var agg = newDstAggregator(t)) do
                agg.copy(ret[ret.domain.orderToIndex(i)], a);
        }

        return ret;
    }

    @arkouda.instantiateAndRegister()
    proc arraySegString(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws {
        const size = msgArgs["size"].toScalar(int),
              rname = st.nextName();

        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "dtype: %? size: %?".format(array_dtype:string,size));

        const a = makeArrayFromBytes(msgArgs.payload, (size,), array_dtype);
        st.addEntry(rname, createSymEntry(a));
        
        try {
            st.checkTable(rname, "arrayMsg");
            var g = st[rname];
            if g.isAssignableTo(SymbolEntryType.TypedArraySymEntry){
                var values = toSymEntry( (g:GenSymEntry), uint(8) );
                var offsets = segmentedCalcOffsets(values.a, values.a.domain);
                var oname = st.nextName();
                var offsetsEntry = createSymEntry(offsets);
                st.addEntry(oname, offsetsEntry);
                const msg = "created " + st.attrib(oname) + "+created " + st.attrib(rname);
                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),msg);
                return new MsgTuple(msg, MsgType.NORMAL);
            } else {
                throw new Error("Unsupported Type %s".format(g.entryType));
            }
        } catch e: Error {
            const msg = "Error creating offsets for SegString";
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),msg);
            return new MsgTuple(msg, MsgType.ERROR);
        }
    }

    /**
     * For creating the Strings/SegString object we can calculate the offsets array on the server
     * by finding the null terminators given the values/bytes array which should have already been
     * converted to uint8
     */
    proc segmentedCalcOffsets(values:[] uint(8), valuesDom): [] int throws {
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Calculating offsets for SegString");
        var nb_locs = forall (i,v) in zip(valuesDom, values) do if v == NULL_STRINGS_VALUE then i+1;
        // We need to adjust nb_locs b/c offsets is really the starting position of each string
        // So allocated a new array of zeros and assign nb_locs offset by one
        var offsets = makeDistArray(nb_locs.domain, int);
        offsets[1..offsets.domain.high] = nb_locs[0..#nb_locs.domain.high];
        return offsets;
    }

    /*
     * Outputs the pdarray as a Numpy ndarray in the form of a 
     * Chapel Bytes object
     */
    @arkouda.instantiateAndRegister
    proc tondarray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
        where array_dtype != bigint
    {
        const array = st[msgArgs["array"]]: borrowed SymEntry(array_dtype, array_nd);
        
        overMemLimit(2 * array.size * typeSize(array_dtype));

        var ptr = allocate(array_dtype, array.size);
        var localA = makeArrayFromPtr(ptr, array.size:uint);
        if array_nd == 1 {
            localA = array.a;
        } else {
            forall (i, a) in zip(localA.domain, localA) with (var agg = newSrcAggregator(array_dtype)) do
                agg.copy(localA[i], array.a[array.a.domain.orderToIndex(i)]);
        }
        const size = array.size*c_sizeof(array_dtype):int;
        return MsgTuple.payload(bytes.createAdoptingBuffer(ptr:c_ptr(uint(8)), size, size));
    }

    /*
     * Utility proc to test casting a string to a specified type
     * :arg c: String to cast
     * :type c: string
     * 
     * :arg toType: the type to cast into
     * :type toType: type
     *
     * :returns: bool true if the cast was successful, false otherwise
     */
    proc checkCast(c:string, type toType): bool {
        try {
            var x:toType = c:toType;
            return true;
        } catch {
            return false;
        }
    }

    proc buildReadAllMsgJson(rnames:list((string, ObjType, string)), allowErrors:bool, fileErrorCount:int, fileErrors:list(string), st: borrowed SymTab): string throws {
        var items: list(map(string, string));

        for rname in rnames {
            var (dsetName, akType, id) = rname;
            var item: map(string, string) = new map(string, string);
            item.add("dataset_name", dsetName.replace(Q, ESCAPED_QUOTES, -1));
            item.add("arkouda_type", akType: string);
            var create_str: string;
            select akType {
                when ObjType.ARRAYVIEW {
                    var (valName, segName) = id.splitMsgToTuple("+", 2);
                create_str = "created " + st.attrib(valName) + "+created " + st.attrib(segName);
                }
                when ObjType.PDARRAY, ObjType.IPV4, ObjType.DATETIME, ObjType.TIMEDELTA {
                    create_str = "created " + st.attrib(id);
                }
                when ObjType.STRINGS {
                    var (segName, nBytes) = id.splitMsgToTuple("+", 2);
                    create_str = "created " + st.attrib(segName) + "+created bytes.size " + nBytes;
                }
                when ObjType.SEGARRAY, ObjType.CATEGORICAL, ObjType.GROUPBY, ObjType.DATAFRAME, ObjType.INDEX, ObjType.MULTIINDEX {
                    create_str = id;
                }
                otherwise {
                    continue;
                }
            }
            item.add("created", create_str);
            items.pushBack(item);
        }
        
        var reply: map(string, string) = new map(string, string);
        reply.add("items", formatJson(items));
        if allowErrors && !fileErrors.isEmpty() { // If configured, build the allowErrors portion
            reply.add("allow_errors", "true");
            reply.add("file_error_count", fileErrorCount:string);
            reply.add("file_errors", formatJson(fileErrors));
        }
        return formatJson(reply);
    }

    /*
     * Simple JSON parser to allow creating a map(string, string) for properly formatted JSON string.
     * REQUIRES THAT DATA DOES NOT CONTAIN : or ". This will only work on JSON that is not nested.
    */
    proc jsonToMap(json: string): map(string, string) throws {
        // remove components not needed for parsing
        var clean_json = json.strip().replace("\"", "").replace("{", "").replace("}", ""); // syntax highlight messed up by \".

        // generate the return map
        var m: map(string, string) = new map(string, string);

        //get each key value pair
        var key_value = clean_json.split(", ");
        for kv in key_value  {
            // split to 2 components key: value
            var x = kv.split(": ");
            m.addOrReplace(x[0], x[1]);
        }
        return m;
    }

}
