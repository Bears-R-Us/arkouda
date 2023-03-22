module GenSymIO {
    use IO;
    use CTypes;
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

    use ArkoudaMapCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const gsLogger = new Logger(logLevel, logChannel);
    config const NULL_STRINGS_VALUE = 0:uint(8);

    /*
     * Creates a pdarray server-side and returns the SymTab name used to
     * retrieve the pdarray from the SymTab.
     */
    proc arrayMsg(cmd: string, msgArgs: borrowed MessageArgs, ref data: bytes, st: borrowed SymTab): MsgTuple throws {
        // Set up our return items
        var msgType = MsgType.NORMAL;
        var msg:string = "";
        var rname:string = "";
        var dtype = DType.UNDEF;
        var size:int;
        var asSegStr = false;
        
        try {
            dtype = str2dtype(msgArgs.getValueOf("dtype"));
            size = msgArgs.get("size").getIntValue();
            asSegStr = msgArgs.get("seg_string").getBoolValue();
        } catch {
            var errorMsg = "Error parsing/decoding either dtypeBytes or size";
            gsLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        overMemLimit(2*size);

        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "dtype: %t size: %i".format(dtype,size));

        proc bytesToSymEntry(size:int, type t, st: borrowed SymTab, ref data:bytes): string throws {
            var entry = new shared SymEntry(size, t);
            var localA = makeArrayFromPtr(data.c_str():c_void_ptr:c_ptr(t), size:uint);
            entry.a = localA;
            var name = st.nextName();
            st.addEntry(name, entry);
            return name;
        }

        if dtype == DType.Int64 {
            rname = bytesToSymEntry(size, int, st, data);
        } else if dtype == DType.UInt64 {
            rname = bytesToSymEntry(size, uint, st, data);
        } else if dtype == DType.Float64 {
            rname = bytesToSymEntry(size, real, st, data);
        } else if dtype == DType.Bool {
            rname = bytesToSymEntry(size, bool, st, data);
        } else if dtype == DType.UInt8 {
            rname = bytesToSymEntry(size, uint(8), st, data);
        } else {
            msg = "Unhandled data type %s".format(msgArgs.getValueOf("dtype"));
            msgType = MsgType.ERROR;
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),msg);
        }

        if asSegStr {
            try {
                st.checkTable(rname, "arrayMsg");
                var g = st.lookup(rname);
                if g.isAssignableTo(SymbolEntryType.TypedArraySymEntry){
                    var values = toSymEntry( (g:GenSymEntry), uint(8) );
                    var offsets = segmentedCalcOffsets(values.a, values.a.domain);
                    var oname = st.nextName();
                    var offsetsEntry = new shared SymEntry(offsets);
                    st.addEntry(oname, offsetsEntry);
                    msg = "created " + st.attrib(oname) + "+created " + st.attrib(rname);
                } else {
                    throw new Error("Unsupported Type %s".format(g.entryType));
                }

            } catch e: Error {
                msg = "Error creating offsets for SegString";
                msgType = MsgType.ERROR;
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),msg);
            }
        }

        if (MsgType.ERROR != msgType) {  // success condition
            // Set up return message indicating SymTab name corresponding to new pdarray
            // If we made a SegString or we encountered an error, the msg will already be populated
            if (msg.isEmpty()) {
                msg = "created " + st.attrib(rname);
            }
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),msg);
        }
        return new MsgTuple(msg, msgType);
    }

    /**
     * For creating the Strings/SegString object we can calculate the offsets array on the server
     * by finding the null terminators given the values/bytes array which should have already been
     * converted to uint8
     */
    proc segmentedCalcOffsets(values:[] uint(8), valuesDom:domain): [] int throws {
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Calculating offsets for SegString");
        var nb_locs = forall (i,v) in zip(valuesDom, values) do if v == NULL_STRINGS_VALUE then i+1;
        // We need to adjust nb_locs b/c offsets is really the starting position of each string
        // So allocated a new array of zeros and assign nb_locs offset by one
        var offsets: [nb_locs.domain] int;
        offsets[1..offsets.domain.high] = nb_locs[0..#nb_locs.domain.high];
        return offsets;
    }

    /*
     * Outputs the pdarray as a Numpy ndarray in the form of a 
     * Chapel Bytes object
     */
    proc tondarrayMsg(cmd: string, msgArgs: borrowed MessageArgs, st: 
                                          borrowed SymTab): bytes throws {
        var arrayBytes: bytes;
        var abstractEntry = st.lookup(msgArgs.getValueOf("array"));
        if !abstractEntry.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
            var errorMsg = "Error: Unhandled SymbolEntryType %s".format(abstractEntry.entryType);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return errorMsg.encode(); // return as bytes
        }
        var entry:borrowed GenSymEntry = abstractEntry: borrowed GenSymEntry;
        
        overMemLimit(2 * entry.getSizeEstimate());

        proc distArrToBytes(A: [?D] ?eltType) {
            var ptr = c_malloc(eltType, D.size);
            var localA = makeArrayFromPtr(ptr, D.size:uint);
            localA = A;
            const size = D.size*c_sizeof(eltType):int;
            return createBytesWithOwnedBuffer(ptr:c_ptr(uint(8)), size, size);
        }

        if entry.dtype == DType.Int64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, int).a);
        } else if entry.dtype == DType.UInt64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, uint).a);
        } else if entry.dtype == DType.Float64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, real).a);
        } else if entry.dtype == DType.Bool {
            arrayBytes = distArrToBytes(toSymEntry(entry, bool).a);
        } else if entry.dtype == DType.UInt8 {
            arrayBytes = distArrToBytes(toSymEntry(entry, uint(8)).a);
        } else {
            var errorMsg = "Error: Unhandled dtype %s".format(entry.dtype);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return errorMsg.encode(); // return as bytes
        }

       return arrayBytes;
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

    proc _buildReadAllMsgJson(rnames:list(3*string), allowErrors:bool, fileErrorCount:int, fileErrors:list(string), st: borrowed SymTab): string throws {
        var items: list(map(string, string));

        for rname in rnames {
            var (dsetName, akType, id) = rname;
            var item: map(string, string) = new map(string, string);
            item.add("dataset_name", dsetName.replace(Q, ESCAPED_QUOTES, -1));
            item.add("arkouda_type", akType);
            var create_str: string;
            if (akType == "ArrayView") {
                var (valName, segName) = id.splitMsgToTuple("+", 2);
                create_str = "created " + st.attrib(valName) + "+created " + st.attrib(segName);
            }
            else if (akType == "pdarray") {
                create_str = "created " + st.attrib(id);
            }
            else if (akType == "seg_string") {
                var (segName, nBytes) = id.splitMsgToTuple("+", 2);
                create_str = "created " + st.attrib(segName) + "+created bytes.size " + nBytes;
            }
            else if (akType == "seg_array") {
                create_str = id;

            } 
            else {
                continue;
            }
            item.add("created", create_str);
            items.append(item);
        }
        
        var reply: map(string, string) = new map(string, string);
        reply.add("items", "%jt".format(items));
        if allowErrors && !fileErrors.isEmpty() { // If configured, build the allowErrors portion
            reply.add("allow_errors", "true");
            reply.add("file_error_count", fileErrorCount:string);
            reply.add("file_errors", "%jt".format(fileErrors));
        }
        return "%jt".format(reply);
    }

}
