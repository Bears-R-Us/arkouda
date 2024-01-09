
module MsgProcessing
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;

    use ArkoudaBigIntCompat;
    use ArkoudaTimeCompat as Time;
    use ArkoudaMathCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const mpLogger = new Logger(logLevel, logChannel);

    /* 
    Parse, execute, and respond to a create message 

    :arg : payload
    :type string: containing (dtype,size)

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple) response message
    */
    @arkouda.registerND(cmd_prefix="create")
    proc createMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        const dtype = str2dtype(msgArgs.getValueOf("dtype")),
              shape = msgArgs.get("shape").getTuple(nd),
              rname = st.nextName();

        var size = 1;
        for s in shape do size *= s;

        if (dtype == DType.UInt8) || (dtype == DType.Bool)
            then overMemLimit(size);
            else overMemLimit(8*size);

        // if verbose print action
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s size: %i new pdarray name: %s".doFormat(
                                                     cmd,dtype2str(dtype),size,rname));
        // create and add entry to symbol table
        st.addEntry(rname, (...shape), dtype);
        // if verbose print result
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                    "created the pdarray %s".doFormat(st.attrib(rname)));

        const repMsg = "created " + st.attrib(rname);
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    // used for "zero-dimensional" array api scalars
    proc createMsg0D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const dtype = str2dtype(msgArgs.getValueOf("dtype")),
              rname = st.nextName();

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "cmd: %s dtype: %s size: 1 new pdarray name: %s".doFormat(
                       cmd,dtype2str(dtype),rname));

        // on the client side, scalar (0D) arrays have a shape of "()" and a size of 1
        // here, we represent that using a 1D array with a shape of (1,) and a size of 1
        var e = toGenSymEntry(st.addEntry(rname, 1, dtype));
        e.size = 1;
        e.shape = "[]";
        e.ndim = 1; // this is 1 rather than 0 s.t. calls to other message handlers treat it as a 1D
                    // array (e.g., we should call 'set1D', not 'set0D' on this array)

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "created the pdarray %s".doFormat(st.attrib(rname)));

        var repMsg = "created " + st.attrib(rname);
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Parse, execute, and respond to a delete message 

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    */
    proc deleteMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("name");
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                     "cmd: %s array: %s".doFormat(cmd,st.attrib(name)));
        // delete entry from symbol table
        if st.deleteEntry(name) {
            repMsg = "deleted %s".doFormat(name);
        }
        else {
            repMsg = "registered symbol, %s, not deleted".doFormat(name);
        }
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);       
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Clear all unregistered symbols and associated data from sym table
    
    :arg reqMsg: request containing (cmd)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
     */
    proc clearMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "cmd: %s".doFormat(cmd));
        st.clear();

        repMsg = "success";
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /* 
    Takes the name of data referenced in a msg and searches for the name in the provided sym table.
    Returns a string of info for the sym entry that is mapped to the provided name.

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
     */
    proc infoMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("names");
 
        // if name == "__AllSymbols__" passes back info on all symbols       
        repMsg = st.info(name);
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    /* 
    query server configuration...
    
    :arg reqMsg: request containing (cmd)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
     */
    proc getconfigMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".doFormat(cmd));
        return new MsgTuple(getConfig(), MsgType.NORMAL);
    }

    /* 
        query server total memory allocated or symbol table data memory

        :arg reqMsg: request containing (cmd)
        :type reqMsg: string

        :arg st: SymTab to act on
        :type st: borrowed SymTab

        :returns: MsgTuple
     */
    proc getmemusedMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        var factor = msgArgs.get("factor").getIntValue();
        var asPercent = msgArgs.get("as_percent").getBoolValue();
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".doFormat(cmd));
        var memUsed = if memTrack then getMemUsed():real * numLocales else st.memUsed():real;
        if asPercent {
            repMsg = mathRound((memUsed / (getMemLimit():real * numLocales)) * 100):uint:string;
        }
        else {
            repMsg = mathRound(memUsed / factor):uint:string;
        }
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
        query server total memory availble

        :arg reqMsg: request containing (cmd)
        :type reqMsg: string

        :arg st: SymTab to act on
        :type st: borrowed SymTab

        :returns: MsgTuple
     */
    proc getmemavailMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        var factor = msgArgs.get("factor").getIntValue();
        var asPercent = msgArgs.get("as_percent").getBoolValue();
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".doFormat(cmd));
        var memUsed = if memTrack then getMemUsed():real * numLocales else st.memUsed():real;
        var totMem = getMemLimit():real * numLocales;
        if asPercent {
            repMsg = (100 - mathRound((memUsed / totMem) * 100)):uint:string;
        }
        else {
            repMsg = mathRound((totMem - memUsed) / factor):uint:string;
        }
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    /**
     * Generate the mapping of server command to function as JSON
     * encoded string.
     *
     * The args are IGNORED. They are only here to match the CommandMap
     * standard function signature, similar to other procs.
     *
     * :arg cmd: Ignored
     * :type cmd: string 
     *
     * :arg payload: Ignored
     * :type payload: string
     *
     * :arg st: Ignored
     * :type st: borrowed SymTab 
     *
     * :returns: MsgTuple containing JSON formatted string of cmd -> function mapping
     */
    proc getCommandMapMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab) throws {
        // We can ignore the args, we just need it to match the CommandMap call signature
        import CommandMap;
        try {
            const json:string = CommandMap.dumpCommandMap();
            return new MsgTuple(CommandMap.dumpCommandMap():string, MsgType.NORMAL);
        } catch {
            var errorMsg = "Error converting CommandMap to JSON";
            mpLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    /* 
    Response to __str__ method in python str convert array data to string 

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string,MsgType)
   */
    proc strMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");

        var printThresh = msgArgs.get("printThresh").getIntValue();
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                              "cmd: %s name: %s threshold: %i".doFormat(
                                               cmd,name,printThresh));  
                                               
        repMsg  = st.datastr(name,printThresh);        
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);  
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    /* Response to __repr__ method in python.
       Repr convert array data to string 
       
       :arg reqMsg: request containing (cmd,name)
       :type reqMsg: string 

       :arg st: SymTab to act on
       :type st: borrowed SymTab 

       :returns: MsgTuple
      */ 
    proc reprMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        var printThresh = msgArgs.get("printThresh").getIntValue();

        repMsg = st.datarepr(name,printThresh);
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    /* 
    Sets all elements in array to a value (broadcast) 

    :arg reqMsg: request containing (cmd,name,dtype,value)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND(cmd_prefix="set")
    proc setMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd = 1): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        const value = msgArgs.get("val");

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "cmd: %s value: %s in pdarray %s".doFormat(cmd,name,st.attrib(name)));

        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gEnt,int, nd);
                var val: int = value.getIntValue();
                e.a = val;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int, nd);
                var val: real = value.getRealValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "cmd: %s name: %s to val: %?".doFormat(cmd,name,val:int));
                e.a = val:int;
                repMsg = "set %s to %?".doFormat(name, val:int);
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int, nd);
                var val: bool = value.getBoolValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "cmd: %s name: %s to val: %?".doFormat(cmd,name,val:int));
                e.a = val:int;
                repMsg = "set %s to %?".doFormat(name, val:int);
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real, nd);
                var val: int = value.getIntValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                      "cmd: %s name: %s to value: %?".doFormat(cmd,name,val:real));
                e.a = val:real;
                repMsg = "set %s to %?".doFormat(name, val:real);
            }
            when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real, nd);
                var val: real = value.getRealValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "cmd: %s name; %s to value: %?".doFormat(cmd,name,val));
                e.a = val;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real, nd);
                var val: bool = value.getBoolValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "cmd: %s name: %s to value: %?".doFormat(cmd,name,val:real));
                e.a = val:real;
                repMsg = "set %s to %?".doFormat(name, val:real);
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool, nd);
                var val: int = value.getIntValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "cmd: %s name: %s to value: %?".doFormat(cmd,name,val:bool));
                e.a = val:bool;
                repMsg = "set %s to %?".doFormat(name, val:bool);
            }
            when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,int, nd);
                var val: real = value.getRealValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                      "cmd: %s name: %s to  value: %?".doFormat(cmd,name,val:bool));
                e.a = val:bool;
                repMsg = "set %s to %?".doFormat(name, val:bool);
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,bool, nd);
                var val: bool = value.getBoolValue();
                mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "cmd: %s name: %s to value: %?".doFormat(cmd,name,val));
                e.a = val;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            when (DType.UInt64, DType.UInt64) {
                var e = toSymEntry(gEnt,uint, nd);
                var val: uint = value.getUIntValue();
                e.a = val;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            when (DType.BigInt, DType.BigInt) {
                var e = toSymEntry(gEnt,bigint, nd);
                var val: bigint = value.getBigIntValue();
                e.a = val;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            when (DType.BigInt, DType.UInt64) {
                var e = toSymEntry(gEnt,bigint, nd);
                var val: uint = value.getUIntValue();
                e.a = val:bigint;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            when (DType.BigInt, DType.Int64) {
                var e = toSymEntry(gEnt,bigint, nd);
                var val: int = value.getIntValue();
                e.a = val:bigint;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            when (DType.BigInt, DType.Bool) {
                var e = toSymEntry(gEnt,bigint, nd);
                var val: bool = value.getBoolValue();
                e.a = val:bigint;
                repMsg = "set %s to %?".doFormat(name, val);
            }
            otherwise {
                mpLogger.error(getModuleName(),getRoutineName(),
                                               getLineNumber(),"dtype: %s".doFormat(msgArgs.getValueOf("dtype")));
                return new MsgTuple(unrecognizedTypeError(pn,msgArgs.getValueOf("dtype")), MsgType.ERROR);
            }
        }
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }


    /*
        Create a "broadcasted" array (of rank 'nd') by copying an array into an
        array of the given shape.

        E.g., given the following broadcast:
        A      (4d array):  8 x 1 x 6 x 1
        B      (3d array):      7 x 1 x 5
        ---------------------------------
        Result (4d array):  8 x 7 x 6 x 5

        Two separate calls would be made to store 'A' and 'B' in arrays with
        result's shape.

        When copying from a singleton dimension, the value is repeated along
        that dimension (e.g., A's 1st and 3rd, or B's 2nd dimension above).
        For non singleton dimensions, the size of the two arrays must match,
        and the values are copied into the result array.

        When prepending a new dimension to increase an array's rank, the
        values from the other dimensions are repeated along the new dimension.

        !!! TODO: Avoid the promoted copies here by leaving the singleton
        dimensions in the result array, and making operations on arrays
        aware that promotion of singleton dimensions may be necessary. E.g.,
        make matrix multiplication aware that it can treat a singleton
        value as a vector of the appropriate length during multiplication.

        (this may require a modification of SymEntry to keep track of
        which dimensions are explicitly singletons)

        NOTE: registration of this procedure is handled specially in
        'serverModuleGen.py' because it has two param fields. The purpose of
        designing "broadcast" this way is to avoid the the need for multiple
        dimensionality param fields in **all** other message handlers (e.g.,
        matrix multiply can be designed to expect matrices of equal rank,
        requiring only one dimensionality param field. As such, the client
        implementation of matrix-multiply may be required to broadcast the array
        arguments up to some common rank (N) before issuing a 'matMult{N}D'
        command to the server)
    */
    proc broadcastNDArray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
        param ndIn: int, // rank of the array to be broadcast
        param ndOut: int // rank of the result array
    ): MsgTuple throws {
        const name = msgArgs.getValueOf("name"),
              shapeOut = msgArgs.get("shape").getTuple(ndOut),
              rname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        proc doAssignment(type dtype): MsgTuple throws {
            var eIn = toSymEntry(gEnt, dtype, ndIn),
                eOut = st.addEntry(rname, (...shapeOut), dtype);

            if ndIn == ndOut && eIn.tupShape == shapeOut {
                // no broadcast necessary, copy the array
                eOut.a = eIn.a;
            } else {
                // ensure that 'shapeOut' is a valid broadcast of 'eIn.tupShape'
                //   and determine which dimensions will require promoted assignment
                var (valid, bcDims) = checkValidBroadcast(eIn.tupShape, shapeOut);

                if !valid {
                    const errorMsg = "Invalid broadcast: " + eIn.tupShape:string + " -> " + shapeOut:string;
                    mpLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                } else {
                    // define a mapping from the output array's indices to the input array's indices
                    inline proc map(idx: int ...ndOut): ndIn*int {
                        var ret: ndIn*int; // 'ret' is initialized to zero (assumes zero-based arrays)
                        for param i in 0..<ndIn do
                            ret[i] = if bcDims[i] then 0 else idx[i];
                        return ret;
                    }

                    // TODO: Is this being auto-aggregated? If not, add explicit aggregation
                    forall idx in eOut.a.domain do
                        if ndOut == 1
                            then eOut.a[idx] = eIn.a[map(idx)];
                            else eOut.a[idx] = eIn.a[map((...idx))];
                }
            }

            const repMsg = "created " + st.attrib(rname);
            mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select gEnt.dtype {
            when DType.Int64 do return doAssignment(int);
            when DType.UInt8 do return doAssignment(uint(8));
            when DType.UInt64 do return doAssignment(uint);
            when DType.Float64 do return doAssignment(real);
            when DType.Bool do return doAssignment(bool);
            otherwise {
                var errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
                mpLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    proc checkValidBroadcast(from: ?Nf*int, to: ?Nt*int): (bool, Nf*bool) {
        var dimsToBroadcast: Nf*bool;
        if Nf > Nt then return (false, dimsToBroadcast);

        for param iIn in 0..<Nf {
            param iOut = Nt - Nf + iIn;
            if from[iIn] == 1 && to[iOut] != 1 {
                dimsToBroadcast[iIn] = true;
            } else if from[iIn] != to[iOut] {
                return (false, dimsToBroadcast);
            }
        }

        return (true, dimsToBroadcast);
    }
}
