
module MsgProcessing
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use BigInteger;
    use Math;
    use Time;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use AryUtil;

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

        overMemLimit(dtypeSize(dtype) * size);

        // if verbose print action
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s size: %i new pdarray name: %s".format(
                                                     cmd,dtype2str(dtype),size,rname));
        if isSupportedDType(dtype) {
            // create and add entry to symbol table
            st.addEntry(rname, (...shape), dtype);
            // if verbose print result
            mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "created the pdarray %s".format(st.attrib(rname)));

            const repMsg = "created " + st.attrib(rname);
            mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        } else {
            const errorMsg = unsupportedTypeError(dtype, getRoutineName());
            mpLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    // used for "zero-dimensional" array api scalars
    proc createMsg0D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const dtype = str2dtype(msgArgs.getValueOf("dtype")),
              rname = st.nextName();

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "cmd: %s dtype: %s size: 1 new pdarray name: %s".format(
                       cmd,dtype2str(dtype),rname));

        // on the client side, scalar (0D) arrays have a shape of "()" and a size of 1
        // here, we represent that using a 1D array with a shape of (1,) and a size of 1
        var e = toGenSymEntry(st.addEntry(rname, 1, dtype));
        e.size = 1;
        e.shape = "[]";
        e.ndim = 1; // this is 1 rather than 0 s.t. calls to other message handlers treat it as a 1D
                    // array (e.g., we should call 'set1D', not 'set0D' on this array)

        // set the value if a 'value' argument is provided
        proc setValue(type t) throws {
            try {
                const valueArg = msgArgs.get("value");
                toSymEntry(e, t, 1).a[0] = valueArg.toScalar(t);
            } catch e: ErrorWithContext {
            } catch e {
                throw e;
            }
        }

        select dtype {
            when DType.Int64 do setValue(int);
            when DType.UInt64 do setValue(uint);
            when DType.Float64 do setValue(real);
            when DType.Bool do setValue(bool);
            when DType.BigInt do setValue(bigint);
            otherwise {
                const errorMsg = unsupportedTypeError(dtype, getRoutineName());
                mpLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "created the pdarray %s".format(st.attrib(rname)));

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
                                     "cmd: %s array: %s".format(cmd,st.attrib(name)));
        // delete entry from symbol table
        if st.deleteEntry(name) {
            repMsg = "deleted %s".format(name);
        }
        else {
            repMsg = "registered symbol, %s, not deleted".format(name);
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
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "cmd: %s".format(cmd));
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
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".format(cmd));
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
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".format(cmd));
        var memUsed = if memTrack then getMemUsed():real * numLocales else st.memUsed():real;
        if asPercent {
            repMsg = Math.round((memUsed / (getMemLimit():real * numLocales)) * 100):uint:string;
        }
        else {
            repMsg = Math.round(memUsed / factor):uint:string;
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
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %s".format(cmd));
        var memUsed = if memTrack then getMemUsed():real * numLocales else st.memUsed():real;
        var totMem = getMemLimit():real * numLocales;
        if asPercent {
            repMsg = (100 - Math.round((memUsed / totMem) * 100)):uint:string;
        }
        else {
            repMsg = Math.round((totMem - memUsed) / factor):uint:string;
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
                                              "cmd: %s name: %s threshold: %i".format(
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
        const name = msgArgs.getValueOf("array"),
              value = msgArgs.get("val");

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "cmd: %s value: %s in pdarray %s".format(cmd,name,st.attrib(name)));

        proc doAssignment(type t): MsgTuple throws
            where isSupportedType(t)
        {
            var e = toSymEntry(gEnt, t, nd);
            const val = value.getScalarValue(t);
            e.a = val;
            mpLogger.debug(getModuleName(),pn,getLineNumber(),
                            "cmd: %s name: %s to val: %?".format(cmd,name,val));

            const repMsg = "set %s to %?".format(name, val);
            mpLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        proc doAssignment(type t): MsgTuple throws
            where !isSupportedType(t)
        {
            const errorMsg = unsupportedTypeError(gEnt.dtype, pn);
            mpLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        select gEnt.dtype {
            when DType.Int8 do return doAssignment(int(8));
            when DType.Int16 do return doAssignment(int(16));
            when DType.Int32 do return doAssignment(int(32));
            when DType.Int64 do return doAssignment(int(64));
            when DType.UInt8 do return doAssignment(uint(8));
            when DType.UInt16 do return doAssignment(uint(16));
            when DType.UInt32 do return doAssignment(uint(32));
            when DType.UInt64 do return doAssignment(uint(64));
            when DType.Float64 do return doAssignment(real(64));
            when DType.Complex64 do return doAssignment(complex(64));
            when DType.Complex128 do return doAssignment(complex(128));
            when DType.Bool do return doAssignment(bool);
            when DType.BigInt do return doAssignment(bigint);
            otherwise {
                mpLogger.error(getModuleName(),getRoutineName(),
                                               getLineNumber(),"dtype: %s".format(msgArgs.getValueOf("dtype")));
                return new MsgTuple(unrecognizedTypeError(pn,msgArgs.getValueOf("dtype")), MsgType.ERROR);
            }
        }
    }

     /*
        Get a list of lists indicating how an array is "chunked" across locales.

        For example, a 100x40 2D array on 4 locales could return: [[0, 50], [0, 20]]
        indicating that the chunks start at indices 0 and 50 in the first dimension,
        and 0 and 20 in the second dimension.
    */
    @arkouda.registerND
    proc chunkInfoMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        const name = msgArgs.getValueOf("array");
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        proc getChunkInfo(type t): MsgTuple throws {
            var blockSizes: [0..<nd] [0..<numLocales] int;
            const e = toSymEntry(gEnt, t, nd);

            coforall loc in Locales do on loc {
                const locDom = e.a.localSubdomain();
                for i in 0..<nd do
                    blockSizes[i][loc.id] = locDom.dim(i).low;
            }

            var msg = "[";
            var first = true;
            for dim in blockSizes {
                if first then first = false; else msg += ", ";
                msg += "[";
                var firstInner = true;
                for locSize in dim {
                    if firstInner then firstInner = false; else msg += ", ";
                    msg += locSize:string;
                }
                msg += "]";
            }
            msg += "]";

            mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),msg);
            return new MsgTuple(msg, MsgType.NORMAL);
        }

        select gEnt.dtype {
            when DType.Int64 do return getChunkInfo(int);
            when DType.UInt64 do return getChunkInfo(uint);
            when DType.Float64 do return getChunkInfo(real);
            when DType.Bool do return getChunkInfo(bool);
            when DType.UInt8 do return getChunkInfo(uint(8));
            otherwise {
                const errorMsg = notImplementedError(getRoutineName(),dtype2str(gEnt.dtype));
                mpLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
        }
    }
}
