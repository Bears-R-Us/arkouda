
module MsgProcessing
{
    use ServerConfig;

    use Math only;
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
        var repMsg: string, // response message
            dtype = str2dtype(msgArgs.getValueOf("dtype")),
            shape = msgArgs.get("shape").getTuple(nd);

        var size = 1;
        for s in shape do size *= s;

        // get next symbol name
        overMemLimit(dtypeSize(dtype) * size);
        const rname = st.nextName();

        // if verbose print action
        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s dtype: %s size: %i new pdarray name: %s".doFormat(
                                                     cmd,dtype2str(dtype),size,rname));
        if isSupportedDType(dtype) {
            // create and add entry to symbol table
            st.addEntry(rname, (...shape), dtype);
            // if verbose print result
            mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "created the pdarray %s".doFormat(st.attrib(rname)));

            repMsg = "created " + st.attrib(rname);
            mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        } else {
            const errorMsg = unsupportedTypeError(dtype, getRoutineName());
            mpLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    // this proc is not technically needed with the 'arkouda.registerND' annotation above
    //  (keeping it for now as a stopgap until the ND array work is further along)
    proc createMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        return createMsg(cmd, msgArgs, st, 1);
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
            repMsg = AutoMath.round((memUsed / (getMemLimit():real * numLocales)) * 100):uint:string;
        }
        else {
            repMsg = AutoMath.round(memUsed / factor):uint:string;
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
            repMsg = (100 - AutoMath.round((memUsed / totMem) * 100)):uint:string;
        }
        else {
            repMsg = AutoMath.round((totMem - memUsed) / factor):uint:string;
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
        const name = msgArgs.getValueOf("array"),
              value = msgArgs.get("val");

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        mpLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "cmd: %s value: %s in pdarray %s".doFormat(cmd,name,st.attrib(name)));

        proc doAssignment(type t): MsgTuple throws {
            var e = toSymEntry(gEnt, t, nd);
            const val = value.getScalarValue(t);
            e.a = val;
            mpLogger.debug(getModuleName(),pn,getLineNumber(),
                            "cmd: %s name: %s to val: %?".doFormat(cmd,name,val));

            const repMsg = "set %s to %?".doFormat(name, val);
            mpLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        proc notImplemented(): MsgTuple throws {
            const errorMsg = unsupportedTypeError(gEnt.dtype, pn);
            mpLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        select gEnt.dtype {
            when DType.Int8 {
                if SupportsInt8
                    then return doAssignment(int(8));
                    else return notImplemented();
            }
            when DType.Int16 {
                if SupportsInt16
                    then return doAssignment(int(16));
                    else return notImplemented();
            }
            when DType.Int32 {
                if SupportsInt32
                    then return doAssignment(int(32));
                    else return notImplemented();
            }
            when DType.Int64 {
                if SupportsInt64
                    then return doAssignment(int(64));
                    else return notImplemented();
            }
            when DType.UInt8 {
                if SupportsUint8
                    then return doAssignment(uint(8));
                    else return notImplemented();
            }
            when DType.UInt16 {
                if SupportsUint16
                    then return doAssignment(uint(16));
                    else return notImplemented();
            }
            when DType.UInt32 {
                if SupportsUint32
                    then return doAssignment(uint(32));
                    else return notImplemented();
            }
            when DType.UInt64 {
                if SupportsUint64
                    then return doAssignment(uint(64));
                    else return notImplemented();
            }
            when DType.Float32 {
                if SupportsFloat32
                    then return doAssignment(real(32));
                    else return notImplemented();
            }
            when DType.Float64 {
                if SupportsFloat64
                    then return doAssignment(real(64));
                    else return notImplemented();
            }
            when DType.Complex64 {
                if SupportsComplex64
                    then return doAssignment(complex(64));
                    else return notImplemented();
            }
            when DType.Complex128 {
                if SupportsComplex128
                    then return doAssignment(complex(128));
                    else return notImplemented();
            }
            when DType.Bool {
                if SupportsBool
                    then return doAssignment(bool);
                    else return notImplemented();
            }
            when DType.BigInt {
                return doAssignment(bigint);
            }
            otherwise {
                mpLogger.error(getModuleName(),getRoutineName(),
                                               getLineNumber(),"dtype: %s".doFormat(msgArgs.getValueOf("dtype")));
                return new MsgTuple(unrecognizedTypeError(pn,msgArgs.getValueOf("dtype")), MsgType.ERROR);
            }
        }
    }

    // this proc is not technically needed with the 'arkouda.registerND' annotation above
    //   (keeping it for now as a stopgap until the ND array work is further along)
    proc setMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        return setMsg(cmd, msgArgs, st, 1);
    }
}
