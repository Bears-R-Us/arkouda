module EncodingMsg {
    use Subprocess;
    use Reflection;
    use Logging;
    use ServerConfig;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use CommAggregation;
    use ServerErrors;
    use ServerErrorStrings;

    use SegmentedString;

    private config const logLevel = ServerConfig.logLevel;
    const emLogger = new Logger(logLevel);

    proc encodeDecodeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string;
        var encoding = msgArgs.getValueOf("encoding");

        var stringsObj = getSegString(msgArgs.getValueOf("obj"), st);

        select encoding.toLower() {
            when "idna" {
                var (offsets, vals) = stringsObj.idnaEncodeDecode(cmd); // use the cmd to trigger correct action
                var encodedStrings = getSegString(offsets, vals, st);
                repMsg = "created " + st.attrib(encodedStrings.name) + "+created bytes.size %t".format(encodedStrings.nBytes);
                
            }
            otherwise {
                var errorMsg = "%s %s not currently supported".format(cmd, encoding);      
                emLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
                return new MsgTuple(errorMsg, MsgType.ERROR);    
            }
        }

        emLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }


    use CommandMap;
    registerFunction("encode", encodeDecodeMsg, getModuleName());
    registerFunction("decode", encodeDecodeMsg, getModuleName());
}