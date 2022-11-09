module GroupByMsg {
    use CTypes;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use SegmentedArray;
    use SegmentedString;
    use ServerErrorStrings;
    use ServerConfig;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use RandArray;
    use IO;
    use Map;
    use GenSymIO;
    use GroupBy;

    private config const logLevel = ServerConfig.logLevel;
    const gmLogger = new Logger(logLevel);

    proc createGroupByMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const assumeSorted = msgArgs.get("assumeSortedStr").getBoolValue();
        var n = msgArgs.get("nkeys").getIntValue();
        var keynames = msgArgs.get("keynames").getList(n);
        var keytypes = msgArgs.get("keytypes").getList(n); 

        var gb = getGroupBy(n, keynames, keytypes, assumeSorted, st);

        // Create message tuple containing the GroupBy obj, segments, permutation, and indexes of unique keys
        var rtnmap: map(string, string) = new map(string, string);
        gb.fillReturnMap(rtnmap, st);
        var repMsg: string = "%jt".format(rtnmap);
        gmLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("createGroupBy", createGroupByMsg, getModuleName());
}