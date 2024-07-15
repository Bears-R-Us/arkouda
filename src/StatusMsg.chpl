module StatusMsg {
    use Reflection;
    use ServerConfig;
    use Logging;    
    use Message;
    use MemoryMgmt;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use IOUtils;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sLogger = new Logger(logLevel, logChannel);
    
    proc getMemoryStatusMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var statuses = formatJson(getLocaleMemoryStatuses());

        sLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      'memory statuses '+formatJson(statuses));
        return new MsgTuple(statuses, MsgType.NORMAL);     
    }
}
