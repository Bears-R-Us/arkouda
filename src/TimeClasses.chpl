module TimeClasses {
  use MultiTypeSymEntry;
  use MultiTypeSymbolTable;
  use Logging;

  private config const logLevel = ServerConfig.logLevel;
  const tcLogger = new Logger(logLevel);
  
  proc getDatetime(name: string, st: borrowed SymTab): shared Datetime throws {
    var abstractEntry = st.lookup(name);
    if !abstractEntry.isAssignableTo(SymbolEntryType.Datetime) {
      var errorMsg = "Error: Cannot interpret %s as Datetime".format(abstractEntry.entryType);
      tcLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
      throw new Error(errorMsg);
    }
    var entry:DatetimeSymEntry = abstractEntry: shared DatetimeSymEntry;
    return entry;
  }
  
}