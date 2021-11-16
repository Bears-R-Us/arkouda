module MyModuleMsg {
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Message;

  proc myTestMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    return new MsgTuple("This is an optional module", MsgType.NORMAL);
  }

  proc registerMe() {
    use arkouda_server;
    registerFunction("test-command", myTestMsg);
  }
}