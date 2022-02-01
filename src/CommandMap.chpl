module CommandMap {
  use ServerRegistration;
  use Map;
  use Message;
  use MultiTypeSymbolTable;

  /**
   * This is a dummy function to get the signature of the Arkouda
   * server FCF. Ideally, the `func()` function would be able to
   * construct the FCF type, but there is no way to generate a
   * FCF that throws using `func()` today.
   */
  proc akMsgSign(a: string, b: string, c: borrowed SymTab): MsgTuple throws {
    var rep = new MsgTuple("dummy-msg", MsgType.NORMAL);
    return rep;
  }

  /**
   * Just like akMsgSign, but Messages which have a binary return
   * require a different signature
   */
  proc akBinMsgSign(a: string, b: string, c: borrowed SymTab): bytes throws {
    var nb = b"\x00";
    return nb;
  }

  private var f = akMsgSign;
  private var b = akBinMsgSign;

  var commandMap: map(string, f.type);
  var commandMapBinary: map(string, b.type);

  /**
   * Register command->function in the CommandMap
   * This binds a server command to its corresponding function matching the standard
   * function signature & MsgTuple return type
   */
  proc registerFunction(cmd: string, fcf: f.type) {
    commandMap.add(cmd, fcf);
  }

  /**
   * Register command->function in the CommandMap for Binary returning functions
   * This binds a server command to its corresponding function matching the standard
   * function signature but returning "bytes"
   */
  proc registerBinaryFunction(cmd: string, fcf: b.type) {
    commandMapBinary.add(cmd, fcf);
  }

  /**
   * Dump the combined contents of the command maps as a single json encoded string
   */
  proc dumpCommandMap(): string throws {
    var cm1:string = "%jt".format(commandMap);
    var cm2:string = "%jt".format(commandMapBinary);
    // Join these two together
    var idx_close = cm1.rfind("}"):int;
    return cm1(0..idx_close-1) + ", " + cm2(1..cm2.size-1);
  }

}
