module CommandMap {
  use ServerRegistration;
  use Map;
  use Message;
  use MultiTypeSymbolTable;

  // This is a dummy function to get the signature of the Arkouda
  // server FCF. Ideally, the `func()` function would be able to
  // construct the FCF type, but there is no way to generate a
  // FCF that throws using `func()` today.
  proc akMsgSign(a: string, b: string, c: borrowed SymTab): MsgTuple throws {
    var rep = new MsgTuple("dummy-msg", MsgType.NORMAL);
    return rep;
  }

  var f = akMsgSign;
  var commandMap: map(string, f.type);
  
  proc registerFunction(cmd: string, fcf: f.type) {
    commandMap.add(cmd, fcf);
  }
}
