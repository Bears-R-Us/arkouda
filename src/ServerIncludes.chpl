module ServerIncludes {
  use ServerRegistration;
  use Map;
  use Message;
  use MultiTypeSymbolTable;
  proc akMsgSign(a: string, b: string, c: borrowed SymTab): MsgTuple throws {
    var rep = new MsgTuple("sports", MsgType.NORMAL);
    return rep;
  }

  var f = akMsgSign;
  var commandMap: map(string, f.type);

  proc registerFunction(cmd: string, fcf: f.type) {
    commandMap.add(cmd, fcf);
  }
}
