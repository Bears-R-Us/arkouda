module CommandMap {
  use Message;
  use MultiTypeSymbolTable;

  use ArkoudaIOCompat;
  use Map;

  /**
   * This is a dummy function to get the signature of the Arkouda
   * server FCF. Ideally, the `func()` function would be able to
   * construct the FCF type, but there is no way to generate a
   * FCF that throws using `func()` today.
   */
  proc akMsgSign(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    return MsgTuple.success("");
  }

  proc arrayMsgSign(cmd: string, msgArgs: borrowed MessageArgs, ref data: bytes, st: borrowed SymTab): MsgTuple throws {
    var rep = new MsgTuple("dummy-msg", MsgType.NORMAL);
    return rep;
  }

  private var f = akMsgSign;
  private var a = arrayMsgSign;

  var commandMap: map(string, f.type);
  var commandMapArray: map(string, a.type);
  var moduleMap: map(string, string);
  use Set;
  var usedModules: set(string);

  /**
   * Register command->function in the CommandMap
   * This binds a server command to its corresponding function matching the standard
   * function signature & MsgTuple return type
   */
  proc registerFunction(cmd: string, fcf: f.type) {
    commandMap.add(cmd, fcf);
  }

  proc registerFunction(cmd: string, fcf: f.type, modName: string) {
    commandMap.add(cmd, fcf);
    moduleMap.add(cmd, modName);
  }

  proc writeUsedModules(fmt: string = "cfg") {
    select fmt {
      when "json" do writeUsedModulesJson(usedModules);
      when "cfg" do writeUsedModulesCfg();
      otherwise {
        writeln("Unrecognized format for used-modules file: '%s'");
        writeln("Use '--usedModulesFmt=\"json\"' or '--usedModulesFmt=\"cfg\"'");
        writeln("Defaulting to json...");
        writeUsedModulesJson(usedModules);
      }
    }
  }

  private proc writeUsedModulesCfg() {
    use IO;
    var newCfgFile = try! open("UsedModules.cfg", ioMode.cw);
    var chnl = try! newCfgFile.writer(locking=false);
    for mod in usedModules do
      try! chnl.write(mod + '\n');
  }

  proc registerArrayFunction(cmd: string, fcf: a.type) {
    commandMapArray.add(cmd, fcf);
  }

  /**
   * Dump the combined contents of the command maps as a single json encoded string
   */
  proc dumpCommandMap(): string throws {
    return formatJson(commandMap);
  }

  proc executeCommand(cmd: string, msgArgs, st) throws {
    var repTuple: MsgTuple;
    if commandMap.contains(cmd) {
      usedModules.add(moduleMap[cmd]);
      repTuple = commandMap[cmd](cmd, msgArgs, st);
    } else {
      repTuple = new MsgTuple("Unrecognized command: %s".doFormat(cmd), MsgType.ERROR);
    }
    return repTuple;
  }

}
