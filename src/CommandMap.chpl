module CommandMap {
  use Message;
  use MultiTypeSymbolTable;
  import ServerDaemon;

  use JSON;
  use IO;
  use IOUtils;
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
  private var f = akMsgSign;

  var commandMap: map(string, f.type);        // cmd-name => function
  var moduleMap: map(string, (string, int));  // cmd-name => (module-name, line number)
  use Set;
  var usedModules: set(string);

  /**
   * Register command->function in the CommandMap
   * This binds a server command to its corresponding function matching the standard
   * function signature & MsgTuple return type
   */
  proc registerFunction(cmd: string, fcf: f.type, modName: string = "", line: int = -1) {
    if commandMap.contains(cmd) {
      writef("Warning: Command '%s' already registered. Ignoring registration from [%s:%d]", cmd, modName, line);
    } else {
      commandMap.add(cmd, fcf);

      if modName != "" then
        moduleMap.add(cmd, (modName, line));
    }
  }

  /* Support for asynchronous checkpointing. */
  proc emptyStartAsyncCheckpointDaemon(sd: borrowed ServerDaemon.DefaultServerDaemon) {
    return false;
  }
  var funStartAsyncCheckpointDaemon = emptyStartAsyncCheckpointDaemon;

  proc writeUsedModulesJson(ref mods: set(string)) {
    const cfgFile = try! open("UsedModules.json", ioMode.cw),
          w = try! cfgFile.writer(locking=false, serializer = new jsonSerializer());

    try! w.write(mods);
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

  /**
   * Dump the combined contents of the command maps as a single json encoded string
   */
  proc dumpCommandMap(): string throws {
    return formatJson(commandMap);
  }

  proc executeCommand(cmd: string, msgArgs, st): MsgTuple throws {
    if commandMap.contains(cmd) {
      if moduleMap.contains(cmd) then usedModules.add(moduleMap[cmd][0]);
      return commandMap[cmd](cmd, msgArgs, st);
    } else {
      return MsgTuple.error("Unrecognized command: %s".format(cmd));
    }
  }
}
