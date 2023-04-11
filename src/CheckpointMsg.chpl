module CheckpointMsg {
  use ServerErrors, ServerConfig;
  use MultiTypeSymbolTable, MultiTypeSymEntry;
  use Message;
  use Reflection;
  use FileSystem;
  use ParquetMsg;
  use ArkoudaMapCompat;

  proc checkpointMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    if !exists('checkpoint') then
      mkdir("checkpoint");
    
    for (k, v) in zip(st.tab.keys(), st.tab.values()) {
      var e = toSymEntry(toGenSymEntry(v), int);
      write1DDistArrayParquet("checkpoint/"+k+"-"+e.size:string, 'asd', "int64", 0, 0, e.a);
    }
    return new MsgTuple("Checkpointed yo", MsgType.NORMAL);
  }

  proc checkpointLoadMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var tmp = glob("checkpoint/*");
    var rnames: list(3*string);
    for filename in tmp {
      var fileSize = filename[24]:int;
      var entryVal = new shared SymEntry(fileSize, int);
      readFilesByName(entryVal.a, [filename], [fileSize], "asd", 0);
      var rname = st.nextName();
      st.addEntry(rname, entryVal);
      rnames.append((rname, "pdarray", rname));
    }

    writeln(rnames);
    use List;
    var l = new list(string);
    use GenSymIO;
    var repMsg = _buildReadAllMsgJson(rnames, false, 0, l, st);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  use CommandMap;
  registerFunction("checkpoint", checkpointMsg, getModuleName());
  registerFunction("loadcheckpoint", checkpointLoadMsg, getModuleName());
}
