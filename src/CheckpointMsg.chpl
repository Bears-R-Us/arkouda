module CheckpointMsg {
  use ServerErrors, ServerConfig;
  use MultiTypeSymbolTable, MultiTypeSymEntry;
  use Message;
  use Reflection;
  use FileSystem;
  use ParquetMsg;
  use List;
  /*use ArkoudaMapCompat;*/

  proc checkpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                     st: borrowed SymTab): MsgTuple throws {
    var path = msgArgs.getValueOf("path");

    writeln("in checkpoint msg");

    if !exists(path) then
      mkdir(path);
    
    for (k, v) in zip(st.tab.keys(), st.tab.values()) {
      writeln(v.type:string);
      var e = toSymEntry(toGenSymEntry(v), int);
      write1DDistArrayParquet(path+"/"+k+"-"+e.size:string, 'asd', "int64",
                              0, 0, e.a);
      writeln(path+"/"+k+"-"+e.size:string);
    }
    return new MsgTuple("Checkpointed yo", MsgType.NORMAL);
  }
  use CommandMap;
  registerFunction("checkpoint", checkpointMsg, getModuleName());

  proc checkpointLoadMsg(cmd: string, msgArgs: borrowed MessageArgs,
                         st: borrowed SymTab): MsgTuple throws {
    var path = msgArgs.getValueOf("path");
    if !exists(path) {
      var errorMsg = "Directory not found: " + path;
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }
    var tmp = glob(path+"/*");
    var rnames: list((string, ObjType, string));
    for filename in tmp {
      var fileSize = filename[filename.find("-")+1..filename.rfind("_")-1]:int;
      var name = filename[filename.find("/")+1..filename.find("-")-1];
      var entryVal = new shared SymEntry(fileSize, int);
      readFilesByName(entryVal.a, [filename], [fileSize], "asd", 0);
      var rname = st.nextName();
      st.addEntry(rname, entryVal);
      rnames.pushBack((name, ObjType.PDARRAY, rname));
    }
    var l = new list(string);
    use GenSymIO;
    var repMsg = buildReadAllMsgJson(rnames, false, 0, l, st);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }
 
  registerFunction("loadcheckpoint", checkpointLoadMsg, getModuleName());
}
