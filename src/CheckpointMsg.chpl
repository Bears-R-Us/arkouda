module CheckpointMsg {
  use Reflection;
  use FileSystem;
  use List;
  import IO, Path;

  use ServerErrors, ServerConfig;
  use MultiTypeSymbolTable, MultiTypeSymEntry;
  use Message;
  use ParquetMsg;
  use IOUtils;

  config param metadataExt = "md";
  config param dataExt = "data";
  config param mdNameMaxLength = 256;

  proc checkpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                     st: borrowed SymTab): MsgTuple throws {
    var path = msgArgs.getValueOf("path");

    writeln("in checkpoint msg");

    if !exists(path) then
      mkdir(path);

    for (name, entry) in zip(st.tab.keys(), st.tab.values()) {
      var e = toSymEntry(toGenSymEntry(entry), int);
      try! saveArr(path, name, e);
    }
    return new MsgTuple("Checkpointed yo", MsgType.NORMAL);
  }

  proc checkpointLoadMsg(cmd: string, msgArgs: borrowed MessageArgs,
                         st: borrowed SymTab): MsgTuple throws {
    var path = msgArgs.getValueOf("path");
    if !exists(path) {
      var errorMsg = "Directory not found: " + path;
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    var rnames: list((string, ObjType, string));

    for mdName in glob(path+"/*"+metadataExt) {
      var (name, entry) = loadArr(path, mdName);
      st.addEntry(name, entry);
      rnames.pushBack((name, ObjType.PDARRAY, name));
    }
    var l = new list(string);
    use GenSymIO;
    var repMsg = buildReadAllMsgJson(rnames, false, 0, l, st);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  private proc saveArr(path, name, entry) throws {
    const mdName = Path.joinPath(path, ".".join(name, metadataExt));

    var mdFile = IO.open(mdName, ioMode.cw);
    var mdWriter = mdFile.writer();
    mdWriter.writeln(name);
    mdWriter.writeln(entry.size);
    mdWriter.writeln(entry.a.targetLocales().size);

    const dataName = Path.joinPath(path, ".".join(name, dataExt));
    const (warnFlag, filenames) = write1DDistArrayParquet(dataName, 'asd',
                                                          "int64", 0, 0,
                                                          entry.a);

    writeln("Data created: ", dataName);

    for f in filenames {
      mdWriter.writef("%i %s\n", f.numCodepoints, f);
    }

    mdWriter.close();
    mdFile.close();

    writeln("Metadata created: ", mdName);
  }

  private proc loadArr(path, mdName) throws {
    var name: string;
    var size: int;
    var numTargetLocales: int;

    var mdFile = IO.open(mdName, ioMode.r);
    var mdReader = mdFile.reader();
    mdReader.read(name, size, numTargetLocales);

    assert(numTargetLocales==1);

    const dataNames: [0..#numTargetLocales] string;

    for name in dataNames {
      const fnSize = mdReader.readThrough(" "):int;
      name = mdReader.readString(maxSize=fnSize);
    }

    var entryVal = new shared SymEntry(size, int);
    readFilesByName(entryVal.a, dataNames, [size], "asd", 0);

    writeln("Data created: ", dataNames);

    mdReader.close();
    mdFile.close();

    return (name, entryVal);
  }

  use CommandMap;
  registerFunction("checkpoint", checkpointMsg, getModuleName());
  registerFunction("loadcheckpoint", checkpointLoadMsg, getModuleName());
}
