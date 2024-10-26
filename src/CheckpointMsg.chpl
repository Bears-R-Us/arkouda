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
  config param serverMetadataName = "server."+metadataExt;

  proc checkpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                     st: borrowed SymTab): MsgTuple throws {
    var path = msgArgs.getValueOf("path");

    writeln("in checkpoint msg");

    if !exists(path) then
      mkdir(path);

    saveServerMetadata(path, st);

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

    var loadedId = loadServerMetadata(path);

    var rnames: list((string, ObjType, string));

    for mdName in glob(path+"/*"+metadataExt) {
      if mdName == serverMetadataName then continue;
      var (name, entry) = loadArr(path, mdName, loadedId);
      writeln("name before replace ", name);
      /*name = name.replace(loadedId, st.serverid);*/
      writeln("name after replace ", name);

      st.addEntry(name, entry);

      writeln("added entry ", name);
      rnames.pushBack((name, ObjType.PDARRAY, name));
    }
    writeln("finished load loop ");
    writeln(rnames);
    var l = new list(string);
    use GenSymIO;
    var repMsg = buildReadAllMsgJson(rnames, false, 0, l, st);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  private proc saveServerMetadata(path, st: borrowed SymTab) throws {
    const mdName = Path.joinPath(path, serverMetadataName);

    var mdFile = IO.open(mdName, ioMode.cw);
    var mdWriter = mdFile.writer();

    mdWriter.writeln(st.serverid);
    mdWriter.writeln(numLocales);
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

  private proc loadServerMetadata(path) throws {
    const mdName = Path.joinPath(path, serverMetadataName);

    var mdFile = IO.open(mdName, ioMode.r);
    var mdReader = mdFile.reader();

    const loadedId = try! mdReader.readThrough(separator="\n");
    writeln("loadedId ", loadedId);

    var loadedNumLocales: int;
    try! mdReader.read(loadedNumLocales);
    writeln("loadedNumLocales ", loadedNumLocales);
    assert(numLocales == loadedNumLocales);

    return loadedId;
  }

  private proc loadArr(path, mdName, loadedId) throws {
    writeln("Reading ", mdName);
    var mdFile = IO.open(mdName, ioMode.r);
    var mdReader = mdFile.reader();

    const name = try! mdReader.readThrough("\n", stripSeparator=true);
    const size = try! mdReader.readThrough("\n", stripSeparator=true):int;
    const numTargetLocales = try! mdReader.readThrough("\n", stripSeparator=true):int;

    writeln("name ", name);
    writeln("size ", size);
    writeln("numTargetLocales ", numTargetLocales);

    assert(numTargetLocales==1);

    const dataNames: [0..#numTargetLocales] string;

    for name in dataNames {
      const fnSize = try! mdReader.readThrough(" "):int;
      name = try! mdReader.readString(maxSize=fnSize);
    }

    var entryVal = new shared SymEntry(size, int);
    readFilesByName(entryVal.a, dataNames, [size], "asd", 0);

    writeln("Data loaded: ", dataNames);

    mdReader.close();
    mdFile.close();

    return (name, entryVal);
  }

  use CommandMap;
  registerFunction("checkpoint", checkpointMsg, getModuleName());
  registerFunction("loadcheckpoint", checkpointLoadMsg, getModuleName());
}
