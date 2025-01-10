module CheckpointMsg {
  use FileSystem;
  use List;
  import IO, Path;
  import Reflection.{getModuleName as M,
                     getRoutineName as R,
                     getLineNumber as L};

  use ServerErrors, ServerConfig;
  use MultiTypeSymbolTable, MultiTypeSymEntry;
  use Message;
  use ParquetMsg;
  use IOUtils;
  use Logging;

  config param metadataExt = "md";
  config param dataExt = "data";
  config param mdNameMaxLength = 256;
  config param serverMetadataName = "server."+metadataExt;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const cpLogger = new Logger(logLevel,logChannel);


  proc saveCheckpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                         st: borrowed SymTab): MsgTuple throws {
    var basePath = msgArgs.getValueOf("path");
    const nameArg = msgArgs.getValueOf("name");

    if !exists(basePath) then
      mkdir(basePath);

    const cpName = if nameArg.isEmpty() then st.serverid else nameArg;
    const cpPath = Path.joinPath(basePath, cpName);

    if !exists(cpPath) then
      mkdir(cpPath);
    else if isDir(cpPath) then
      rmTree(cpPath);
    else
      remove(cpPath); // warn?

    saveServerMetadata(cpPath, st);

    for (name, entry) in zip(st.tab.keys(), st.tab.values()) {
      var e = toSymEntry(toGenSymEntry(entry), int);
      saveArr(cpPath, name, e);
    }
    return Msg.send(cpName);
  }

  proc loadCheckpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                         st: borrowed SymTab): MsgTuple throws {
    var basePath = msgArgs.getValueOf("path");
    const nameArg = msgArgs.getValueOf("name");

    if !exists(basePath) {
      return Msg.error("The base save directory not found: " + basePath);
    }

    const cpPath = Path.joinPath(basePath, nameArg);

    if !exists(cpPath) {
      return Msg.error("The Arkouda save directory not found: " + cpPath);
    }

    var loadedId: string;
    try {
      loadedId = loadServerMetadata(cpPath);
    }
    catch e: FileNotFoundError {
      return Msg.error("Can't find the server metadata in the saved session.");
    }

    var rnames: list((string, ObjType, string));

    // iterate over metadata while loading individual data for each metadata
    for mdName in glob(cpPath+"/*"+metadataExt) {
      // skip the server metadata
      if mdName == Path.joinPath(cpPath, serverMetadataName) then continue;

      // load the array (data and metadata)
      var (name, entry) = loadArr(cpPath, mdName, loadedId);

      st.addEntry(name, entry);

    }
    return Msg.send(nameArg);
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

    cpLogger.debug(M(), R(), L(), "Data created: %s".format(dataName));

    for f in filenames {
      mdWriter.writef("%i %s\n", f.numCodepoints, f);
    }

    mdWriter.close();
    mdFile.close();

    cpLogger.debug(M(), R(), L(), "Metadata created: %s".format(mdName));
  }

  private proc loadServerMetadata(path) throws {
    const mdName = Path.joinPath(path, serverMetadataName);

    var mdFile = IO.open(mdName, ioMode.r);
    var mdReader = mdFile.reader();

    const loadedId = mdReader.readThrough(separator="\n");

    var loadedNumLocales: int;
    mdReader.read(loadedNumLocales);

    // TODO
    assert(numLocales == loadedNumLocales);

    return loadedId;
  }

  private proc loadArr(path, mdName, loadedId) throws {
    cpLogger.debug(M(), R(), L(), "Reading %s".format(mdName));

    var mdFile = IO.open(mdName, ioMode.r);
    var mdReader = mdFile.reader();

    const name = mdReader.readThrough("\n", stripSeparator=true);
    const size = mdReader.readThrough("\n", stripSeparator=true):int;
    const numTargetLocales = mdReader.readThrough("\n", stripSeparator=true):int;

    // TODO
    assert(numTargetLocales == numLocales);

    const dataNames: [0..#numTargetLocales] string;

    for name in dataNames {
      const fnSize = mdReader.readThrough(" "):int;
      name = mdReader.readString(maxSize=fnSize);
    }

    var entryVal = new shared SymEntry(size, int);
    readFilesByName(entryVal.a, dataNames, [size], "asd", 0);

    cpLogger.debug(M(), R(), L(), "Data loaded %s".format(dataNames));

    mdReader.close();
    mdFile.close();

    return (name, entryVal);
  }

  use CommandMap;
  registerFunction("save_checkpoint", saveCheckpointMsg, M());
  registerFunction("load_checkpoint", loadCheckpointMsg, M());

  module Msg {
    use Message;
    inline proc error(msg: string) {
      return new MsgTuple(msg, MsgType.ERROR);
    }

    inline proc send(msg) {
      return new MsgTuple(msg, MsgType.NORMAL);
    }
  }
}
