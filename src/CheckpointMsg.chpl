module CheckpointMsg {
  use FileSystem;
  use List;
  use ArkoudaJSONCompat;
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

  config const dsetname = "data";

  /* Perform checkpointing automatically when used memory exceeds this many percent
     of available memory. No auto-checkpointing if 0 or below. */
  config const ckptMemPct = 0;

  // Prevent further auto-checkpoints upon exceeding 'ckptMemPct'
  // until memory usage drops below the limit, or upon loadCheckpointMsg.
  private var skipAutoCkpt = false;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const cpLogger = new Logger(logLevel,logChannel);


  private proc saveCheckpointImpl(basePath, cpName, mode, st) throws {
    if !exists(basePath) then
      mkdir(basePath);

    const cpPath = Path.joinPath(basePath, cpName);

    if exists(cpPath) {
      if mode=="error" {
        throw new SaveCheckpointError(
            ("A file already exists in %s. If you want to overwrite, use " +
             "`mode=\"overwrite\"`, which will delete the file").format(cpPath)
        );
      }

      if isDir(cpPath) {
        rmTree(cpPath);
      }
      else {
        remove(cpPath);
      }
    }

    mkdir(cpPath);

    saveServerMetadata(cpPath, st);

    for (name, entry) in zip(st.tab.keys(), st.tab.values()) {
      try {
        var gse = toGenSymEntry(entry);

        // TODO we can expand this
        if gse.ndim != 1 then continue;

        // These types are the only types parquet IO supports right now.
        select gse.dtype {
          when DType.UInt64  do saveArr(cpPath, name, gse: getSEType(uint(64)));
          when DType.Int64   do saveArr(cpPath, name, gse: getSEType(int(64)));
          when DType.Float64 do saveArr(cpPath, name, gse: getSEType(real(64)));
          when DType.Bool    do saveArr(cpPath, name, gse: getSEType(bool));
          otherwise          do continue;
        }
        cpLogger.debug(M(), R(), L(), "Saved entry %s".format(name));
      }
      catch err: ClassCastError {
        // we couldn't build a symentry, not saving this entry
        cpLogger.debug(M(), R(), L(), "Cannot save %s".format(name));
      }
    }
  }

  proc saveCheckpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                         st: borrowed SymTab): MsgTuple throws {
    const basePath = msgArgs.getValueOf("path");
    const nameArg = msgArgs.getValueOf("name");
    const cpName = if nameArg.isEmpty() then st.serverid else nameArg;
    const mode = msgArgs.getValueOf("mode");

    saveCheckpointImpl(basePath, cpName, mode, st);

    return Msg.send(cpName);
  }

  proc loadCheckpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                         st: borrowed SymTab): MsgTuple throws {
    skipAutoCkpt = true;
    var basePath = msgArgs.getValueOf("path");
    const nameArg = msgArgs.getValueOf("name");

    if !exists(basePath) {
      return Msg.error("The base save directory not found: " + basePath);
    }

    const cpPath = Path.joinPath(basePath, nameArg);

    if !exists(cpPath) {
      return Msg.error("The Arkouda save directory not found: " + cpPath);
    }

    cpLogger.debug(M(), R(), L(), "Save directory found: %s".format(cpPath));

    var loadedId: string;
    try {
      loadedId = loadServerMetadata(cpPath);
    }
    catch e: FileNotFoundError {
      return Msg.error("Can't find the server metadata in the saved session.");
    }

    cpLogger.debug(M(), R(), L(), "Metadata loaded");

    var rnames: list((string, ObjType, string));

    // iterate over metadata while loading individual data for each metadata
    for mdName in glob(cpPath+"/*"+metadataExt) {
      // skip the server metadata
      if mdName == Path.joinPath(cpPath, serverMetadataName) then continue;

      cpLogger.debug(M(), R(), L(),
                     "Loading array with metadata %s".format(mdName));
      // load the array (data and metadata)
      var (name, entry) = loadArr(cpPath, mdName, loadedId);

      st.addEntry(name, entry);

      cpLogger.debug(M(), R(), L(),
                     "Loaded array with metadata %s".format(mdName));

    }
    return Msg.send(nameArg);
  }

  private proc needMemBasedCkpt() {
    if ckptMemPct <= 0 then return false;

    // check whether memory use exceeds ckptMemPct
    if (getMemUsed():real / getMemLimit()) > ckptMemPct / 100.0 {
      return if skipAutoCkpt then false else true;
    } else {
      skipAutoCkpt = false;
      return false;
    }
  }

  proc autoCheckpointMsg(cmd: string, msgArgs: borrowed MessageArgs,
                         st: borrowed SymTab): MsgTuple throws {
    select msgArgs.payload {
        when b"idle start" {
          if needMemBasedCkpt() {
            const basePath = ".akdata";
            const cpName = "auto_checkpoint";
            cpLogger.info(M(), R(), L(), "starting autoCheckpoint: memory exceeded %i %%; saving into %s/%s"
                          .format(ckptMemPct, basePath, cpName));
            saveCheckpointImpl(basePath, cpName, "overwrite", st);
            cpLogger.info(M(), R(), L(), "finished autoCheckpoint into %s/%s".format(basePath, cpName));
            skipAutoCkpt = true;
            return MsgTuple.success("autoCheckpoint: completed");
          }
        }
//        when b"idle start" {
//          // nothing for now
//        }
        otherwise {
          cpLogger.debug(M(), R(), L(), "Unrecognized auto_checkpoint command: %?".format(msgArgs));
          return MsgTuple.error("Unrecognized auto_checkpoint command: %?".format(msgArgs));
        }
     }

    return MsgTuple.success("autoCheckpoint: no action taken");
  }

  private proc getSEType(type t) type {
    return borrowed SymEntry(t, dimensions=1);
  }

  private proc saveServerMetadata(path, st: borrowed SymTab) throws {
    const serverMD = new serverMetadata(st.serverid, numLocales);

    const mdName = Path.joinPath(path, serverMetadataName);
    var mdFile = IO.open(mdName, ioMode.cw);
    var mdWriter = mdFile.writer();
    mdWriter.writeln(toJson(serverMD));
  }

  private proc saveArr(path, name, entry) throws {
    const arrMD = new arrayMetadata(name, entry.size,
                                    entry.a.targetLocales().size);

    // write the array
    const dataName = Path.joinPath(path, ".".join(name, dataExt));
    const (warnFlag, filenames, numElems) =
        write1DDistArrayParquet(filename=dataName,
                                dsetname=dsetname,
                                dtype="int64",
                                compression=0,
                                mode=TRUNCATE,
                                A=entry.a);

    cpLogger.debug(M(), R(), L(), "Data created: %s".format(dataName));

    // write the metadata
    const mdName = Path.joinPath(path, ".".join(name, metadataExt));
    var mdFile = IO.open(mdName, ioMode.cw);
    var mdWriter = mdFile.writer();
    mdWriter.writeln(toJson(arrMD));

    for data in zip(filenames, numElems) {
      const metadata = new chunkMetadata((...data));
      mdWriter.writeln(toJson(metadata));
    }

    mdWriter.close();
    mdFile.close();

    cpLogger.debug(M(), R(), L(), "Metadata created: %s".format(mdName));
  }

  private proc loadServerMetadata(path) throws {
    const mdName = Path.joinPath(path, serverMetadataName);

    var mdFile = IO.open(mdName, ioMode.r);
    var mdReader = mdFile.reader();

    const metadata: serverMetadata;
    try {
      metadata = fromJsonThrow(mdReader.readAll(string), serverMetadata);
    }
    catch IllegalArgumentError {
      throw new owned LoadCheckpointError(
          "Server metadata has incorrect format (%s) ".format(mdName)
      );
    }

    if metadata.numLocales!=numLocales {
      throw new owned LoadCheckpointError(
          ("Attempting to load a checkpoint that was made with a different " +
           "number of locales (%i) then the current execution (%i)" +
           "").format(metadata.numLocales, numLocales));
    }

    return metadata.serverid;
  }

  private proc loadArr(path, mdName, loadedId) throws {
    cpLogger.debug(M(), R(), L(), "Reading %s".format(mdName));

    var mdFile = IO.open(mdName, ioMode.r);
    var mdReader = mdFile.reader();

    const metadata: arrayMetadata;
    try {
      metadata = fromJsonThrow(mdReader.readThrough("\n"), arrayMetadata);
    }
    catch IllegalArgumentError {
      throw new owned LoadCheckpointError(
          "Array metadata header has incorrect format (%s) ".format(mdName)
      );
    }

    cpLogger.debug(M(), R(), L(), "Metadata read %s".format(mdName));

    const ref name = metadata.name;
    const ref size = metadata.size;
    const ref numTargetLocales = metadata.numTargetLocales;

    if numTargetLocales!=numLocales {
      throw new owned LoadCheckpointError(
          ("Attempting to load a checkpoint that was made with a different " +
           "number of locales (%i) then the current execution (%i)" +
           "").format(numTargetLocales, numLocales));
    }

    var filenames: [0..#numTargetLocales] string;
    var numElems: [0..#numTargetLocales] int;

    var line: string;
    var curChunk = 0;
    while mdReader.readLine(line) {
      const metadata: chunkMetadata;
      try {
        metadata = fromJsonThrow(line, chunkMetadata);
      }
      catch IllegalArgumentError {
        throw new owned LoadCheckpointError(
            "Array chunk metadata has incorrect format (%s) ".format(mdName)
        );
      }

      filenames[curChunk] = metadata.filename;
      numElems[curChunk] = metadata.numElems;
      curChunk += 1;
    }

    if curChunk != numTargetLocales {
      cpLogger.debug(M(), R(), L(), "Chunk count mismatch, will throw");
      throw new owned LoadCheckpointError(
          ("Array metadata (%s) does not contain correct number of chunks " +
           "(%i found, %i expected).").format(mdName, curChunk,
                                              numTargetLocales)
      );
    }

    cpLogger.debug(M(), R(), L(), "Filenames loaded %s".format(mdName));

    var entryVal = new shared SymEntry(size, int);
    readFilesByName(A=entryVal.a,
                    filenames=filenames,
                    sizes=numElems,
                    dsetname=dsetname,
                    ty=0);

    cpLogger.debug(M(), R(), L(), "Data loaded %s".format(mdName));

    mdReader.close();
    mdFile.close();

    return (name, entryVal);
  }

  use CommandMap;
  registerFunction("save_checkpoint", saveCheckpointMsg, M());
  registerFunction("load_checkpoint", loadCheckpointMsg, M());
  registerFunction("auto_checkpoint", autoCheckpointMsg, M());

  /* Thrown while loading a checkpoint. The cases for this error type is limited
     to the logic of checkpointing itself. Other errors related to IO etc can
     also be thrown during loading. */
  class LoadCheckpointError: Error {
    proc init(msg: string) {
      super.init(msg);
    }
  }

  /* Thrown while saving a checkpoint. The cases for this error type is limited
     to the logic of checkpointing itself. Other errors related to IO etc can
     also be thrown during saving. */
  class SaveCheckpointError: Error {
    proc init(msg: string) {
      super.init(msg);
    }
  }

  /* Representation of server metadata. Created per checkpoint. Will be saved to
     a metadata file in JSON format.
  */
  record serverMetadata {
    const serverid: string;
    const numLocales: int;
  }

  /* Representation of array metadata. Created per array per checkpoint. Will be
     saved to a metadata file in JSON format along with chunk metadata.
  */
  record arrayMetadata {
    const name: string;
    const size: int;
    const numTargetLocales: int;
  }

  /* Representation of array metadata. Created per locale per array per
     checkpoint. Each chunk will result in a JSON object being written to the
     array metadata.
  */
  record chunkMetadata {
    const filename: string;
    const numElems: int;
  }

  // this works around an issue in Chapel's standard fromJson, where some errors
  // cannot be caught. https://github.com/chapel-lang/chapel/pull/26656 will be
  // the fix for that. This helper can be removed or moved to compatibility
  // modules when that fix goes in.
  private proc fromJsonThrow(s: string, type t) throws {
    var fileReader = openStringReader(s, deserializer=new jsonDeserializer());
    var ret: t;
    fileReader.read(ret);
    return ret;
  }

  /* Convenience wrapper for sending messages to the client. */
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
