module CheckpointMsg {
  use FileSystem;
  use List;
  use ArkoudaJSONCompat;
  import IO, Path, Time;
  import Reflection.{getModuleName as M,
                     getRoutineName as R,
                     getLineNumber as L};

  use ServerErrors, ServerConfig;
  import ServerDaemon.DefaultServerDaemon;
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

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const cpLogger = new Logger(logLevel,logChannel);

  /* Perform checkpointing automatically when memory usage exceeds this many percent
     of available memory. By default / if <=0, auto-checkpointing is not triggered by memory usage. */
  config const checkpointMemPct = 0;

  /* When memory exceeds the `checkpointMemPct` threhsold, wait for this many seconds
     of idle time before checkpointing. By default / if <=0, uses 5 seconds. */
  config var checkpointMemPctDelay = 0;

  /* Perform checkpointing automatically when the server is idle for this many seconds.
     By default / if <=0, auto-checkpointing is not triggered by idle time. */
  config const checkpointIdleTime = 0;

  /* The auto-checkpoint implementation will wake up every this many seconds to check
     if a checkpoint needs to be saved due to memory usage or idle time.
     By default / if <=0, uses min(checkpointMemPctDelay,checkpointIdleTime). */
  config var checkpointCheckInterval = 0;

  // The smaller of the active delays, 0 if no checking is requested.
  private var minRequestedDelay = 0;

  // The biggest of the active delays.
  private var maxRequestedDelay = 0;

  /* Automatic checkpointing due to memory usage or idle time will wait for at least this many seconds
     after any completed checkpoint save or load operations, whether automatic or client-initiated.
     By default / if <=0, uses 3600 (one hour). This avoids overly-frequent auto-checkpointing.
     The server does not need not be idle during this interval.
     Checkpoint operations requested by the client are not subject to this delay. */
  config var checkpointInterval = 0;

  /* The directory to save the automatic checkpoint, like the `path` argument of save_checkpoint(). */
  config const checkpointPath = ".akdata";

  /* The name for the automatic checkpoint, like the `name` argument of save_checkpoint(). */
  config const checkpointName = "auto_checkpoint";

  /* The mode for automatic checkpointing, see ak.save_checkpoint(). */
  config const checkpointMode = "preserve_previous";

  // Time stamp of a successful checkpointing operation.
  // Note: this is shared across all daemons.
  private var lastCkptCompletion: atomic real = 0;

  private proc updateLastCkptCompletion() {
    lastCkptCompletion.write(Time.timeSinceEpoch().totalSeconds());
  }


  private proc removeIt(path) throws do
    if isDir(path) then rmTree(path); else remove(path);

  private proc renameIt(from, to) throws do
    if isDir(from) then moveDir(from, to); else rename(from, to);


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

      if mode=="preserve_previous" {
        const prevPath = Path.joinPath(basePath, cpName + ".prev");
        if exists(prevPath)
          then removeIt(prevPath);
        renameIt(cpPath, prevPath);
      }
      else {
        removeIt(cpPath);
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

    updateLastCkptCompletion();
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
    updateLastCkptCompletion();
    return Msg.send(nameArg);
  }

  // the smaller of those argument that are positive
  private proc minOfPos(x, y: x.type) {
    if x <= 0 && y <= 0 then return 0: x.type;
    if x <= 0 then return y;
    if y <= 0 then return x;
    return min(x, y);
  }

  // returns true if the daemon was started
  proc startAsyncCheckpointDaemon(sd: borrowed DefaultServerDaemon) {
    if checkpointMemPct <= 0 && checkpointIdleTime <= 0 {
      cpLogger.info(M(), R(), L(), "asynchronous checkpointing was not requested");
      return false;
    }

    // tidy up the delays and 'minRequestedDelay'

    if checkpointMemPct <= 0 then checkpointMemPctDelay = 0;
    else if checkpointMemPctDelay <= 0 then checkpointMemPctDelay = 5;

    if checkpointInterval <= 0 then checkpointInterval = 3600;

    minRequestedDelay = minOfPos(checkpointMemPctDelay, checkpointIdleTime);

    if checkpointCheckInterval <= 0 then checkpointCheckInterval = minRequestedDelay;

    maxRequestedDelay = max(checkpointInterval, checkpointCheckInterval,
                            checkpointIdleTime, checkpointMemPctDelay);

    // start the asynchronous task
    begin asyncCheckpointDaemon(sd);

    cpLogger.info(M(), R(), L(), "started the asynchronous checkpointing daemon");
    return true;
  }

  //
  // This function runs asynchronously. It creates automatic checkpoints.
  //
  private proc asyncCheckpointDaemon(sd: borrowed DefaultServerDaemon) {
    var idleStartForLastCheckpoint: real = 0;
    var idleStartForLastMemCheck: real = 0;
    var delay: real = minRequestedDelay;

    // The only way to end this task is exit() in DefaultServerDaemon.shutdown().
    while true {
      try {
        Time.sleep(max(delay, checkpointCheckInterval));

        // Check and checkpoint within the same mutex region
        // to avoid the server firing up when we are about to checkpoint.
        sd.activityMutex.writeEF("async checkpointing");
        defer { sd.activityMutex.readFE(); }

        const curTime = Time.timeSinceEpoch().totalSeconds();
        const lastCkpt = lastCkptCompletion.read();

        if lastCkpt > 0 && curTime < lastCkpt + checkpointInterval {
          // There was a recent checkpoint activity. Wait.
          delay = lastCkpt + checkpointInterval - curTime;
          continue;
        }

        const idleStart = sd.idlePeriodStart.read();
        if idleStart == 0 {
          // The server is not idle. Do not checkpoint now.
          delay = minRequestedDelay;
          continue;
        }

        if idleStart == idleStartForLastCheckpoint {
          // There has been no action on the server since we last checkpointed.
          delay = minRequestedDelay;
          continue;
        }

        const idleTime = curTime - idleStart;
        if idleTime < minRequestedDelay {
          // The server has not been idle long enough. Check back later.
          delay = minRequestedDelay - idleTime;
          continue;
        }

        if ! sd.seenNotableActivity.read() {
          // There has been no action on the server since it started.
          delay = minRequestedDelay;
          continue;
        }

        const ckptReason = needToCheckpoint(idleTime, idleStart, idleStartForLastMemCheck);
        if ! ckptReason.isEmpty() {
          // Save a checkpoint.
          const ref basePath = checkpointPath;
          const ref cpName = checkpointName;
          cpLogger.info(M(), R(), L(), "starting autoCheckpoint: " + ckptReason +
                        "; saving into " + basePath + "/" + cpName);

          saveCheckpointImpl(basePath, cpName, checkpointMode, sd.st);

          idleStartForLastCheckpoint = idleStart;
          cpLogger.info(M(), R(), L(), "finished autoCheckpoint into " +
                        basePath + "/" + cpName);

          // Fulfill all waiting periods before a next checkpoint.
          delay = maxRequestedDelay;
          continue;
        }

        // Nothing came up. Check back later.
        delay = checkpointCheckInterval;

      } catch err {
          cpLogger.error(M(), R(), L(), err.message());
      }
    }
  }

  private proc needToCheckpoint(idleTime, idleStart, ref idleStartForLastMemCheck) {
    if checkpointIdleTime > 0 && idleTime >= checkpointIdleTime then
      return "server was idle for over " + checkpointIdleTime:string + " seconds";

    if checkpointMemPctDelay > 0 && idleTime >= checkpointMemPctDelay &&
        // no need to check if no server action since we last checked
        idleStart > idleStartForLastMemCheck {
      idleStartForLastMemCheck = idleStart;
      // check whether memory use exceeds checkpointMemPct
      if (getMemUsed():real / getMemLimit()) >= checkpointMemPct / 100.0 then
        return "memory usage exceeded " + checkpointMemPct:string + "%";
    }

    // no need to checkpoint
    return "";
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
  funStartAsyncCheckpointDaemon = startAsyncCheckpointDaemon;

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
