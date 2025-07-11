module CheckpointMsg {
  use FileSystem;
  use List, Random;
  use ArkoudaJSONCompat;
  import IO, Path, Time;
  import Reflection.{getModuleName as M,
                     getRoutineName as R,
                     getLineNumber as L};
  import Reflection.getNumFields;

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


  private param SymEntryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
  private param GenEntryType = SymbolEntryType.GeneratorSymEntry;

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
      if name != entry.name then
        cpLogger.error(M(), R(), L(), "SymTab name ", name,
                       " differs from entry.name ", entry.name, ".");

      const entryMD = new entryMetadata(name, entry.entryType:string);
      const mdName = Path.joinPath(cpPath, ".".join(name, metadataExt));
      var mdWriter = IO.open(mdName, ioMode.cw).writer(locking=false);
      writeJson(mdWriter, entryMD, "entry metadata", mdName);

      // actual work is done here
      entry.saveEntry(name, cpPath, mdName, mdWriter);
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
      return Msg.error("The base checkpoint directory not found: " + basePath);
    }

    const cpPath = Path.joinPath(basePath, nameArg);

    if !exists(cpPath) {
      return Msg.error("The checkpoint directory not found: " + cpPath);
    }

    loadServerMetadata(cpPath);

    // iterate over metadata files, loading an entry for each metadata
    for mdName in glob(cpPath+"/*"+metadataExt) {
      // skip the server metadata
      if mdName == Path.joinPath(cpPath, serverMetadataName) then continue;

      cpLogger.debug(M(), R(), L(), "Loading entry metadata from ", mdName);
      var mdReader = IO.open(mdName, ioMode.r).reader(locking=false);
      const entryMD = readJson(mdReader, entryMetadata, "entry metadata", mdName);

      const entry: shared AbstractSymEntry;
      select entryMD.entryType {
        when SymEntryType:string do
          entry = loadSymEntry(mdName, entryMD, mdReader);
        when GenEntryType:string do
          entry = loadGenerator(mdName, entryMD, mdReader);
        otherwise do
          // we should be able to load everything we saved
          throw new NotImplementedError(cpNotImplementedMsg("Loading",
            entryMD.entryName, entryMD.entryType, mdName), L(), R(), M());
      }

      if entry.name.isEmpty() then
        throw new LoadCheckpointError("Entry in " + mdName + " has empty name");

      st.addEntry(entry.name, entry);
      cpLogger.debug(M(), R(), L(), "Added entry with metadata ", mdName);
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

  private proc saveServerMetadata(path, st: borrowed SymTab) throws {
    const serverMD = new serverMetadata(st.serverid, numLocales);
    const mdName = Path.joinPath(path, serverMetadataName);
    writeJson(mdName, serverMD, "server metadata");
    cpLogger.debug(M(), R(), L(), "Server metadata saved into ", mdName);
  }

  private proc loadServerMetadata(path) throws {
    const mdName = Path.joinPath(path, serverMetadataName);
    cpLogger.debug(M(), R(), L(), "Loading server metadata from ", mdName);
    const serverMD = readJson(mdName, serverMetadata, "server metadata");

    if serverMD.numLocales != numLocales {
      throw new owned LoadCheckpointError(
          ("Attempting to load a checkpoint that was made with a different " +
           "number of locales (%i) than the current execution (%i)" +
           "").format(serverMD.numLocales, numLocales));
    }
  }

  private proc cpNotImplementedMsg(direction, entryName, entryType, mdName) {
    return " ".join(direction, "a checkpoint is not implemented for entry",
                    entryName, "which is a", entryType, "with metadata file", mdName);
  }

  // to be implemented by concrete subclasses
  proc AbstractSymEntry.saveEntry(name, path, mdName, mdWriter) throws {
    // this is not an error at the moment.
    cpLogger.debug(M(), R(), L(), cpNotImplementedMsg("Saving",
                          name, this.entryType:string, mdName));
  }

  private proc checkEntryType(entry: borrowed AbstractSymEntry,
                              expected: SymbolEntryType, direction: string) {
    if entry.entryType != expected then
      cpLogger.error(M(), R(), L(), "unexpected entry.entryType=",
                     entry.entryType:string, ", expected: ", expected:string,
                     " when ", direction, " an entry with name:", entry.name);
  }

  /******************************************************************/
  /*** SymEntry / PrimitiveTypedArraySymEntry aka SymEntryType ***/
  /******************************************************************/

  override proc SymEntry.saveEntry(name, path, mdName, mdWriter) throws {
    const entry = this;
    checkEntryType(entry, SymEntryType, "saving");

    // SymEntry.saveEntry() will be instantiated by the compiler for each
    // (etype, dimensions) combination that the program can create.
    // So we just need to check whether our implementation can handle
    // the given instantiation. Cf. the comment for loadSymEntry().

    if entry.dimensions > 1 {
        cpLogger.debug(M(), R(), L(), cpNotImplementedMsg(" ".join(
          "Saving an array with >1 dimensions into"),
          name, entry.entryType:string, mdName));
        return;
    }
    proc isSupportedEtype(type t) do return isIntegral(t) || isReal(t) || isBool(t);
    if ! isSupportedEtype(entry.etype) {
        cpLogger.debug(M(), R(), L(), cpNotImplementedMsg(" ".join(
          "Saving an array with", entry.etype:string, "element type into"),
          name, entry.entryType:string, mdName));
        return;
    }

    // write the array
    const dataName = Path.joinPath(path, ".".join(name, dataExt));
    const (warnFlag, filenames, numElems) =
        write1DDistArrayParquet(filename=dataName,
                                dsetname=dsetname,
                                dtype=type2str(entry.etype),
                                compression=0,
                                mode=TRUNCATE,
                                A=entry.a);

    cpLogger.debug(M(), R(), L(), "Data created: %s".format(dataName));

    const arrayMD = new arrayMetadata(dtype=dtype2str(entry.dtype), size=entry.size,
                                      numTargetLocales=entry.a.targetLocales().size,
                                      ndim=entry.ndim, numChunks=numElems.size);
    writeJson(mdWriter, arrayMD, "array metadata", mdName);

    for data in zip(filenames, numElems) {
      const chunkMD = new chunkMetadata((...data));
      writeJson(mdWriter, chunkMD, "chunk metadata", mdName);
    }

    cpLogger.debug(M(), R(), L(), "SymEntry metadata created in", mdName);
  }

  // load and return a SymEntry / PrimitiveTypedArraySymEntry
  private proc loadSymEntry(mdName, entryMD, mdReader) : shared GenSymEntry throws {
    cpLogger.debug(M(), R(), L(), "Reading %s".format(mdName));

    const arrayMD = readJson(mdReader, arrayMetadata, "array metadata", mdName);
    cpLogger.debug(M(), R(), L(), "Metadata read %s".format(mdName));

    const numTargetLocales = arrayMD.numTargetLocales;

    if numTargetLocales!=numLocales {
      throw new owned LoadCheckpointError(
          ("Attempting to load a checkpoint that was made with a different " +
           "number of locales (%i) then the current execution (%i)" +
           "").format(numTargetLocales, numLocales));
    }

    // Below we instantiate loadHelperSE() for each (dtype, ndim) combination
    // provided by the registration framework. This may not cover all the
    // combinations for which `SymEntry.saveEntry()`is instantiated.

    for param dimIdx in 0..arrayDimensionsTy.size-1 do
      if arrayMD.ndim == arrayDimensionsTy[dimIdx].size then
        for param eltyIdx in 0..arrayElementsTy.size-1 do
          if arrayMD.dtype == type2str(arrayElementsTy[eltyIdx]) then
            return loadHelperSE(mdName, entryMD, arrayMD, mdReader,
              arrayDimensionsTy[dimIdx].size, arrayElementsTy[eltyIdx]);

    // we should be able to load everything we saved
    throw new NotImplementedError(cpNotImplementedMsg(" ".join(
      "Loading an array with", arrayMD.ndim:string,
      "dimensions and", arrayMD.dtype, "element type from"),
      entryMD.entryName, entryMD.entryType, mdName), L(), R(), M());
  }

  private proc loadHelperSE(mdName, entryMD, arrayMD, mdReader,
                            param ndim, type etype) : shared GenSymEntry throws {
    if ndim > 1 then
      throw new NotImplementedError(cpNotImplementedMsg(" ".join(
      "Loading an array with >1 dimensions from"),
      entryMD.entryName, entryMD.entryType, mdName), L(), R(), M());

    const numTargetLocales = arrayMD.numTargetLocales;
    var filenames: [0..#numTargetLocales] string;
    var numElems: [0..#numTargetLocales] int;

    for curChunk in 0..#arrayMD.numChunks {
      const metadata = readJson(mdReader, chunkMetadata,
                         "chunk " + (curChunk+1):string + " metadata", mdName);
      filenames[curChunk] = metadata.filename;
      numElems[curChunk] = metadata.numElems;
    }

    cpLogger.debug(M(), R(), L(), "Filenames loaded %s".format(mdName));

    const entryVal = new shared SymEntry(arrayMD.size, etype);
    entryVal.name = entryMD.entryName;
    readFilesByName(A=entryVal.a,
                    filenames=filenames,
                    sizes=numElems,
                    dsetname=dsetname,
                    ty=0);

    cpLogger.debug(M(), R(), L(), "Data loaded %s".format(mdName));
    checkEntryType(entryVal, SymEntryType, "loading");
    return entryVal;
  }

  /******************************************************************/
  /*** GeneratorSymEntry aka GenEntryType ***/
  /******************************************************************/

  //
  // This helper record is binary-serializable "out of the box".
  // Cf. `randomStream` has a serializer so the compiler is "hands off"
  // and does not create all the needed methods to de/serialize it properly.
  // See https://github.com/chapel-lang/chapel/issues/27363
  //
  /* private */ record generatorSerializer
  {
    type rngsType;    // the type of randomStream.PCGRandomStreamPrivate_rngs
    var state: int;
    // fields of randomStream
    var seed: int;
    var rngs: rngsType;
    var count: int(64);
  }

  private proc rngsTypeForEtype(type eltType) type {
    var rs = new randomStream(eltType);
    return rs.PCGRandomStreamPrivate_rngs.type;
  }

  private proc toSerializable(gen: GeneratorSymEntry(?)) {
    const ref rs = gen.generator;

    // A failure in the following asserts is a signal to update the code
    // to changes in `randomStream` and/or `GeneratorSymEntry`.
    // randomStream: eltType, seed, rngs, count
    compilerAssert(getNumFields(rs.type) == 4);
    // GeneratorSymEntry: etype, generator, state
    compilerAssert(getNumFields(gen.type) == 3);

    return new generatorSerializer(
      rngsType = rngsTypeForEtype(rs.eltType),
      state    = gen.state,
      seed     = rs.seed,
      rngs     = rs.PCGRandomStreamPrivate_rngs,
      count    = rs.PCGRandomStreamPrivate_count);
  }

  override proc GeneratorSymEntry.saveEntry(name, path, mdName, mdWriter) throws {
    const entry = this;
    checkEntryType(entry, GenEntryType, "saving");
    mdWriter.writeln(entry.etype:string);
    const data = toSerializable(entry);
    mdWriter.withSerializer(binarySerializer).write(data);
    mdWriter.writeln();
    cpLogger.debug(M(), R(), L(), "GeneratorSymEntry metadata and data created in", mdName);
  }

  // Is it OK to create a randomStream of this type?
  private proc isOkGeneratorType(type etype) param do
    return isNumericType(etype) || isBoolType(etype);

  private proc loadGenerator(mdName, entryMD, mdReader) : shared AbstractSymEntry throws {
    const etypeStr = mdReader.read(string);
    mdReader.readThrough("\n");

    for param eltyIdx in 0..arrayElementsTy.size-1 {
      type etype = arrayElementsTy[eltyIdx];
      if isOkGeneratorType(etype) &&
         etypeStr == etype:string then
        return loadHelperGen(mdName, entryMD, mdReader, arrayElementsTy[eltyIdx]);
    }

    throw new NotImplementedError(cpNotImplementedMsg(" ".join(
      "Loading a generator with elements of type", etypeStr, "from"),
      entryMD.entryName, entryMD.entryType, mdName), L(), R(), M());
  }

  private proc loadHelperGen(mdName, entryMD, mdReader, type etype) throws {
    type deserType = generatorSerializer(rngsType = rngsTypeForEtype(etype));
    const data = mdReader.withDeserializer(binaryDeserializer).read(deserType);
    var rs = new randomStream(etype, data.seed);
    rs.PCGRandomStreamPrivate_rngs = data.rngs;
    rs.PCGRandomStreamPrivate_count = data.count;

    const entryVal = new shared GeneratorSymEntry(rs, data.state);
    entryVal.name = entryMD.entryName;
    cpLogger.debug(M(), R(), L(), "Generator data loaded from ", mdName);
    checkEntryType(entryVal, GenEntryType, "loading");
    return entryVal;
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

  /* Entry metadata: name and entryType. Saved as the first line
     in each entry's metadata type. */
  record entryMetadata {
    const entryName: string;
    const entryType: string;
  }

  /* Representation of array metadata. Created per array per checkpoint. Will be
     saved to a metadata file in JSON format along with chunk metadata.
  */
  record arrayMetadata {
    const dtype: string;
    const size: int;
    const ndim: int;
    const numTargetLocales: int;
    const numChunks: int;
  }

  /* Representation of array metadata. Created per locale per array per
     checkpoint. Each chunk will result in a JSON object being written to the
     array metadata.
  */
  record chunkMetadata {
    const filename: string;
    const numElems: int;
  }

  // Write 'data' to 'w' in JSON, followed by a newline, by convention.
  private proc writeJson(w: fileWriter(?), data,
                        description: string, mdName: string) throws {
    w.withSerializer(jsonSerializer).write(data);
    w.writeln();
  }

  // Write 'data' to file 'mdName' in JSON, followed by a newline.
  private proc writeJson(mdName: string, data, description: string) throws {
    writeJson(IO.open(mdName, ioMode.cw).writer(locking=false),
              data, description, mdName);
  }

  // Read JSON data of type 'resultType' from 'r'.
  // Apply the workaround in https://github.com/chapel-lang/chapel/pull/26656
  private proc readJson(r: fileReader(?), type resultType,
                        description: string, mdName: string) throws {
    var result: resultType;
    var ok: bool;
    try {
      ok = r.withDeserializer(jsonDeserializer).read(result);
      cpLogger.debug(M(), R(), L(), "readJson -> ",
        if ok then "%s=%?".format(resultType:string, result) else "failure");
    }
    catch e {
      throw new LoadCheckpointError("".join("invalid ", description, " in ",
                                            mdName, ": ", e.message()));
    }
    if !ok {
      throw new LoadCheckpointError("".join("could not read ", description,
                                            " from ", mdName));
    }
    return result;
  }

  // Read JSON data of type 'resultType' from file 'mdName',
  // assuming it is located at start of the file.
  private proc readJson(mdName: string, type resultType, description: string) throws {
    return readJson(IO.open(mdName, ioMode.r).reader(locking=false),
                    resultType, description, mdName);
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
