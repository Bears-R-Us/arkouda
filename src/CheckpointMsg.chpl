module CheckpointMsg {
  use FileSystem;
  use Types, List;
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
  use BigInteger, CTypes, GMP;  // for checkpointing of bigint arrays

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

  private proc checkSymEntryType(entry: SymEntry(?), direction: string) {
    if entry.entryType != SymEntryType then
      cpLogger.error(M(), R(), L(), "unexpected SymEntry.entryType=",
                     entry.entryType:string, ", expected: ", SymEntryType:string,
                     " when ", direction, " an entry with name:", entry.name);
  }

  private proc hasPrimitiveElements(entry) param do
    return isIntegral(entry.etype) || isReal(entry.etype) || isBool(entry.etype);

  override proc SymEntry.saveEntry(name, path, mdName, mdWriter) throws {
    const entry = this;
    checkSymEntryType(entry, "saving");

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

    if hasPrimitiveElements(entry) {
        saveSymEntryPrimitive(entry, name, path, mdName, mdWriter);

    } else if entry.etype == bigint {
        saveSymEntryBigint(entry, name, path, mdName, mdWriter);

    } else {
        cpLogger.debug(M(), R(), L(), cpNotImplementedMsg(" ".join(
          "Saving an array with", entry.etype:string, "element type into"),
          name, entry.entryType:string, mdName));
        return;
    }
  }

  private proc saveSymEntryPrimitive(entry, name, path, mdName, mdWriter) throws {
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

    cpLogger.debug(M(), R(), L(), "Metadata created: ", mdName);
  }

  private proc localeName(baseName, locIdx) do
    return "".join(baseName, ".loc", locIdx:string, ".bin");

  private proc saveSymEntryBigint(entry, name, path, mdName, mdWriter) throws {
    // Metadata is similar to saveSymEntryPrimitive(), except no "chunks".
    const arrayMD = new arrayMetadata(dtype=dtype2str(entry.dtype), size=entry.size,
                                      numTargetLocales=entry.a.targetLocales().size,
                                      ndim=entry.ndim, numChunks=0);
    writeJson(mdWriter, arrayMD, "bigint array metadata", mdName);
    cpLogger.debug(M(), R(), L(), "bigint SymEntry metadata created in ", mdName);

    // Now save each locale's elements.
    const baseName = Path.joinPath(path, name);
    mdWriter.writeln(baseName);
    coforall (loc, locIdx) in zip(entry.a.targetLocales(), 1..) do on loc {
        bigintWriteArray(localeName(baseName, locIdx),
                         entry.a.localSlice[entry.a.localSubdomain()]);
    }
    cpLogger.debug(M(), R(), L(), "bigint SymEntry elements written to ", baseName, '.*');
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

    const entryVal = new shared SymEntry(arrayMD.size, etype);
    entryVal.name = entryMD.entryName;

    if hasPrimitiveElements(entryVal) {
      loadHelperSEPrimitive(entryVal, mdName, arrayMD, mdReader);

    } else if entryVal.etype == bigint {
      loadHelperSEBigint(entryVal, arrayMD, mdReader);

    } else {
      compilerError("load_checkpoint is not implemented for elements of type "
                    + entryVal.etype:string);
    }

    cpLogger.debug(M(), R(), L(), "Data loaded %s".format(mdName));
    checkSymEntryType(entryVal, "loading");
    return entryVal;
  }

  private proc loadHelperSEPrimitive(entryVal, mdName, arrayMD, mdReader) throws {
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

    readFilesByName(A=entryVal.a,
                    filenames=filenames,
                    sizes=numElems,
                    dsetname=dsetname,
                    ty=0);
  }

  private proc loadHelperSEBigint(entry, arrayMD, mdReader) throws {
    mdReader.readLine(string); // consume the newline
    const baseName = mdReader.readLine(string, stripNewline=true);
    coforall (loc, locIdx) in zip(entry.a.targetLocales(), 1..) do on loc {
        bigintReadArray(localeName(baseName, locIdx),
                        entry.a.localSlice[entry.a.localSubdomain()]);
    }
  }

  /******************************************************************/
  /*** saving/loading bigint data ***/
  /******************************************************************/

  /*private*/ type mpz_count_type = uint(32);
  /*private*/ type mpz_numelems_type = uint;

  /* Writes 'count' and 'sgn' (sign) into 'w'.
     'count' is assumed to be non-negative.
     If 'sgn' is 0 then 'count' is recorded as 0 as well.
  */
  private proc bigintWriteCountSgn(w: fileWriter(?),
                                   count: c_size_t, sgn: c_int) throws {
    // Use the highest bit for "negative".
    param sgnBit = (1 << (numBits(mpz_count_type)-1)) : mpz_count_type;
    param countMask = ~sgnBit;
    var toWrite = (count: mpz_count_type) & countMask;

    if (toWrite: uint(64)) != (count: uint(64)) then
      throw new NotImplementedError(
        "checkpointing is not implemented for bigint that need >=2**31 bytes",
         L(), R(), M());

    if sgn == 0 then toWrite = 0;
    else if sgn < 0 then toWrite |= sgnBit;

    w.write(toWrite);
  }

  /* Reads from 'r' into 'countP' and 'sgnP'. Returns false upon EOF. */
  private proc bigintReadCountSgn(r: fileReader(?),
                        ref countP: c_size_t, ref sgnP: c_int): bool throws {
    param sgnBit = (1 << (numBits(mpz_count_type)-1)) : mpz_count_type;
    param countMask = ~sgnBit;
    var count: mpz_count_type;
    if ! r.read(count) then return false;

    countP = count & countMask;
    sgnP = ( if count == 0 then 0 else
               if (count & sgnBit) == 0 then 1 else -1 ): c_int;
    return true;
  }

  // GMP import, export: https://gmplib.org/manual/Integer-Import-and-Export

  private extern proc mpz_import(ref rop: mpz_t, count: c_size_t,
    order: c_int, size: c_size_t, endian: c_int, nails: c_size_t,
    op: c_ptr(void)): void;

  private extern proc mpz_export(rop: c_ptr(void), ref countp: c_size_t,
    order: c_int, size: c_size_t, endian: c_int, nails: c_size_t,
    const ref op: mpz_t): c_ptr(void);

  // could tweak these
  /*private*/ type mpz_imex_elt = uint(8);
  private param imex_order = 1,
                imex_size = numBytes(mpz_imex_elt),
                imex_endian = 0,
                imex_nails = 0,
                imex_capacity_start = 300;  // sufficient for most uses?

  /* 'memBuf' provides resizeable memory for exporting and importing GMP ints.
     This avoids doing alloc+free for each exported/imported bigint.
   */
  /*private*/ record memBuf {
    var dom = { 1:uint..imex_capacity_start:uint };
    var arr: [dom] mpz_imex_elt;
    inline proc ref addr do
      return c_ptrTo(arr);
    proc ref this(i: uint) ref do
      return arr[i];
    proc ref ensureCapacity(numBytes: uint) {
      const currentSize = arr.sizeAs(uint);
      if currentSize >= numBytes then return;
      else dom = { 1..max(currentSize*2, numBytes) };  //resizes 'arr'
    }
  }

  private proc bigintWriteOneNumber(w: fileWriter(?), ref mem: memBuf,
                                    bi: bigint): void throws {
    const ref birep = bi.mpz;
    const sgn = mpz_sgn(birep);
    if sgn == 0 {
      bigintWriteCountSgn(w, 0, 0);
      return;
    }
    const numBits = mpz_sizeinbase(birep, 2);
    const numBytes = (numBits + 7) / 8;
    mem.ensureCapacity(numBytes);

    // "export" the bigint into 'mem'
    var count: c_size_t;
    const edata = mpz_export(mem.addr, count,
                             imex_order, imex_size, imex_endian, imex_nails,
                             birep);
    // now count==numBytes and edata==mem.addr

    // write out the outcome to 'w'
    bigintWriteCountSgn(w, count, sgn);
    for i in 1..count do
      w.writeByte(mem[i]);
  }

  private proc bigintReadOneNumber(r: fileReader(?), ref mem: memBuf,
                                   ref result: bigint): bool throws {
    var count: c_size_t;
    var sgn: c_int;
    if ! bigintReadCountSgn(r, count, sgn) then
      return false;
    if sgn == 0 {
      result = 0; return true;
    }

    mem.ensureCapacity(count);
    for i in 1..count do
      if ! r.readByte(mem[i]) then
        throw new BadFormatError("", "readOneBigint(): expected more bytes");

    ref birep = result.mpz;
    mpz_import(birep, count,
               imex_order, imex_size, imex_endian, imex_nails,
               mem.addr);
    if sgn < 0 then
      mpz_neg(birep, birep);
    return true;
  }

  /* Writes all bigints from 'vals' out to 'fname' in binary,
     after the first line containing their count in plain text.
     Overwrites 'fname' if it exists.
   */
  private proc bigintWriteArray(fname, const vals: [] bigint) throws {
    var mem1: memBuf;
    var ckpt1 = open(fname, ioMode.cw);
    var writer = ckpt1.writer(serializer=new binarySerializer(),
                              locking=false);
    writer.write(vals.sizeAs(mpz_numelems_type));
    // the main loop for writing
    for bi in vals do
      bigintWriteOneNumber(writer, mem1, bi);
    writer.close();
    ckpt1.close();
  }

  /* Reads all bigints from 'fname' into 'vals',
     which must contain the correct number of elements.
   */
  private proc bigintReadArray(fname, ref vals: [] bigint) throws {
    var mem2: memBuf;
    var ckpt2 = open(fname, ioMode.r);
    var reader = ckpt2.reader(deserializer=new binaryDeserializer(),
                              locking=false);
    const count = reader.read(mpz_numelems_type);
    if count != vals.sizeAs(mpz_numelems_type) then
      throw new IllegalArgumentError("".join(
        "count mismatch when reading from ", fname, ": expected ",
        vals.size:string, " bigints while the file contains ",
        count:string, " bigints"));

    // the main loop for reading
    for ci in vals do
      if ! bigintReadOneNumber(reader, mem2, ci) then
        throw new BadFormatError("", "unexpected EOF while reading bigints from " + fname);

    var dummy: uint(8);
    if reader.readByte(dummy) then
      cpLogger.error(M(), R(), L(), "unexpected contents after reading ", count:string, " bigints from ", fname);

    reader.close();
    ckpt2.close();
  }

  /******************************************************************/
  /*** registration and utilities ***/
  /******************************************************************/

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
