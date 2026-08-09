module ParquetMsg {
  use IO;
  use ServerErrors, ServerConfig;
  use FileIO;
  use FileSystem;
  use GenSymIO;
  use List;
  use Logging;
  use Message;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use NumPyDType;
  use Sort;
  use AryUtil;
  use CTypes;
  use Map;
  use SegmentedString;
  use IOUtils;
  use ParquetSharedEnums;

  // The core Parquet I/O primitives now live in the standalone Mason `Parquet`
  // package. ParquetMsg delegates to that API and only retains Arkouda-server
  // specific plumbing (the message handlers, the optimized read-all path, and
  // the multi-column writer) plus the few helpers Mason does not provide.
  // These symbols are re-exported so existing importers (e.g. CheckpointMsg)
  // keep resolving them through ParquetMsg.
  public use Parquet only CompressionType, ArrowTypes,
                          TRUNCATE, APPEND, ROWGROUPS,
                          ARROWINT64, ARROWINT32, ARROWUINT64, ARROWUINT32,
                          ARROWBOOLEAN, ARROWSTRING, ARROWFLOAT, ARROWDOUBLE,
                          ARROWLIST, ARROWDECIMAL, ARROWERROR,
                          readFilesByName, readStrFilesByName,
                          readListFilesByName, readColumn, readAllCols,
                          calcListSizesandOffset, calcStrSizesAndOffset,
                          calcStrListSizesAndOffset, getSubdomains,
                          getNullIndices, getStrColSize, getStrListColSize,
                          getArrSize, typeFromCType, typeToCType, getArrType,
                          getListData, getNumCols, getAllTypes,
                          populateTagData, getDatasets, getByteLength,
                          getVersionInfo,
                          write1DDistArrayParquet as masonWrite1DDistArrayParquet,
                          writeStringsColumn, writeListColumn,
                          writeStrListColumn, filesExistForWrite, pqWriteOp;

  // Use reflection for error information
  import Reflection.{getModuleName as getM,
                     getRoutineName as getR,
                     getLineNumber as getL};
  // The C++ Parquet prerequisites (headers and objects) are supplied on the
  // `chpl` command line by scripts/get_parquet_package.sh via the Makefile, so
  // no `require` statements are needed here.

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const pqLogger = new Logger(logLevel, logChannel);

  // Undocumented for now, just for internal experiments
  private config const batchSize = getEnvInt("ARKOUDA_SERVER_PARQUET_BATCH_SIZE", 8192);

  class ParquetReadError: ErrorWithContext {
    proc init(msg: string, moduleName: string, routineName: string,
              lineNumber: int(64)) {
      super.init(msg, lineNumber, routineName, moduleName,
                 errorClass="ParquetReadError");
    }
  }

  proc processFilelist(Filelist: [] string) throws {
    if Filelist.size == 1 {
      if Filelist[0].strip().size == 0 then
          throw new ParquetReadError("Filelist was empty", getM(), getR(), getL());

      var GlobRes = glob(Filelist[0]);
      pqLogger.debug(getM(), getR(), getL(),
                     "glob expanded %s to %i files".format(Filelist[0],
                                                           GlobRes.size));
      if GlobRes.size == 0 {
          throw new ParquetReadError("The wildcarded filename %s either " +
                                     "corresponds to files inaccessible to " +
                                     "Arkouda server or files of an invalid " +
                                     "format".format(Filelist[0]),
                                     getM(), getR(), getL());
      }

      // Glob returns filenames in weird order. Sort for consistency
      sort(GlobRes);
      return GlobRes;
    } else {
      return Filelist;
    }
  }

  // TODO: do we want to add offset writing for Parquet string writes?
  //       if we do, then we need to add the load offsets functionality
  //       in the string reading function
  proc write1DDistStrings(filename: string, mode: int, dsetName: string,
                          entry: SegStringSymEntry, compression: int) throws {
    var segString = new SegString("", entry);
    ref ss = segString;
    return writeStringsColumn(filename, dsetName, ss.offsets.a, ss.values.a,
                              compression: CompressionType, mode);
  }

  // `dtype` is retained for backward compatibility with existing callers
  // (e.g. CheckpointMsg and pdarray_toParquetMsg); the Mason writer infers the
  // Arrow type from the Chapel array element type, so it is no longer needed.
  proc write1DDistArrayParquet(filename: string, dsetname, dtype, compression, mode, A) throws {
    return masonWrite1DDistArrayParquet(filename, dsetname,
                                        compression: CompressionType, mode, A);
  }

  proc parseListDataset(filenames: [] string, dsetname: string, ty, len: int, sizes: [] int, st: borrowed SymTab) throws {
    var rtnmap: map(string, string) = new map(string, string);
    // len here is our segment size
    var filedom = filenames.domain;
    var seg_sizes = makeDistArray(len, int);
    var listSizes: [filedom] int = calcListSizesandOffset(seg_sizes, filenames, sizes, dsetname);
    var segments = (+ scan seg_sizes) - seg_sizes; // converts segment sizes into offsets
    var sname = st.nextName();
    st.addEntry(sname, createSymEntry(segments));
    rtnmap.add("segments", "created " + st.attrib(sname));

    var vname = st.nextName();
    if ty == ArrowTypes.int64 || ty == ArrowTypes.int32 {
      var values = makeDistArray((+ reduce listSizes), int);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, createSymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.uint64 || ty == ArrowTypes.uint32 {
      var values = makeDistArray((+ reduce listSizes), uint);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, createSymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.double || ty == ArrowTypes.float {
      var values = makeDistArray((+ reduce listSizes), real);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, createSymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.boolean {
      var values = makeDistArray((+ reduce listSizes), bool);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, createSymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.stringArr {
      var entrySeg = createSymEntry((+ reduce listSizes), int);
      var byteSizes = calcStrListSizesAndOffset(entrySeg.a, filenames, listSizes, dsetname);
      entrySeg.a = (+ scan entrySeg.a) - entrySeg.a;

      var entryVal = createSymEntry((+ reduce byteSizes), uint(8));
      readListFilesByName(entryVal.a, sizes, seg_sizes, segments, filenames, byteSizes, dsetname, ty);
      var stringsEntry = assembleSegStringFromParts(entrySeg, entryVal, st);
      rtnmap.add("values", "created %s+created bytes.size %?".format(st.attrib(stringsEntry.name), stringsEntry.nBytes));
    }
    else {
      throw getErrorWithContext(getL(), getM(), getR(), msg="Invalid Arrow Type",
                                errorClass='IllegalArgumentError');
    }
    return formatJson(rtnmap);
  }

  proc readAllColsParquetMsg(cmd: string, msgArgs: borrowed MessageArgs,
                             st: borrowed SymTab): MsgTuple throws {
    var repMsg: string;
    var tagData: bool = msgArgs.get("tag_data").getBoolValue();
    var strictTypes: bool = msgArgs.get("strict_types").getBoolValue();

    var fixedLen = msgArgs.get('fixed_len').getIntValue() + 1;

    var allowErrors: bool = msgArgs.get("allow_errors").getBoolValue(); // default is false
    var hasNonFloatNulls: bool = msgArgs.get("has_non_float_nulls").getBoolValue();
    var nullHandlingArg: string = msgArgs.get("null_handling").getValue();

    pqLogger.debug(getM(),getR(),getL(), "handled args");

    if allowErrors {
        pqLogger.warn(getM(), getR(), getL(), "Allowing file read errors");
    }

    var nullMode: NullMode;
    select nullHandlingArg {
      when "none" { nullMode = NullMode.noNulls; }
      when "only floats" { nullMode = NullMode.onlyFloats; }
      when "all" { nullMode = NullMode.all; }
      otherwise { throw new NotImplementedError(
          "null_handling=%s is not implemented "+
          "in the server".format(nullHandlingArg), getL(), getR(), getM());
      }
    }

    pqLogger.debug(getM(),getR(),getL(), "handled null");

    var nfiles = msgArgs.get("filename_size").getIntValue();
    var filelist: [0..#nfiles] string;

    try {
        filelist = msgArgs.get("filenames").getList(nfiles);
    } catch {
        // limit length of file names to 2000 chars
        var n: int = 1000;
        var jsonfiles = msgArgs.getValueOf("filenames");
        var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
        var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, files);
        pqLogger.error(getM(),getR(),getL(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    var Filenames = processFilelist(filelist);

    
    var fileErrors: list(string);
    var fileErrorCount:int = 0;
    var fileErrorMsg:string = "";
    var Sizes: [Filenames.domain] int;
    var Offsets: [Filenames.domain] int;
    
    var rnames: list((string, ObjType, string));

    var len: int;
    
    pqLogger.debug(getM(),getR(),getL(), "will process sizes");
    for (size, filename, idx) in zip(Sizes, Filenames, Filenames.domain) {
      var hadError = false;
      try {
        size = getArrSize(filename);
        len += size;
        if idx>Filenames.domain.low {
          Offsets[idx] = len-size;
        }
      } catch e : Error {
        // This is only type of error thrown by Parquet
        fileErrorMsg =
            "Other error in accessing file %s: %s".format(filename,
                                                          e.message());
        pqLogger.error(getM(), getR(), getL(), fileErrorMsg);
        hadError = true;

        if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
      }

      // This may need to be adjusted for this all-in-one approach
      if hadError {
        // Keep running total, but we'll only report back the first 10
        if fileErrorCount < 10 {
          fileErrors.pushBack(fileErrorMsg.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip());
        }
        fileErrorCount += 1;
      }
    }

    const numCols = getNumCols(Filenames[0]);

    const ColDomain = {0..#numCols};
    const CTypes: [ColDomain] c_int = getAllTypes(Filenames[0]);

    pqLogger.debug(getM(),getR(),getL(), "will try tagging data");

    // If tagging is turned on, tag the data
    if tagData {
      throw new NotImplementedError("Reading all columns while tagging " +
                                    "data is not implemented.", getL(), getR(), getM());
    }

    var op = new pqReadColOp(Filenames, len, CTypes, Sizes, Offsets,
                             hasNonFloatNulls, nullMode);

    if op.isOptimizable() {
      pqLogger.debug(getM(),getR(),getL(), "doing optimized reads");
      var Entries = op.generateEntries();
      op.readInto(Entries);

      const datasets = getDatasets(Filenames[0]);

      var thrownError: owned Error?;
      for (rawEntry, t, colIdx, colName) in zip(Entries, op.ty,
                                                op.colDom, datasets) {
        var entry = op.postProcess(rawEntry, t, colIdx);

        var valName = st.nextName();
        try {
          // can't throw from a non-inlined iterator, I'll handle this myself
          st.addEntry(valName, entry);
        } catch e {
          thrownError = e;
          break;
        }
        rnames.pushBack((colName, ObjType.PDARRAY, valName));
      }

      if thrownError != nil then throw thrownError;
    }
    else {
      // TODO this is a bad workaround
      // I want to refactor the core of the function we are calling and call
      // that core from here, but I have to gradually get there.
      return readAllParquetMsg(cmd, msgArgs, st);
    }

    repMsg = buildReadAllMsgJson(rnames, false, 0, fileErrors, st);
    pqLogger.debug(getM(),getR(),getL(),repMsg);
    return new MsgTuple(repMsg,MsgType.NORMAL);
  }

  proc validIntType(type t) param {
    return t==int(32) || t==int(64);
  }

  proc validUIntType(type t) param {
    return t==uint(32) || t==uint(64);
  }

  proc validRealType(type t) param {
    return t==real(32) || t==real(64);
  }

  enum pqReadReqKind { oneCol, allCols };

  record pqReadColOp {
    param kind: pqReadReqKind;

    var filesDom = {1..0};
    const filenames: [filesDom] string;
    const len: int;
    const ty; // TODO should probably be an array of colDom
    const dsetname: string;
    const sizes: [filesDom] int;
    const offsets: [filesDom] int;
    const hasNonFloatNulls: bool;
    const nullMode: NullMode;

    var colDom = {1..0};
    var whereNullDom = makeDistDom(len);
    var _whereNull: [colDom][whereNullDom] bool;

    var hasOffsets = false;

    proc init(filenames: [?filesDom] string, len: int, ty: c_int,
              dsetname: string, sizes: [] int, hasNonFloatNulls: bool,
              nullMode: NullMode) {
      this.kind = pqReadReqKind.oneCol;

      this.filesDom = filesDom;
      this.filenames = filenames;
      this.len = len;
      this.ty = ty;
      this.dsetname = dsetname;
      this.sizes = sizes;
      this.hasNonFloatNulls = hasNonFloatNulls;
      this.nullMode = nullMode;

      this.colDom = {0..0};
    }

    proc init(filenames: [?filesDom] string, len: int, ty: [] c_int,
              sizes: [] int, offsets: [] int, hasNonFloatNulls: bool,
              nullMode: NullMode) {
      this.kind = pqReadReqKind.allCols;

      this.filesDom = filesDom;
      this.filenames = filenames;
      this.len = len;
      this.ty = ty;
      this.sizes = sizes;
      this.offsets = offsets;
      this.hasNonFloatNulls = hasNonFloatNulls;
      this.nullMode = nullMode;

      this.colDom = {0..#ty.size};

      this.hasOffsets = true;
    }

    proc isOneCol() param do return kind==pqReadReqKind.oneCol;

    proc ref whereNull ref where isOneCol() do
      return _whereNull[0];

    proc ref whereNull(colIdx) ref where !isOneCol() do
      return _whereNull[colIdx];

    proc type canHandleType(ty: c_int) {
      return ty == ARROWINT64   || ty == ARROWINT32  ||
             ty == ARROWUINT64  || ty == ARROWUINT32 ||
             ty == ARROWDOUBLE  || ty == ARROWFLOAT  ||
             ty == ARROWBOOLEAN ||
             ty == ARROWDECIMAL;
    }

    proc isOptimizable() where kind==pqReadReqKind.allCols {
      return && reduce this.type.canHandleType(this.ty);
    }

    proc generateEntryHelp(t): shared GenSymEntry? throws {
      // Arkouda typically doesn't want to support 32-bit arrays to save
      // compilation time. Even if the type is 32-bits we are creating arrays
      // of 64-bit elements here.
      select t {
        when ARROWINT64   do return createSymEntry(this.len, int(64));
        when ARROWINT32   do return createSymEntry(this.len, int(64));
        when ARROWUINT64  do return createSymEntry(this.len, uint(64));
        when ARROWUINT32  do return createSymEntry(this.len, uint(64));
        when ARROWDOUBLE  do return createSymEntry(this.len, real(64));
        when ARROWFLOAT   do return createSymEntry(this.len, real(64));
        when ARROWBOOLEAN do return createSymEntry(this.len, bool);
        when ARROWDECIMAL do return createSymEntry(this.len, real(64));
        otherwise do
          throw new NotImplementedError("Unexpected column type while " +
                                        "reading parquet file", getL(), getR(), getM());
      }
    }

    proc generateEntry(): shared GenSymEntry? throws
        where kind==pqReadReqKind.oneCol {
      return generateEntryHelp(ty);
    }

    proc generateEntry(): shared GenSymEntry? throws {
      compilerError("generateEntry is called for a parquet operation that " +
                    "generates multiple entries. " +
                    "Did you mean 'generateEntries'?");
    }

    proc generateEntries(): [] shared GenSymEntry? throws
        where kind==pqReadReqKind.allCols {
      var Entries: [this.ty.domain] shared GenSymEntry?;

      for (entry, t) in zip(Entries, this.ty) {
        entry = generateEntryHelp(t);
      }

      return Entries;
    }

    proc generateEntries(): [] shared GenSymEntry? throws {
      compilerError("generateEntries is called for a parquet operation that " +
                    "generates a single entry. " +
                    "Did you mean 'generateEntry'?");
    }

    // read a single column into a single entry
    proc ref readInto(ref e: shared GenSymEntry?) throws 
        where this.kind == pqReadReqKind.oneCol {

      // TODO in the all-col implementation, this is handled at CPP
      const byteLength = if ty == ARROWDECIMAL
                            then getByteLength(filenames[0], dsetname)
                            else -1;

      // lazily allocate the whereNulls array
      if hasNonFloatNulls {
        this.whereNullDom = makeDistDom(this.len);
      }

      var subdoms = getSubdomains(sizes);
      
      // TODO we can always feed offsets to avoid computing here
      var fileOffsets = if this.hasOffsets
                            then offsets
                            else (+ scan sizes) - sizes;

      const ref Dom = getDomain(e);
      coforall loc in Dom.targetLocales() with (ref this) do on loc {
        var locFiles = filenames;
        var locFiledoms = subdoms;
        var locOffsets = fileOffsets;
        
        forall (off, filedom, filename) in zip(locOffsets,
                                               locFiledoms,
                                               locFiles) with (ref this) {
          for locdom in Dom.localSubdomains() {
            const intersection = domain_intersection(locdom, filedom);
            if intersection.size > 0 {
              var whereNullPtr = if hasNonFloatNulls
                                    then c_ptrTo(whereNull[intersection.low])
                                    else nil;
              readColumn(filename=filename,
                         colName=dsetname,
                         ptr=getPtr(e, intersection.low),
                         whereNullPtr=whereNullPtr,
                         numElems=intersection.size,
                         startIdx=intersection.low - off,
                         batchSize=batchSize,
                         byteLength=byteLength,
                         hasNonFloatNulls=hasNonFloatNulls);
            }
          }
        }
      }
    }

    // read all columns into multiple entries
    proc ref readInto(ref e: [] shared GenSymEntry?) throws
        where this.kind == pqReadReqKind.allCols {

      var subdoms = getSubdomains(this.sizes);

      pqLogger.debug(getM(), getR(), getL(),
        "readInto with e.domain=%?, ty=%?, offsets=%?, subdoms=%?, filenames=%?"
        .format(e.domain, this.ty, this.offsets, subdoms, this.filenames));

      coforall loc in Locales with (ref this) do on loc {
        const LocTypes = this.ty;

        const ref DataDom = getDomain(e);

        forall (off, filedom, filename) in zip(this.offsets, subdoms, this.filenames) with (ref this) {
          var CPtrsToData: [e.domain] c_ptr(void);
          var CPtrsToWhereNulls: [e.domain] c_ptr(void);
          for locdom in DataDom.localSubdomains() {
            const intersection = domain_intersection(locdom, filedom);
            if intersection.size > 0 {
              for colIdx in e.domain {
                pqLogger.debug(getM(), getR(), getL(), "Locale " + here.id:string +
                               " will read " + intersection:string, " for column " + colIdx:string);
                pqLogger.debug(getM(), getR(), getL(), "\tCTypes %?\n".format(LocTypes));
                pqLogger.debug(getM(), getR(), getL(), "\tnullMode %?\n".format(nullMode));
                CPtrsToData[colIdx] = getPtr(e=e[colIdx],
                                             off=intersection.low,
                                             dtype=ty[colIdx]);
                if nullMode == NullMode.all {
                  CPtrsToWhereNulls[colIdx] =
                      c_ptrTo(this.whereNull[colIdx][intersection.low]);
                }

              }

              const startIdx = intersection.low - off;
              pqLogger.debug(getM(), getR(), getL(), "about to call c_readAllCols for locale %? with filename=%?, startIdx=%?, ptrs=%?"
                             .format(here.id, filename, startIdx, CPtrsToData));
              readAllCols(filename, CPtrsToData, LocTypes,
                          CPtrsToWhereNulls,
                          numElems=intersection.size,
                          startIdx=intersection.low-off,
                          batchSize=batchSize,
                          nullMode=nullMode:int);
            }
          }
        }
      }
    }


    // TODO this no longer needs to be a method, it can be a private helper
    proc getDomainImpl(e: GenSymEntry?, dtype: c_int): domain(?) throws {
      type SE = borrowed SymEntry(?);

      select dtype {
        when ARROWINT64   do return (e:(SE(int(64) , 1))).a.domain;
        when ARROWINT32   do return (e:(SE(int(64) , 1))).a.domain;
        when ARROWUINT64  do return (e:(SE(uint(64), 1))).a.domain;
        when ARROWUINT32  do return (e:(SE(uint(64), 1))).a.domain;
        when ARROWDOUBLE  do return (e:(SE(real(64), 1))).a.domain;
        when ARROWFLOAT   do return (e:(SE(real(64), 1))).a.domain;
        when ARROWBOOLEAN do return (e:(SE(bool    , 1))).a.domain;
        when ARROWDECIMAL do return (e:(SE(real(64), 1))).a.domain;
        otherwise do
          throw new NotImplementedError("Unexpected column type while " +
                                        "reading parquet file", getL(), getR(), getM());
      }
    }

    inline proc getDomain(e: [] GenSymEntry?): domain(?) throws {
      // just assume that they all have the same domain
      return getDomainImpl(e.first, this.ty.first);
    }

    inline proc getDomain(e: GenSymEntry?): domain(?) throws {
      return getDomainImpl(e, this.ty);
    }

    proc getPtr(e: GenSymEntry?, off: int, dtype: c_int): c_ptr(void) throws {
      type BSE = borrowed SymEntry(?);
      select dtype {
        when ARROWINT64   do return c_ptrTo((e:(BSE(int(64) , 1))).a[off]);
        when ARROWINT32   do return c_ptrTo((e:(BSE(int(64) , 1))).a[off]);
        when ARROWUINT64  do return c_ptrTo((e:(BSE(uint(64), 1))).a[off]);
        when ARROWUINT32  do return c_ptrTo((e:(BSE(uint(64), 1))).a[off]);
        when ARROWDOUBLE  do return c_ptrTo((e:(BSE(real(64), 1))).a[off]);
        when ARROWFLOAT   do return c_ptrTo((e:(BSE(real(64), 1))).a[off]);
        when ARROWBOOLEAN do return c_ptrTo((e:(BSE(bool    , 1))).a[off]);
        when ARROWDECIMAL do return c_ptrTo((e:(BSE(real(64), 1))).a[off]);
        otherwise do
          throw new NotImplementedError("Unexpected column type while " +
                                        "reading parquet file", getL(), getR(), getM());
      }
    }

    inline proc getPtr(e: GenSymEntry?, off: int): c_ptr(void) throws {
      return getPtr(e, off, this.ty);
    }

    // postProcess is called per-entry to handle some special cases
    // Engin: when I first started implementing, there were more of these
    // special cases. I have eliminated all, but null handling. We could
    // consider making integral arrays always floats if the user enabled null
    // handling.
    proc postProcess(e: shared GenSymEntry?): shared AbstractSymEntry throws {
      return postProcess(e, ty, 0);
    }

    proc postProcess(e: shared GenSymEntry?, ty,
                     colIdx): shared AbstractSymEntry throws {
      type SSE = shared SymEntry(?);
      select ty {
        when ARROWINT64   do return _postProcess(e:(SSE(int(64) , 1)), colIdx);
        when ARROWINT32   do return _postProcess(e:(SSE(int(64) , 1)), colIdx);
        when ARROWUINT64  do return _postProcess(e:(SSE(uint(64), 1)), colIdx);
        when ARROWUINT32  do return _postProcess(e:(SSE(uint(64), 1)), colIdx);
        when ARROWDOUBLE  do return _postProcess(e:(SSE(real(64), 1)), colIdx);
        when ARROWFLOAT   do return _postProcess(e:(SSE(real(64), 1)), colIdx);
        when ARROWBOOLEAN do return _postProcess(e:(SSE(bool    , 1)), colIdx);
        when ARROWDECIMAL do return _postProcess(e:(SSE(real(64), 1)), colIdx);
        otherwise do
          throw new NotImplementedError("Unexpected column type while " +
                                        "reading parquet file", getL(), getR(), getM());
      }
    }

    // I wanted to name these `postProcess` but ran into a compiler bug
    // TODO instead of `colIdx`, we can actually pass `whereNull`
    proc _postProcess(se: SymEntry(?t), colIdx) throws where validIntType(t) ||
                                                             validUIntType(t) ||
                                                             t==bool {
      const ref myWhereNull = _whereNull[colIdx];
      const handleNonFloatNulls = if isOneCol()
                                    then hasNonFloatNulls
                                    else nullMode == NullMode.all;
      if handleNonFloatNulls && (|| reduce myWhereNull) {
        // if we have non-float nulls and there's at least one null
        var floatEntry = createSymEntry(se.size, real);
        floatEntry.a = (se.a):real;
        ref fa = floatEntry.a;
        [(t, f) in zip(myWhereNull, fa)] if t then f = nan;

        return floatEntry: AbstractSymEntry;
      }
      else {
        return se: AbstractSymEntry;
      }
    }

    inline proc _postProcess(se: SymEntry(?t), colIdx) throws {
      return se;
    }
  }


  proc readAllParquetMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var repMsg: string;
    var tagData: bool = msgArgs.get("tag_data").getBoolValue();
    var strictTypes: bool = msgArgs.get("strict_types").getBoolValue();

    var fixedLen = msgArgs.get('fixed_len').getIntValue() + 1;

    var allowErrors: bool = msgArgs.get("allow_errors").getBoolValue(); // default is false
    var hasNonFloatNulls: bool = msgArgs.get("has_non_float_nulls").getBoolValue();
    if allowErrors {
        pqLogger.warn(getM(), getR(), getL(), "Allowing file read errors");
    }
    
    var ndsets = msgArgs.get("dset_size").getIntValue();
    var nfiles = msgArgs.get("filename_size").getIntValue();
    var dsetlist: [0..#ndsets] string;
    var filelist: [0..#nfiles] string;

    try {
        dsetlist = msgArgs.get("dsets").getList(ndsets);
    } catch {
        // limit length of dataset names to 2000 chars
        var n: int = 1000;
        var jsondsets = msgArgs.getValueOf("dsets");
        var dsets: string = if jsondsets.size > 2*n then jsondsets[0..#n]+'...'+jsondsets[jsondsets.size-n..#n] else jsondsets;
        var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".format(
                                            ndsets, dsets);
        pqLogger.error(getM(),getR(),getL(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    try {
        filelist = msgArgs.get("filenames").getList(nfiles);
    } catch {
        // limit length of file names to 2000 chars
        var n: int = 1000;
        var jsonfiles = msgArgs.getValueOf("filenames");
        var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
        var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, files);
        pqLogger.error(getM(),getR(),getL(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    var dsetdom = dsetlist.domain;
    var filedom = filelist.domain;
    var dsetnames: [dsetdom] string;
    var filenames: [filedom] string;
    dsetnames = dsetlist;

    if filelist.size == 1 {
      if filelist[0].strip().size == 0 {
          var errorMsg = "filelist was empty.";
          pqLogger.error(getM(),getR(),getL(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
      var tmp = glob(filelist[0]);
      pqLogger.debug(getM(),getR(),getL(),
                            "glob expanded %s to %i files".format(filelist[0], tmp.size));
      if tmp.size == 0 {
          var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
          pqLogger.error(getM(),getR(),getL(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
      // Glob returns filenames in weird order. Sort for consistency
      sort(tmp);
      filedom = tmp.domain;
      filenames = tmp;
    } else {
        filenames = filelist;
    }
    
    var fileErrors: list(string);
    var fileErrorCount:int = 0;
    var fileErrorMsg:string = "";
    var sizes: [filedom] int;
    var types: [dsetdom] ArrowTypes;
    var byteSizes: [filedom] int;
    
    var rnames: list((string, ObjType, string)); // tuple (dsetName, item type, id)

    
    for (dsetidx, dsetname) in zip(dsetdom, dsetnames) {
        types[dsetidx] = getArrType(filenames[0], dsetname);
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                sizes[i] = getArrSize(fname);
            } catch e : Error {
                // This is only type of error thrown by Parquet
                fileErrorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                pqLogger.error(getM(),getR(),getL(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            }

            // This may need to be adjusted for this all-in-one approach
            if hadError {
              // Keep running total, but we'll only report back the first 10
              if fileErrorCount < 10 {
                fileErrors.pushBack(fileErrorMsg.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip());
              }
              fileErrorCount += 1;
            }
        }
        
        var len = + reduce sizes;
        var ty = types[dsetidx];

        // If tagging is turned on, tag the data
        if tagData {
          pqLogger.debug(getM(),getR(),getL(), "Tagging Data with File Code");
          var tagEntry = createSymEntry(len, int);
          populateTagData(tagEntry.a, filenames, sizes);
          var rname = st.nextName();
          st.addEntry(rname, tagEntry);
          rnames.pushBack(("Filename_Codes", ObjType.PDARRAY, rname));
          tagData = false; // turn off so we only run once
        }

        const cty = typeToCType(ty);

        if pqReadColOp.canHandleType(cty) {
          var dummyNullMode: NullMode; // ignored for single col reads for now
          var op = new pqReadColOp(filenames, len, cty, dsetname, sizes,
                                   hasNonFloatNulls, dummyNullMode);
          var entryVal = op.generateEntry();
          var valName = st.nextName();

          op.readInto(entryVal);

          st.addEntry(valName, op.postProcess(entryVal));
          rnames.pushBack((dsetname, ObjType.PDARRAY, valName));
        } else if ty == ArrowTypes.stringArr {
          var entrySeg = createSymEntry(len, int);

          // Calculate byte sizes by reading or fixed length
          if fixedLen < 2 {
            byteSizes = calcStrSizesAndOffset(entrySeg.a, filenames, sizes, dsetname);
          } else {
            entrySeg.a = fixedLen;
            for i in sizes.domain do
              byteSizes[i] = fixedLen*sizes[i];
          }
          entrySeg.a = (+ scan entrySeg.a) - entrySeg.a;

          // Read into distributed array
          var entryVal = new shared SymEntry((+ reduce byteSizes), uint(8));
          readStrFilesByName(entryVal.a, filenames, byteSizes, dsetname);
          
          var stringsEntry = assembleSegStringFromParts(entrySeg, entryVal, st);
          rnames.pushBack((dsetname, ObjType.STRINGS,
                           "%s+%?".format(stringsEntry.name,
                                          stringsEntry.nBytes)));
        } else if ty == ArrowTypes.list {
          var list_ty = getListData(filenames[0], dsetname,
                                    unsupportedAsNotImplemented=true);
          // check for and skip further nested datasets
          if list_ty == ArrowTypes.notimplemented {
            pqLogger.info(getM(),getR(),getL(),
                          "Invalid list datatype found in %s. Skipping.".format(dsetname));
          }
          else {
            var create_str: string = parseListDataset(filenames, dsetname,
                                                      list_ty, len, sizes, st);
            rnames.pushBack((dsetname, ObjType.SEGARRAY, create_str));
          }
        } else {
          var errorMsg = "DType %s not supported for Parquet reading".format(ty);
          pqLogger.error(getM(),getR(),getL(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    repMsg = buildReadAllMsgJson(rnames, false, 0, fileErrors, st);
    pqLogger.debug(getM(),getR(),getL(),repMsg);
    return new MsgTuple(repMsg,MsgType.NORMAL);
  }

  proc pdarray_toParquetMsg(msgArgs: MessageArgs, st: borrowed SymTab): bool throws {
    var mode = msgArgs.get("mode").getIntValue();
    var filename: string = msgArgs.getValueOf("prefix");
    var entry = st[msgArgs.getValueOf("values")];
    var dsetname = msgArgs.getValueOf("dset");
    var dataType = str2dtype(msgArgs.getValueOf("dtype"));
    var dtypestr = msgArgs.getValueOf("dtype");
    var compression = msgArgs.getValueOf("compression").toUpper(): CompressionType;

    if (!entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry)) {
      var errorMsg = "ObjType (PDARRAY) does not match SymEntry Type: %s".format(entry.entryType);
      throw getErrorWithContext(getL(), getM(), getR(), msg=errorMsg,
                                errorClass='TypeError');
    }

    var warnFlag: bool;
    select dataType {
      when DType.Int64 {
        var e = toSymEntry(toGenSymEntry(entry), int);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr,
                                           compression:int, mode, e.a)[0];
      }
      when DType.UInt64 {
        var e = toSymEntry(toGenSymEntry(entry), uint);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr,
                                           compression:int, mode, e.a)[0];
      }
      when DType.Bool {
        var e = toSymEntry(toGenSymEntry(entry), bool);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr,
                                           compression:int, mode, e.a)[0];
      } when DType.Float64 {
        var e = toSymEntry(toGenSymEntry(entry), real);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr,
                                           compression:int, mode, e.a)[0];
      } otherwise {
        var errorMsg = "Writing Parquet files not supported for %s type".format(msgArgs.getValueOf("dtype"));
        pqLogger.error(getM(),getR(),getL(),errorMsg);
        throw getErrorWithContext(getL(), getM(), getR(), msg=errorMsg,
                                  errorClass='DataTypeError');
      }
    }
    return warnFlag;
  }

  proc strings_toParquetMsg(msgArgs: MessageArgs, st: borrowed SymTab): bool throws {
    var mode = msgArgs.get("mode").getIntValue();
    var filename: string = msgArgs.getValueOf("prefix");
    var entry = st[msgArgs.getValueOf("values")];
    var dsetname = msgArgs.getValueOf("dset");
    var dataType = msgArgs.getValueOf("dtype");
    var compression = msgArgs.getValueOf("compression").toUpper(): CompressionType;

    if (!entry.isAssignableTo(SymbolEntryType.SegStringSymEntry)) {
      var errorMsg = "ObjType (STRINGS) does not match SymEntry Type: %s".format(entry.entryType);
      throw getErrorWithContext(getL(), getM(), getR(), msg=errorMsg,
                                errorClass='TypeError');
    }

    var segString:SegStringSymEntry = toSegStringSymEntry(entry);
    var warnFlag: bool = write1DDistStrings(filename, mode, dsetname,
                        segString, compression:int);
    return warnFlag;
  }

  proc writeSegArrayParquet(filename: string, dsetName: string, c_dtype, segments_entry, values_entry, compression: int): bool throws {
    // Delegates to the Mason Parquet package. `c_dtype` is retained for the
    // caller's dispatch but is unused here: writeListColumn infers the Arrow
    // type from the value array's Chapel element type.
    return writeListColumn(filename, dsetName, segments_entry.a,
                           values_entry.a, compression: CompressionType);
  }

  proc writeStrSegArrayParquet(filename: string, dsetName: string, segments_entry, values_entry, compression: int): bool throws {
    // Delegates to the Mason Parquet package. For a SegArray of strings,
    // `segments` indexes into the string offsets, `offsets` into the raw byte
    // values, matching Arkouda's SegString layout.
    return writeStrListColumn(filename, dsetName, segments_entry.a,
                              values_entry.offsetsEntry.a,
                              values_entry.bytesEntry.a,
                              compression: CompressionType);
  }

  proc segarray_toParquetMsg(msgArgs: MessageArgs, st: borrowed SymTab): bool throws {
    var mode = msgArgs.get("mode").getIntValue();
    var filename: string = msgArgs.getValueOf("prefix");
    var entry = st[msgArgs.getValueOf("values")];
    var dsetname = msgArgs.getValueOf("dset");
    var compression = msgArgs.getValueOf("compression").toUpper(): CompressionType;

    // because append has been depreacted, support is not being added for SegArray. 
    if mode == APPEND {
      throw getErrorWithContext(getL(), getM(), getR(),
                                msg="APPEND write mode is not supported for SegArray.",
                                errorClass='WriteModeError');
    }

    // segments is always int64
    var segments = toSymEntry(toGenSymEntry(st[msgArgs.getValueOf("segments")]), int);

    var genVal = toGenSymEntry(st[msgArgs.getValueOf("values")]);
    
    var warnFlag: bool;
    select genVal.dtype {
      when DType.Int64 {
        var values = toSymEntry(genVal, int);
        warnFlag = writeSegArrayParquet(filename, dsetname, ARROWINT64, segments, values, compression:int);
      }
      when DType.UInt64 {
        var values = toSymEntry(genVal, uint);
        warnFlag = writeSegArrayParquet(filename, dsetname, ARROWUINT64, segments, values, compression:int);
      }
      when DType.Bool {
        var values = toSymEntry(genVal, bool);
        warnFlag = writeSegArrayParquet(filename, dsetname, ARROWBOOLEAN, segments, values, compression:int);
      } when DType.Float64 {
        var values = toSymEntry(genVal, real);
        warnFlag = writeSegArrayParquet(filename, dsetname, ARROWDOUBLE, segments, values, compression:int);
      } when DType.Strings {
        var values = toSegStringSymEntry(genVal);
        warnFlag = writeStrSegArrayParquet(filename, dsetname, segments, values, compression:int);
      } otherwise {
        var errorMsg = "Writing Parquet files not supported for %s type".format(genVal.dtype);
        pqLogger.error(getM(),getR(),getL(),errorMsg);
        throw getErrorWithContext(getL(), getM(), getR(), msg=errorMsg,
                                  errorClass='DataTypeError');
      }
    }
    return warnFlag;
  }

  proc toparquetMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var objType: ObjType = msgArgs.getValueOf("objType").toUpper(): ObjType; // pdarray, Strings, SegArray
    
    var warnFlag: bool;
    try {
      select objType {
        when ObjType.PDARRAY {
          // call handler for pdarray write
          warnFlag = pdarray_toParquetMsg(msgArgs, st);
        }
        when ObjType.STRINGS {
          // call handler for strings write
          warnFlag = strings_toParquetMsg(msgArgs, st);
        }
        when ObjType.SEGARRAY {
          // call handler for strings write
          warnFlag = segarray_toParquetMsg(msgArgs, st);
        }
        otherwise {
            var errorMsg = "Unable to write object type %s to Parquet file.".format(objType);
            pqLogger.error(getM(),getR(),getL(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    } catch e: FileNotFoundError {
      var errorMsg = "Unable to open %s for writing: %s".format(msgArgs.getValueOf("filename"),e.message());
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: MismatchedAppendError {
      var errorMsg = "Mismatched append %s".format(e.message());
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: WriteModeError {
      var errorMsg = "Write mode error %s".format(e.message());
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: Error {
      var errorMsg = "problem writing to file %s".format(e.message());
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    if warnFlag {
      var warnMsg: string = "Warning: possibly overwriting existing files matching filename pattern";
      pqLogger.debug(getM(),getR(),getL(),warnMsg);
      return new MsgTuple(warnMsg, MsgType.WARNING);
    } else {
      var repMsg: string = "Dataset written successfully!";
      pqLogger.debug(getM(),getR(),getL(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
  }

  private proc registerParquetColumns(ref op, colNames: [] string,
                                      symNames: [] string,
                                      colObjTypes: [] string,
                                      st: borrowed SymTab) throws {
    for (colName, symName, objTypeName) in
        zip(colNames, symNames, colObjTypes) {
      select objTypeName.toUpper(): ObjType {
        when ObjType.PDARRAY {
          const entry = getGenericTypedArrayEntry(symName, st);
          select entry.dtype {
            when DType.Int64 do
              op.registerColumn(toSymEntry(entry, int).a, colName);
            when DType.UInt64 do
              op.registerColumn(toSymEntry(entry, uint).a, colName);
            when DType.Float64 do
              op.registerColumn(toSymEntry(entry, real).a, colName);
            when DType.Bool do
              op.registerColumn(toSymEntry(entry, bool).a, colName);
            otherwise do
              throw getErrorWithContext(getL(), getM(), getR(),
                  msg="Unsupported PDArray DType for writing to Parquet, " +
                      entry.dtype:string,
                  errorClass='DataTypeError');
          }
        }
        when ObjType.STRINGS {
          const entry = toSegStringSymEntry(st[symName]);
          op.registerStrColumn(entry.offsetsEntry.a, entry.bytesEntry.a,
                               colName);
        }
        when ObjType.SEGARRAY {
          const components = jsonToMap(symName);
          const segments = toSymEntry(
              getGenericTypedArrayEntry(components["segments"], st), int);
          const values = getGenericTypedArrayEntry(components["values"], st);

          select values.dtype {
            when DType.Int64 do
              op.registerListColumn(segments.a,
                                    toSymEntry(values, int).a, colName);
            when DType.UInt64 do
              op.registerListColumn(segments.a,
                                    toSymEntry(values, uint).a, colName);
            when DType.Float64 do
              op.registerListColumn(segments.a,
                                    toSymEntry(values, real).a, colName);
            when DType.Bool do
              op.registerListColumn(segments.a,
                                    toSymEntry(values, bool).a, colName);
            when DType.Strings {
              const strings = toSegStringSymEntry(values);
              op.registerStrListColumn(segments.a, strings.offsetsEntry.a,
                                       strings.bytesEntry.a, colName);
            }
            otherwise do
              throw getErrorWithContext(getL(), getM(), getR(),
                  msg="Unsupported SegArray DType for writing to Parquet, " +
                      values.dtype:string,
                  errorClass='DataTypeError');
          }
        }
        otherwise do
          throw getErrorWithContext(getL(), getM(), getR(),
              msg="Writing Parquet files does not support " +
                  objTypeName + " columns.",
              errorClass='DataTypeError');
      }
    }
  }

  private proc writeMultiColWithDomain(filename: string,
                                       colNames: [] string,
                                       symNames: [] string,
                                       colObjTypes: [] string,
                                       compression: int,
                                       st: borrowed SymTab,
                                       const ref sharedDom) throws {
    const filesExist = filesExistForWrite(
        filename, sharedDom.targetLocales().size, TRUNCATE);

    var op = new pqWriteOp(filename, sharedDom);
    op.compression = compression;
    op.distributed = true;
    registerParquetColumns(op, colNames, symNames, colObjTypes, st);
    op.write();
    return filesExist;
  }

  private proc writeMultiColWithMason(filename: string,
                                      colNames: [] string,
                                      symNames: [] string,
                                      colObjTypes: [] string,
                                      compression: int,
                                      st: borrowed SymTab) throws {
    const firstName = symNames[symNames.domain.low];
    select colObjTypes[colObjTypes.domain.low].toUpper(): ObjType {
      when ObjType.STRINGS {
        const first = toSegStringSymEntry(st[firstName]);
        return writeMultiColWithDomain(filename, colNames, symNames,
                                       colObjTypes, compression, st,
                                       first.offsetsEntry.a.domain);
      }
      when ObjType.SEGARRAY {
        const components = jsonToMap(firstName);
        const first = toSymEntry(
            getGenericTypedArrayEntry(components["segments"], st), int);
        return writeMultiColWithDomain(filename, colNames, symNames,
                                       colObjTypes, compression, st,
                                       first.a.domain);
      }
      when ObjType.PDARRAY {
        const first = getGenericTypedArrayEntry(firstName, st);
        select first.dtype {
          when DType.Int64 do
            return writeMultiColWithDomain(filename, colNames, symNames,
                colObjTypes, compression, st,
                toSymEntry(first, int).a.domain);
          when DType.UInt64 do
            return writeMultiColWithDomain(filename, colNames, symNames,
                colObjTypes, compression, st,
                toSymEntry(first, uint).a.domain);
          when DType.Float64 do
            return writeMultiColWithDomain(filename, colNames, symNames,
                colObjTypes, compression, st,
                toSymEntry(first, real).a.domain);
          when DType.Bool do
            return writeMultiColWithDomain(filename, colNames, symNames,
                colObjTypes, compression, st,
                toSymEntry(first, bool).a.domain);
          otherwise do
            throw getErrorWithContext(getL(), getM(), getR(),
                msg="Unsupported PDArray DType for writing to Parquet, " +
                    first.dtype:string,
                errorClass='DataTypeError');
        }
      }
      otherwise do
        throw getErrorWithContext(getL(), getM(), getR(),
            msg="Writing Parquet files does not support " +
                colObjTypes[colObjTypes.domain.low] + " columns.",
            errorClass='DataTypeError');
    }
    return false;
  }

  proc toParquetMultiColMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    const filename: string = msgArgs.getValueOf("filename");
    const ncols: int = msgArgs.get("num_cols").getIntValue();

    // get list of the names for the columns
    const col_names: [0..#ncols] string = msgArgs.get("col_names").getList(ncols);

    // get list of sym entry names holding column data
    const sym_names: [0..#ncols] string = msgArgs.get("columns").getList(ncols); // note SegArrays will be JSON

    // get list of objTypes for the names 
    const col_objType_strs: [0..#ncols] string = msgArgs.get("col_objtypes").getList(ncols);

    // compression format as integer
    const compression = msgArgs.getValueOf("compression").toUpper(): CompressionType;

    var warnFlag: bool;
    try {
      warnFlag = writeMultiColWithMason(filename, col_names, sym_names,
                                        col_objType_strs, compression:int, st);
    } catch e: FileNotFoundError {
      var errorMsg = "Unable to open %s for writing: %s".format(filename,e.message());
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: WriteModeError {
      var errorMsg = "Write mode error %s".format(e.message());
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: Error {
      var errorMsg = "problem writing to file %s".format(e.message());
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    if warnFlag {
      var warnMsg = "Warning: possibly overwriting existing files matching filename pattern";
      return new MsgTuple(warnMsg, MsgType.WARNING);
    } else {
      var repMsg = "File written successfully!";
      pqLogger.debug(getM(),getR(),getL(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
  }

  proc lspqMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    // reqMsg: "lshdf [<json_filename>]"
    var repMsg: string;

    // determine if read nested flag is set
    var read_nested: bool = msgArgs.get("read_nested").getBoolValue();

    // Retrieve filename from payload
    var filename: string = msgArgs.getValueOf("filename");
    if filename.isEmpty() {
      var errorMsg = "Filename was Empty";
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    // If the filename represents a glob pattern, retrieve the locale 0 filename
    if isGlobPattern(filename) {
      // Attempt to interpret filename as a glob expression and ls the first result
      var tmp = glob(filename);

      if tmp.size <= 0 {
        var errorMsg = "Cannot retrieve filename from glob expression %s, check file name or format".format(filename);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
            
      // Set filename to globbed filename corresponding to locale 0
      filename = tmp[tmp.domain.first];
    }
        
    // Check to see if the file exists. If not, return an error message
    if !exists(filename) {
      var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filename);
      return new MsgTuple(errorMsg,MsgType.ERROR);
    }
        
    try {
      repMsg = formatJson(getDatasets(filename, readNested=read_nested));
    } catch e : Error {
      var errorMsg = "Failed to process Parquet file %?".format(e.message());
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc nullIndicesMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var repMsg: string;

    var ndsets = msgArgs.get("dset_size").getIntValue();
    var nfiles = msgArgs.get("filename_size").getIntValue();
    var dsetlist: [0..#ndsets] string;
    var filelist: [0..#nfiles] string;

    try {
      dsetlist = msgArgs.get("dsets").getList(ndsets);
    } catch {
      var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".format(
                                                                                              1, msgArgs.getValueOf("dsets"));
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    try {
      filelist = msgArgs.get("filenames").getList(nfiles);
    } catch {
      // limit length of file names to 2000 chars
      var n: int = 1000;
      var jsonfiles = msgArgs.getValueOf("filenames");
      var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
      var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, files);
      pqLogger.error(getM(),getR(),getL(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    var dsetdom = dsetlist.domain;
    var filedom = filelist.domain;
    var dsetnames: [dsetdom] string;
    var filenames: [filedom] string;
    dsetnames = dsetlist;

    if filelist.size == 1 {
      if filelist[0].strip().size == 0 {
        var errorMsg = "filelist was empty.";
        pqLogger.error(getM(),getR(),getL(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
      var tmp = glob(filelist[0]);
      pqLogger.debug(getM(),getR(),getL(),
                     "glob expanded %s to %i files".format(filelist[0], tmp.size));
      if tmp.size == 0 {
        var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
        pqLogger.error(getM(),getR(),getL(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
      // Glob returns filenames in weird order. Sort for consistency
      sort(tmp);
      filedom = tmp.domain;
      filenames = tmp;
    } else {
      filenames = filelist;
    }

    var fileErrors: list(string);
    var fileErrorCount:int = 0;
    var fileErrorMsg:string = "";
    var sizes: [filedom] int;
    var types: [dsetdom] ArrowTypes;
    var byteSizes: [filedom] int;

    var rnames: list((string, ObjType, string)); // tuple (dsetName, item type, id)
    
    for (dsetidx, dsetname) in zip(dsetdom, dsetnames) do {
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                types[dsetidx] = getArrType(fname, dsetname);
                sizes[i] = getArrSize(fname);
            } catch e : Error {
                // This is only type of error thrown by Parquet
                fileErrorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                pqLogger.error(getM(),getR(),getL(),fileErrorMsg);
                hadError = true;
                return new MsgTuple(fileErrorMsg, MsgType.ERROR);
            }

            // This may need to be adjusted for this all-in-one approach
            if hadError {
              // Keep running total, but we'll only report back the first 10
              if fileErrorCount < 10 {
                fileErrors.pushBack(fileErrorMsg.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip());
              }
              fileErrorCount += 1;
            }
        }
        var len = + reduce sizes;
        var ty = types[dsetidx];
        
        if ty == ArrowTypes.stringArr {
          var entryVal = createSymEntry(len, int);
          getNullIndices(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.pushBack((dsetname, ObjType.PDARRAY, valName));
        } else {
          var errorMsg = "Null indices only supported on Parquet string columns, not %? columns".format(ty);
          pqLogger.error(getM(),getR(),getL(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    repMsg = buildReadAllMsgJson(rnames, false, 0, fileErrors, st);
    pqLogger.debug(getM(),getR(),getL(),repMsg);
    return new MsgTuple(repMsg,MsgType.NORMAL);
  }

  use CommandMap;
  registerFunction("readAllParquet", readAllParquetMsg, getM());
  registerFunction("readAllColsParquet", readAllColsParquetMsg, getM());
  registerFunction("toParquet_multi", toParquetMultiColMsg, getM());
  registerFunction("writeParquet", toparquetMsg, getM());
  registerFunction("lspq", lspqMsg, getM());
  registerFunction("getnullparquet", nullIndicesMsg, getM());
  ServerConfig.appendToConfigStr("ARROW_VERSION", getVersionInfo());
}
