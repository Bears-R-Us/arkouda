module ParquetMsg {
  use SysCTypes, CPtr, IO;
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


  // Use reflection for error information
  use Reflection;
  require "ArrowFunctions.h";
  require "ArrowFunctions.o";

  private config const logLevel = ServerConfig.logLevel;
  const pqLogger = new Logger(logLevel);
  
  private config const ROWGROUPS = 512*1024*1024 / numBytes(int); // 512 mb of int64
  // Undocumented for now, just for internal experiments
  private config const batchSize = getEnvInt("ARKOUDA_SERVER_PARQUET_BATCH_SIZE", 8192);

  extern var ARROWINT64: c_int;
  extern var ARROWINT32: c_int;
  extern var ARROWUINT64: c_int;
  extern var ARROWUNDEFINED: c_int;
  extern var ARROWERROR: c_int;

  enum ArrowTypes { int64, int32, uint64, notimplemented };

  record parquetErrorMsg {
    var errMsg: c_ptr(uint(8));
    proc init() {
      errMsg = c_nil;
    }
    
    proc deinit() {
      extern proc c_free_string(ptr);
      c_free_string(errMsg);
    }

    proc parquetError(lineNumber, routineName, moduleName) throws {
      extern proc strlen(a): int;
      var err: string;
      try {
        err = createStringWithNewBuffer(errMsg, strlen(errMsg));
      } catch e {
        err = "Error converting Parquet error message to Chapel string";
      }
      throw getErrorWithContext(
                     msg=err,
                     lineNumber,
                     routineName,
                     moduleName,
                     errorClass="ParquetError");
    }
  }
  
  proc getVersionInfo() {
    extern proc c_getVersionInfo(): c_string;
    extern proc strlen(str): c_int;
    extern proc c_free_string(ptr);
    var cVersionString = c_getVersionInfo();
    defer {
      c_free_string(cVersionString: c_void_ptr);
    }
    var ret: string;
    try {
      ret = createStringWithNewBuffer(cVersionString,
                                strlen(cVersionString));
    } catch e {
      ret = "Error converting Arrow version message to Chapel string";
    }
    return ret;
  }
  
  proc getSubdomains(lengths: [?FD] int) {
    var subdoms: [FD] domain(1);
    var offset = 0;
    for i in FD {
      subdoms[i] = {offset..#lengths[i]};
      offset += lengths[i];
    }
    return (subdoms, (+ reduce lengths));
  }

  proc readFilesByName(A: [] ?t, filenames: [] string, sizes: [] int, dsetname: string, ty) throws {
    extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, batchSize, errMsg): int;
    var (subdoms, length) = getSubdomains(sizes);
    
    coforall loc in A.targetLocales() do on loc {
      var locFiles = filenames;
      var locFiledoms = subdoms;
      forall (filedom, filename) in zip(locFiledoms, locFiles) {
        for locdom in A.localSubdomains() {
          const intersection = domain_intersection(locdom, filedom);
          if intersection.size > 0 {
            var pqErr = new parquetErrorMsg();
            var col: [filedom] t;
            if c_readColumnByName(filename.localize().c_str(), c_ptrTo(col),
                                  dsetname.localize().c_str(), filedom.size, batchSize,
                                  c_ptrTo(pqErr.errMsg)) == ARROWERROR {
              pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
            }
            A[filedom] = col;
          }
        }
      }
    }
  }

  proc getArrSize(filename: string) throws {
    extern proc c_getNumRows(chpl_str, errMsg): int;
    var pqErr = new parquetErrorMsg();
    
    var size = c_getNumRows(filename.localize().c_str(),
                            c_ptrTo(pqErr.errMsg));
    if size == ARROWERROR {
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
    }
    return size;
  }

  proc getArrType(filename: string, colname: string) throws {
    extern proc c_getType(filename, colname, errMsg): c_int;
    var pqErr = new parquetErrorMsg();
    var arrType = c_getType(filename.localize().c_str(),
                            colname.localize().c_str(),
                            c_ptrTo(pqErr.errMsg));
    if arrType == ARROWERROR {
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
    }
    
    if arrType == ARROWINT64 then return ArrowTypes.int64;
    else if arrType == ARROWINT32 then return ArrowTypes.int32;
    else if arrType == ARROWUINT64 then return ArrowTypes.uint64;
    return ArrowTypes.notimplemented;
  }

  proc writeDistArrayToParquet(A, filename, dsetname, dtype, rowGroupSize, compressed) throws {
    extern proc c_writeColumnToParquet(filename, chpl_arr, colnum,
                                       dsetname, numelems, rowGroupSize,
                                       dtype, compressed, errMsg): int;
    var filenames: [0..#A.targetLocales().size] string;
    var dtypeRep = if dtype == "int64" then 1 else 2;
    for i in 0..#A.targetLocales().size {
      var suffix = '%04i'.format(i): string;
      filenames[i] = filename + "_LOCALE" + suffix + ".parquet";
    }
    var matchingFilenames = glob("%s_LOCALE*%s".format(filename, ".parquet"));

    var warnFlag = processParquetFilenames(filenames, matchingFilenames);
    
    coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) do on loc {
        var pqErr = new parquetErrorMsg();
        const myFilename = filenames[idx];

        var locDom = A.localSubdomain();
        var locArr = A[locDom];
        if c_writeColumnToParquet(myFilename.localize().c_str(), c_ptrTo(locArr), 0,
                                  dsetname.localize().c_str(), locDom.size, rowGroupSize,
                                  dtypeRep, compressed, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
          pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
        }
      }
    return warnFlag;
  }
      

  proc processParquetFilenames(filenames: [] string, matchingFilenames: [] string) throws {
    var warnFlag: bool;
    if matchingFilenames.size > 0 {
      warnFlag = true;
    } else {
      warnFlag = false;
    }
    return warnFlag;
  }

  proc write1DDistArrayParquet(filename: string, dsetname, dtype, compressed, A) throws {
    return writeDistArrayToParquet(A, filename, dsetname, dtype, ROWGROUPS, compressed);
  }

  proc readAllParquetMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var repMsg: string;
    // May need a more robust delimiter then " | "
    var (strictFlag, ndsetsStr, nfilesStr, allowErrorsFlag, arraysStr) = payload.splitMsgToTuple(5);
    var strictTypes: bool = true;
    if (strictFlag.toLower().strip() == "false") {
      strictTypes = false;
    }

    var allowErrors: bool = "true" == allowErrorsFlag.toLower(); // default is false
    if allowErrors {
        pqLogger.warn(getModuleName(), getRoutineName(), getLineNumber(), "Allowing file read errors");
    }

    // Test arg casting so we can send error message instead of failing
    if (!checkCast(ndsetsStr, int)) {
        var errMsg = "Number of datasets:`%s` could not be cast to an integer".format(ndsetsStr);
        pqLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errMsg);
        return new MsgTuple(errMsg, MsgType.ERROR);
    }
    if (!checkCast(nfilesStr, int)) {
      var errMsg = "Number of files:`%s` could not be cast to an integer".format(nfilesStr);
      pqLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errMsg);
      return new MsgTuple(errMsg, MsgType.ERROR);
    }

    var (jsondsets, jsonfiles) = arraysStr.splitMsgToTuple(" | ",2);
    var ndsets = ndsetsStr:int; // Error checked above
    var nfiles = nfilesStr:int; // Error checked above
    var dsetlist: [0..#ndsets] string;
    var filelist: [0..#nfiles] string;

    try {
        dsetlist = jsonToPdArray(jsondsets, ndsets);
    } catch {
        var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".format(
                                            1, jsondsets);
        pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    try {
        filelist = jsonToPdArray(jsonfiles, nfiles);
    } catch {
        var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
        pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
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
            pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        var tmp = glob(filelist[0]);
        pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "glob expanded %s to %i files".format(filelist[0], tmp.size));
        if tmp.size == 0 {
            var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
            pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
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
    var ty = getArrType(filenames[filedom.low],
                        dsetlist[dsetdom.low]);
    var rnames: list((string, string, string)); // tuple (dsetName, item type, id)

    for dsetname in dsetnames do {
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                // not using the type for now since it is only implemented for ints
                // also, since Parquet files have a `numRows` that isn't specifc
                // to dsetname like for HDF5, we only need to get this once per
                // file, regardless of how many datasets we are reading
                sizes[i] = getArrSize(fname);
            } catch e: FileNotFoundError {
                fileErrorMsg = "File %s not found".format(fname);
                pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e: PermissionError {
                fileErrorMsg = "Permission error %s opening %s".format(e.message(),fname);
                pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e: DatasetNotFoundError {
                fileErrorMsg = "Dataset %s not found in file %s".format(dsetname,fname);
                pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e: SegArrayError {
                fileErrorMsg = "SegmentedArray error: %s".format(e.message());
                pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e : Error {
                fileErrorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            }

            // This may need to be adjusted for this all-in-one approach
            if hadError {
              // Keep running total, but we'll only report back the first 10
              if fileErrorCount < 10 {
                fileErrors.append(fileErrorMsg.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip());
              }
              fileErrorCount += 1;
            }
        }
        // This is handled in the readFilesByName() function
        var subdoms: [filedom] domain(1);
        var len: int;
        var nSeg: int;
        len = + reduce sizes;

        // Only integer is implemented for now, do nothing if the Parquet
        // file has a different type
        if ty == ArrowTypes.int64 || ty == ArrowTypes.int32 {
          var entryVal = new shared SymEntry(len, int);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.append((dsetname, "pdarray", valName));
        } else if ty == ArrowTypes.uint64 {
          var entryVal = new shared SymEntry(len, uint(64));
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.append((dsetname, "pdarray", valName));
        }
    }

    repMsg = _buildReadAllMsgJson(rnames, false, 0, fileErrors, st);
    pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg,MsgType.NORMAL);
  }

  proc toparquetMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var (arrayName, dsetname,  jsonfile, dataType, isCompressed)= payload.splitMsgToTuple(5);
    var filename: string;
    var entry = getGenericTypedArrayEntry(arrayName, st);

    var compressed = try! isCompressed.toLower():bool;

    try {
      filename = jsonToPdArray(jsonfile, 1)[0];
    } catch {
      var errorMsg = "Could not decode json filenames via tempfile " +
        "(%i files: %s)".format(1, jsonfile);
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    var warnFlag: bool;

    try {
      select entry.dtype {
          when DType.Int64 {
            var e = toSymEntry(entry, int);
            warnFlag = write1DDistArrayParquet(filename, dsetname, dataType, compressed, e.a);
          }
          when DType.UInt64 {
            var e = toSymEntry(entry, uint(64));
            warnFlag = write1DDistArrayParquet(filename, dsetname, dataType, compressed, e.a);
          }
          otherwise {
            var errorMsg = "Writing Parquet files is only supported for int arrays";
            pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
    } catch e: FileNotFoundError {
      var errorMsg = "Unable to open %s for writing: %s".format(filename,e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: MismatchedAppendError {
      var errorMsg = "Mismatched append %s".format(e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: WriteModeError {
      var errorMsg = "Write mode error %s".format(e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: Error {
      var errorMsg = "problem writing to file %s".format(e);
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }
    if warnFlag {
      var warnMsg = "Warning: possibly overwriting existing files matching filename pattern";
      return new MsgTuple(warnMsg, MsgType.WARNING);
    } else {
      var repMsg = "wrote array to file";
      pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
  }

  proc registerMe() {
    use CommandMap;
    registerFunction("readAllParquet", readAllParquetMsg, getModuleName());
    registerFunction("writeParquet", toparquetMsg, getModuleName());
    ServerConfig.appendToConfigStr("ARROW_VERSION", getVersionInfo());
  }

}
