module ParquetMsg {
  use CTypes, IO;
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

  use SegmentedArray;


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
  extern var ARROWBOOLEAN: c_int;
  extern var ARROWSTRING: c_int;
  extern var ARROWFLOAT: c_int;
  extern var ARROWDOUBLE: c_int;
  extern var ARROWERROR: c_int;

  enum ArrowTypes { int64, int32, uint64, stringArr,
                    timestamp, boolean, double,
                    float, notimplemented };

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
    extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, errMsg): int;
    var (subdoms, length) = getSubdomains(sizes);
    var fileOffsets = (+ scan sizes) - sizes;
    
    coforall loc in A.targetLocales() do on loc {
      var locFiles = filenames;
      var locFiledoms = subdoms;
      var locOffsets = fileOffsets;
      
      try {
        forall (off, filedom, filename) in zip(locOffsets, locFiledoms, locFiles) {
          for locdom in A.localSubdomains() {
            const intersection = domain_intersection(locdom, filedom);
            
            if intersection.size > 0 {
              var pqErr = new parquetErrorMsg();
              if c_readColumnByName(filename.localize().c_str(), c_ptrTo(A[intersection.low]),
                                    dsetname.localize().c_str(), intersection.size, intersection.low - off,
                                    batchSize,
                                    c_ptrTo(pqErr.errMsg)) == ARROWERROR {
                pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
              }
            }
          }
        }
      } catch e {
        throw e;
      }
    }
  }

  proc readStrFilesByName(A: [] ?t, filenames: [] string, sizes: [] int, dsetname: string, ty) throws {
    extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, errMsg): int;
    var (subdoms, length) = getSubdomains(sizes);
    var fileOffsets = (+ scan sizes) - sizes;
    
    coforall loc in A.targetLocales() do on loc {
      var locFiles = filenames;
      var locFiledoms = subdoms;
      var locOffsets = fileOffsets;

      try {
        forall (off, filedom, filename) in zip(locOffsets, locFiledoms, locFiles) {
          for locdom in A.localSubdomains() {
            const intersection = domain_intersection(locdom, filedom);
            var startByte = intersection.low - filedom.low;

            if intersection.size > 0 {
              var pqErr = new parquetErrorMsg();

              if c_readColumnByName(filename.localize().c_str(), c_ptrTo(A[intersection.low]),
                                    dsetname.localize().c_str(), intersection.size, startByte,
                                    batchSize, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
                pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
              }
            }
          }
        }
      } catch e {
        throw e;
      }
    }
  }

  proc calcSizesAndOffset(offsets: [] ?t, byteSizes: [] int, filenames: [] string, sizes: [] int, dsetname: string) throws {
    var (subdoms, length) = getSubdomains(sizes);
    var fileOffsets = (+ scan sizes) - sizes;

    coforall loc in offsets.targetLocales() do on loc {
      var locFiles = filenames;
      var locFiledoms = subdoms;
      var locOffsets = fileOffsets;
      
      try {
        forall (i, off, filedom, filename) in zip(sizes.domain, locOffsets, locFiledoms, locFiles) {
          for locdom in offsets.localSubdomains() {
            const intersection = domain_intersection(locdom, filedom);
            if intersection.size > 0 {
              var pqErr = new parquetErrorMsg();
              getStrColSize(filename, dsetname, offsets, intersection.size,
                            intersection.low, intersection.low - off);
            }
          }
        }
      } catch e {
        throw e;
      }
    }
    var currIdx = 0;
    for i in 0..#sizes.size {
      byteSizes[i] = (+ reduce offsets[currIdx..#sizes[i]]);
      currIdx += sizes[i];
    }
  }

  proc getStrColSize(filename: string, dsetname: string, offsets: [] int, numElems: int, chplStartIdx: int, cStartIdx: int) throws {
    extern proc c_getStringColumnNumBytes(filename, colname, offsets, numElems, startIdx, errMsg): int;
    var pqErr = new parquetErrorMsg();
    
    if c_getStringColumnNumBytes(filename.localize().c_str(),
                                 dsetname.localize().c_str(),
                                 c_ptrTo(offsets[chplStartIdx]),
                                 numElems, cStartIdx,
                                 c_ptrTo(pqErr.errMsg)) == ARROWERROR then
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
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
    else if arrType == ARROWBOOLEAN then return ArrowTypes.boolean;
    else if arrType == ARROWSTRING then return ArrowTypes.stringArr;
    else if arrType == ARROWDOUBLE then return ArrowTypes.double;
    else if arrType == ARROWFLOAT then return ArrowTypes.float;
    throw getErrorWithContext(
                  msg="Unrecognized Parquet data type",
                  getLineNumber(),
                  getRoutineName(),
                  getModuleName(),
                  errorClass="ParquetError");
    return ArrowTypes.notimplemented;
  }

  proc toCDtype(dtype: string) throws {
    select dtype {
      when 'int64' {
        return ARROWINT64;
      } when 'uint64' {
        return ARROWUINT64;
      } when 'bool' {
        return ARROWBOOLEAN;
      } when 'float64' {
        return ARROWDOUBLE;
      } otherwise {
         throw getErrorWithContext(
                msg="Trying to convert unrecognized dtype to Parquet type",
                getLineNumber(),
                getRoutineName(),
                getModuleName(),
                errorClass="ParquetError");
        return ARROWERROR;
      }
    }
  }

  proc writeDistArrayToParquet(A, filename, dsetname, dtype, rowGroupSize, compressed) throws {
    extern proc c_writeColumnToParquet(filename, chpl_arr, colnum,
                                       dsetname, numelems, rowGroupSize,
                                       dtype, compressed, errMsg): int;
    var filenames: [0..#A.targetLocales().size] string;
    var dtypeRep = toCDtype(dtype);
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
    var types: [dsetdom] ArrowTypes;
    var byteSizes: [filedom] int;

    var rnames: list((string, string, string)); // tuple (dsetName, item type, id)
    
    for (dsetidx, dsetname) in zip(dsetdom, dsetnames) do {
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                types[dsetidx] = getArrType(fname, dsetname);
                sizes[i] = getArrSize(fname);
            } catch e : Error {
                // This is only type of error thrown by Parquet
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
        var len = + reduce sizes;
        var ty = types[dsetidx];

        // Only integer is implemented for now, do nothing if the Parquet
        // file has a different type
        if ty == ArrowTypes.int64 || ty == ArrowTypes.int32 {
          var entryVal = new shared SymEntry(len, int);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.append((dsetname, "pdarray", valName));
        } else if ty == ArrowTypes.uint64 {
          var entryVal = new shared SymEntry(len, uint);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.append((dsetname, "pdarray", valName));
        } else if ty == ArrowTypes.boolean {
          var entryVal = new shared SymEntry(len, bool);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.append((dsetname, "pdarray", valName));
        } else if ty == ArrowTypes.stringArr {
          var entrySeg = new shared SymEntry(len, int);
          calcSizesAndOffset(entrySeg.a, byteSizes, filenames, sizes, dsetname);
          entrySeg.a = (+ scan entrySeg.a) - entrySeg.a;
          
          var entryVal = new shared SymEntry((+ reduce byteSizes), uint(8));
          readStrFilesByName(entryVal.a, filenames, byteSizes, dsetname, ty);
          
          var stringsEntry = assembleSegStringFromParts(entrySeg, entryVal, st);
          rnames.append((dsetname, "seg_string", "%s+%t".format(stringsEntry.name, stringsEntry.nBytes)));
        } else if ty == ArrowTypes.double || ty == ArrowTypes.float {
          var entryVal = new shared SymEntry(len, real);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.append((dsetname, "pdarray", valName));
        } else {
          var errorMsg = "DType %s not supported for Parquet reading".format(ty);
          pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
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
            var e = toSymEntry(entry, uint);
            warnFlag = write1DDistArrayParquet(filename, dsetname, dataType, compressed, e.a);
          }
          when DType.Bool {
            var e = toSymEntry(entry, bool);
            warnFlag = write1DDistArrayParquet(filename, dsetname, dataType, compressed, e.a);
          } when DType.Float64 {
            var e = toSymEntry(entry, real);
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

  proc lspqMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    // reqMsg: "lshdf [<json_filename>]"
    var repMsg: string;
    var (jsonfile) = payload.splitMsgToTuple(1);

    // Retrieve filename from payload
    var filename: string;
    try {
      filename = jsonToPdArray(jsonfile, 1)[0];
      if filename.isEmpty() {
        throw new IllegalArgumentError("filename was empty");  // will be caught by catch block
      }
    } catch {
      var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(
                                                                                          1, jsonfile);
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
      extern proc c_getDatasetNames(filename, dsetResult, errMsg): int(32);
      extern proc strlen(a): int;
      var pqErr = new parquetErrorMsg();
      var res: c_ptr(uint(8));
      defer {
        extern proc c_free_string(ptr);
        c_free_string(res);
      }
      if c_getDatasetNames(filename.c_str(), c_ptrTo(res),
                           c_ptrTo(pqErr.errMsg)) == ARROWERROR {
        pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
      }
      try! repMsg = createStringWithNewBuffer(res, strlen(res));
      var items = new list(repMsg.split(",")); // convert to json

      // TODO: There is a bug with json formatting of lists in Chapel 1.24.x fixed in 1.25
      //       See: https://github.com/chapel-lang/chapel/issues/18156
      //       Below works in 1.25, but until we are fully off of 1.24 we should format json manually for lists
      // repMsg = "%jt".format(items); // Chapel >= 1.25.0
      repMsg = "[";  // Manual json building Chapel <= 1.24.1
      var first = true;
      for i in items {
        i = i.replace(Q, ESCAPED_QUOTES, -1);
        if first {
          first = false;
        } else {
          repMsg += ",";
        }
        repMsg += Q + i + Q;
      }
      repMsg += "]";
    } catch e : Error {
      var errorMsg = "Failed to process Parquet file %t".format(e.message());
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc registerMe() {
    use CommandMap;
    registerFunction("readAllParquet", readAllParquetMsg, getModuleName());
    registerFunction("writeParquet", toparquetMsg, getModuleName());
    registerFunction("lspq", lspqMsg, getModuleName());
    ServerConfig.appendToConfigStr("ARROW_VERSION", getVersionInfo());
  }

}
