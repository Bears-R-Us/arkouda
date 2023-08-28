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
  use CommAggregation;
  use AryUtil;

  use SegmentedString;

  use Map;
  use ArkoudaCTypesCompat;
  use ArkoudaListCompat;
  use ArkoudaStringBytesCompat;
  use ArkoudaIOCompat;

  enum CompressionType {
    NONE=0,
    SNAPPY=1,
    GZIP=2,
    BROTLI=3,
    ZSTD=4,
    LZ4=5
  };


  // Use reflection for error information
  use Reflection;
  require "ArrowFunctions.h";
  require "ArrowFunctions.o";

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const pqLogger = new Logger(logLevel, logChannel);
  config const TRUNCATE: int = 0;
  config const APPEND: int = 1;
  
  private config const ROWGROUPS = 512*1024*1024 / numBytes(int); // 512 mb of int64
  // Undocumented for now, just for internal experiments
  private config const batchSize = getEnvInt("ARKOUDA_SERVER_PARQUET_BATCH_SIZE", 8192);

  extern var ARROWINT64: c_int;
  extern var ARROWINT32: c_int;
  extern var ARROWUINT64: c_int;
  extern var ARROWUINT32: c_int;
  extern var ARROWBOOLEAN: c_int;
  extern var ARROWSTRING: c_int;
  extern var ARROWFLOAT: c_int;
  extern var ARROWLIST: c_int;
  extern var ARROWDOUBLE: c_int;
  extern var ARROWERROR: c_int;

  enum ArrowTypes { int64, int32, uint64, uint32,
                    stringArr, timestamp, boolean,
                    double, float, list, notimplemented };

  record parquetErrorMsg {
    var errMsg: c_ptr(uint(8));
    proc init() {
      errMsg = nil;
    }
    
    proc deinit() {
      extern proc c_free_string(ptr);
      c_free_string(errMsg);
    }

    proc parquetError(lineNumber, routineName, moduleName) throws {
      extern proc strlen(a): int;
      var err: string;
      try {
        err = string.createCopyingBuffer(errMsg, strlen(errMsg));
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
      ret = string.createCopyingBuffer(cVersionString,
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
      
    }
  }

  proc readStrFilesByName(A: [] ?t, filenames: [] string, sizes: [] int, dsetname: string, ty) throws {
    extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, errMsg): int;
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
                                  dsetname.localize().c_str(), intersection.size, 0,
                                  batchSize, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
              pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
            }
            A[filedom] = col;
          }
        }
      }
    }
  }

  proc computeIdx(offsets: [] int, val: int): int throws {
    var (v, idx) = maxloc reduce zip(offsets > val, offsets.domain);
    return if v then idx-1 else offsets.size-1;
  }

  proc computeEmptySegs(seg_sizes: [] int, offsets: [] int, s: int, intersection: domain(1), fileoffset: int): int throws {
    // Compute the number of empty segments preceeding the current chunk
    if (intersection.low-fileoffset == 0){ //starts the file, no shift needed
      return 0;
    }
    var highidx = computeIdx(offsets, intersection.low);
    var sub_segs = seg_sizes[s..highidx];
    var empty_segs = sub_segs == 0;
    return (+ reduce empty_segs);
  }

  proc readListFilesByName(A: [] ?t, rows_per_file: [] int, seg_sizes: [] int, offsets: [] int, filenames: [] string, sizes: [] int, dsetname: string, ty) throws {
    extern proc c_readListColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, errMsg): int;
    var (subdoms, length) = getSubdomains(sizes);
    var fileOffsets = (+ scan sizes) - sizes;
    var segmentOffsets = (+ scan rows_per_file) - rows_per_file;
    
    coforall loc in A.targetLocales() do on loc {
      var locFiles = filenames;
      var locFiledoms = subdoms;
      var locOffsets = fileOffsets; // value count offset
      var locSegOffsets = segmentOffsets; // indicates which segment index is first for the file

      forall (s, off, filedom, filename) in zip(locSegOffsets, locOffsets, locFiledoms, locFiles) {
        for locdom in A.localSubdomains() {
          const intersection = domain_intersection(locdom, filedom);
          
          if intersection.size > 0 {
            var pqErr = new parquetErrorMsg();
            var col: [filedom] t;
            if c_readListColumnByName(filename.localize().c_str(), c_ptrTo(col),
                                  dsetname.localize().c_str(), filedom.size, 0,
                                  batchSize, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
              pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
            }
            A[filedom] = col;
          }
        }
      }
    }
  }

  proc calcListSizesandOffset(seg_sizes: [] ?t, filenames: [] string, sizes: [] int, dsetname: string) throws {
    var (subdoms, length) = getSubdomains(sizes);

    var listSizes: [filenames.domain] int;
    var file_offset: int = 0;
    coforall loc in seg_sizes.targetLocales() do on loc{
      var locFiles = filenames;
      var locFiledoms = subdoms;
      
      forall (i, filedom, filename) in zip(sizes.domain, locFiledoms, locFiles) {
        for locdom in seg_sizes.localSubdomains() {
          const intersection = domain_intersection(locdom, filedom);
          if intersection.size > 0 {
            var col: [filedom] t;
            listSizes[i] = getListColSize(filename, dsetname, col);
            seg_sizes[filedom] = col; // this is actually segment sizes here
          }
        }
      }
    }
    return listSizes;
  }

  proc calcStrSizesAndOffset(offsets: [] ?t, filenames: [] string, sizes: [] int, dsetname: string) throws {
    var (subdoms, length) = getSubdomains(sizes);

    var byteSizes: [filenames.domain] int;

    coforall loc in offsets.targetLocales() do on loc {
      var locFiles = filenames;
      var locFiledoms = subdoms;
      
      forall (i, filedom, filename) in zip(sizes.domain, locFiledoms, locFiles) {
        for locdom in offsets.localSubdomains() {
          const intersection = domain_intersection(locdom, filedom);
          if intersection.size > 0 {
            var col: [filedom] t;
            byteSizes[i] = getStrColSize(filename, dsetname, col);
            offsets[filedom] = col;
          }
        }
      }
    }
    return byteSizes;
  }

  proc getNullIndices(A: [] ?t, filenames: [] string, sizes: [] int, dsetname: string, ty) throws {
    extern proc c_getStringColumnNullIndices(filename, colname, chpl_nulls, errMsg): int;
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
            if c_getStringColumnNullIndices(filename.localize().c_str(), dsetname.localize().c_str(),
                                            c_ptrTo(col), pqErr.errMsg) {
              pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
            }
            A[filedom] = col;
          }
        }
      }
    }
  }

  proc getStrColSize(filename: string, dsetname: string, offsets: [] int) throws {
    extern proc c_getStringColumnNumBytes(filename, colname, offsets, numElems, startIdx, errMsg): int;
    var pqErr = new parquetErrorMsg();

    var byteSize = c_getStringColumnNumBytes(filename.localize().c_str(),
                                             dsetname.localize().c_str(),
                                             c_ptrTo(offsets),
                                             offsets.size, 0,
                                             c_ptrTo(pqErr.errMsg));
    
    if byteSize == ARROWERROR then
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
    return byteSize;
  }

  proc getListColSize(filename: string, dsetname: string, seg_sizes: [] int) throws {
    extern proc c_getListColumnSize(filename, colname, seg_sizes, numElems, startIdx, errMsg): int;
    var pqErr = new parquetErrorMsg();

    var listSize = c_getListColumnSize(filename.localize().c_str(),
                                             dsetname.localize().c_str(),
                                             c_ptrTo(seg_sizes),
                                             seg_sizes.size, 0,
                                             c_ptrTo(pqErr.errMsg));
    
    if listSize == ARROWERROR then
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
    return listSize;
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
    else if arrType == ARROWUINT32 then return ArrowTypes.uint32;
    else if arrType == ARROWUINT64 then return ArrowTypes.uint64;
    else if arrType == ARROWBOOLEAN then return ArrowTypes.boolean;
    else if arrType == ARROWSTRING then return ArrowTypes.stringArr;
    else if arrType == ARROWDOUBLE then return ArrowTypes.double;
    else if arrType == ARROWFLOAT then return ArrowTypes.float;
    else if arrType == ARROWLIST then return ArrowTypes.list;
    throw getErrorWithContext(
                  msg="Unrecognized Parquet data type",
                  getLineNumber(),
                  getRoutineName(),
                  getModuleName(),
                  errorClass="ParquetError");
    return ArrowTypes.notimplemented;
  }

  proc getListData(filename: string, dsetname: string) throws {
    extern proc c_getListType(filename, dsetname, errMsg): c_int;
    var pqErr = new parquetErrorMsg();
    
    var t = c_getListType(filename.localize().c_str(), dsetname.localize().c_str(), c_ptrTo(pqErr.errMsg));
    if t == ARROWINT64 then return ArrowTypes.int64;
    else if t == ARROWINT32 then return ArrowTypes.int32;
    else if t == ARROWUINT32 then return ArrowTypes.uint32;
    else if t == ARROWUINT64 then return ArrowTypes.uint64;
    else if t == ARROWBOOLEAN then return ArrowTypes.boolean;
    else if t == ARROWSTRING then return ArrowTypes.stringArr;
    else if t == ARROWDOUBLE then return ArrowTypes.double;
    else if t == ARROWFLOAT then return ArrowTypes.float;
    return ArrowTypes.notimplemented;
  }

  proc toCDtype(dtype: string) throws {
    select dtype {
      when 'int64' {
        return ARROWINT64;
      } when 'uint32' {
        return ARROWUINT32;
      } when 'uint64' {
        return ARROWUINT64;
      } when 'bool' {
        return ARROWBOOLEAN;
      } when 'float64' {
        return ARROWDOUBLE;
      } when 'str' {
        return ARROWSTRING;
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

  proc writeDistArrayToParquet(A, filename, dsetname, dtype, rowGroupSize, compression, mode) throws {
    extern proc c_writeColumnToParquet(filename, chpl_arr, colnum,
                                       dsetname, numelems, rowGroupSize,
                                       dtype, compression, errMsg): int;
    extern proc c_appendColumnToParquet(filename, chpl_arr,
                                        dsetname, numelems,
                                        dtype, compression,
                                        errMsg): int;
    var dtypeRep = toCDtype(dtype);
    var prefix: string;
    var extension: string;
  
    (prefix, extension) = getFileMetadata(filename);

    // Generate the filenames based upon the number of targetLocales.
    var filenames = generateFilenames(prefix, extension, A.targetLocales().size);

    //Generate a list of matching filenames to test against. 
    var matchingFilenames = getMatchingFilenames(prefix, extension);

    var filesExist = processParquetFilenames(filenames, matchingFilenames, mode);

    if mode == APPEND {
      if filesExist {
        var datasets = getDatasets(filenames[0]);
        if datasets.contains(dsetname) then
          throw getErrorWithContext(
                    msg="A column with name " + dsetname + " already exists in Parquet file",
                    lineNumber=getLineNumber(), 
                    routineName=getRoutineName(), 
                    moduleName=getModuleName(), 
                    errorClass='WriteModeError');
      }
    }
    
    coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) do on loc {
        var pqErr = new parquetErrorMsg();
        const myFilename = filenames[idx];

        var locDom = A.localSubdomain();
        var locArr = A[locDom];
        var valPtr: c_void_ptr = nil;
        if locArr.size != 0 {
          valPtr = c_ptrTo(locArr);
        }
        if mode == TRUNCATE || !filesExist {
          if c_writeColumnToParquet(myFilename.localize().c_str(), valPtr, 0,
                                    dsetname.localize().c_str(), locDom.size, rowGroupSize,
                                    dtypeRep, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
            pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
          }
        } else {
          if c_appendColumnToParquet(myFilename.localize().c_str(), valPtr,
                                     dsetname.localize().c_str(), locDom.size,
                                     dtypeRep, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
            pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
          }
        }
      }
    // Only warn when files are being overwritten in truncate mode
    return filesExist && mode == TRUNCATE;
  }

  proc createEmptyParquetFile(filename: string, dsetname: string, dtype: int, compression: int) throws {
    extern proc c_createEmptyParquetFile(filename, dsetname, dtype,
                                         compression, errMsg): int;
    var pqErr = new parquetErrorMsg();
    if c_createEmptyParquetFile(filename.localize().c_str(), dsetname.localize().c_str(),
                                dtype, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
    }
  }
  
  // TODO: do we want to add offset writing for Parquet string writes?
  //       if we do, then we need to add the load offsets functionality
  //       in the string reading function
  proc write1DDistStringsAggregators(filename: string, mode: int, dsetName: string, entry: SegStringSymEntry, compression: int) throws {
    var segString = new SegString("", entry);
    ref ss = segString;
    var A = ss.offsets.a;

    var prefix: string;
    var extension: string;
  
    (prefix, extension) = getFileMetadata(filename);

    // Generate the filenames based upon the number of targetLocales.
    var filenames = generateFilenames(prefix, extension, A.targetLocales().size);

    //Generate a list of matching filenames to test against. 
    var matchingFilenames = getMatchingFilenames(prefix, extension);

    var filesExist = processParquetFilenames(filenames, matchingFilenames, mode);

    if mode == APPEND {
      if filesExist {
        var datasets = getDatasets(filenames[0]);
        if datasets.contains(dsetName) then
          throw getErrorWithContext(
                   msg="A column with name " + dsetName + " already exists in Parquet file",
                   lineNumber=getLineNumber(), 
                   routineName=getRoutineName(), 
                   moduleName=getModuleName(), 
                   errorClass='WriteModeError');
      }
    }
    
    const extraOffset = ss.values.size;
    const lastOffset = if A.size == 0 then 0 else A[A.domain.high]; // prevent index error when empty
    const lastValIdx = ss.values.a.domain.high;
    // For each locale gather the string bytes corresponding to the offsets in its local domain
    coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with (ref ss) do on loc {
        const myFilename = filenames[idx];

        const locDom = A.localSubdomain();
        var dims: [0..#1] int;
        dims[0] = locDom.size: int;

        if (locDom.isEmpty() || locDom.size <= 0) {
          if mode == APPEND && filesExist then
            throw getErrorWithContext(
                 msg="Parquet columns must each have the same length: " + myFilename,
                 lineNumber=getLineNumber(), 
                 routineName=getRoutineName(), 
                 moduleName=getModuleName(), 
                 errorClass='WriteModeError');
          createEmptyParquetFile(myFilename, dsetName, ARROWSTRING, compression);
        } else {
          var localOffsets = A[locDom];
          var startValIdx = localOffsets[locDom.low];

          var endValIdx = if (lastOffset == localOffsets[locDom.high]) then lastValIdx else A[locDom.high + 1] - 1;
                
          var valIdxRange = startValIdx..endValIdx;
          var localVals: [valIdxRange] uint(8);
          ref olda = ss.values.a;
          forall (localVal, valIdx) in zip(localVals, valIdxRange) with (var agg = newSrcAggregator(uint(8))) {
            // Copy the remote value at index position valIdx to our local array
            agg.copy(localVal, olda[valIdx]); // in SrcAgg, the Right Hand Side is REMOTE
          }
          var locOffsets: [0..#locDom.size+1] int;
          locOffsets[0..#locDom.size] = A[locDom];
          if locDom.high == A.domain.high then
            locOffsets[locOffsets.domain.high] = extraOffset;
          else
            locOffsets[locOffsets.domain.high] = A[locDom.high+1];
          
          writeStringsComponentToParquet(myFilename, dsetName, localVals, locOffsets, ROWGROUPS, compression, mode, filesExist);
        }
      }
    return filesExist && mode == TRUNCATE;
  }

  private proc writeStringsComponentToParquet(filename, dsetname, values: [] uint(8), offsets: [] int, rowGroupSize, compression, mode, filesExist) throws {
    extern proc c_writeStrColumnToParquet(filename, chpl_arr, chpl_offsets,
                                          dsetname, numelems, rowGroupSize,
                                          dtype, compression, errMsg): int;
    extern proc c_appendColumnToParquet(filename, chpl_arr,
                                        dsetname, numelems,
                                        dtype, compression,
                                        errMsg): int;
    var pqErr = new parquetErrorMsg();
    var dtypeRep = ARROWSTRING;
    if mode == TRUNCATE || !filesExist {
      if c_writeStrColumnToParquet(filename.localize().c_str(), c_ptrTo(values), c_ptrTo(offsets),
                                   dsetname.localize().c_str(), offsets.size-1, rowGroupSize,
                                   dtypeRep, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
        pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
      }
    } else if mode == APPEND {
      if c_appendColumnToParquet(filename.localize().c_str(), c_ptrTo(values),
                                 dsetname.localize().c_str(), offsets.size-1,
                                 dtypeRep, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
        pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
      }
    }
  }

  proc processParquetFilenames(filenames: [] string, matchingFilenames: [] string, mode: int) throws {
    var filesExist: bool = true;
    if mode == APPEND {
      if matchingFilenames.size == 0 {
        // Files do not exist, so we can just create the files
        filesExist = false;
      }
      else if matchingFilenames.size != filenames.size {
        throw getErrorWithContext(
                   msg="Appending to existing files must be done with the same number " +
                      "of locales. Try saving with a different directory or filename prefix?",
                   lineNumber=getLineNumber(), 
                   routineName=getRoutineName(), 
                   moduleName=getModuleName(), 
                   errorClass='MismatchedAppendError'
              );
      }
    } else if mode == TRUNCATE {
      if matchingFilenames.size > 0 {
        filesExist = true;
      } else {
        filesExist = false;
      }
    } else {
      throw getErrorWithContext(
                 msg="The mode %? is invalid".doFormat(mode),
                 lineNumber=getLineNumber(), 
                 routineName=getRoutineName(), 
                 moduleName=getModuleName(), 
                 errorClass='IllegalArgumentError');
    }
    return filesExist;
  }

  proc write1DDistArrayParquet(filename: string, dsetname, dtype, compression, mode, A) throws {
    return writeDistArrayToParquet(A, filename, dsetname, dtype, ROWGROUPS, compression, mode);
  }

  proc parseListDataset(filenames: [] string, dsetname: string, ty, len: int, sizes: [] int, st: borrowed SymTab) throws {
    var rtnmap: map(string, string) = new map(string, string);
    // len here is our segment size
    var filedom = filenames.domain;
    var seg_sizes = makeDistArray(len, int);
    var listSizes: [filedom] int = calcListSizesandOffset(seg_sizes, filenames, sizes, dsetname);
    var segments = (+ scan seg_sizes) - seg_sizes; // converts segment sizes into offsets
    var sname = st.nextName();
    st.addEntry(sname, new shared SymEntry(segments));
    rtnmap.add("segments", "created " + st.attrib(sname));

    var vname = st.nextName();
    if ty == ArrowTypes.int64 || ty == ArrowTypes.int32 {
      var values = makeDistArray((+ reduce listSizes), int);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, new shared SymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.uint64 || ty == ArrowTypes.uint32 {
      var values = makeDistArray((+ reduce listSizes), uint);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, new shared SymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.double || ty == ArrowTypes.float {
      var values = makeDistArray((+ reduce listSizes), real);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, new shared SymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.boolean {
      var values = makeDistArray((+ reduce listSizes), bool);
      readListFilesByName(values, sizes, seg_sizes, segments, filenames, listSizes, dsetname, ty);
      st.addEntry(vname, new shared SymEntry(values));
      rtnmap.add("values", "created " + st.attrib(vname));
    }
    else if ty == ArrowTypes.stringArr {
      var entrySeg = createSymEntry((+ reduce listSizes), int);
      var byteSizes = calcStrSizesAndOffset(entrySeg.a, filenames, listSizes, dsetname);
      entrySeg.a = (+ scan entrySeg.a) - entrySeg.a;

      var entryVal = createSymEntry((+ reduce byteSizes), uint(8));
      readListFilesByName(entryVal.a, sizes, seg_sizes, segments, filenames, byteSizes, dsetname, ty);
      var stringsEntry = assembleSegStringFromParts(entrySeg, entryVal, st);
      rtnmap.add("values", "created %s+created bytes.size %?".doFormat(st.attrib(stringsEntry.name), stringsEntry.nBytes));
    }
    else {
      throw getErrorWithContext(
                 msg="Invalid Arrow Type",
                 lineNumber=getLineNumber(), 
                 routineName=getRoutineName(), 
                 moduleName=getModuleName(), 
                 errorClass='IllegalArgumentError');
    }
    return formatJson(rtnmap);
  }

  proc populateTagData(A, filenames: [?fD] string, sizes) throws {
    var (subdoms, length) = getSubdomains(sizes);
    var fileOffsets = (+ scan sizes) - sizes;
    
    coforall loc in A.targetLocales() do on loc {
      var locFiles = filenames;
      var locFiledoms = subdoms;
      var locOffsets = fileOffsets;
      
      try {
        forall (off, filedom, filename, tag) in zip(locOffsets, locFiledoms, locFiles, 0..) {
          for locdom in A.localSubdomains() {
            const intersection = domain_intersection(locdom, filedom);

            if intersection.size > 0 {
              // write the tag into the entry
              A[intersection] = tag;
            }
          }
        }
      }
    }
  }

  proc readAllParquetMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    var repMsg: string;
    var tagData: bool = msgArgs.get("tag_data").getBoolValue();
    var strictTypes: bool = msgArgs.get("strict_types").getBoolValue();

    var allowErrors: bool = msgArgs.get("allow_errors").getBoolValue(); // default is false
    if allowErrors {
        pqLogger.warn(getModuleName(), getRoutineName(), getLineNumber(), "Allowing file read errors");
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
        var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".doFormat(
                                            ndsets, dsets);
        pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    try {
        filelist = msgArgs.get("filenames").getList(nfiles);
    } catch {
        // limit length of file names to 2000 chars
        var n: int = 1000;
        var jsonfiles = msgArgs.getValueOf("filenames");
        var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
        var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".doFormat(nfiles, files);
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
                            "glob expanded %s to %i files".doFormat(filelist[0], tmp.size));
      if tmp.size == 0 {
          var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".doFormat(filelist[0]);
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
    
    var rnames: list((string, ObjType, string)); // tuple (dsetName, item type, id)
    
    for (dsetidx, dsetname) in zip(dsetdom, dsetnames) do {
        types[dsetidx] = getArrType(filenames[0], dsetname);
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                sizes[i] = getArrSize(fname);
            } catch e : Error {
                // This is only type of error thrown by Parquet
                fileErrorMsg = "Other error in accessing file %s: %s".doFormat(fname,e.message());
                pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
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
          pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Tagging Data with File Code");
          var tagEntry = createSymEntry(len, int);
          populateTagData(tagEntry.a, filenames, sizes);
          var rname = st.nextName();
          st.addEntry(rname, tagEntry);
          rnames.pushBack(("Filename_Codes", ObjType.PDARRAY, rname));
          tagData = false; // turn off so we only run once
        }

        // Only integer is implemented for now, do nothing if the Parquet
        // file has a different type
        if ty == ArrowTypes.int64 || ty == ArrowTypes.int32 {
          var entryVal = createSymEntry(len, int);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.pushBack((dsetname, ObjType.PDARRAY, valName));
        } else if ty == ArrowTypes.uint64 || ty == ArrowTypes.uint32 {
          var entryVal = createSymEntry(len, uint);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          if (ty == ArrowTypes.uint32){ // correction for high bit 
            ref ea = entryVal.a;
            // Access the high bit (64th bit) and shift it into the high bit for uint32 (32nd bit)
            // Apply 32 bit mask to drop top 32 bits and properly store uint32
            entryVal.a = ((ea & (2**63))>>32 | ea) & (2**32 -1);
          }
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.pushBack((dsetname, ObjType.PDARRAY, valName));
        } else if ty == ArrowTypes.boolean {
          var entryVal = createSymEntry(len, bool);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.pushBack((dsetname, ObjType.PDARRAY, valName));
        } else if ty == ArrowTypes.stringArr {
          var entrySeg = createSymEntry(len, int);
          byteSizes = calcStrSizesAndOffset(entrySeg.a, filenames, sizes, dsetname);
          entrySeg.a = (+ scan entrySeg.a) - entrySeg.a;
          
          var entryVal = createSymEntry((+ reduce byteSizes), uint(8));
          readStrFilesByName(entryVal.a, filenames, byteSizes, dsetname, ty);
          
          var stringsEntry = assembleSegStringFromParts(entrySeg, entryVal, st);
          rnames.pushBack((dsetname, ObjType.STRINGS, "%s+%?".doFormat(stringsEntry.name, stringsEntry.nBytes)));
        } else if ty == ArrowTypes.double || ty == ArrowTypes.float {
          var entryVal = createSymEntry(len, real);
          readFilesByName(entryVal.a, filenames, sizes, dsetname, ty);
          var valName = st.nextName();
          st.addEntry(valName, entryVal);
          rnames.pushBack((dsetname, ObjType.PDARRAY, valName));
        } else if ty == ArrowTypes.list {
          var list_ty = getListData(filenames[0], dsetname);
          if list_ty == ArrowTypes.notimplemented { // check for and skip further nested datasets
            pqLogger.info(getModuleName(),getRoutineName(),getLineNumber(),"Invalid list datatype found in %s. Skipping.".doFormat(dsetname));
          }
          else {
            var create_str: string = parseListDataset(filenames, dsetname, list_ty, len, sizes, st);
            rnames.pushBack((dsetname, ObjType.SEGARRAY, create_str));
          }
        } else {
          var errorMsg = "DType %s not supported for Parquet reading".doFormat(ty);
          pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    repMsg = _buildReadAllMsgJson(rnames, false, 0, fileErrors, st);
    pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg,MsgType.NORMAL);
  }

  proc getDatasets(filename) throws {
    extern proc c_getDatasetNames(filename, dsetResult, readNested, errMsg): int(32);
    extern proc strlen(a): int;
    var pqErr = new parquetErrorMsg();
    var res: c_ptr(uint(8));
    defer {
      extern proc c_free_string(ptr);
      c_free_string(res);
    }
    if c_getDatasetNames(filename.c_str(), c_ptrTo(res), false,
                         c_ptrTo(pqErr.errMsg)) == ARROWERROR {
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
    }
    var datasets: string;
    try! datasets = string.createCopyingBuffer(res, strlen(res));
    return new list(datasets.split(","));
  }

  proc pdarray_toParquetMsg(msgArgs: MessageArgs, st: borrowed SymTab): bool throws {
    var mode = msgArgs.get("mode").getIntValue();
    var filename: string = msgArgs.getValueOf("prefix");
    var entry = st.lookup(msgArgs.getValueOf("values"));
    var dsetname = msgArgs.getValueOf("dset");
    var dataType = str2dtype(msgArgs.getValueOf("dtype"));
    var dtypestr = msgArgs.getValueOf("dtype");
    var compression = msgArgs.getValueOf("compression").toUpper(): CompressionType;

    if (!entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry)) {
      var errorMsg = "ObjType (PDARRAY) does not match SymEntry Type: %s".doFormat(entry.entryType);
      throw getErrorWithContext(
                   msg=errorMsg,
                   lineNumber=getLineNumber(), 
                   routineName=getRoutineName(), 
                   moduleName=getModuleName(), 
                   errorClass='TypeError');
    }

    var warnFlag: bool;
    select dataType {
      when DType.Int64 {
        var e = toSymEntry(toGenSymEntry(entry), int);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr, compression:int, mode, e.a);
      }
      when DType.UInt64 {
        var e = toSymEntry(toGenSymEntry(entry), uint);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr, compression:int, mode, e.a);
      }
      when DType.Bool {
        var e = toSymEntry(toGenSymEntry(entry), bool);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr, compression:int, mode, e.a);
      } when DType.Float64 {
        var e = toSymEntry(toGenSymEntry(entry), real);
        warnFlag = write1DDistArrayParquet(filename, dsetname, dtypestr, compression:int, mode, e.a);
      } otherwise {
        var errorMsg = "Writing Parquet files not supported for %s type".doFormat(msgArgs.getValueOf("dtype"));
        pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        throw getErrorWithContext(
                   msg=errorMsg,
                   lineNumber=getLineNumber(), 
                   routineName=getRoutineName(), 
                   moduleName=getModuleName(), 
                   errorClass='DataTypeError');
      }
    }
    return warnFlag;
  }

  proc strings_toParquetMsg(msgArgs: MessageArgs, st: borrowed SymTab): bool throws {
    var mode = msgArgs.get("mode").getIntValue();
    var filename: string = msgArgs.getValueOf("prefix");
    var entry = st.lookup(msgArgs.getValueOf("values"));
    var dsetname = msgArgs.getValueOf("dset");
    var dataType = msgArgs.getValueOf("dtype");
    var compression = msgArgs.getValueOf("compression").toUpper(): CompressionType;

    if (!entry.isAssignableTo(SymbolEntryType.SegStringSymEntry)) {
      var errorMsg = "ObjType (STRINGS) does not match SymEntry Type: %s".doFormat(entry.entryType);
      throw getErrorWithContext(
                   msg=errorMsg,
                   lineNumber=getLineNumber(), 
                   routineName=getRoutineName(), 
                   moduleName=getModuleName(), 
                   errorClass='TypeError');
    }

    var segString:SegStringSymEntry = toSegStringSymEntry(entry);
    var warnFlag: bool = write1DDistStringsAggregators(filename, mode, dsetname, segString, compression:int);
    return warnFlag;
  }

  proc createEmptyListParquetFile(filename: string, dsetname: string, dtype: int, compression: int) throws {
    extern proc c_createEmptyListParquetFile(filename, dsetname, dtype,
                                         compression, errMsg): int;
    var pqErr = new parquetErrorMsg();
    if c_createEmptyListParquetFile(filename.localize().c_str(), dsetname.localize().c_str(),
                                dtype, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
      pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
    }
  }

  proc writeSegArrayComponent(filename: string, dsetname: string, const ref distVals: [] ?t, valIdxRange, segments, locDom, 
                              extraOffset, lastOffset, lastValId, c_dtype, compression) throws {
    extern proc c_writeListColumnToParquet(filename, chpl_arr, chpl_offsets,
                                          dsetname, numelems, rowGroupSize,
                                          dtype, compression, errMsg): int;
    var localVals: [valIdxRange] t = distVals[valIdxRange];
    var locOffsets: [0..#locDom.size+1] int;
    locOffsets[0..#locDom.size] = segments[locDom];
    if locDom.high == segments.domain.high then
      locOffsets[locOffsets.domain.high] = extraOffset;
    else
      locOffsets[locOffsets.domain.high] = segments[locDom.high+1];

    var pqErr = new parquetErrorMsg();

    var valPtr: c_void_ptr = nil;
    if localVals.size != 0 {
      valPtr = c_ptrTo(localVals);
    }

    if c_writeListColumnToParquet(filename.localize().c_str(), c_ptrTo(locOffsets), valPtr,
                                   dsetname.localize().c_str(), locOffsets.size-1, ROWGROUPS,
                                   c_dtype, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
        pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
      }
  }

  proc writeSegArrayParquet(filename: string, dsetName: string, c_dtype, segments_entry, values_entry, compression: int): bool throws {
    // get the array of segments
    var segments = segments_entry.a;

    var prefix: string;
    var extension: string;
  
    (prefix, extension) = getFileMetadata(filename);

    // Generate the filenames based upon the number of targetLocales.
    var filenames = generateFilenames(prefix, extension, segments.targetLocales().size);

    //Generate a list of matching filenames to test against. 
    var matchingFilenames = getMatchingFilenames(prefix, extension);

    var filesExist = processParquetFilenames(filenames, matchingFilenames, TRUNCATE);
    
    const extraOffset = values_entry.size;
    const lastOffset = if segments.size == 0 then 0 else segments[segments.domain.high]; // prevent index error when empty
    const lastValIdx = values_entry.a.domain.high;
    ref olda = values_entry.a;

    // pull values to the locale of the offset
    coforall (loc, idx) in zip(segments.targetLocales(), filenames.domain) with (ref olda) do on loc {
      const myFilename = filenames[idx];
      const locDom = segments.localSubdomain();
      var dims: [0..#1] int;
      dims[0] = locDom.size: int;

      if (locDom.isEmpty() || locDom.size <= 0) {
        // we know append is not supported so creating new empty file
        createEmptyListParquetFile(myFilename, dsetName, c_dtype, compression);
      } else {
        var localSegments = segments[locDom];
        var startValIdx = localSegments[locDom.low];

        var endValIdx = if (lastOffset == localSegments[locDom.high]) then lastValIdx else segments[locDom.high + 1] - 1;
              
        var valIdxRange = startValIdx..endValIdx;
        writeSegArrayComponent(myFilename, dsetName, olda, valIdxRange, segments, locDom, extraOffset, lastOffset, lastValIdx, c_dtype, compression);
      }
    }
    return filesExist; // trigger warning if overwrite occuring
  }

  proc writeStrSegArrayParquet(filename: string, dsetName: string, segments_entry, values_entry, compression: int): bool throws {
    extern proc c_writeStrListColumnToParquet(filename, chpl_segs, chpl_offsets, chpl_arr,
                                          dsetname, numelems, rowGroupSize,
                                          dtype, compression, errMsg): int;
    // get the array of segments
    var segments = segments_entry.a;

    var prefix: string;
    var extension: string;
  
    (prefix, extension) = getFileMetadata(filename);

    // Generate the filenames based upon the number of targetLocales.
    var filenames = generateFilenames(prefix, extension, segments.targetLocales().size);

    //Generate a list of matching filenames to test against. 
    var matchingFilenames = getMatchingFilenames(prefix, extension);

    var filesExist = processParquetFilenames(filenames, matchingFilenames, TRUNCATE); // we know append is not supported

    // Note - seg/segment refers to segarray offsets and off/offsets refers to string object offsets
    ref oldOff = values_entry.offsetsEntry.a;
    ref oldVal = values_entry.bytesEntry.a;
    const extraSegment = values_entry.offsetsEntry.size;
    const extraOffset = values_entry.bytesEntry.size;
    const lastOffset = if segments.size == 0 then 0 else segments[segments.domain.high]; // prevent index error when empty
    const lastOffsetIdx = oldOff.domain.high;
    const lastValIdx = oldVal.domain.high;

    // pull values to the locale of the offset
    coforall (loc, idx) in zip(segments.targetLocales(), filenames.domain) with (ref oldOff, ref oldVal) do on loc {
      const myFilename = filenames[idx];

      const locDom = segments.localSubdomain();

      if (locDom.isEmpty() || locDom.size <= 0) {
        // we know append is not supported so creating new empty file
        var c_dtype = ARROWSTRING;
        createEmptyListParquetFile(myFilename, dsetName, c_dtype, compression);
      }
      else {
        var localSegments = segments[locDom];        
        var locSegments: [0..#locDom.size+1] int;
        locSegments[0..#locDom.size] = segments[locDom];
        if locDom.high == segments.domain.high then
          locSegments[locSegments.domain.high] = extraSegment;
        else
          locSegments[locSegments.domain.high] = segments[locDom.high+1];

        var startOffsetIdx = localSegments[locDom.low];
        var endOffsetIdx = if (lastOffset == localSegments[locDom.high]) then lastOffsetIdx else segments[locDom.high + 1] - 1;
        var offIdxRange = startOffsetIdx..endOffsetIdx;

        var pqErr = new parquetErrorMsg();
        var dtypeRep = ARROWSTRING;
        var valPtr: c_void_ptr = nil;
        var offPtr: c_void_ptr = nil;

        // need to get the local string values
        if offIdxRange.size > 0 {
          var localOffsets: [offIdxRange] int = oldOff[offIdxRange];
          var startValIdx = oldOff[offIdxRange.low];
          var endValIdx = if (lastOffsetIdx == offIdxRange.high) then lastValIdx else oldOff[offIdxRange.high + 1] - 1;
          var valIdxRange = startValIdx..endValIdx;
          var localVals: [valIdxRange] uint(8) = oldVal[valIdxRange];

          var locOffsets: [0..#offIdxRange.size+1] int;
          locOffsets[0..#offIdxRange.size] = oldOff[offIdxRange];
          
          if offIdxRange.high == oldOff.domain.high {
            locOffsets[locOffsets.domain.high] = extraOffset;
          } else {
            locOffsets[locOffsets.domain.high] = oldOff[offIdxRange.high+1];
          }
          
          if localVals.size > 0 {
            valPtr = c_ptrTo(localVals);
          }
          if locOffsets.size > 0 {
            offPtr = c_ptrTo(locOffsets);
          }
          // the call to c must be within the if block so the arrays stay in scope
          if c_writeStrListColumnToParquet(myFilename.localize().c_str(), c_ptrTo(locSegments), offPtr, 
                                      valPtr, dsetName.localize().c_str(), locSegments.size-1, 
                                      ROWGROUPS, dtypeRep, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
            pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
          }
        }
        else {
          // empty segment case
          if c_writeStrListColumnToParquet(myFilename.localize().c_str(), c_ptrTo(locSegments), offPtr, 
                                      valPtr, dsetName.localize().c_str(), locSegments.size-1, 
                                      ROWGROUPS, dtypeRep, compression, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
            pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
          }
        }
        
      }
    }
    return filesExist; // trigger warning if overwrite occuring
  }

  proc segarray_toParquetMsg(msgArgs: MessageArgs, st: borrowed SymTab): bool throws {
    var mode = msgArgs.get("mode").getIntValue();
    var filename: string = msgArgs.getValueOf("prefix");
    var entry = st.lookup(msgArgs.getValueOf("values"));
    var dsetname = msgArgs.getValueOf("dset");
    var compression = msgArgs.getValueOf("compression").toUpper(): CompressionType;

    // because append has been depreacted, support is not being added for SegArray. 
    if mode == APPEND {
      throw getErrorWithContext(
                msg="APPEND write mode is not supported for SegArray.",
                lineNumber=getLineNumber(), 
                routineName=getRoutineName(), 
                moduleName=getModuleName(), 
                errorClass='WriteModeError');
    }

    // segments is always int64
    var segments = toSymEntry(toGenSymEntry(st.lookup(msgArgs.getValueOf("segments"))), int);

    var genVal = toGenSymEntry(st.lookup(msgArgs.getValueOf("values")));
    
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
        var errorMsg = "Writing Parquet files not supported for %s type".doFormat(genVal.dtype);
        pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        throw getErrorWithContext(
                   msg=errorMsg,
                   lineNumber=getLineNumber(), 
                   routineName=getRoutineName(), 
                   moduleName=getModuleName(), 
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
            var errorMsg = "Unable to write object type %s to Parquet file.".doFormat(objType);
            pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    } catch e: FileNotFoundError {
      var errorMsg = "Unable to open %s for writing: %s".doFormat(msgArgs.getValueOf("filename"),e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: MismatchedAppendError {
      var errorMsg = "Mismatched append %s".doFormat(e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: WriteModeError {
      var errorMsg = "Write mode error %s".doFormat(e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: Error {
      var errorMsg = "problem writing to file %s".doFormat(e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    if warnFlag {
      var warnMsg: string = "Warning: possibly overwriting existing files matching filename pattern";
      pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),warnMsg);
      return new MsgTuple(warnMsg, MsgType.WARNING);
    } else {
      var repMsg: string = "Dataset written successfully!";
      pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
  }

  proc writeMultiColParquet(filename: string, col_names: [] string, 
                              ncols: int, sym_names: [] string, col_objTypes: [] string, targetLocales: [] locale, 
                              compression: int, st: borrowed SymTab): bool throws {

    extern proc c_writeMultiColToParquet(filename, column_names, ptr_arr, offset_arr, objTypes,
                                      datatypes, segArr_sizes, colnum, numelems, rowGroupSize, compression, errMsg): int;

    var prefix: string;
    var extension: string;
    (prefix, extension) = getFileMetadata(filename);

    // Generate the filenames based upon the number of targetLocales.
    var filenames = generateFilenames(prefix, extension, targetLocales.size);

    //Generate a list of matching filenames to test against. 
    var matchingFilenames = getMatchingFilenames(prefix, extension);

    // TODO when APPEND is fully deprecated update this to not need the mode.
    var filesExist = processParquetFilenames(filenames, matchingFilenames, TRUNCATE); // set to truncate. We will not be supporting appending. 

    coforall (loc, idx) in zip(targetLocales, filenames.domain) do on loc {
      var pqErr = new parquetErrorMsg();
      const fname = filenames[idx];

      var ptrList: [0..#ncols] c_void_ptr;
      var segmentPtr: [0..#ncols] c_void_ptr; // ptrs to offsets for SegArray. Know number of rows so we know where to stop
      var objTypes: [0..#ncols] int; // ObjType enum integer values
      var datatypes: [0..#ncols] int;
      var sizeList: [0..#ncols] int;
      var segarray_sizes: [0..#ncols] int; // track # of values in each column. Used to determine last segment size.

      var my_column_names = col_names;
      var c_names: [0..#ncols] c_string;

      var segment_ct: [0..#ncols] int;
      var seg_sizes_str: [0..#ncols] int; // Track the sizes of string columns and # of segments in segarray str column
      var val_sizes_str: [0..#ncols] int; // Track # of values making strings coming to locale
      var seg_sizes_int: [0..#ncols] int; // only fill in sizes for int, uint segarray columns
      var seg_sizes_real: [0..#ncols] int; // only fill in sizes for float segarray columns
      var seg_sizes_bool: [0..#ncols] int; // only fill in sizes for bool segarray columns
      forall (i, column, ot) in zip(0..#ncols, sym_names, col_objTypes) {
        var x: int;
        var objType = ot.toUpper(): ObjType;

        if objType == ObjType.STRINGS {
          var entry = st.lookup(column);
          var e: SegStringSymEntry = toSegStringSymEntry(entry);
          var segStr = new SegString("", e);
          ref ss = segStr;
          var lens = ss.getLengths();
          const locDom = ss.offsets.a.localSubdomain();
          for d in locDom do x += lens[d];
          seg_sizes_str[i] = x;
        }
        else if objType == ObjType.SEGARRAY {
          // parse the json in column to get the component pdarrays
          var components: map(string, string) = jsonToMap(column);
          var seg_entry = getGenericTypedArrayEntry(components["segments"], st);
          var segments = toSymEntry(seg_entry, int);
          ref sa = segments.a;
          const saD = sa.domain;
          var lens: [saD] int;
          const high = saD.high;
          const locDom = sa.localSubdomain();
          var values = getGenericTypedArrayEntry(components["values"], st);

          segment_ct[i] += locDom.size;
          if values.dtype == DType.Strings && locDom.size > 0 {
            var e: SegStringSymEntry = toSegStringSymEntry(values);
            var segStr = new SegString("", e);
            ref ss = segStr;
            var lens = ss.getLengths();
            const lastOffset = if sa.size == 0 then 0 else sa[high];
            const lastOffsetIdx = segStr.offsets.a.domain.high;
            var startOffsetIdx = sa[locDom.low];
            var endOffsetIdx = if (lastOffset == sa[locDom.high]) then lastOffsetIdx else sa[locDom.high + 1] - 1;
            var offIdxRange = startOffsetIdx..endOffsetIdx;
            var str_bytes: int;
            for d in offIdxRange do str_bytes += lens[d];

            seg_sizes_str[i] = str_bytes;
          }

          lens = [(i, s) in zip (saD, sa)] if i == high then values.size - s else sa[i+1] - s;
          for d in locDom do x += lens[d];
          select values.dtype {
            when DType.Int64 {
              seg_sizes_int[i] = x;
            }
            when DType.UInt64 {
              seg_sizes_int[i] = x;
            }
            when DType.Float64 {
              seg_sizes_real[i] = x;
            }
            when DType.Bool {
              seg_sizes_bool[i] = x;
            }
            when DType.Strings {
              val_sizes_str[i] = x;
            }
            otherwise {
              throw getErrorWithContext(
                    msg="Unsupported SegArray DType for writing to Parquet, ".doFormat(values.dtype: string),
                    lineNumber=getLineNumber(), 
                    routineName=getRoutineName(), 
                    moduleName=getModuleName(), 
                    errorClass='DataTypeError'
                  );
            }
          }
        }
      }

      var totalSegs: int = + reduce segment_ct; // total # of offsets on locale
      var segment_tracking: [0..#totalSegs] int; // array to write offset values into after adjusting for locale
      var segment_idx = (+ scan segment_ct) - segment_ct; // offset start indexes for each column

      var locSize_str: int = + reduce seg_sizes_str;
      var str_vals: [0..#locSize_str] uint(8);
      var locSize_int: int = + reduce seg_sizes_int;
      var int_vals: [0..#locSize_int] int; //int and uint written the same so no conversions will be needed
      var locSize_real: int = + reduce seg_sizes_real;
      var real_vals: [0..#locSize_real] real;
      var locSize_bool: int = + reduce seg_sizes_bool;
      var bool_vals: [0..#locSize_bool] bool;

      // indexes for which values go to which columns
      var str_idx = (+ scan seg_sizes_str) - seg_sizes_str;
      var int_idx = (+ scan seg_sizes_int) - seg_sizes_int;
      var real_idx = (+ scan seg_sizes_real) - seg_sizes_real;
      var bool_idx = (+ scan seg_sizes_bool) - seg_sizes_bool;

      // populate data based on object and data types
      forall (i, column, ot, si, ui, ri, bi, segidx) in zip(0..#ncols, sym_names, col_objTypes, str_idx, int_idx, real_idx, bool_idx, segment_idx) {
        // generate the local c string list of column names
        c_names[i] = my_column_names[i].localize().c_str();

        select ot.toUpper(): ObjType {
          when ObjType.STRINGS {
            var entry = st.lookup(column);
            var e: SegStringSymEntry = toSegStringSymEntry(entry);
            var segStr = new SegString("", e);
            ref ss = segStr;
            var A = ss.offsets.a;
            const lastOffset = if A.size == 0 then 0 else A[A.domain.high]; // prevent index error when empty
            const lastValIdx = ss.values.a.domain.high;
            const locDom = ss.offsets.a.localSubdomain();

            objTypes[i] = ObjType.STRINGS: int;
            datatypes[i] = ARROWSTRING;

            if locDom.size > 0 {
              var localOffsets = A[locDom];
              var startValIdx = localOffsets[locDom.low];
              var endValIdx = if (lastOffset == localOffsets[locDom.high]) then lastValIdx else A[locDom.high + 1] - 1;
              var valIdxRange = startValIdx..endValIdx;
              ref olda = ss.values.a;
              str_vals[si..#valIdxRange.size] = olda[valIdxRange];
              ptrList[i] = c_ptrTo(str_vals[si]): c_void_ptr;
              sizeList[i] = locDom.size;
            }
          }
          when ObjType.SEGARRAY {
            // parse the json in column to get the component pdarrays
            var components: map(string, string) = jsonToMap(column);

            // access segments symentry
            var seg_entry = getGenericTypedArrayEntry(components["segments"], st);
            var segments = toSymEntry(seg_entry, int);

            ref S = segments.a;
            const locDom = segments.a.localSubdomain();
            objTypes[i] = ObjType.SEGARRAY: int;            

            if locDom.size > 0 {
              const lastSegment = if S.size == 0 then 0 else S[S.domain.high]; // prevent index error when empty;
              const localSegments = S[locDom];
              const startValIdx = localSegments[locDom.low];
              sizeList[i] = locDom.size;
              segment_tracking[segidx..#locDom.size] = localSegments - startValIdx;
              segmentPtr[i] = c_ptrTo(segment_tracking[segidx]);
              
              var valEntry = getGenericTypedArrayEntry(components["values"], st);
              select valEntry.dtype {
                when DType.Int64 {
                  segarray_sizes[i] = seg_sizes_int[i];
                  var values = toSymEntry(valEntry, int);
                  const lastValIdx = values.a.domain.high;

                  datatypes[i] = ARROWINT64;

                  const endValIdx = if (lastSegment == localSegments[locDom.high]) then lastValIdx else S[locDom.high + 1] - 1;
                  var valIdxRange = startValIdx..endValIdx;
                  ref olda = values.a;
                  int_vals[ui..#valIdxRange.size] = olda[valIdxRange];
                  ptrList[i] = c_ptrTo(int_vals[ui]): c_void_ptr;
                }
                when DType.UInt64 {
                  segarray_sizes[i] = seg_sizes_int[i];
                  var values = toSymEntry(valEntry, uint);
                  const lastValIdx = values.a.domain.high;

                  datatypes[i] = ARROWUINT64;

                  var endValIdx = if (lastSegment == localSegments[locDom.high]) then lastValIdx else S[locDom.high + 1] - 1;
                  var valIdxRange = startValIdx..endValIdx;
                  ref olda = values.a;
                  int_vals[ui..#valIdxRange.size] = olda[valIdxRange]: int;
                  ptrList[i] = c_ptrTo(int_vals[ui]): c_void_ptr;
                }
                when DType.Float64 {
                  segarray_sizes[i] = seg_sizes_real[i];
                  var values = toSymEntry(valEntry, real);
                  const lastValIdx = values.a.domain.high;

                  datatypes[i] = ARROWDOUBLE;

                  var endValIdx = if (lastSegment == localSegments[locDom.high]) then lastValIdx else S[locDom.high + 1] - 1;
                  var valIdxRange = startValIdx..endValIdx;
                  ref olda = values.a;
                  real_vals[ri..#valIdxRange.size] = olda[valIdxRange];
                  ptrList[i] = c_ptrTo(real_vals[ri]): c_void_ptr;
                }
                when DType.Bool {
                  segarray_sizes[i] = seg_sizes_bool[i];
                  var values = toSymEntry(valEntry, bool);
                  const lastValIdx = values.a.domain.high;

                  datatypes[i] = ARROWBOOLEAN;

                  var endValIdx = if (lastSegment == localSegments[locDom.high]) then lastValIdx else S[locDom.high + 1] - 1;
                  var valIdxRange = startValIdx..endValIdx;
                  ref olda = values.a;
                  bool_vals[bi..#valIdxRange.size] = olda[valIdxRange];
                  ptrList[i] = c_ptrTo(bool_vals[bi]): c_void_ptr;
                }
                when DType.Strings {
                  segarray_sizes[i] = val_sizes_str[i];
                  var values = toSegStringSymEntry(valEntry);
                  ref oldOff = values.offsetsEntry.a;
                  ref oldVal = values.bytesEntry.a;

                  const extraSegment = oldOff.size;
                  const extraOffset = oldVal.size;
                  const lastSegment = if S.size == 0 then 0 else S[S.domain.high];
                  const lastOffsetIdx = oldOff.domain.high;
                  const lastValIdx = oldVal.domain.high;

                  datatypes[i] = ARROWSTRING;
                  
                  var startOffsetIdx = localSegments[locDom.low];
                  var endOffsetIdx = if (lastSegment == localSegments[locDom.high]) then lastOffsetIdx else S[locDom.high + 1] - 1;
                  var offIdxRange = startOffsetIdx..endOffsetIdx;
                  if offIdxRange.size > 0 {

                    var localOffsets: [offIdxRange] int = oldOff[offIdxRange];
                    var startValIdx = localOffsets[offIdxRange.low];
                    
                    var endValIdx = if (lastOffsetIdx == offIdxRange.high) then lastValIdx else oldOff[offIdxRange.high + 1] - 1;
                    var valIdxRange = startValIdx..endValIdx;
                    str_vals[si..#valIdxRange.size] = oldVal[valIdxRange];
                    ptrList[i] = c_ptrTo(str_vals[si]): c_void_ptr;
                  }
                }
                otherwise {
                  throw getErrorWithContext(
                    msg="Unsupported SegArray DType for writing to Parquet, ".doFormat(valEntry.dtype: string),
                    lineNumber=getLineNumber(), 
                    routineName=getRoutineName(), 
                    moduleName=getModuleName(), 
                    errorClass='DataTypeError'
                  );
                }
              }
            }
            else {
              // set the datatype for empty locales to ensure that metadata is correct in all files
              var valEntry = getGenericTypedArrayEntry(components["values"], st);
              select valEntry.dtype {
                when DType.Int64 {
                  datatypes[i] = ARROWINT64;
                }
                when DType.UInt64 {
                  datatypes[i] = ARROWUINT64;
                }
                when DType.Float64 {
                  datatypes[i] = ARROWDOUBLE;
                }
                when DType.Bool {
                  datatypes[i] = ARROWBOOLEAN;
                }
                when DType.Strings {
                  datatypes[i] = ARROWSTRING;
                }
                otherwise {
                  throw getErrorWithContext(
                    msg="Unsupported SegArray DType for writing to Parquet, ".doFormat(valEntry.dtype: string),
                    lineNumber=getLineNumber(), 
                    routineName=getRoutineName(), 
                    moduleName=getModuleName(), 
                    errorClass='DataTypeError'
                  );
                }
              }
            }
          }
          when ObjType.PDARRAY {
            var entry = getGenericTypedArrayEntry(column, st);
            select entry.dtype {
              when DType.Int64 {
                var e = toSymEntry(toGenSymEntry(entry), int);
                var locDom = e.a.localSubdomain();
                objTypes[i] = ObjType.PDARRAY: int;
                datatypes[i] = ARROWINT64;
                // set the pointer to the entry array in the list of Pointers
                if locDom.size > 0 {
                  ptrList[i] = c_ptrTo(e.a[locDom]): c_void_ptr;
                  sizeList[i] = locDom.size;
                }
              }
              when DType.UInt64 {
                var e = toSymEntry(toGenSymEntry(entry), uint);
                var locDom = e.a.localSubdomain();
                objTypes[i] = ObjType.PDARRAY: int;
                datatypes[i] = ARROWUINT64;
                // set the pointer to the entry array in the list of Pointers
                if locDom.size > 0 {
                  ptrList[i] = c_ptrTo(e.a[locDom]): c_void_ptr;
                  sizeList[i] = locDom.size;
                }
              }
              when DType.Float64 {
                var e = toSymEntry(toGenSymEntry(entry), real);
                var locDom = e.a.localSubdomain();
                objTypes[i] = ObjType.PDARRAY: int;
                datatypes[i] = ARROWDOUBLE;
                // set the pointer to the entry array in the list of Pointers
                if locDom.size > 0 {
                  ptrList[i] = c_ptrTo(e.a[locDom]): c_void_ptr;
                  sizeList[i] = locDom.size;
                }
              }
              when DType.Bool {
                var e = toSymEntry(toGenSymEntry(entry), bool);
                var locDom = e.a.localSubdomain();
                objTypes[i] = ObjType.PDARRAY: int;
                datatypes[i] = ARROWBOOLEAN;
                // set the pointer to the entry array in the list of Pointers
                if locDom.size > 0 {
                  ptrList[i] = c_ptrTo(e.a[locDom]): c_void_ptr;
                  sizeList[i] = locDom.size;
                }
              }
              otherwise {
                throw getErrorWithContext(
                  msg="Unsupported PDArray DType for writing to Parquet, ".doFormat(entry.dtype: string),
                  lineNumber=getLineNumber(), 
                  routineName=getRoutineName(), 
                  moduleName=getModuleName(), 
                  errorClass='DataTypeError'
                );
              }
            }
          }
          otherwise {
            throw getErrorWithContext(
              msg="Writing Parquet files (multi-column) does not support %s columns.".doFormat(ot),
              lineNumber=getLineNumber(), 
              routineName=getRoutineName(), 
              moduleName=getModuleName(), 
              errorClass='DataTypeError'
            );
          }
        }
      }
      // validate all elements same size
      var numelems: int = sizeList[0];
      if !(&& reduce (sizeList==numelems)) {
        throw getErrorWithContext(
              msg="Parquet columns must be the same size",
              lineNumber=getLineNumber(), 
              routineName=getRoutineName(), 
              moduleName=getModuleName(), 
              errorClass='WriteModeError'
        );
      }
      
      var result: int = c_writeMultiColToParquet(fname.localize().c_str(), c_ptrTo(c_names), c_ptrTo(ptrList), c_ptrTo(segmentPtr), c_ptrTo(objTypes), c_ptrTo(datatypes), c_ptrTo(segarray_sizes), ncols, numelems, ROWGROUPS, compression, c_ptrTo(pqErr.errMsg));
      if result == ARROWERROR {
        pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
      }
    }
    return filesExist;
  }

  proc _identifyTargetLocales(name: string, objType: string, st: borrowed SymTab) throws {
    var targetLocales;
    select objType.toUpper(): ObjType {
      when ObjType.STRINGS {
        var entry = st.lookup(name);
        var e: SegStringSymEntry = toSegStringSymEntry(entry);
        var segStr = new SegString("", e);
        targetLocales = segStr.offsets.a.targetLocales();
      }
      when ObjType.SEGARRAY {
        // parse the json in column to get the component pdarrays
        var components: map(string, string) = jsonToMap(name);

        // access segments symentry
        var seg_entry = getGenericTypedArrayEntry(components["segments"], st);
        var segments = toSymEntry(seg_entry, int);
        targetLocales = segments.a.targetLocales();
      }
      when ObjType.PDARRAY {
        var entry = st.lookup(name);
        var entryDtype = (entry: borrowed GenSymEntry).dtype;
        select entryDtype {
          when DType.Int64 {
            var e = toSymEntry(toGenSymEntry(entry), int);
            targetLocales = e.a.targetLocales();
          }
          when DType.UInt64 {
            var e = toSymEntry(toGenSymEntry(entry), uint);
            targetLocales = e.a.targetLocales();
          }
          when DType.Float64 {
            var e = toSymEntry(toGenSymEntry(entry), real);
            targetLocales = e.a.targetLocales();
          }
          when DType.Bool {
            var e = toSymEntry(toGenSymEntry(entry), bool);
            targetLocales = e.a.targetLocales();
          }
          otherwise {
            throw getErrorWithContext(
              msg="Writing Parquet files (multi-column) does not support columns of type %s".doFormat(entryDtype: string),
              lineNumber=getLineNumber(), 
              routineName=getRoutineName(), 
              moduleName=getModuleName(), 
              errorClass='DataTypeError'
            );
          }
        }
      }
      otherwise {
        throw getErrorWithContext(
          msg="Writing Parquet files (multi-column) does not support %s columns.".doFormat(objType),
          lineNumber=getLineNumber(), 
          routineName=getRoutineName(), 
          moduleName=getModuleName(), 
          errorClass='DataTypeError'
        );
      }
    }
    return targetLocales;
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

    // use the first entry to identify target locales. Assuming all have same distribution
    var targetLocales = _identifyTargetLocales(sym_names[0], col_objType_strs[0], st);
    
    var warnFlag: bool;
    try {
      warnFlag = writeMultiColParquet(filename, col_names, ncols, sym_names, col_objType_strs, targetLocales, compression:int, st);
    } catch e: FileNotFoundError {
      var errorMsg = "Unable to open %s for writing: %s".doFormat(filename,e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: WriteModeError {
      var errorMsg = "Write mode error %s".doFormat(e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: Error {
      var errorMsg = "problem writing to file %s".doFormat(e.message());
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    if warnFlag {
      var warnMsg = "Warning: possibly overwriting existing files matching filename pattern";
      return new MsgTuple(warnMsg, MsgType.WARNING);
    } else {
      var repMsg = "File written successfully!";
      pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
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
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    // If the filename represents a glob pattern, retrieve the locale 0 filename
    if isGlobPattern(filename) {
      // Attempt to interpret filename as a glob expression and ls the first result
      var tmp = glob(filename);

      if tmp.size <= 0 {
        var errorMsg = "Cannot retrieve filename from glob expression %s, check file name or format".doFormat(filename);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
            
      // Set filename to globbed filename corresponding to locale 0
      filename = tmp[tmp.domain.first];
    }
        
    // Check to see if the file exists. If not, return an error message
    if !exists(filename) {
      var errorMsg = "File %s does not exist in a location accessible to Arkouda".doFormat(filename);
      return new MsgTuple(errorMsg,MsgType.ERROR);
    }
        
    try {
      extern proc c_getDatasetNames(filename, dsetResult, readNested, errMsg): int(32);
      extern proc strlen(a): int;
      var pqErr = new parquetErrorMsg();
      var res: c_ptr(uint(8));
      defer {
        extern proc c_free_string(ptr);
        c_free_string(res);
      }
      if c_getDatasetNames(filename.c_str(), c_ptrTo(res), read_nested,
                           c_ptrTo(pqErr.errMsg)) == ARROWERROR {
        pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
      }
      try! repMsg = string.createCopyingBuffer(res, strlen(res));
      var items = new list(repMsg.split(",")); // convert to json

      repMsg = formatJson(items);
    } catch e : Error {
      var errorMsg = "Failed to process Parquet file %?".doFormat(e.message());
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
      var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".doFormat(
                                                                                              1, msgArgs.getValueOf("dsets"));
      pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    try {
      filelist = msgArgs.get("filenames").getList(nfiles);
    } catch {
      // limit length of file names to 2000 chars
      var n: int = 1000;
      var jsonfiles = msgArgs.getValueOf("filenames");
      var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
      var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".doFormat(nfiles, files);
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
                     "glob expanded %s to %i files".doFormat(filelist[0], tmp.size));
      if tmp.size == 0 {
        var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".doFormat(filelist[0]);
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

    var rnames: list((string, ObjType, string)); // tuple (dsetName, item type, id)
    
    for (dsetidx, dsetname) in zip(dsetdom, dsetnames) do {
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                types[dsetidx] = getArrType(fname, dsetname);
                sizes[i] = getArrSize(fname);
            } catch e : Error {
                // This is only type of error thrown by Parquet
                fileErrorMsg = "Other error in accessing file %s: %s".doFormat(fname,e.message());
                pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
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
          var errorMsg = "Null indices only supported on Parquet string columns, not %? columns".doFormat(ty);
          pqLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }

    repMsg = _buildReadAllMsgJson(rnames, false, 0, fileErrors, st);
    pqLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg,MsgType.NORMAL);
  }

  use CommandMap;
  registerFunction("readAllParquet", readAllParquetMsg, getModuleName());
  registerFunction("toParquet_multi", toParquetMultiColMsg, getModuleName());
  registerFunction("writeParquet", toparquetMsg, getModuleName());
  registerFunction("lspq", lspqMsg, getModuleName());
  registerFunction("getnullparquet", nullIndicesMsg, getModuleName());
  ServerConfig.appendToConfigStr("ARROW_VERSION", getVersionInfo());
}
