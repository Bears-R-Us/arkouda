module Parquet {
  use SysCTypes, CPtr, IO;
  use ServerErrors, ServerConfig;
  // Use reflection for error information
  use Reflection;
  if hasParquetSupport {
    require "ArrowFunctions.h";
    require "ArrowFunctions.o";
  }

  private config const ROWGROUPS = 512*1024*1024 / numBytes(int); // 512 mb of int64

  extern var ARROWINT64: c_int;
  extern var ARROWINT32: c_int;
  extern var ARROWUNDEFINED: c_int;
  extern var ARROWERROR: c_int;

  enum ArrowTypes { int64, int32, notimplemented };

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

  proc domain_intersection(d1: domain(1), d2: domain(1)) {
    var low = max(d1.low, d2.low);
    var high = min(d1.high, d2.high);
    if (d1.stride !=1) && (d2.stride != 1) {
      //TODO: change this to throw
      halt("At least one domain must have stride 1");
    }
    var stride = max(d1.stride, d2.stride);
    return {low..high by stride};
  }
  
  proc readFilesByName(A, filenames: [] string, sizes: [] int, dsetname: string) throws {
    extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, errMsg): int;
    var (subdoms, length) = getSubdomains(sizes);

    coforall loc in A.targetLocales() do on loc {
      var pqErr = new parquetErrorMsg();
      var locFiles = filenames;
      var locFiledoms = subdoms;
      for (filedom, filename) in zip(locFiledoms, locFiles) {
        for locdom in A.localSubdomains() {
          const intersection = domain_intersection(locdom, filedom);
          if intersection.size > 0 {
            var col: [filedom] int;
            if c_readColumnByName(filename.localize().c_str(), c_ptrTo(col),
                                  dsetname.localize().c_str(), filedom.size,
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
    return ArrowTypes.notimplemented;
  }

  proc writeDistArrayToParquet(A, filename, dsetname, rowGroupSize) throws {
    extern proc c_writeColumnToParquet(filename, chpl_arr, colnum,
                                       dsetname, numelems, rowGroupSize,
                                       errMsg): int;
    var filenames: [0..#A.targetLocales().size] string;
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
                                  c_ptrTo(pqErr.errMsg)) == ARROWERROR {
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

  proc write1DDistArrayParquet(filename: string, dsetname, A) throws {
    return writeDistArrayToParquet(A, filename, dsetname, ROWGROUPS);
  }
}
