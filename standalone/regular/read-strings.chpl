use CTypes;
use Reflection;
use Time;
require "ArrowFunctions.h";
require "ArrowFunctions.o";

config const filename = '/Users/ben.mcdonald/arkouda/test-file.parquet';
config const colname = 'strings_array';
config const batchSize = 100;
extern var ARROWERROR: c_int;

proc main() {
  var totalT: stopwatch;
  var t1: stopwatch;
  var t2: stopwatch;
  var t3: stopwatch;

  totalT.start();
  
  var filenames: [0..#1] string;
  filenames[0] = filename;

  t1.start();
  var len = getArrSize(filenames[0]);
  t1.stop();

  t2.start();
  var seg: [0..#len] int;
  var sizes: [0..#1] int = len;
  var byteSizes = calcStrSizesAndOffset(seg, filenames, sizes, colname);
  seg = (+ scan seg) - seg;
  var ret: [0..#(+ reduce byteSizes)] uint(8);
  t2.stop();
  
  t3.start();
  readStrFilesByName(ret, filenames, byteSizes, colname);
  t3.stop();

  totalT.stop();
  writeln("Total time     : ", totalT.elapsed());
  writeln("Get size       : ", t1.elapsed());
  writeln("Created seg    : ", t2.elapsed());
  writeln("Read files     : ", t3.elapsed());
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

proc calcStrSizesAndOffset(offsets: [] ?t, filenames: [] string, sizes: [] int, dsetname: string) throws {
  var (subdoms, length) = getSubdomains(sizes);

  var byteSizes: [filenames.domain] int;

  coforall loc in offsets.targetLocales() with (ref byteSizes) do on loc {
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

proc readStrFilesByName(A: [] ?t, filenames: [] string, sizes: [] int, dsetname: string) throws {
  extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, byteLength, errMsg): int;
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
                                  batchSize, -1, c_ptrTo(pqErr.errMsg)) == ARROWERROR {
              pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
            }
            A[filedom] = col;
          }
        }
      }
    }
}

proc getStrColSize(filename: string, dsetname: string, ref offsets: [] int) throws {
  extern proc c_getStringColumnNumBytes(filename, colname, offsets, numElems, startIdx, batchSize, errMsg): int;
  var pqErr = new parquetErrorMsg();

  var byteSize = c_getStringColumnNumBytes(filename.localize().c_str(),
                                           dsetname.localize().c_str(),
                                           c_ptrTo(offsets),
                                           offsets.size, 0, 256,
                                           c_ptrTo(pqErr.errMsg));
    
  if byteSize == ARROWERROR then
    pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
  return byteSize;
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

type stridableRange = range(strides=strideKind.any);
proc stridable(a) param {
  return !(a.strides==strideKind.one);
}

proc domain_intersection(d1: domain(1), d2: domain(1)) {
  var low = max(d1.low, d2.low);
  var high = min(d1.high, d2.high);
  if (d1.stride !=1) && (d2.stride != 1) {
    //TODO: change this to throw
    halt("At least one domain must have stride 1");
  }
  if !stridable(d1) && !stridable(d2) {
    return {low..high};
  } else {
    var stride = max(d1.stride, d2.stride);
    return {low..high by stride};
  }
}

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
    throw new Error(err);
  }
}
