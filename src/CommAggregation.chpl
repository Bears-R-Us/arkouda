module CommAggregation {
  use SysCTypes;
  use UnorderedCopy;
  private use CommPrimitives;

  // TODO these parameters need to be tuned and size should be user-settable at
  // creation time. iters before yield should be based on numLocales & buffSize
  private config const maxItersBeforeYield = 4096;
  private config const dstBuffSize = 4096;
  private config const srcBuffSize = 4096;


  /* Creates a new destination aggregator (dst/lhs will be remote). */
  proc newDstAggregator(type elemType, param useUnorderedCopy=false) {
    if CHPL_COMM == "none" || useUnorderedCopy {
      return new DstUnorderedAggregator(elemType);
    } else {
      return new DstAggregator(elemType);
    }
  }

  /* Creates a new source aggregator (src/rhs will be remote). */
  proc newSrcAggregator(type elemType, param useUnorderedCopy=false) {
    // SrcAggregator is not currently optimized for ugni
    if CHPL_COMM == "none" || CHPL_COMM =="ugni" || useUnorderedCopy {
      return new SrcUnorderedAggregator(elemType);
    } else {
      return new SrcAggregator(elemType);
    }
  }

  /*
   * Aggregates copy(ref dst, src). Optimized for when src is local.
   * Not parallel safe and is expected to be created on a per-task basis
   * High memory usage since there are per-destination buffers
   */
  record DstAggregator {
    type elemType;
    const bufferSize = dstBuffSize;
    const myLocaleSpace = LocaleSpace;
    var itersSinceYield: int;
    var buffer: [myLocaleSpace][0..#bufferSize] (c_void_ptr, elemType);
    var bufferIdxs: [myLocaleSpace] int;

    proc deinit() {
      flush();
    }

    proc flush() {
      for loc in myLocaleSpace {
        _flushBuffer(loc, bufferIdxs[loc]);
      }
    }

    inline proc copy(ref dst: elemType, const in srcVal: elemType) {
      // Get the locale of dst and the local address on that locale
      const loc = dst.locale.id;
      const dstAddr = getAddr(dst);

      // Get our current index into the buffer for dst's locale
      ref bufferIdx = bufferIdxs[loc];

      // Buffer the address and desired value
      buffer[loc][bufferIdx] = (dstAddr, srcVal);
      bufferIdx += 1;

      // Flush our buffer if it's full. If it's been a while since we've let
      // other tasks run, yield so that we're not blocking remote tasks from
      // flushing their buffers.
      if bufferIdx == bufferSize {
        _flushBuffer(loc, bufferIdx);
      } else if itersSinceYield % maxItersBeforeYield == 0 {
        chpl_task_yield();
        itersSinceYield = 0;
      }
      itersSinceYield += 1;
    }

    proc _flushBuffer(loc: int, ref bufferIdx) {
      const myBufferIdx = bufferIdx;
      if myBufferIdx == 0 then return;

      //
      // Allocate a remote buffer, PUT our data there, and then process the
      // remote buffer. We do this with 2 ons + PUT instead of 1 on + GET in
      // order to limit the lifetime of remote tasks for configurations that
      // yield while doing comm.
      //

      // Allocate a remote buffer (and capture the address to it)
      const origLoc = here.locale.id;
      var remBufferPtr: c_ptr((c_void_ptr, elemType));
      var remBufferPtrAddr = c_ptrTo(remBufferPtr);
      on Locales[loc] {
        var bufferPtr = c_malloc((c_void_ptr, elemType), myBufferIdx);
        PUT(c_ptrTo(bufferPtr), origLoc, remBufferPtrAddr, c_sizeof(bufferPtr.type));
      }

      // Send our buffered data to the remote node's buffer
      const size = myBufferIdx:size_t * c_sizeof((c_void_ptr, elemType));
      PUT(c_ptrTo(buffer[loc][0]), loc, remBufferPtr, size);

      // Process the remote buffer on the remote node
      on Locales[loc] {
        var bufferPtr = remBufferPtr;
        for i in 0..myBufferIdx-1 {
          var (dstAddr, srcVal) = bufferPtr[i];
          (dstAddr:c_ptr(elemType)).deref() = srcVal;
        }
       c_free(bufferPtr);
      }
      bufferIdx = 0;
    }
  }



  /* "Aggregator" that uses unordered copy instead of actually aggregating */
  pragma "no doc"
  record DstUnorderedAggregator {
    type elemType;

    proc deinit() {
      flush();
    }
    proc flush() {
      unorderedCopyTaskFence();
    }
    inline proc copy(ref dst: elemType, const in srcVal: elemType) {
      unorderedCopyWrapper(dst, srcVal);
    }
  }


  /*
   * Aggregates copy(ref dst, const ref src). Only works when dst is local.
   * Not parallel safe and is expected to be created on a per task basis
   * High memory usage since there are per-destination buffers
   */
  record SrcAggregator {
    type elemType;
    const bufferSize = srcBuffSize;
    const myLocaleSpace = LocaleSpace;
    var itersSinceYield: int;
    var dstBuffer: [myLocaleSpace][0..#bufferSize] c_void_ptr;
    var srcBuffer: [myLocaleSpace][0..#bufferSize] c_void_ptr;
    var bufferIdxs: [myLocaleSpace] int;

    proc deinit() {
      flush();
    }

    proc flush() {
      for loc in myLocaleSpace {
        _flushBuffer(loc, bufferIdxs[loc]);
      }
    }

    inline proc copy(ref dst: elemType, const ref src: elemType) {
      assert(dst.locale.id == here.id);
      const dstAddr = getAddr(dst);

      const loc = src.locale.id;
      const srcAddr = getAddr(src);

      ref bufferIdx = bufferIdxs[loc];
      srcBuffer[loc][bufferIdx] = srcAddr;
      dstBuffer[loc][bufferIdx] = dstAddr;
      bufferIdx += 1;

      if bufferIdx == bufferSize {
        _flushBuffer(loc, bufferIdx);
      } else if itersSinceYield % maxItersBeforeYield == 0 {
        chpl_task_yield();
        itersSinceYield = 0;
      }
      itersSinceYield += 1;
    }

    proc _flushBuffer(loc: int, ref bufferIdx) {
      const myBufferIdx = bufferIdx;
      if myBufferIdx == 0 then return;

      // Create an array to store the src values.
      var srcVals: [0..#myBufferIdx] elemType;
      on Locales[loc] {
        // GET the src addrs
        const localSrcAddrs = srcBuffer[loc][0..#myBufferIdx];
        // Create a local array to store the src values
        var localSrcVals:  [0..#myBufferIdx] elemType;
        for (srcVal, srcAddr) in zip (localSrcVals, localSrcAddrs) {
          srcVal = (srcAddr:c_ptr(elemType)).deref();
        }
        // PUT the src values back
        srcVals = localSrcVals;
      }

      // Assign the srcVal to the dstAddrs
      for (dstAddr, srcVal) in zip (dstBuffer[loc][0..#myBufferIdx], srcVals) {
        (dstAddr:c_ptr(elemType)).deref() = srcVal;
      }

      bufferIdx = 0;
    }
  }

  /* "Aggregator" that uses unordered copy instead of actually aggregating */
  pragma "no doc"
  record SrcUnorderedAggregator {
    type elemType;

    proc deinit() {
      flush();
    }
    proc flush() {
      unorderedCopyTaskFence();
    }
    inline proc copy(ref dst: elemType, const ref src: elemType) {
      assert(dst.locale.id == here.id);
      unorderedCopyWrapper(dst, src);
    }
  }

  //
  // Helper routines
  //

  // Unordered copy wrapper that also supports tuples. In 1.20 this will call
  // unorderedCopy for each tuple element, but in 1.21 it is a single call.
  private inline proc unorderedCopyWrapper(ref dst, const ref src): void {
    use Reflection;
    // Always resolves in 1.21, only resolves for numeric/bool types in 1.20
    if canResolve("unorderedCopy", dst, src) {
      unorderedCopy(dst, src);
    } else if isTuple(dst) && isTuple(src) {
      for param i in 1..dst.size {
        unorderedCopyWrapper(dst(i), src(i));
      }
    } else {
      compilerWarning("Missing optimized unorderedCopy for " + dst.type:string);
      dst = src;
    }
  }
}
