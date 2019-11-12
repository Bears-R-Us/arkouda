module CommAggregation {

  // TODO these parameters need to be tuned and size should be user-settable at
  // creation time. iters before yield should be based on numLocales & buffSize
  private config const maxItersBeforeYield = 4096;
  private config const dstBuffSize = 4096;
  private config const srcBuffSize = 4096;

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

    proc init(type elemType) {
      this.elemType = elemType;
    }

    proc deinit() {
      flush();
    }

    proc flush() {
      for loc in myLocaleSpace {
        _flushBuffer(loc, bufferIdxs[loc]);
      }
    }

    inline proc copy(ref dst: elemType, srcVal: elemType) {
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
        itersSinceYield = 0;
      } else if itersSinceYield % maxItersBeforeYield == 0 {
        chpl_task_yield();
        itersSinceYield = 0;
      } else {
        itersSinceYield += 1;
      }
    }

    proc _flushBuffer(loc: int, ref bufferIdx) {
      const myBufferIdx = bufferIdx;
      if myBufferIdx == 0 then return;
      on Locales[loc] {
        // GET the buffered dst addrs and src values, and assign
        var localBuffer = buffer[loc][0..#myBufferIdx];
        for (dstAddr, srcVal) in localBuffer {
          (dstAddr:c_ptr(elemType)).deref() = srcVal;
        }
      }
      bufferIdx = 0;
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

    proc init(type elemType) {
      this.elemType = elemType;
    }

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
        itersSinceYield = 0;
      } else if itersSinceYield % maxItersBeforeYield == 0 {
        chpl_task_yield();
        itersSinceYield = 0;
      } else {
        itersSinceYield += 1;
      }
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

  // TODO can this use c_ptrTo?
  private inline proc getAddr(const ref p): c_ptr(p.type) {
    return __primitive("_wide_get_addr", p): c_ptr(p.type);
  }
}
