module CustomCopyAggregation {
  use AggregationPrimitives;
  use ChplConfig;
  use CTypes;
  use CopyAggregation;

  private config param verboseAggregation = false;

  private param defaultBuffSize = if CHPL_TARGET_PLATFORM == "hpe-cray-ex" then 1024
                                                                  else if CHPL_COMM == "ugni" then 4096
                                                                  else 8192;

  private const yieldFrequency = getEnvInt("CHPL_AGGREGATION_YIELD_FREQUENCY", 1024);
  private const dstBuffSize = getEnvInt("CHPL_AGGREGATION_DST_BUFF_SIZE", defaultBuffSize);
  private const srcBuffSize = getEnvInt("CHPL_AGGREGATION_SRC_BUFF_SIZE", defaultBuffSize);
  private config param aggregate = CHPL_COMM != "none";

  record remoteHandler {
    type t;
    var original: _to_nilable(t);
    var localHandler: _to_nilable(_to_unmanaged(original!.sourceCopy().type));
    var loc: int;

    proc init(type t) {
      this.t = t;
      this.localHandler = nil;
    }

    proc init(type t, ref original, loc) {
      this.t = t;
      this.original = original;
      this.localHandler = nil;
      this.loc = loc;
    }

    proc deinit() {
      delete localHandler;
    }

    inline proc ref get() ref {
      if localHandler == nil {
        on Locales[loc] {
          localHandler = original!.sourceCopy();
        }
      }
      return localHandler;
    }
  }

  /*
    Aggregates ``copy(ref dst, src)``. Optimized for when src is local.
    Not parallel safe and is expected to be created on a per-task basis.
    High memory usage since there are per-destination buffers.
  */
  record CustomDstAggregator {
    type elemType;
    param custom = false;

    @chpldoc.nodoc
    var agg: if aggregate then CustomDstAggregatorImpl(elemType, ?) else nothing;

    proc init(type elemType) {
      this.elemType = elemType;
      if aggregate then this.agg = new CustomDstAggregatorImpl(elemType, 0);
    }

    proc init(ref handler) {
      this.elemType = handler.elemType;
      this.custom = true;
      if aggregate then this.agg = new CustomDstAggregatorImpl(elemType, handler);
    }

    /*
      Sets ``dst = srcVal`` in a way that aggregates such updates
      to improve communication efficiency assuming that ``dst`` is remote
      and ``srcVal`` is local.
    */
    inline proc ref copy(ref dst: elemType, const in srcVal: elemType) {
      if aggregate then agg.copy(dst, srcVal);
                   else dst = srcVal;
    }

    /*
      Copy method for custom aggregator.
    */
    inline proc ref copy(const in srcVal: elemType) {
      compilerAssert(aggregate);
      agg.copy(srcVal);
    }

    /*
      Flushes the aggregator & completes the updates queued up from the
      :proc:`DstAggregator.copy` calls.

      :arg freeBuffers: if ``true``, deallocates buffers used by this
                        aggregator. If ``false``, the buffers will remain
                        allocated after this ``flush`` (to support further
                        :proc:`DstAggregator.copy` calls) and deallocated
                        when the aggregator variable is deinitialized.
    */
    inline proc ref flush(freeBuffers=false) {
      if aggregate then agg.flush(freeBuffers=freeBuffers);
    }
  }

  @chpldoc.nodoc
  record CustomDstAggregatorImpl {
    type elemType;
    var handler;
    type aggType = (c_ptr(elemType), elemType);
    const bufferSize = dstBuffSize;
    const myLocaleSpace = 0..<numLocales;
    var lastLocale: int;
    var opsUntilYield = yieldFrequency;
    var lBuffers: c_ptr(c_ptr(aggType));
    var rBuffers: [myLocaleSpace] remoteBuffer(aggType);
    var rHandlers: [myLocaleSpace] remoteHandler(handler.type);
    var bufferIdxs: c_ptr(int);

    proc ref postinit() {
      lBuffers = allocate(c_ptr(aggType), numLocales);
      bufferIdxs = bufferIdxAlloc();
      for loc in myLocaleSpace {
        lBuffers[loc] = allocate(aggType, bufferSize);
        bufferIdxs[loc] = 0;
        rBuffers[loc] = new remoteBuffer(aggType, bufferSize, loc);
        rHandlers[loc] = new remoteHandler(handler.type, handler, loc);
      }
    }

    proc ref deinit() {
      flush(freeBuffers=true);
      for loc in myLocaleSpace {
        deallocate(lBuffers[loc]);
      }
      deallocate(lBuffers);
      deallocate(bufferIdxs);
    }

    proc ref flush(freeBuffers: bool) {
      for offsetLoc in myLocaleSpace + lastLocale {
        const loc = offsetLoc % numLocales;
        _flushBuffer(loc, bufferIdxs[loc], freeData=freeBuffers);
      }
    }

    inline proc ref copy(ref dst: elemType, const in srcVal: elemType) {
      if verboseAggregation {
        writeln("DstAggregator.copy is called");
      }
      // Get the locale of dst and the local address on that locale
      const loc = dst.locale.id;
      lastLocale = loc;
      const dstAddr = getAddr(dst);

      // Get our current index into the buffer for dst's locale
      ref bufferIdx = bufferIdxs[loc];

      // Buffer the address and desired value
      lBuffers[loc][bufferIdx] = (dstAddr, srcVal);
      bufferIdx += 1;

      // Flush our buffer if it's full. If it's been a while since we've let
      // other tasks run, yield so that we're not blocking remote tasks from
      // flushing their buffers.
      if bufferIdx == bufferSize {
        _flushBuffer(loc, bufferIdx, freeData=false);
        opsUntilYield = yieldFrequency;
      } else if opsUntilYield == 0 {
        currentTask.yieldExecution();
        opsUntilYield = yieldFrequency;
      } else {
        opsUntilYield -= 1;
      }
    }

    inline proc ref copy(const in srcVal: elemType) {
      if verboseAggregation {
        writeln("DstAggregator.copy is called");
      }
      // Get the locale of srcVal and set dstAddr to nil
      const loc = handler.getDestinationLocale(srcVal).id;
      lastLocale = loc;
      const dstAddr = nil;

      // Get our current index into the buffer for dst's locale
      ref bufferIdx = bufferIdxs[loc];

      // Buffer the address and desired value
      lBuffers[loc][bufferIdx] = (dstAddr,srcVal);
      bufferIdx += 1;

      // Flush our buffer if it's full. If it's been a while since we've let
      // other tasks run, yield so that we're not blocking remote tasks from
      // flushing their buffers.
      if bufferIdx == bufferSize {
        _flushBuffer(loc, bufferIdx, freeData=false);
        opsUntilYield = yieldFrequency;
      } else if opsUntilYield == 0 {
        currentTask.yieldExecution();
        opsUntilYield = yieldFrequency;
      } else {
        opsUntilYield -= 1;
      }
    }

    proc ref _flushBuffer(loc: int, ref bufferIdx, freeData) {
      const myBufferIdx = bufferIdx;
      if myBufferIdx == 0 then return;

      // Allocate a remote buffer
      ref rBuffer = rBuffers[loc];
      const remBufferPtr = rBuffer.cachedAlloc();
      ref rHandler = rHandlers[loc].get();

      // Copy local buffer to remote buffer
      rBuffer.PUT(lBuffers[loc], myBufferIdx);

      // Process remote buffer
      on Locales[loc] {
        if !isNothing(handler) then rHandler!.flush(rBuffer, remBufferPtr, myBufferIdx);
      }
      if freeData {
        rBuffer.markFreed();
      }
      bufferIdx = 0;
    }
  }
}
