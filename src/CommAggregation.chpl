module CommAggregation {
  use CTypes;
  use ServerConfig;
  use UnorderedCopy;
  use CommPrimitives;
  use ChplConfig;

  // TODO should tune these values at startup
  param defaultBuffSize = if CHPL_COMM == "ugni" then 4096 else 8192;
  private config const yieldFrequency = getEnvInt("ARKOUDA_SERVER_AGGREGATION_YIELD_FREQUENCY", 1024);
  private config const dstBuffSize = getEnvInt("ARKOUDA_SERVER_AGGREGATION_DST_BUFF_SIZE", defaultBuffSize);
  private config const srcBuffSize = getEnvInt("ARKOUDA_SERVER_AGGREGATION_SRC_BUFF_SIZE", defaultBuffSize);


  /* Creates a new destination aggregator (dst/lhs will be remote). */
  proc newDstAggregator(type elemType, param useUnorderedCopy=false) {
    use BigInteger, BigIntegerAggregation;
    if CHPL_COMM == "none" || useUnorderedCopy {
      return new DstUnorderedAggregator(elemType);
    } else if elemType == bigint {
      return new DstAggregatorBigint();
    } else {
      return new DstAggregator(elemType);
    }
  }

  /* Creates a new source aggregator (src/rhs will be remote). */
  proc newSrcAggregator(type elemType, param useUnorderedCopy=false) {
    use BigInteger, BigIntegerAggregation;
    if CHPL_COMM == "none" || useUnorderedCopy {
      return new SrcUnorderedAggregator(elemType);
    } else if elemType == bigint {
      return new SrcAggregatorBigint();
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
    type aggType = (c_ptr(elemType), elemType);
    const bufferSize = dstBuffSize;
    const myLocaleSpace = 0..<numLocales;
    var lastLocale: int;
    var opsUntilYield = yieldFrequency;
    var lBuffers: c_ptr(c_ptr(aggType));
    var rBuffers: [myLocaleSpace] remoteBuffer(aggType);
    var bufferIdxs: c_ptr(int);

    proc postinit() {
      lBuffers = c_malloc(c_ptr(aggType), numLocales);
      bufferIdxs = bufferIdxAlloc();
      for loc in myLocaleSpace {
        lBuffers[loc] = c_malloc(aggType, bufferSize);
        bufferIdxs[loc] = 0;
        rBuffers[loc] = new remoteBuffer(aggType, bufferSize, loc);
      }
    }

    proc deinit() {
      flush();
      for loc in myLocaleSpace {
        c_free(lBuffers[loc]);
      }
      c_free(lBuffers);
      c_free(bufferIdxs);
    }

    proc flush() {
      for offsetLoc in myLocaleSpace + lastLocale {
        const loc = offsetLoc % numLocales;
        _flushBuffer(loc, bufferIdxs[loc], freeData=true);
      }
    }

    inline proc copy(ref dst: elemType, const in srcVal: elemType) {
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
        chpl_task_yield();
        opsUntilYield = yieldFrequency;
      } else {
        opsUntilYield -= 1;
      }
    }

    proc _flushBuffer(loc: int, ref bufferIdx, freeData) {
      const myBufferIdx = bufferIdx;
      if myBufferIdx == 0 then return;

      // Allocate a remote buffer
      ref rBuffer = rBuffers[loc];
      const remBufferPtr = rBuffer.cachedAlloc();

      // Copy local buffer to remote buffer
      rBuffer.PUT(lBuffers[loc], myBufferIdx);

      // Process remote buffer
      on Locales[loc] {
        for (dstAddr, srcVal) in rBuffer.localIter(remBufferPtr, myBufferIdx) {
          dstAddr.deref() = srcVal;
        }
        if freeData {
          rBuffer.localFree(remBufferPtr);
        }
      }
      if freeData {
        rBuffer.markFreed();
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
      if isPOD(elemType) then unorderedCopyTaskFence();
    }
    inline proc copy(ref dst: elemType, const in srcVal: elemType) {
      if isPOD(elemType) then unorderedCopy(dst, srcVal);
                         else dst = srcVal;
    }
  }


  /*
   * Aggregates copy(ref dst, const ref src). Only works when dst is local.
   * Not parallel safe and is expected to be created on a per task basis
   * High memory usage since there are per-destination buffers
   */
  record SrcAggregator {
    type elemType;
    type aggType = c_ptr(elemType);
    const bufferSize = srcBuffSize;
    const myLocaleSpace = 0..<numLocales;
    var lastLocale: int;
    var opsUntilYield = yieldFrequency;
    var dstAddrs: c_ptr(c_ptr(aggType));
    var lSrcAddrs: c_ptr(c_ptr(aggType));
    var lSrcVals: [myLocaleSpace][0..#bufferSize] elemType;
    var rSrcAddrs: [myLocaleSpace] remoteBuffer(aggType);
    var rSrcVals: [myLocaleSpace] remoteBuffer(elemType);
    var bufferIdxs: c_ptr(int);

    proc postinit() {
      dstAddrs = c_malloc(c_ptr(aggType), numLocales);
      lSrcAddrs = c_malloc(c_ptr(aggType), numLocales);
      bufferIdxs = bufferIdxAlloc();
      for loc in myLocaleSpace {
        dstAddrs[loc] = c_malloc(aggType, bufferSize);
        lSrcAddrs[loc] = c_malloc(aggType, bufferSize);
        bufferIdxs[loc] = 0;
        rSrcAddrs[loc] = new remoteBuffer(aggType, bufferSize, loc);
        rSrcVals[loc] = new remoteBuffer(elemType, bufferSize, loc);
      }
    }

    proc deinit() {
      flush();
      for loc in myLocaleSpace {
        c_free(dstAddrs[loc]);
        c_free(lSrcAddrs[loc]);
      }
      c_free(dstAddrs);
      c_free(lSrcAddrs);
      c_free(bufferIdxs);
    }

    proc flush() {
      for offsetLoc in myLocaleSpace + lastLocale {
        const loc = offsetLoc % numLocales;
        _flushBuffer(loc, bufferIdxs[loc], freeData=true);
      }
    }

    inline proc copy(ref dst: elemType, const ref src: elemType) {
      if boundsChecking {
        assert(dst.locale.id == here.id);
      }
      const dstAddr = getAddr(dst);

      const loc = src.locale.id;
      lastLocale = loc;
      const srcAddr = getAddr(src);

      ref bufferIdx = bufferIdxs[loc];
      lSrcAddrs[loc][bufferIdx] = srcAddr;
      dstAddrs[loc][bufferIdx] = dstAddr;
      bufferIdx += 1;

      if bufferIdx == bufferSize {
        _flushBuffer(loc, bufferIdx, freeData=false);
        opsUntilYield = yieldFrequency;
      } else if opsUntilYield == 0 {
        chpl_task_yield();
        opsUntilYield = yieldFrequency;
      } else {
        opsUntilYield -= 1;
      }
    }

    proc _flushBuffer(loc: int, ref bufferIdx, freeData) {
      const myBufferIdx = bufferIdx;
      if myBufferIdx == 0 then return;

      ref myLSrcVals = lSrcVals[loc];
      ref myRSrcAddrs = rSrcAddrs[loc];
      ref myRSrcVals = rSrcVals[loc];

      // Allocate remote buffers
      const rSrcAddrPtr = myRSrcAddrs.cachedAlloc();
      const rSrcValPtr = myRSrcVals.cachedAlloc();

      // Copy local addresses to remote buffer
      myRSrcAddrs.PUT(lSrcAddrs[loc], myBufferIdx);

      // Process remote buffer, copying the value of our addresses into a
      // remote buffer
      on Locales[loc] {
        for i in 0..<myBufferIdx {
          rSrcValPtr[i] = rSrcAddrPtr[i].deref();
        }
        if freeData {
          myRSrcAddrs.localFree(rSrcAddrPtr);
        }
      }
      if freeData {
        myRSrcAddrs.markFreed();
      }

      // Copy remote values into local buffer
      myRSrcVals.GET(myLSrcVals, myBufferIdx);

      // Assign the srcVal to the dstAddrs
      var dstAddrPtr = c_ptrTo(dstAddrs[loc][0]);
      var srcValPtr = c_ptrTo(myLSrcVals[0]);
      for i in 0..<myBufferIdx {
        dstAddrPtr[i].deref() = srcValPtr[i];
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
      if isPOD(elemType) then unorderedCopyTaskFence();
    }
    inline proc copy(ref dst: elemType, const ref src: elemType) {
      assert(dst.locale.id == here.id);
      if isPOD(elemType) then unorderedCopy(dst, src);
                         else dst = src;
    }
  }


  // A remote buffer with lazy allocation
  record remoteBuffer {
    type elemType;
    var size: int;
    var loc: int;
    var data: c_ptr(elemType);

    // Allocate a buffer on loc if we haven't already. Return a c_ptr to the
    // remote locales buffer
    proc cachedAlloc(): c_ptr(elemType) {
      if data == c_nil {
        const rvf_size = size;
        on Locales[loc] do {
          data = c_malloc(elemType, rvf_size);
        }
      }
      return data;
    }

    // Iterate through buffer elements, must be running on loc. data is passed
    // in to avoid communication.
    iter localIter(data: c_ptr(elemType), size: int) ref : elemType {
      if boundsChecking {
        assert(this.loc == here.id);
        assert(this.data == data);
        assert(data != c_nil);
      }
      for i in 0..<size {
        yield data[i];
      }
    }

    // Free the data, must be running on the owning locale, data is passed in
    // to avoid communication. Data is freed'd automatically when this record
    // goes out of scope, but this is an optimization to free when already
    // running on loc
    inline proc localFree(data: c_ptr(elemType)) {
      if boundsChecking {
        assert(this.loc == here.id);
        assert(this.data == data);
        assert(data != c_nil);
      }
      c_free(data);
    }

    // After free'ing the data, need to nil out the records copy of the pointer
    // so we don't double-free on deinit
    inline proc markFreed() {
      if boundsChecking {
        assert(this.locale.id == here.id);
      }
      data = c_nil;
    }

    // Copy size elements from lArr to the remote buffer. Must be running on
    // lArr's locale.
    proc PUT(lArr: [] elemType, size: int) where lArr.isDefaultRectangular() {
      if boundsChecking {
        assert(size <= this.size);
        assert(this.size == lArr.size);
        assert(lArr.domain.low == 0);
        assert(lArr.locale.id == here.id);
      }
      const byte_size = size:c_size_t * c_sizeof(elemType);
      CommPrimitives.PUT(c_ptrTo(lArr[0]), loc, data, byte_size);
    }

    proc PUT(lArr: c_ptr(elemType), size: int) {
      if boundsChecking {
        assert(size <= this.size);
      }
      const byte_size = size:c_size_t * c_sizeof(elemType);
      CommPrimitives.PUT(lArr, loc, data, byte_size);
    }

    proc GET(lArr: [] elemType, size: int) where lArr.isDefaultRectangular() {
      if boundsChecking {
        assert(size <= this.size);
        assert(this.size == lArr.size);
        assert(lArr.domain.low == 0);
        assert(lArr.locale.id == here.id);
      }
      const byte_size = size:c_size_t * c_sizeof(elemType);
      CommPrimitives.GET(c_ptrTo(lArr[0]), loc, data, byte_size);
    }

    proc deinit() {
      if data != c_nil {
        const rvf_data=data;
        on Locales[loc] {
          localFree(rvf_data);
        }
        markFreed();
      }
    }
  }

  //
  // Helper routines
  //

  // Cacheline aligned and padded allocation to avoid false-sharing
  inline proc bufferIdxAlloc() {
    const cachePaddedLocales = (numLocales + 7) & ~7;
    return c_aligned_alloc(int, 64, cachePaddedLocales);
  }

  module BigIntegerAggregation {
    use CTypes;
    use CommPrimitives;
    use CommAggregation;

    // procs to get at internal mpz fields (copied from chapel GMP)
    use BigInteger, GMP;
    private extern proc chpl_gmp_mpz_struct_sign_size(from: __mpz_struct) : mp_size_t;
    private extern proc chpl_gmp_mpz_struct_limbs(from: __mpz_struct) : c_ptr(mp_limb_t);
    private extern proc chpl_gmp_mpz_set_sign_size(ref dst:mpz_t, sign_size:mp_size_t);

    // At minimum, need to store dst address, size+size of src, and 1 limb.
    // Could be more limbs, but size to make comm comparisons to int64 agg.
    private proc minInlineBigintSize() {
      return (c_sizeof(c_ptr(bigint)) + c_sizeof(mp_size_t) + c_sizeof(mp_limb_t)):int;
    }

    record DstAggregatorBigint {
      type aggType = uint(8);
      const bufferSize = dstBuffSize * minInlineBigintSize();
      const myLocaleSpace = 0..<numLocales;
      var lastLocale: int;
      var opsUntilYield = yieldFrequency;
      var lBuffers: c_ptr(c_ptr(aggType));
      var rBuffers: [myLocaleSpace] remoteBuffer(aggType);
      var bufferIdxs: c_ptr(int);

      proc postinit() {
        lBuffers = c_malloc(c_ptr(aggType), numLocales);
        bufferIdxs = bufferIdxAlloc();
        for loc in myLocaleSpace {
          lBuffers[loc] = c_malloc(aggType, bufferSize);
          bufferIdxs[loc] = 0;
          rBuffers[loc] = new remoteBuffer(aggType, bufferSize, loc);
        }
      }

      proc deinit() {
        flush();
        for loc in myLocaleSpace {
          c_free(lBuffers[loc]);
        }
        c_free(lBuffers);
        c_free(bufferIdxs);
      }

      proc flush() {
        for offsetLoc in myLocaleSpace + lastLocale {
          const loc = offsetLoc % numLocales;
          _flushBuffer(loc, bufferIdxs[loc], freeData=true);
        }
      }

      inline proc copy(ref dst: bigint, const ref src: bigint) {
        // Get the locale of dst and the local address on that locale
        // TODO only works when record wrapper and mpz have same locale..
        // should be true for most of arkouda but not guaranteed
        const loc = dst.locale.id;
        lastLocale = loc;
        var dstAddr = getAddr(dst);

        // Get our current index into the buffer for dst's locale
        ref bufferIdx = bufferIdxs[loc];

        // Get the src sign+size and pointer to the limbs
        var sign_size = chpl_gmp_mpz_struct_sign_size(src.getImpl());
        var src_limbs = chpl_gmp_mpz_struct_limbs(src.getImpl());

        // compute sizes of addr, sign+size, and limbs
        var addr_bytes = c_sizeof(c_ptr(bigint));
        var size_bytes = c_sizeof(mp_size_t);
        var limb_bytes = abs(sign_size) * c_sizeof(mp_limb_t);

        // Just do direct assignment if dst is local
        // TODO also if size will exceed max buffer
        if loc == here.id {
          dst = src;
          return;
        }

        // Flush our buffer if this entry will exceed capacity
        if bufferIdx + addr_bytes + size_bytes + limb_bytes > bufferSize {
          _flushBuffer(loc, bufferIdx, freeData=false);
          opsUntilYield = yieldFrequency;
        }

        // Buffer the address and the serialized value (sign_sign, limbs)
        c_memcpy(c_ptrTo(lBuffers[loc][bufferIdx]), c_ptrTo(dstAddr), addr_bytes);
        bufferIdx += addr_bytes:int;
        c_memcpy(c_ptrTo(lBuffers[loc][bufferIdx]), c_ptrTo(sign_size), size_bytes);
        bufferIdx += size_bytes:int;
        c_memcpy(c_ptrTo(lBuffers[loc][bufferIdx]), src_limbs, limb_bytes);
        bufferIdx += limb_bytes:int;

        // If it's been a while since we've let other tasks run, yield so that
        // we're not blocking remote tasks from flushing their buffers.
        if opsUntilYield == 0 {
          chpl_task_yield();
          opsUntilYield = yieldFrequency;
        } else {
          opsUntilYield -= 1;
        }
      }

      proc _flushBuffer(loc: int, ref bufferIdx, freeData) {
        const myBufferIdx = bufferIdx;
        if myBufferIdx == 0 then return;

        // Allocate a remote buffer
        ref rBuffer = rBuffers[loc];
        const remBufferPtr = rBuffer.cachedAlloc();

        // Copy local buffer to remote buffer
        rBuffer.PUT(lBuffers[loc], myBufferIdx);

        // Process remote buffer
        on Locales[loc] {
          var curBufferIdx = 0;
          while curBufferIdx < myBufferIdx {
            var dstAddr: c_ptr(bigint);
            var sign_size: mp_size_t;
            var src_limbs: c_ptr(mp_limb_t);

            var addr_bytes = c_sizeof(c_ptr(bigint));
            var size_bytes = c_sizeof(mp_size_t);

            // Copy addr and size out of buffer
            c_memcpy(c_ptrTo(dstAddr), c_ptrTo(remBufferPtr[curBufferIdx]), addr_bytes);
            curBufferIdx += addr_bytes: int;
            c_memcpy(c_ptrTo(sign_size), c_ptrTo(remBufferPtr[curBufferIdx]), size_bytes);
            curBufferIdx += size_bytes:int;

            // extract size from sign+size and compute limb bytes
            var n = abs(sign_size);
            var limb_bytes = n * c_sizeof(mp_limb_t);

            // reallocate target bigint
            _mpz_realloc(dstAddr.deref().mpz, n);

            // extract pointer to target bigint limbs, and copy buffered limbs into it
            var xp = chpl_gmp_mpz_struct_limbs(dstAddr.deref().getImpl());
            c_memcpy(xp, c_ptrTo(remBufferPtr[curBufferIdx]), limb_bytes);
            curBufferIdx += limb_bytes:int;

            // update the sign+size of target bigint
            chpl_gmp_mpz_set_sign_size(dstAddr.deref().mpz, sign_size);
          }

          if freeData {
            rBuffer.localFree(remBufferPtr);
          }
        }
        if freeData {
          rBuffer.markFreed();
        }
        bufferIdx = 0;
      }
    }

    record SrcAggregatorBigint {
      type elemType = bigint;
      type aggType = c_ptr(elemType);
      const bufferSize = srcBuffSize;
      const myLocaleSpace = 0..<numLocales;
      var lastLocale: int;
      var opsUntilYield = yieldFrequency;
      var dstAddrs: c_ptr(c_ptr(aggType));
      var lSrcAddrs: c_ptr(c_ptr(aggType));
      var lSrcVals: [myLocaleSpace][0..#bufferSize] elemType;
      var rSrcAddrs: [myLocaleSpace] remoteBuffer(aggType);
      var rSrcVals: [myLocaleSpace] remoteBuffer(elemType);
      var bufferIdxs: c_ptr(int);

      proc postinit() {
        dstAddrs = c_malloc(c_ptr(aggType), numLocales);
        lSrcAddrs = c_malloc(c_ptr(aggType), numLocales);
        bufferIdxs = bufferIdxAlloc();
        for loc in myLocaleSpace {
          dstAddrs[loc] = c_malloc(aggType, bufferSize);
          lSrcAddrs[loc] = c_malloc(aggType, bufferSize);
          bufferIdxs[loc] = 0;
          rSrcAddrs[loc] = new remoteBuffer(aggType, bufferSize, loc);
          rSrcVals[loc] = new remoteBuffer(elemType, bufferSize, loc);
        }
      }

      proc deinit() {
        flush();
        for loc in myLocaleSpace {
          c_free(dstAddrs[loc]);
          c_free(lSrcAddrs[loc]);
        }
        c_free(dstAddrs);
        c_free(lSrcAddrs);
        c_free(bufferIdxs);
      }

      proc flush() {
        for offsetLoc in myLocaleSpace + lastLocale {
          const loc = offsetLoc % numLocales;
          _flushBuffer(loc, bufferIdxs[loc], freeData=true);
        }
      }

      inline proc copy(ref dst: elemType, const ref src: elemType) {
        // TODO aggregation not supported today, just do plain assignment
        dst = src;
        return;

        if boundsChecking {
          assert(dst.locale.id == here.id);
        }
        const dstAddr = getAddr(dst);

        const loc = src.locale.id;
        lastLocale = loc;
        const srcAddr = getAddr(src);

        ref bufferIdx = bufferIdxs[loc];
        lSrcAddrs[loc][bufferIdx] = srcAddr;
        dstAddrs[loc][bufferIdx] = dstAddr;
        bufferIdx += 1;

        if bufferIdx == bufferSize {
          _flushBuffer(loc, bufferIdx, freeData=false);
          opsUntilYield = yieldFrequency;
        } else if opsUntilYield == 0 {
          chpl_task_yield();
          opsUntilYield = yieldFrequency;
        } else {
          opsUntilYield -= 1;
        }
      }

      proc _flushBuffer(loc: int, ref bufferIdx, freeData) {
        const myBufferIdx = bufferIdx;
        if myBufferIdx == 0 then return;

        ref myLSrcVals = lSrcVals[loc];
        ref myRSrcAddrs = rSrcAddrs[loc];
        ref myRSrcVals = rSrcVals[loc];

        // Allocate remote buffers
        const rSrcAddrPtr = myRSrcAddrs.cachedAlloc();
        const rSrcValPtr = myRSrcVals.cachedAlloc();

        // Copy local addresses to remote buffer
        myRSrcAddrs.PUT(lSrcAddrs[loc], myBufferIdx);

        // Process remote buffer, copying the value of our addresses into a
        // remote buffer
        on Locales[loc] {
          for i in 0..<myBufferIdx {
            rSrcValPtr[i] = rSrcAddrPtr[i].deref();
          }
          if freeData {
            myRSrcAddrs.localFree(rSrcAddrPtr);
          }
        }
        if freeData {
          myRSrcAddrs.markFreed();
        }

        // Copy remote values into local buffer
        myRSrcVals.GET(myLSrcVals, myBufferIdx);

        // Assign the srcVal to the dstAddrs
        var dstAddrPtr = c_ptrTo(dstAddrs[loc][0]);
        var srcValPtr = c_ptrTo(myLSrcVals[0]);
        for i in 0..<myBufferIdx {
          dstAddrPtr[i].deref() = srcValPtr[i];
        }

        bufferIdx = 0;
      }
    }
  }

}
