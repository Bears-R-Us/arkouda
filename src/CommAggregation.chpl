module CommAggregation {
  use ServerConfig;
  use UnorderedCopy;
  use CommPrimitives;
  use ChplConfig;
  use OS.POSIX;

  // TODO should tune these values at startup
  private param defaultBuffSize =
    if CHPL_TARGET_PLATFORM == "hpe-cray-ex" then 1024
    else if CHPL_COMM == "ugni" then 4096
    else 8192;
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

    proc ref postinit() {
      lBuffers = allocate(c_ptr(aggType), numLocales);
      bufferIdxs = bufferIdxAlloc();
      for loc in myLocaleSpace {
        lBuffers[loc] = allocate(aggType, bufferSize);
        bufferIdxs[loc] = 0;
        rBuffers[loc] = new remoteBuffer(aggType, bufferSize, loc);
      }
    }

    proc ref deinit() {
      flush();
      for loc in myLocaleSpace {
        deallocate(lBuffers[loc]);
      }
      deallocate(lBuffers);
      deallocate(bufferIdxs);
    }

    proc ref flush() {
      for offsetLoc in myLocaleSpace + lastLocale {
        const loc = offsetLoc % numLocales;
        flushBuffer(loc, bufferIdxs[loc], freeData=true);
      }
    }

    inline proc ref copy(ref dst: elemType, const in srcVal: elemType) {
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
        flushBuffer(loc, bufferIdx, freeData=false);
        opsUntilYield = yieldFrequency;
      } else if opsUntilYield == 0 {
        currentTask.yieldExecution();
        opsUntilYield = yieldFrequency;
      } else {
        opsUntilYield -= 1;
      }
    }

    proc ref flushBuffer(loc: int, ref bufferIdx, freeData) {
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

    proc ref postinit() {
      dstAddrs = allocate(c_ptr(aggType), numLocales);
      lSrcAddrs = allocate(c_ptr(aggType), numLocales);
      bufferIdxs = bufferIdxAlloc();
      for loc in myLocaleSpace {
        dstAddrs[loc] = allocate(aggType, bufferSize);
        lSrcAddrs[loc] = allocate(aggType, bufferSize);
        bufferIdxs[loc] = 0;
        rSrcAddrs[loc] = new remoteBuffer(aggType, bufferSize, loc);
        rSrcVals[loc] = new remoteBuffer(elemType, bufferSize, loc);
      }
    }

    proc ref deinit() {
      flush();
      for loc in myLocaleSpace {
        deallocate(dstAddrs[loc]);
        deallocate(lSrcAddrs[loc]);
      }
      deallocate(dstAddrs);
      deallocate(lSrcAddrs);
      deallocate(bufferIdxs);
    }

    proc ref flush() {
      for offsetLoc in myLocaleSpace + lastLocale {
        const loc = offsetLoc % numLocales;
        flushBuffer(loc, bufferIdxs[loc], freeData=true);
      }
    }

    inline proc ref copy(ref dst: elemType, const ref src: elemType) {
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
        flushBuffer(loc, bufferIdx, freeData=false);
        opsUntilYield = yieldFrequency;
      } else if opsUntilYield == 0 {
        currentTask.yieldExecution();
        opsUntilYield = yieldFrequency;
      } else {
        opsUntilYield -= 1;
      }
    }

    proc ref flushBuffer(loc: int, ref bufferIdx, freeData) {
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
    proc ref cachedAlloc(): c_ptr(elemType) {
      if data == nil {
        const rvf_size = size;
        on Locales[loc] do {
          data = allocate(elemType, rvf_size);
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
        assert(data != nil);
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
        assert(data != nil);
      }
      deallocate(data);
    }

    // After free'ing the data, need to nil out the records copy of the pointer
    // so we don't double-free on deinit
    inline proc ref markFreed() {
      if boundsChecking {
        assert(this.locale.id == here.id);
      }
      data = nil;
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
      CommPrimitives.PUT(data, c_ptrTo(lArr[0]), loc, byte_size);
    }

    proc PUT(lArr: c_ptr(elemType), size: int) {
      if boundsChecking {
        assert(size <= this.size);
      }
      const byte_size = size:c_size_t * c_sizeof(elemType);
      CommPrimitives.PUT(data, lArr, loc, byte_size);
    }

    proc ref GET(ref lArr: [] elemType, size: int) where lArr.isDefaultRectangular() {
      if boundsChecking {
        assert(size <= this.size);
        assert(this.size == lArr.size);
        assert(lArr.domain.low == 0);
        assert(lArr.locale.id == here.id);
      }
      const byte_size = size:c_size_t * c_sizeof(elemType);
      CommPrimitives.GET(c_ptrTo(lArr[0]), data, loc, byte_size);
    }

    proc ref deinit() {
      if data != nil {
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
    return allocate(int, cachePaddedLocales, alignment=64);
  }

  module BigIntegerAggregation {
    use CTypes;
    use CommPrimitives;
    use CommAggregation;
    use BigInteger, GMP;
    use Math;
    use OS.POSIX;

    proc bigint.serializedSize() {
      extern proc chpl_gmp_mpz_struct_sign_size(from: __mpz_struct) : mp_size_t;

      var sign_size = chpl_gmp_mpz_struct_sign_size(this.getImpl());

      var size_bytes = c_sizeof(mp_size_t):int;
      var limb_bytes = Math.abs(sign_size:int) * c_sizeof(mp_limb_t):int;

      return size_bytes + limb_bytes;
    }

    proc bigint.serializeInto(x: c_ptr(uint(8))) {
      extern proc chpl_gmp_mpz_struct_sign_size(from: __mpz_struct) : mp_size_t;
      extern proc chpl_gmp_mpz_struct_limbs(from: __mpz_struct) : c_ptr(mp_limb_t);

      var sign_size = chpl_gmp_mpz_struct_sign_size(this.getImpl());

      var size_bytes = c_sizeof(mp_size_t):int;
      var limb_bytes = Math.abs(sign_size:int) * c_sizeof(mp_limb_t):int;

      var limb_ptr = chpl_gmp_mpz_struct_limbs(this.getImpl());

      memcpy(x, c_ptrTo(sign_size), size_bytes);
      memcpy(x+size_bytes, limb_ptr, limb_bytes);
    }

    proc ref bigint.deserializeFrom(x: c_ptr(uint(8))) {
      extern proc chpl_gmp_mpz_struct_limbs(from: __mpz_struct) : c_ptr(mp_limb_t);
      extern proc chpl_gmp_mpz_set_sign_size(ref dst:mpz_t, sign_size:mp_size_t);

      var sign_size: mp_size_t;
      var src_limbs: c_ptr(mp_limb_t);

      var size_bytes = c_sizeof(mp_size_t);

      memcpy(c_ptrTo(sign_size), x, size_bytes);

      var nlimbs = Math.abs(sign_size:int);
      var limb_bytes = nlimbs * c_sizeof(mp_limb_t):int;

      _mpz_realloc(this.mpz, nlimbs);
      var xp = chpl_gmp_mpz_struct_limbs(this.getImpl());
      memcpy(xp, x+size_bytes, limb_bytes);

      chpl_gmp_mpz_set_sign_size(this.mpz, sign_size);

      return size_bytes:int + limb_bytes:int;
    }

    record DstAggregatorBigint {
      type aggType = uint(8);
      const bufferSize = dstBuffSize * (c_sizeof(c_ptr(bigint)) + c_sizeof(mp_size_t) + c_sizeof(mp_limb_t)): int;
      const myLocaleSpace = 0..<numLocales;
      var lastLocale: int;
      var opsUntilYield = yieldFrequency;
      var lBuffers: c_ptr(c_ptr(aggType));
      var rBuffers: [myLocaleSpace] remoteBuffer(aggType);
      var bufferIdxs: c_ptr(int);

      proc ref postinit() {
        lBuffers = allocate(c_ptr(aggType), numLocales);
        bufferIdxs = bufferIdxAlloc();
        for loc in myLocaleSpace {
          lBuffers[loc] = allocate(aggType, bufferSize);
          bufferIdxs[loc] = 0;
          rBuffers[loc] = new remoteBuffer(aggType, bufferSize, loc);
        }
      }

      proc ref deinit() {
        flush();
        for loc in myLocaleSpace {
          deallocate(lBuffers[loc]);
        }
        deallocate(lBuffers);
        deallocate(bufferIdxs);
      }

      proc ref flush() {
        for offsetLoc in myLocaleSpace + lastLocale {
          const loc = offsetLoc % numLocales;
          flushBuffer(loc, bufferIdxs[loc], freeData=true);
        }
      }

      inline proc ref copy(ref dst: bigint, const ref src: bigint) {
        if boundsChecking { assert(src.locale.id == here.id && src.localeId == here.id); }
        // Note this is the locale of the record wrapper, and required to be
        // the same as the mpz storage itself.
        const loc = dst.locale.id;

        const serialize_bytes = src.serializedSize();

        // Just do direct assignment if dst is local or src size is large
        if loc == here.id || serialize_bytes > (bufferSize >> 2) {
          dst = src;
          return;
        }

        lastLocale = loc;

        var dstAddr = getAddr(dst);
        var addr_bytes = c_sizeof(c_ptr(bigint)): int;

        // Get our current index into the buffer for dst's locale
        ref bufferIdx = bufferIdxs[loc];

        // Flush our buffer if this entry will exceed capacity
        if bufferIdx + addr_bytes + serialize_bytes > bufferSize {
          flushBuffer(loc, bufferIdx, freeData=false);
          opsUntilYield = yieldFrequency;
        }

        // Buffer the address and the serialized value
        memcpy(c_ptrTo(lBuffers[loc][bufferIdx]), c_ptrTo(dstAddr), addr_bytes);
        src.serializeInto(c_ptrTo(lBuffers[loc][bufferIdx+addr_bytes]));
        bufferIdx += addr_bytes + serialize_bytes;

        // If it's been a while since we've let other tasks run, yield so that
        // we're not blocking remote tasks from flushing their buffers.
        if opsUntilYield == 0 {
          currentTask.yieldExecution();
          opsUntilYield = yieldFrequency;
        } else {
          opsUntilYield -= 1;
        }
      }

      proc ref flushBuffer(loc: int, ref bufferIdx, freeData) {
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
            var addr_bytes = c_sizeof(c_ptr(bigint)): int;

            // Copy addr out of buffer
            memcpy(c_ptrTo(dstAddr), c_ptrTo(remBufferPtr[curBufferIdx]), addr_bytes);
            // assert that record locality matches mpz locality
            if boundsChecking { assert(dstAddr.deref().localeId == here.id); }
            // deserialize into bigint
            var ser_bytes = dstAddr.deref().deserializeFrom(c_ptrTo(remBufferPtr[curBufferIdx+addr_bytes]));
            curBufferIdx += addr_bytes + ser_bytes;
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
      type aggType = c_ptr(bigint);
      const bufferSize = srcBuffSize;
      const uintBufferSize = srcBuffSize * (c_sizeof(mp_size_t) + c_sizeof(mp_limb_t)): int;
      const myLocaleSpace = 0..<numLocales;
      var lastLocale: int;
      var opsUntilYield = yieldFrequency;
      var dstAddrs: c_ptr(c_ptr(aggType));
      var lSrcAddrs: c_ptr(c_ptr(aggType));
      var lSrcVals: [myLocaleSpace][0..#uintBufferSize] uint(8);
      var rSrcAddrs: [myLocaleSpace] remoteBuffer(aggType);
      var rSrcVals: [myLocaleSpace] remoteBuffer(uint(8));
      var bufferIdxs: c_ptr(int);

      proc ref postinit() {
        dstAddrs = allocate(c_ptr(aggType), numLocales);
        lSrcAddrs = allocate(c_ptr(aggType), numLocales);
        bufferIdxs = bufferIdxAlloc();
        for loc in myLocaleSpace {
          dstAddrs[loc] = allocate(aggType, bufferSize);
          lSrcAddrs[loc] = allocate(aggType, bufferSize);
          bufferIdxs[loc] = 0;
          rSrcAddrs[loc] = new remoteBuffer(aggType, bufferSize, loc);
          rSrcVals[loc] = new remoteBuffer(uint(8), uintBufferSize, loc);
        }
      }

      proc ref deinit() {
        flush();
        for loc in myLocaleSpace {
          deallocate(dstAddrs[loc]);
          deallocate(lSrcAddrs[loc]);
        }
        deallocate(dstAddrs);
        deallocate(lSrcAddrs);
        deallocate(bufferIdxs);
      }

      proc ref flush() {
        for offsetLoc in myLocaleSpace + lastLocale {
          const loc = offsetLoc % numLocales;
          flushBuffer(loc, bufferIdxs[loc], freeData=true);
        }
      }

      inline proc ref copy(ref dst: bigint, const ref src: bigint) {
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
          flushBuffer(loc, bufferIdx, freeData=false);
          opsUntilYield = yieldFrequency;
        } else if opsUntilYield == 0 {
          currentTask.yieldExecution();
          opsUntilYield = yieldFrequency;
        } else {
          opsUntilYield -= 1;
        }
      }

      proc ref flushBuffer(loc: int, ref bufferIdx, freeData) {
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

        var bytesValsWritten: 2*int;
        var addrBufferIdx = 0;

        while bytesValsWritten(1) < myBufferIdx {
          // Process remote buffer, copying the value of our addresses into a
          // remote buffer
          const cbytesValsWritten = bytesValsWritten;
          const myBufferSize = uintBufferSize;
          on Locales[loc] {
            const mycbytesValsWritten = cbytesValsWritten;
            var (valueBufferIdx, addrBufferIdx) = mycbytesValsWritten;
            valueBufferIdx = 0;

            while valueBufferIdx < myBufferSize && addrBufferIdx < myBufferIdx {
              var srcAddr = rSrcAddrPtr[addrBufferIdx];

              var ser_size = srcAddr.deref().serializedSize();
              if ser_size > myBufferSize {
                // Halt if the size is too big. A slow fallback would be to
                // just break and have the initiator see the bytes written
                // didn't advance and do the assignment. Smarter would be to
                // serialize size and enough to call `chpl_gmp_get_mpz`
                halt("size of bigint exceeds max serialized size");
              }
              if valueBufferIdx + ser_size > myBufferSize {
                break;
              }

              // copy value for current address into value array
              srcAddr.deref().serializeInto(c_ptrTo(rSrcValPtr[valueBufferIdx]));

              valueBufferIdx += ser_size;
              addrBufferIdx += 1;
            }
            bytesValsWritten = (valueBufferIdx, addrBufferIdx);

            if freeData && addrBufferIdx == myBufferIdx {
              myRSrcAddrs.localFree(rSrcAddrPtr);
            }
          }

          // Copy remote values into local buffer
          myRSrcVals.GET(myLSrcVals, bytesValsWritten(0));

          // Assign the srcVal to the dstAddrs
          var dstAddrPtr = c_ptrTo(dstAddrs[loc][0]);
          var srcValPtr = c_ptrTo(myLSrcVals[0]);
          var curBufferIdx = 0;
          while addrBufferIdx < bytesValsWritten(1) {
            var dstAddr = dstAddrPtr[addrBufferIdx];
            var ser_size = dstAddr.deref().deserializeFrom(c_ptrTo(srcValPtr[curBufferIdx]));
            curBufferIdx += ser_size:int;
            addrBufferIdx += 1;
          }
        }

        if freeData {
          myRSrcAddrs.markFreed();
        }


        bufferIdx = 0;
      }
    }
  }

}
