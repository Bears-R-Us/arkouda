module ArkoudaRandomCompat {
  use Random;
  use PCGRandom;
  use PCGRandomLib;

  proc fillRandom(arr: [], min: arr.eltType, max: arr.eltType,
                  seed: int(64) = SeedGenerator.oddCurrentTime) {
    var randNums = createRandomStream(seed=seed, eltType=arr.eltType,
                                      parSafe=false, algorithm=defaultRNG);
    randNums.fillRandom(arr, min, max);
  }
  
  proc PCGRandomStream.fillRandom(arr: [], min: arr.eltType, max:arr.eltType) {
    if(!arr.isRectangular()) then
      compilerError("fillRandom does not support non-rectangular arrays");
    
    forall (x, r) in zip(arr, iterate(arr.domain, arr.eltType, min, max)) do
      x = r;
  }

  proc PCGRandomStream.iterate(D: domain, type resultType=eltType,
                               min: resultType, max: resultType) {
    _lock();
    const start = PCGRandomStreamPrivate_count;
    PCGRandomStreamPrivate_count += D.sizeAs(int);
    PCGRandomStreamPrivate_skipToNth_noLock(PCGRandomStreamPrivate_count-1);
    _unlock();
    return PCGRandomPrivate_iterate_bounded(resultType, D, seed, start,
                                            min, max);
  }

  proc PCGRandomStream.iterate(D: domain, type resultType=eltType,
                               min: resultType, max: resultType, param tag)
    where tag == iterKind.leader
  {
    // Note that proc iterate() for the serial case (i.e. the one above)    
    // is going to be invoked as well, so we should not be taking           
    // any actions here other than the forwarding.                          
    const start = PCGRandomStreamPrivate_count;
    return PCGRandomPrivate_iterate_bounded(resultType, D, seed, start,
                                            min, max, tag);
  }


  pragma "no doc"
  iter PCGRandomPrivate_iterate_bounded(type resultType, D: domain,
                                        seed: int(64), start: int(64),
                                        min: resultType, max: resultType) {
    var cursor = randlc_skipto(resultType, seed, start);
    var count = start;
    for i in D {
      yield randlc_bounded(resultType, cursor, seed, count, min, max);
      count += 1;
    }
  }

  pragma "no doc"
  iter PCGRandomPrivate_iterate_bounded(type resultType, D: domain,
                                        seed: int(64),
                                        start: int(64),
                                        min: resultType, max: resultType,
                                        param tag: iterKind)
  where tag == iterKind.leader {
    for block in D.these(tag=iterKind.leader) do
      yield block;
  }
  
  pragma "no doc"
  iter PCGRandomPrivate_iterate_bounded(type resultType, D: domain,
                                        seed: int(64), start: int(64),
                                        min: resultType, max: resultType,
                                        param tag: iterKind, followThis)
  where tag == iterKind.follower {
    use DSIUtil;
    param multiplier = 1;
    const ZD = computeZeroBasedDomain(D);
    const innerRange = followThis(ZD.rank-1);
    for outer in outer(followThis) {
      var myStart = start;
      if ZD.rank > 1 then
        myStart += multiplier * ZD.indexOrder(((...outer), innerRange.low)).safeCast(int(64));
      else
        myStart += multiplier * ZD.indexOrder(innerRange.low).safeCast(int(64));
      if !innerRange.stridable {
        var cursor = randlc_skipto(resultType, seed, myStart);
        var count = myStart;
        for i in innerRange {
          yield randlc_bounded(resultType, cursor, seed, count, min, max);
          count += 1;
        }
      } else {
        myStart -= innerRange.low.safeCast(int(64));
        for i in innerRange {
          var count = myStart + i.safeCast(int(64)) * multiplier;
          var cursor = randlc_skipto(resultType, seed, count);
          yield randlc_bounded(resultType, cursor, seed, count, min, max);
        }
      }
    }
  }

  private proc randlc_skipto(type resultType, seed: int(64), n: integral) {
    var states: numGenerators(resultType) * pcg_setseq_64_xsh_rr_32_rng;

    for param i in 0..states.size-1 {
      param inc = pcg_getvalid_inc(i+1);
      states[i].srandom(seed:uint(64), inc);
      states[i].advance(inc, (n - 1):uint(64));
    }
    return states;
  }

  private
  proc numGenerators(type t) param {
    if isBoolType(t) then return 1;
    else return (numBits(t)+31) / 32;
  }

  private inline
  proc randlc_bounded(type resultType,
                      ref states, seed:int(64), count:int(64),
                      min, max) {

    checkSufficientBitsAndAdvanceOthers(resultType, states);

    if resultType == complex(128) {
      return (randToReal64(rand64_1(states), min.re, max.re),
              randToReal64(rand64_2(states), min.im, max.im)):complex(128);
    } else if resultType == complex(64) {
      return (randToReal32(rand32_1(states), min.re, max.re),
              randToReal32(rand32_2(states), min.im, max.im)):complex(64);
    } else if resultType == imag(64) {
      return _r2i(randToReal64(rand64_1(states), _i2r(min), _i2r(max)));
    } else if resultType == imag(32) {
      return _r2i(randToReal32(rand32_1(states), _i2r(min), _i2r(max)));
    } else if resultType == real(64) {
      return randToReal64(rand64_1(states), min, max);
    } else if resultType == real(32) {
      return randToReal32(rand32_1(states), min, max);
    } else if resultType == uint(64) || resultType == int(64) {
      return (boundedrand64_1(states, seed, count, (max-min):uint(64)) + min:uint(64)):resultType;
    } else if resultType == uint(32) || resultType == int(32) {
      return (boundedrand32_1(states, seed, count, (max-min):uint(32)) + min:uint(32)):resultType;
    } else if(resultType == uint(16) ||
              resultType == int(16)) {
      return (boundedrand32_1(states, seed, count, (max-min):uint(32)) + min:uint(32)):resultType;
    } else if(resultType == uint(8) ||
              resultType == int(8)) {
      return (boundedrand32_1(states, seed, count, (max-min):uint(32)) + min:uint(32)):resultType;
    } else if isBoolType(resultType) {
      compilerError("bounded rand with boolean type");
      return false;
    }
  }

  private
  proc checkSufficientBitsAndAdvanceOthers(type resultType, ref states) {
    // Note - this error could be eliminated if we used                       
    // the same strategy as bounded_rand_vary_inc and                         
    // just computed the RNGs at the later incs                               
    param numGenForResultType = numGenerators(resultType);
    param numGen = states.size;
    if numGenForResultType > numGen then
      compilerError("PCGRandomStream cannot produce " +
                    resultType:string +
                    " (requiring " +
                    (32*numGenForResultType):string +
                    " bits) from a stream configured for " +
                    (32*numGen):string +
                    " bits of output");

    // Step each RNG that is not involved in the output.                      
    for i in numGenForResultType+1..numGen {
      states[i-1].random(pcg_getvalid_inc(i:uint));
    }
  }

  private proc boundedrand64_1(ref states, seed:int(64), count:int(64),
                               bound:uint):uint
  {
    if bound > max(uint(32)):uint {
      var toprand = 0:uint;
      var botrand = 0:uint;
      
      // compute the bounded number in two calls to a 32-bit RNG              
      toprand = boundedrand32_1(states, seed, count, (bound >> 32):uint(32));
      botrand = boundedrand32_2(states, seed, count, (bound & max(uint(32))):uint(32));
      return (toprand << 32) | botrand;
    } else {
      // Generate a # with RNG 1 but ignore it, to keep the                   
      // stepping consistent.                                                 
      rand32_1(states);
      return boundedrand32_2(states, seed, count, bound:uint(32));
    }
  }
  
  private inline
  proc boundedrand32_1(ref states, seed:int(64), count:int(64),
                       bound:uint(32)):uint(32) {
      // just get 32 random bits if bound+1 is not representable.               
    if bound == max(uint(32)) then return rand32_1(states);
                              else return states[0].bounded_random_vary_inc(
                                pcg_getvalid_inc(1), bound + 1,
                                seed:uint(64), (count - 1):uint(64),
                                101, 4);
  }

  private inline
  proc rand32_1(ref states):uint(32) {
    return states[0].random(pcg_getvalid_inc(1));
  }

  private inline
  proc boundedrand32_2(ref states, seed:int(64), count:int(64),
                       bound:uint(32)):uint(32) {
    // just get 32 random bits if bound+1 is not representable.               
    if bound == max(uint(32)) then return rand32_2(states);
                              else return states[1].bounded_random_vary_inc(
                                pcg_getvalid_inc(2), bound + 1,
                                seed:uint(64), (count - 1):uint(64),
                                102, 4);
  }

  private inline
  proc rand32_2(ref states):uint(32) {
    return states[1].random(pcg_getvalid_inc(2));
  }
}