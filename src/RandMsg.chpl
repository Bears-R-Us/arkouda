module RandMsg
{
    use ServerConfig;
    
    use Time;
    use Math;
    use Reflection;
    use ServerErrors;
    use ServerConfig;
    use Logging;
    use Message;
    use RandArray;
    use CommAggregation;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use ArkoudaRandomCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const randLogger = new Logger(logLevel, logChannel);

    /*
    parse, execute, and respond to randint message
    uniform int in half-open interval [min,max)

    :arg reqMsg: message to process (contains cmd,aMin,aMax,len,dtype)
    */
    @arkouda.registerND
    proc randintMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        
        const shape = msgArgs.get("shape").getTuple(nd);
        const dtype = str2dtype(msgArgs.getValueOf("dtype"));
        const seed = msgArgs.getValueOf("seed");
        const low = msgArgs.get("low");
        const high = msgArgs.get("high");

        var len = 1;
        for s in shape do len *= s;

        // get next symbol name
        const rname = st.nextName();

        // if verbose print action
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
               "cmd: %s len: %i dtype: %s rname: %s aMin: %s: aMax: %s".doFormat(
                                           cmd,len,dtype2str(dtype),rname,low.getValue(),high.getValue()));

        proc doFillRand(type t, param sub: t): MsgTuple throws {
            overMemLimit(len);
            const aMin = low.getScalarValue(t),
                  aMax = high.getScalarValue(t) - sub,
                  t1 = Time.timeSinceEpoch().totalSeconds();
            var e = st.addEntry(rname, (...shape), t);
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "alloc time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
            const t2 = Time.timeSinceEpoch().totalSeconds();
            fillRand(e.a, aMin, aMax, seed);
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "compute time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t2));

            const repMsg = "created " + st.attrib(rname);
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        inline proc notImplemented(): MsgTuple throws {
            const errorMsg = unsupportedTypeError(dtype, pn);
            randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        select dtype {
            when DType.Int8 {
                if SupportsInt8
                    then return doFillRand(int(8), 1);
                    else return notImplemented();
            }
            when DType.Int16 {
                if SupportsInt16
                    then return doFillRand(int(16), 1);
                    else return notImplemented();
            }
            when DType.Int32 {
                if SupportsInt32
                    then return doFillRand(int(32), 1);
                    else return notImplemented();
            }
            when DType.Int64 {
                if SupportsInt64
                    then return doFillRand(int, 1);
                    else return notImplemented();
            }
            when DType.UInt8 {
                if SupportsUint8
                    then return doFillRand(uint(8), 1);
                    else return notImplemented();
            }
            when DType.UInt16 {
                if SupportsUint16
                    then return doFillRand(uint(16), 1);
                    else return notImplemented();
            }
            when DType.UInt32 {
                if SupportsUint32
                    then return doFillRand(uint(32), 1);
                    else return notImplemented();
            }
            when DType.UInt64 {
                if SupportsUint64
                    then return doFillRand(uint, 1);
                    else return notImplemented();
            }
            when DType.Float32 {
                if SupportsFloat32
                    then return doFillRand(real(32), 0.0);
                    else return notImplemented();
            }
            when DType.Float64 {
                if SupportsFloat64
                    then return doFillRand(real, 0.0);
                    else return notImplemented();
            }
            when DType.Complex64 {
                if SupportsComplex64
                    then return doFillRand(complex(64), 0.0 + 0.0i);
                    else return notImplemented();
            }
            when DType.Complex128 {
                if SupportsComplex128
                    then return doFillRand(complex, 0.0 + 0.0i);
                    else return notImplemented();
            }
            when DType.Bool {
                if SupportsBool {
                    overMemLimit(len);
                    const t1 = Time.timeSinceEpoch().totalSeconds();
                    var e = st.addEntry(rname, (...shape), bool);
                    randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "alloc time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t1));
                    const t2 = Time.timeSinceEpoch().totalSeconds();
                    fillBool(e.a, seed);
                    randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "compute time = %i sec".doFormat(Time.timeSinceEpoch().totalSeconds() - t2));

                    const repMsg = "created " + st.attrib(rname);
                    randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                }
                else return notImplemented();
            }
            otherwise {
                var errorMsg = notImplementedError(pn,dtype);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    proc randomNormalMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var pn = Reflection.getRoutineName();
        const len = msgArgs.get("size").getIntValue();
        // Result + 2 scratch arrays
        overMemLimit(3*8*len);
        var rname = st.nextName();
        var entry = createSymEntry(len, real);
        fillNormal(entry.a, msgArgs.getValueOf("seed"));
        st.addEntry(rname, entry);

        var repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
     * Creates a generator server-side and returns the SymTab name used to
     * retrieve the generator from the SymTab.
     */
    proc createGeneratorMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName();
        var rname: string;
        const hasSeed = msgArgs.get("has_seed").getBoolValue();
        const seed = if hasSeed then msgArgs.get("seed").getIntValue() else -1;
        const dtypeStr = msgArgs.getValueOf("dtype");
        const dtype = str2dtype(dtypeStr);
        const state = msgArgs.get("state").getIntValue();

        if hasSeed {
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "dtype: %? seed: %i state: %i".doFormat(dtypeStr,seed,state));
        }
        else {
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "dtype: %? state: %i".doFormat(dtypeStr,state));
        }

        proc creationHelper(type t, seed, state, st: borrowed SymTab): string throws {
            var generator = if hasSeed then new randomStream(t, seed) else new randomStream(t);
            if state != 1 {
                // you have to skip to one before where you want to be
                generator.skipTo(state-1);
            }
            var entry = new shared GeneratorSymEntry(generator, state);
            var name = st.nextName();
            st.addEntry(name, entry);
            return name;
        }

        select dtype {
            when DType.Int64 {
                rname = creationHelper(int, seed, state, st);
            }
            when DType.UInt64 {
                rname = creationHelper(uint, seed, state, st);
            }
            when DType.Float64 {
                rname = creationHelper(real, seed, state, st);
            }
            when DType.Bool {
                rname = creationHelper(bool, seed, state, st);
            }
            otherwise {
                var errorMsg = "Unhandled data type %s".doFormat(dtypeStr);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }

        const repMsg = st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc uniformGeneratorMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName();
        var rname = st.nextName();
        const name = msgArgs.getValueOf("name");
        const size = msgArgs.get("size").getIntValue();
        const dtypeStr = msgArgs.getValueOf("dtype");
        const dtype = str2dtype(dtypeStr);
        const state = msgArgs.get("state").getIntValue();

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? size %i dtype: %? state %i".doFormat(name, size, dtypeStr, state));

        st.checkTable(name);

        proc uniformHelper(type t, low, high, state, st: borrowed SymTab) throws {
            var generatorEntry: borrowed GeneratorSymEntry(t) = toGeneratorSymEntry(st.lookup(name), t);
            ref rng = generatorEntry.generator;
            if state != 1 {
                // you have to skip to one before where you want to be
                rng.skipTo(state-1);
            }
            var uniformEntry = createSymEntry(size, t);
            if t != bool {
                rng.fill(uniformEntry.a, low, high);
            }
            else {
                // chpl doesn't support bounded random with boolean type
                rng.fill(uniformEntry.a);
            }
            st.addEntry(rname, uniformEntry);
        }

        select dtype {
            when DType.Int64 {
                const low = msgArgs.get("low").getIntValue();
                const high = msgArgs.get("high").getIntValue();
                uniformHelper(int, low, high, state, st);
            }
            when DType.UInt64 {
                const low = msgArgs.get("low").getIntValue();
                const high = msgArgs.get("high").getIntValue();
                uniformHelper(uint, low, high, state, st);
            }
            when DType.Float64 {
                const low = msgArgs.get("low").getRealValue();
                const high = msgArgs.get("high").getRealValue();
                uniformHelper(real, low, high, state, st);
            }
            when DType.Bool {
                // chpl doesn't support bounded random with boolean type
                uniformHelper(bool, 0, 1, state, st);
            }
            otherwise {
                var errorMsg = "Unhandled data type %s".doFormat(dtypeStr);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }
        var repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc standardNormalGeneratorMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName(),
              name = msgArgs.getValueOf("name"),                // generator name
              size = msgArgs.get("size").getIntValue(),         // population size
              state = msgArgs.get("state").getIntValue(),       // rng state
              rname = st.nextName();

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? size %i state %i".doFormat(name, size, state));

        st.checkTable(name);

        var generatorEntry: borrowed GeneratorSymEntry(real) = toGeneratorSymEntry(st.lookup(name), real);
        ref rng = generatorEntry.generator;
        if state != 1 {
            // you have to skip to one before where you want to be
            rng.skipTo(state-1);
        }

        // uses Box–Muller transform
        // generates values drawn from the standard normal distribution using
        // 2 uniformly distributed random numbers
        var u1 = makeDistArray(size, real);
        var u2 = makeDistArray(size, real);
        rng.fill(u1);
        rng.fill(u2);

        var standNorm = sqrt(-2*log(u1))*cos(2*pi*u2);
        st.addEntry(rname, createSymEntry(standNorm));

        var repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc segmentedSampleMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName(),
              genName = msgArgs.getValueOf("genName"),                          // generator name
              permName = msgArgs.getValueOf("perm"),                            // values array name
              segsName = msgArgs.getValueOf("segs"),                            // segments array name
              segLensName = msgArgs.getValueOf("segLens"),                      // segment lengths array name
              weightsName = msgArgs.getValueOf("weights"),                      // permuted weights array name
              numSamplesName = msgArgs.getValueOf("numSamples"),                // number of samples per segment array name
              replace = msgArgs.get("replace").getBoolValue(),                  // sample with replacement
              hasWeights = msgArgs.get("hasWeights").getBoolValue(),            // flag indicating whether weighted sample
              hasSeed = msgArgs.get("hasSeed").getBoolValue(),                  // flag indicating if generator is seeded
              seed = if hasSeed then msgArgs.get("seed").getIntValue() else -1, // value of seed if present
              state = msgArgs.get("state").getIntValue(),                       // rng state
              rname = st.nextName();

        randLogger.debug(getModuleName(),pn,getLineNumber(),
                         "genName: %? permName %? segsName: %? weightsName: %? numSamplesName %? replace %i hasWeights %i state %i rname %?"
                         .doFormat(genName, permName, segsName, weightsName, numSamplesName, replace, hasWeights, state, rname));

        st.checkTable(permName);
        st.checkTable(segsName);
        st.checkTable(segLensName);
        st.checkTable(numSamplesName);
        const permutation = toSymEntry(getGenericTypedArrayEntry(permName, st),int).a;
        const segments = toSymEntry(getGenericTypedArrayEntry(segsName, st),int).a;
        const segLens = toSymEntry(getGenericTypedArrayEntry(segLensName, st),int).a;
        const numSamples = toSymEntry(getGenericTypedArrayEntry(numSamplesName, st),int).a;

        const sampleOffset = (+ scan numSamples) - numSamples;
        var sampledPerm: [makeDistDom(+ reduce numSamples)] int;

        if hasWeights {
            st.checkTable(weightsName);
            const weights = toSymEntry(getGenericTypedArrayEntry(weightsName, st),real).a;

            forall (segOff, segLen, sampleOff, numSample) in zip(segments, segLens, sampleOffset, numSamples)
                                                 with (var rs = if hasSeed then new randomStream(real, seed) else new randomStream(real)) {
                if state != 1 then rs.skipTo((state+sampleOff) - 1); else rs.skipTo(sampleOff);
                const ref segPerm = permutation[segOff..#segLen];
                const ref segWeights = weights[segOff..#segLen];
                sampledPerm[sampleOff..#numSample] = randSampleWeights(rs, segPerm, segWeights, numSample, replace);
            }
        }
        else {
            forall (segOff, segLen, sampleOff, numSample) in zip(segments, segLens, sampleOffset, numSamples)
                                                 with (var rs = if hasSeed then new randomStream(int, seed) else new randomStream(int)) {
                if state != 1 then rs.skipTo((state+sampleOff) - 1); else rs.skipTo(sampleOff);
                const ref segPerm = permutation[segOff..#segLen];
                sampledPerm[sampleOff..#numSample] = rs.sample(segPerm, numSample, replace);
            }
        }

        st.addEntry(rname, createSymEntry(sampledPerm));
        const repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc choiceMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName(),
              gName = msgArgs.getValueOf("gName"),                      // generator name
              aName = msgArgs.getValueOf("aName"),                      // values array name
              wName = msgArgs.getValueOf("wName"),                      // weights array name
              numSamples = msgArgs.get("numSamples").getIntValue(),     // number of samples
              replace = msgArgs.get("replace").getBoolValue(),          // sample with replacement
              hasWeights = msgArgs.get("hasWeights").getBoolValue(),    // flag indicating whether weighted sample
              isDom = msgArgs.get("isDom").getBoolValue(),              // flag indicating whether return is domain or array
              popSize  = msgArgs.get("popSize").getIntValue(),          // population size
              dtypeStr = msgArgs.getValueOf("dtype"),                   // string version of dtype
              dtype = str2dtype(dtypeStr),                              // DType enum
              state = msgArgs.get("state").getIntValue(),               // rng state
              rname = st.nextName();

        randLogger.debug(getModuleName(),pn,getLineNumber(),
                         "gname: %? aname %? wname: %? numSamples %i replace %i hasWeights %i isDom %i dtype %? popSize %? state %i rname %?"
                         .doFormat(gName, aName, wName, numSamples, replace, hasWeights, isDom, dtypeStr, popSize, state, rname));

        proc weightedIdxHelper() throws {
            var generatorEntry = toGeneratorSymEntry(st.lookup(gName), real);
            ref rng = generatorEntry.generator;

            if state != 1 then rng.skipTo(state-1);

            st.checkTable(wName);
            const weights = toSymEntry(getGenericTypedArrayEntry(wName, st),real).a;
            return sampleDomWeighted(rng, numSamples, weights, replace);
        }

        proc idxHelper() throws {
            var generatorEntry = toGeneratorSymEntry(st.lookup(gName), int);
            ref rng = generatorEntry.generator;

            if state != 1 then rng.skipTo(state-1);

            const choiceDom = {0..<popSize};
            return rng.sample(choiceDom, numSamples, replace);
        }

        proc choiceHelper(type t) throws {
            // I had to break these 2 helpers out into seprate functions since they have different types for generatorEntry
            const choiceIdx = if hasWeights then weightedIdxHelper() else idxHelper();

            if isDom {
                const choiceEntry = createSymEntry(choiceIdx);
                st.addEntry(rname, choiceEntry);
            }
            else {
                var choiceArr: [makeDistDom(numSamples)] t;
                st.checkTable(aName);
                const myArr = toSymEntry(getGenericTypedArrayEntry(aName, st),t).a;

                forall (ca,idx) in zip(choiceArr, choiceIdx) with (var agg = newSrcAggregator(t)) {
                    agg.copy(ca, myArr[idx]);
                }

                const choiceEntry = createSymEntry(choiceArr);
                st.addEntry(rname, choiceEntry);
            }
            const repMsg = "created " + st.attrib(rname);
            randLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select dtype {
            when DType.Int64 {
                return choiceHelper(int);
            }
            when DType.UInt64 {
                return choiceHelper(uint);
            }
            when DType.Float64 {
                return choiceHelper(real);
            }
            when DType.Bool {
                return choiceHelper(bool);
            }
            otherwise {
                const errorMsg = "Unhandled data type %s".doFormat(dtypeStr);
                randLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }
    }

    proc permutationMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName();
        var rname = st.nextName();
        const name = msgArgs.getValueOf("name");
        const xName = msgArgs.getValueOf("x");
        const size = msgArgs.get("size").getIntValue();
        const dtypeStr = msgArgs.getValueOf("dtype");
        const dtype = str2dtype(dtypeStr);
        const state = msgArgs.get("state").getIntValue();
        const isDomPerm = msgArgs.get("isDomPerm").getBoolValue();

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? size %i dtype: %? state %i isDomPerm %?".doFormat(name, size, dtypeStr, state, isDomPerm));

        st.checkTable(name);

        proc permuteHelper(type t) throws {
            // we need the int generator in order for permute(domain) to work correctly
            var intGeneratorEntry: borrowed GeneratorSymEntry(int) = toGeneratorSymEntry(st.lookup(name), int);
            ref intRng = intGeneratorEntry.generator;

            if state != 1 {
                // you have to skip to one before where you want to be
                intRng.skipTo(state-1);
            }
            const permutedDom = makeDistDom(size);
            const permutedIdx = intRng.permute(permutedDom);

            if isDomPerm {
                const permutedEntry = createSymEntry(permutedIdx);
                st.addEntry(rname, permutedEntry);
            }
            else {
                // permute requires that the stream's eltType is coercible to the array/domain's idxType,
                // so we use permute(dom) and use that to gather the permuted vals
                var permutedArr: [permutedDom] t;
                ref myArr = toSymEntry(getGenericTypedArrayEntry(xName, st),t).a;

                forall (pa,idx) in zip(permutedArr, permutedIdx) with (var agg = newSrcAggregator(t)) {
                    agg.copy(pa, myArr[idx]);
                }

                const permutedEntry = createSymEntry(permutedArr);
                st.addEntry(rname, permutedEntry);
            }
        }

        select dtype {
            when DType.Int64 {
                permuteHelper(int);
            }
            when DType.UInt64 {
                permuteHelper(uint);
            }
            when DType.Float64 {
                permuteHelper(real);
            }
            when DType.Bool {
                permuteHelper(bool);
            }
            otherwise {
                var errorMsg = "Unhandled data type %s".doFormat(dtypeStr);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }
        var repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc poissonGeneratorMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName(),
              name = msgArgs.getValueOf("name"),                                // generator name
              isSingleLam = msgArgs.get("is_single_lambda").getBoolValue(),     // boolean indicating if lambda is a single value or array
              lamStr = msgArgs.getValueOf("lam"),                               // lambda for poisson distribution
              size = msgArgs.get("size").getIntValue(),                         // number of values to be generated
              hasSeed = msgArgs.get("has_seed").getBoolValue(),                  // boolean indicating if the generator has a seed
              state = msgArgs.get("state").getIntValue(),                       // rng state
              rname = st.nextName();


        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? size %i isSingleLam %? lamStr %? state %i".doFormat(name, size, isSingleLam, lamStr, state));

        st.checkTable(name);

        var generatorEntry: borrowed GeneratorSymEntry(real) = toGeneratorSymEntry(st.lookup(name), real);
        ref rng = generatorEntry.generator;
        if state != 1 {
            // you have to skip to one before where you want to be
            rng.skipTo(state-1);
        }

        // uses the algorithm from knuth found here:
        // https://en.wikipedia.org/wiki/Poisson_distribution#Random_variate_generation
        // generates values drawn from poisson distribution using a stream of uniformly distributed random numbers
        var poissonArr = makeDistArray(size, int);

        if hasSeed {
            // use a fixed number of elements per stream instead of relying on number of locales or numTasksPerLoc because these
            // can vary from run to run / machine to mahchine. And it's important for the same seed to give the same results
            const generatorSeed = (rng.next() * 2**62):int,
                minPerStream = 256,
                elemsPerStream = max(minPerStream, 2**(2 * ceil(log10(size)):int));

            if isSingleLam {
                const lam = lamStr:real;
                // using nested coforalls over locales and tasks so we know how to generate taskSeed
                coforall loc in Locales do on loc {
                    const locSubDom = poissonArr.localSubdomain(),
                        offset = if loc.id == 0 then 0 else elemsPerStream - (locSubDom.low % elemsPerStream);

                    // skip if all the values were pulled to previous locale
                    if offset <= locSubDom.high {
                        // we take the ceil in chunk calculation because if elemsPerStream doesn't evenly divide along locale boundaries, the remainder is pulled to the previous locale
                        const chunksAlreadyDone = if loc.id == 0 then 0 else ceil((locSubDom.low + 1) / elemsPerStream:real):int,  // number of chunks handled by previous locales
                            thisLocsNumChunks = ceil((locSubDom.high + 1 - locSubDom.low + offset) / elemsPerStream:real):int;  // number of chunks this locale needs to handle

                        coforall streamID in 0..<thisLocsNumChunks {
                            const taskSeed = generatorSeed + chunksAlreadyDone + streamID,  // initial seed offset by other locales threads plus current thread id
                                startIdx = (streamID * elemsPerStream) + offset,
                                stopIdx = min(startIdx + elemsPerStream, (poissonArr.domain.size - locSubDom.low));  // continue past end of localSubDomain to read full block to avoid seed sharing

                            var rs = new randomStream(real, taskSeed);
                            for i in startIdx..<stopIdx {
                                var L = exp(-lam);
                                var k = 0;
                                var p = 1.0;

                                do {
                                    k += 1;
                                    p = p * rs.next(0, 1);
                                } while p > L;
                                poissonArr[locSubDom.low + i] = k - 1;
                            }
                        }
                    }
                }
            }
            else {
                st.checkTable(lamStr);
                const lamArr = toSymEntry(getGenericTypedArrayEntry(lamStr, st),real).a;

                // using nested coforalls over locales and tasks so we know how to generate taskSeed
                coforall loc in Locales do on loc {
                    const locSubDom = poissonArr.localSubdomain(),
                        offset = if loc.id == 0 then 0 else elemsPerStream - (locSubDom.low % elemsPerStream);

                    // skip if all the values were pulled to previous locale
                    if offset <= locSubDom.high {
                        // we take the ceil in chunk calculation because if elemsPerStream doesn't evenly divide along locale boundaries, the remainder is pulled to the previous locale
                        const chunksAlreadyDone = if loc.id == 0 then 0 else ceil((locSubDom.low + 1) / elemsPerStream:real):int,  // number of chunks handled by previous locales
                            thisLocsNumChunks = ceil((locSubDom.high + 1 - locSubDom.low + offset) / elemsPerStream:real):int;  // number of chunks this locale needs to handle

                        coforall streamID in 0..<thisLocsNumChunks {
                            const taskSeed = generatorSeed + chunksAlreadyDone + streamID,  // initial seed offset by other locales threads plus current thread id
                                startIdx = (streamID * elemsPerStream) + offset,
                                stopIdx = min(startIdx + elemsPerStream, (poissonArr.domain.size - locSubDom.low));  // continue past end of localSubDomain to read full block to avoid seed sharing

                            var rs = new randomStream(real, taskSeed);
                            for i in startIdx..<stopIdx {
                                const lam = lamArr[locSubDom.low + i];
                                var L = exp(-lam);
                                var k = 0;
                                var p = 1.0;

                                do {
                                    k += 1;
                                    p = p * rs.next(0, 1);
                                } while p > L;
                                poissonArr[locSubDom.low + i] = k - 1;
                            }
                        }
                    }
                }
            }
        }
        else {  // non-seeded case, we can just use task private variables for our random streams
            if isSingleLam {
                const lam = lamStr:real;
                forall pv in poissonArr with (var rs = new randomStream(real)) {
                    var L = exp(-lam);
                    var k = 0;
                    var p = 1.0;

                    do {
                        k += 1;
                        p = p * rs.next(0, 1);
                    } while p > L;
                    pv = k - 1;
                }
            }
            else {
                st.checkTable(lamStr);
                const lamArr = toSymEntry(getGenericTypedArrayEntry(lamStr, st),real).a;
                forall (pv, lam) in zip(poissonArr, lamArr) with (var rs = new randomStream(real)) {
                    var L = exp(-lam);
                    var k = 0;
                    var p = 1.0;

                    do {
                        k += 1;
                        p = p * rs.next(0, 1);
                    } while p > L;
                    pv = k - 1;
                }
            }
        }
        st.addEntry(rname, createSymEntry(poissonArr));
        const repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc shuffleMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("name");
        const xName = msgArgs.getValueOf("x");
        const size = msgArgs.get("size").getIntValue();
        const dtypeStr = msgArgs.getValueOf("dtype");
        const dtype = str2dtype(dtypeStr);
        const state = msgArgs.get("state").getIntValue();

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? size %i dtype: %? state %i".doFormat(name, size, dtypeStr, state));

        st.checkTable(name);

        proc shuffleHelper(type t) throws {
            var generatorEntry: borrowed GeneratorSymEntry(int) = toGeneratorSymEntry(st.lookup(name), int);
            ref rng = generatorEntry.generator;

            if state != 1 {
                // you have to skip to one before where you want to be
                rng.skipTo(state-1);
            }

            ref myArr = toSymEntry(getGenericTypedArrayEntry(xName, st),t).a;
            rng.shuffle(myArr);
        }

        select dtype {
            when DType.Int64 {
                shuffleHelper(int);
            }
            when DType.UInt64 {
                shuffleHelper(uint);
            }
            when DType.Float64 {
                shuffleHelper(real);
            }
            when DType.Bool {
                shuffleHelper(bool);
            }
            otherwise {
                var errorMsg = "Unhandled data type %s".doFormat(dtypeStr);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }
        var repMsg = "created " + st.attrib(xName);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("randomNormal", randomNormalMsg, getModuleName());
    registerFunction("createGenerator", createGeneratorMsg, getModuleName());
    registerFunction("uniformGenerator", uniformGeneratorMsg, getModuleName());
    registerFunction("standardNormalGenerator", standardNormalGeneratorMsg, getModuleName());
    registerFunction("segmentedSample", segmentedSampleMsg, getModuleName());
    registerFunction("choice", choiceMsg, getModuleName());
    registerFunction("permutation", permutationMsg, getModuleName());
    registerFunction("poissonGenerator", poissonGeneratorMsg, getModuleName());
    registerFunction("shuffle", shuffleMsg, getModuleName());
}
