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
    use RandUtil;
    use Random;
    use CommAggregation;
    use ZigguratConstants;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use ArkoudaRandomCompat;

    import BigInteger;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const randLogger = new Logger(logLevel, logChannel);

    /*
    parse, execute, and respond to randint message
    uniform int in half-open interval [min,max)

    :arg reqMsg: message to process (contains cmd,aMin,aMax,len,dtype)
    */
    @arkouda.instantiateAndRegister
    proc randint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
        where array_dtype != BigInteger.bigint
    {
        const shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              seed = msgArgs["seed"].toScalar(int);

        const low = msgArgs["low"].toScalar(array_dtype),
              high = msgArgs["high"].toScalar(array_dtype) - if isIntegralType(array_dtype) then 1 else 0;

        var len = 1;
        for s in shape do len *= s;
        overMemLimit(len);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s len: %i dtype: %s aMin: %?: aMax: %?".format(
                         cmd,len,type2str(array_dtype),low,high));

        var t = new stopwatch();
        t.start();

        var e = createSymEntry((...shape), array_dtype);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "alloc time = %? sec".format(t.elapsed()));
        t.restart();

        if array_dtype == bool {
            if seed == -1
                then fillRandom(e.a);
                else fillRandom(e.a, seed);
        } else {
            if seed == -1
                then fillRandom(e.a, low, high);
                else fillRandom(e.a, low, high, seed);
        }

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "compute time = %i sec".format(t.elapsed()));

        return st.insert(e);
    }

    proc randint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
        where array_dtype == BigInteger.bigint
    {
        return MsgTuple.error("randint does not support the bigint dtype");
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
                                            "dtype: %? seed: %i state: %i".format(dtypeStr,seed,state));
        }
        else {
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "dtype: %? state: %i".format(dtypeStr,state));
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
                var errorMsg = "Unhandled data type %s".format(dtypeStr);
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
                                "name: %? size %i dtype: %? state %i".format(name, size, dtypeStr, state));

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
                var errorMsg = "Unhandled data type %s".format(dtypeStr);
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
                                "name: %? size %i state %i".format(name, size, state));

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

    /*
        Use the ziggurat method (https://en.wikipedia.org/wiki/Ziggurat_algorithm#Theory_of_operation)
        to generate exponentially distributed numbers using n (num of rectangles) = 256.
        The number of rectangles only impacts how quick the algorithm is, not it's accuracy.
        This is relatively fast because the common case is not computationally expensive

        In this algorithm we choose uniformly and then decide whether to accept the candidate or not
        depending on whether it falls under the pdf of our distribution

        A good explaination of the ziggurat algorithm is:
        https://blogs.mathworks.com/cleve/2015/05/18/the-ziggurat-random-normal-generator/

        This implementation based on numpy's:
        https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c

        Uses constants from lists of size 256 (number of rectangles) found in ZigguratConstants.chpl
    */
    inline proc standardExponentialZig(ref realRng, ref uintRng): real {
        // modified from numpy to use while loop instead of recusrsion so we can inline the proc
        var count = 0;
        // to guarantee completion this should be while true, but limiting to 100 tries will make the chance
        // of failure near zero while avoiding the possibility of an infinite loop
        // the odds of failing 100 times in a row is (1 - .989)**100 = 1.3781e-196
        while count <= 100 {
            var ri = uintRng.next();
            ri >>= 3;

            // AND with 0xFF (255) to get our index into our 256 long const arrays
            var idx = ri & 0xFF;
            ri >>= 8;
            var x = ri * we_double[idx];
            if ri < ke_double[idx] {
                // the point fell in the core of one of our rectangular slices, so we're guaranteed
                // it falls under the pdf curve. We can return it as a sample from our distribution.
                // We will return here 98.9% of the time on the 1st try
                return x;
            }

            // The fall back algorithm for calculating if the sample point lies under the pdf.
            // Either in the tip of one of the rectangles or in the tail of the distribution.
            // See https://blogs.mathworks.com/cleve/2015/05/18/the-ziggurat-random-normal-generator/
            // the tip calculation is based on standardExponentialUnlikely from numpy defined here:
            // https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c

            // candidate point did not fall in the core of any rectangular slices. Either it lies in the
            // first slice (which doesn't have a core), in the tail of the distribution, or in the tip of one of slices.
            if idx == 0 {
                // first rectangular slice; let x = x1 − ln(U1)
                return ziggurat_exp_r - log1p(-realRng.next());
            }
            else if (fe_double[idx-1] - fe_double[idx]) * realRng.next() + fe_double[idx] < exp(-x) {
                // tip calculation
                return x;
            }
            // reject sample and retry
        }
        return -1.0;  // we failed 100 times in a row which should practically never happen
    }

    proc standardExponentialMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName(),
              name = msgArgs.getValueOf("name"),                // generator name
              size = msgArgs.get("size").getIntValue(),         // population size
              method = msgArgs.getValueOf("method"),            // method to use to generate exponential samples
              hasSeed = msgArgs.get("has_seed").getBoolValue(), // boolean indicating if the generator has a seed
              state = msgArgs.get("state").getIntValue(),       // rng state
              rname = st.nextName();

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? size %i method %s state %i".format(name, size, method, state));

        st.checkTable(name);

        var generatorEntry: borrowed GeneratorSymEntry(real) = toGeneratorSymEntry(st.lookup(name), real);
        ref rng = generatorEntry.generator;
        if state != 1 {
            // you have to skip to one before where you want to be
            rng.skipTo(state-1);
        }

        select method {
            when "ZIG" {
                var exponentialArr = makeDistArray(size, real);
                uniformStreamPerElem(exponentialArr, rng, GenerationFunction.ExponentialGenerator, hasSeed);
                st.addEntry(rname, createSymEntry(exponentialArr));
            }
            when "INV" {
                var u1 = makeDistArray(size, real);
                rng.fill(u1);

                // calculate the exponential by doing the inverse of the cdf
                var exponentialArr = -log1p(-u1);
                st.addEntry(rname, createSymEntry(exponentialArr));
            }
            otherwise {
                var errorMsg = "Only ZIG and INV are supported for method. Recieved: %s".format(method);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }

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
                         .format(genName, permName, segsName, weightsName, numSamplesName, replace, hasWeights, state, rname));

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
                         .format(gName, aName, wName, numSamples, replace, hasWeights, isDom, dtypeStr, popSize, state, rname));

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
                const errorMsg = "Unhandled data type %s".format(dtypeStr);
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
                                "name: %? size %i dtype: %? state %i isDomPerm %?".format(name, size, dtypeStr, state, isDomPerm));

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
                var errorMsg = "Unhandled data type %s".format(dtypeStr);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }
        var repMsg = "created " + st.attrib(rname);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    inline proc poissonGenerator(lam: real, ref rs) {
        // the algorithm from knuth found here:
        // https://en.wikipedia.org/wiki/Poisson_distribution#Random_variate_generation
        // generates values drawn from poisson distribution using a stream of uniformly distributed random numbers
        var L = exp(-lam);
        var k = 0;
        var p = 1.0;

        do {
            k += 1;
            p = p * rs.next(0, 1);
        } while p > L;
        return k - 1;
    }

    proc poissonGeneratorMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName(),
              name = msgArgs.getValueOf("name"),                                // generator name
              isSingleLam = msgArgs.get("is_single_lambda").getBoolValue(),     // boolean indicating if lambda is a single value or array
              lamStr = msgArgs.getValueOf("lam"),                               // lambda for poisson distribution
              size = msgArgs.get("size").getIntValue(),                         // number of values to be generated
              hasSeed = msgArgs.get("has_seed").getBoolValue(),                 // boolean indicating if the generator has a seed
              state = msgArgs.get("state").getIntValue(),                       // rng state
              rname = st.nextName();


        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "name: %? size %i hasSeed %? isSingleLam %? lamStr %? state %i".format(name, size, hasSeed, isSingleLam, lamStr, state));

        st.checkTable(name);

        var generatorEntry: borrowed GeneratorSymEntry(real) = toGeneratorSymEntry(st.lookup(name), real);
        ref rng = generatorEntry.generator;
        if state != 1 {
            // you have to skip to one before where you want to be
            rng.skipTo(state-1);
        }
        var poissonArr = makeDistArray(size, int);
        const lam = new scalarOrArray(lamStr, !isSingleLam, st);

        uniformStreamPerElem(poissonArr, rng, GenerationFunction.PoissonGenerator, hasSeed, lam);
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
                                "name: %? size %i dtype: %? state %i".format(name, size, dtypeStr, state));

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
                var errorMsg = "Unhandled data type %s".format(dtypeStr);
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
    registerFunction("standardExponential", standardExponentialMsg, getModuleName());
    registerFunction("segmentedSample", segmentedSampleMsg, getModuleName());
    registerFunction("choice", choiceMsg, getModuleName());
    registerFunction("permutation", permutationMsg, getModuleName());
    registerFunction("poissonGenerator", poissonGeneratorMsg, getModuleName());
    registerFunction("shuffle", shuffleMsg, getModuleName());
}
