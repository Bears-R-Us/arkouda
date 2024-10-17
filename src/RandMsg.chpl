module RandMsg
{
    use ServerConfig;

    use Time;
    use Math;
    use Random;
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

    @arkouda.instantiateAndRegister
    proc randomNormal(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd: int): MsgTuple throws {
        const shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              seed = msgArgs["seed"].toScalar(string);

        var entry = createSymEntry((...shape), real);
        fillNormal(entry.a, seed);
        return st.insert(entry);
    }

    /*
     * Creates a generator server-side and returns the SymTab name used to
     * retrieve the generator from the SymTab.
     */
    @arkouda.instantiateAndRegister
    proc createGenerator(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws
        where array_dtype != BigInteger.bigint
    {
        const hasSeed = msgArgs["has_seed"].toScalar(bool),
              seed = if hasSeed then msgArgs["seed"].toScalar(int) else -1,
              state = msgArgs["state"].toScalar(int);

        if hasSeed {
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "dtype: %? seed: %i state: %i".format(type2str(array_dtype),seed,state));
        }
        else {
            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "dtype: %? state: %i".format(type2str(array_dtype),state));
        }

        var generator = if hasSeed then new randomStream(array_dtype, seed) else new randomStream(array_dtype);
        if state != 1 then generator.skipTo(state-1);
        return st.insert(new shared GeneratorSymEntry(generator, state));
    }

    @arkouda.instantiateAndRegister
    proc uniformGenerator(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
        where array_dtype != BigInteger.bigint
    {
        const name = msgArgs["name"],
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              state = msgArgs["state"].toScalar(int);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? dtype: %? state %i".format(name, shape, type2str(array_dtype), state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(array_dtype); 
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        var uniformEntry = createSymEntry((...shape), array_dtype);
        if array_dtype == bool {
            // chpl doesn't support bounded random with boolean type
            rng.fill(uniformEntry.a);
        }
        else {
            const low = msgArgs["low"].toScalar(array_dtype),
                  high = msgArgs["high"].toScalar(array_dtype);
            rng.fill(uniformEntry.a, low, high);
        }
        return st.insert(uniformEntry);
    }


    /*
        Use the ziggurat method (https://en.wikipedia.org/wiki/Ziggurat_algorithm#Theory_of_operation)
        to generate normally distributed numbers using n (num of rectangles) = 256.
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
    inline proc standardNormZig(ref realRng, ref uintRng): real {
        // modified from numpy to use while loop instead of recusrsion so we can inline the proc
        var count = 0;
        // to guarantee completion this should be while true, but limiting to 100 tries will make the chance
        // of failure near zero while avoiding the possibility of an infinite loop
        // the odds of failing 100 times in a row is (1 - .993)**100 = 3.2345e-216
        while count <= 100 {
            var ri = uintRng.next();

            // AND with 0xFF (255) to get our index into our 256 long const arrays
            var idx = ri & 0xFF;
            ri >>= 8;

            var rabs = (ri >> 1) & 0x000fffffffffffff;
            var x = rabs * wi_double[idx];

            // ziggurat only works on monotonically decreasing, which normal isn't but since stardard normal 
            // is symmetric about x=0, we can do it on one half and then choose positive or negative
            var sign = ri & 0x1;
            if (sign & 0x1) {
                x = -x;
            }
            if (rabs < ki_double[idx]) {
                // the point fell in the core of one of our rectangular slices, so we're guaranteed
                // it falls under the pdf curve. We can return it as a sample from our distribution.
                // We will return here 99.3% of the time on the 1st try
                return x;
            }

            // The fall back algorithm for calculating if the sample point lies under the pdf.
            // Either in the tip of one of the rectangles or in the tail of the distribution.
            // See https://blogs.mathworks.com/cleve/2015/05/18/the-ziggurat-random-normal-generator/

            // candidate point did not fall in the core of any rectangular slices. Either it lies in the
            // first slice (which doesn't have a core), in the tail of the distribution, or in the tip of one of slices.
            if (idx == 0) {
                // first rectangular slice

                // this was written as an infinite inner loop in numpy, but we want to avoid that possibility
                //  since we don't have a probability of success, we'll just loop at most 10**6 times
                var innerCount = 0;
                while innerCount <= 10**6 {
                    const xx = -ziggurat_nor_inv_r * log1p(-realRng.next());
                    const yy = -log1p(-realRng.next());
                    if (yy + yy > xx * xx) {
                        return if ((rabs >> 8) & 0x1) then -(ziggurat_nor_r + xx) else ziggurat_nor_r + xx;
                    }
                    innerCount += 1;
                }
            } else {
                if (((fi_double[idx - 1] - fi_double[idx]) * realRng.next() + fi_double[idx]) < exp(-0.5 * x * x)) {
                    // tip calculation
                    return x;
                }
            }
            // reject sample and retry
            count += 1;
        }
        return -1.0;  // we failed 100 times in a row which should practically never happen
    }

    inline proc standardNormBoxMuller(shape: ?t, ref rng) throws {
        // uses Box–Muller transform
        // generates values drawn from the standard normal distribution using
        // 2 uniformly distributed random numbers

        var u1 = if t == int then makeDistArray(shape, real) else makeDistArray((...shape), real);
        var u2 = if t == int then makeDistArray(shape, real) else makeDistArray((...shape), real);

        rng.fill(u1);
        rng.fill(u2);

        return sqrt(-2*log(u1))*cos(2*pi*u2);
    }

    @arkouda.instantiateAndRegister
    proc standardNormalGenerator(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd): MsgTuple throws
        do return standardNormalGeneratorHelp(cmd, msgArgs, st, array_nd);

    proc standardNormalGeneratorHelp(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd): MsgTuple throws
        where array_nd == 1
    {
        const name = msgArgs["name"],                           // generator name
              shape = msgArgs["shape"].toScalar(int),           // population size
              method = msgArgs["method"].toScalar(string),      // method to use to generate exponential samples
              hasSeed = msgArgs["has_seed"].toScalar(bool),     // boolean indicating if the generator has a seed
              state = msgArgs["state"].toScalar(int);           // rng state

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %i state %i".format(name, shape, state));

        var generatorEntry= st[name]: borrowed GeneratorSymEntry(real);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        select method {
            when "ZIG" {
                // TODO before this can be adapted to handle multidim, we need to figure out how to modify uniformStreamPerElem
                var standNormArr = makeDistArray(shape, real);
                uniformStreamPerElem(standNormArr, rng, GenerationFunction.NormalGenerator, hasSeed);
                return st.insert(createSymEntry(standNormArr));
            }
            when "BOX" {
                const standNormArr = standardNormBoxMuller(shape, rng);
                return st.insert(createSymEntry(standNormArr));
            }
            otherwise {
                var errorMsg = "Only ZIG and BOX are supported for method. Recieved: %s".format(method);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return MsgTuple.error(errorMsg);
            }
        }
    }


    proc standardNormalGeneratorHelp(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd): MsgTuple throws
        where array_nd > 1
    {
        const name = msgArgs["name"],                           // generator name
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),           // population size
              method = msgArgs["method"].toScalar(string),      // method to use to generate exponential samples
              hasSeed = msgArgs["has_seed"].toScalar(bool),     // boolean indicating if the generator has a seed
              state = msgArgs["state"].toScalar(int);           // rng state

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? state %i".format(name, shape, state));

        var generatorEntry= st[name]: borrowed GeneratorSymEntry(real);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        select method {
            when "BOX" {
                const standNomr = standardNormBoxMuller(shape, rng);
                return st.insert(createSymEntry(standNomr));
            }
            otherwise {
                var errorMsg = "Only BOX is supported for method on multidimensional arrays. Recieved: %s".format(method);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return MsgTuple.error(errorMsg);
            }
        }
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
            count += 1;
        }
        return -1.0;  // we failed 100 times in a row which should practically never happen
    }

    inline proc standardExponentialInvCDF(shape: ?t, ref rng) throws {
        var u1 = if t == int then makeDistArray(shape, real) else makeDistArray((...shape), real);
        rng.fill(u1);

        // calculate the exponential by doing the inverse of the cdf
        return -log1p(-u1);
    }

    @arkouda.instantiateAndRegister
    proc standardExponential(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd): MsgTuple throws
        do return standardExponentialHelp(cmd, msgArgs, st, array_nd);

    proc standardExponentialHelp(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd): MsgTuple throws
        where array_nd == 1
    {
        const name = msgArgs["name"],                                   // generator name
              size = msgArgs["size"].toScalar(int),                     // population size
              method = msgArgs["method"].toScalar(string),              // method to use to generate exponential samples
              hasSeed = msgArgs["has_seed"].toScalar(bool),             // boolean indicating if the generator has a seed
              state = msgArgs["state"].toScalar(int);                   // rng state

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? size %i method %s state %i".format(name, size, method, state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(real);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        select method {
            when "ZIG" {
                // TODO before this can be adapted to handle multidim, we need to figure out how to modify uniformStreamPerElem
                var exponentialArr = makeDistArray(size, real);
                uniformStreamPerElem(exponentialArr, rng, GenerationFunction.ExponentialGenerator, hasSeed);
                return st.insert(createSymEntry(exponentialArr));
            }
            when "INV" {
                const exponentialArr = standardExponentialInvCDF(size, rng);
                return st.insert(createSymEntry(exponentialArr));
            }
            otherwise {
                var errorMsg = "Only ZIG and INV are supported for method. Recieved: %s".format(method);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return MsgTuple.error(errorMsg);
            }
        }
    }

    proc standardExponentialHelp(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd): MsgTuple throws
        where array_nd > 1
    {
        const name = msgArgs["name"],                                   // generator name
              shape = msgArgs["size"].toScalarTuple(int, array_nd),     // population size
              method = msgArgs["method"].toScalar(string),              // method to use to generate exponential samples
              hasSeed = msgArgs["has_seed"].toScalar(bool),             // boolean indicating if the generator has a seed
              state = msgArgs["state"].toScalar(int);                   // rng state

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? method %s state %i".format(name, shape, method, state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(real);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        select method {
            when "INV" {
                const exponentialArr = standardExponentialInvCDF(shape, rng);
                return st.insert(createSymEntry(exponentialArr));
            }
            otherwise {
                var errorMsg = "Only INV is supported for method on multidimensional arrays. Recieved: %s".format(method);
                randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return MsgTuple.error(errorMsg);
            }
        }
    }

    proc segmentedSampleMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const genName = msgArgs["genName"],                                 // generator name
              permName = msgArgs["perm"],                                   // values array name
              segsName = msgArgs["segs"],                                   // segments array name
              segLensName = msgArgs["segLens"],                             // segment lengths array name
              weightsName = msgArgs["weights"],                             // permuted weights array name
              numSamplesName = msgArgs["numSamples"],                       // number of samples per segment array name
              replace = msgArgs["replace"].toScalar(bool),                  // sample with replacement
              hasWeights = msgArgs["hasWeights"].toScalar(bool),            // flag indicating whether weighted sample
              hasSeed = msgArgs["hasSeed"].toScalar(bool),                   // flag indicating if generator is seeded
              seed = if hasSeed then msgArgs["seed"].toScalar(int) else -1, // value of seed if present
              state = msgArgs["state"].toScalar(int);                       // rng state

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "genName: %? permName %? segsName: %? weightsName: %? numSamplesName %? replace %i hasWeights %i state %i"
                         .format(genName, permName, segsName, weightsName, numSamplesName, replace, hasWeights, state));

        // TODO replace this with instantiateAndRegister array_nd once this is adapted to handle multi-dim
        param array_nd = 1;
        const permutation = (st[permName]: SymEntry(int, array_nd)).a,
              segments = (st[segsName]: SymEntry(int, array_nd)).a,
              segLens = (st[segLensName]: SymEntry(int, array_nd)).a,
              numSamples = (st[numSamplesName]: SymEntry(int, array_nd)).a;

        const sampleOffset = (+ scan numSamples) - numSamples;
        var sampledPerm: [makeDistDom(+ reduce numSamples)] int;

        if hasWeights {
            const weights = (st[weightsName]: SymEntry(real, array_nd)).a;

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
        return st.insert(createSymEntry(sampledPerm));
    }

    @arkouda.instantiateAndRegister
    proc choice(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws
        where array_dtype != BigInteger.bigint
    {
        const gName = msgArgs["gName"],                             // generator name
              aName = msgArgs["aName"],                             // values array name
              wName = msgArgs["wName"],                             // weights array name
              numSamples = msgArgs["numSamples"].toScalar(int),     // number of samples
              replace = msgArgs["replace"].toScalar(bool),          // sample with replacement
              hasWeights = msgArgs["hasWeights"].toScalar(bool),    // flag indicating whether weighted sample
              isDom = msgArgs["isDom"].toScalar(bool),              // flag indicating whether return is domain or array
              popSize  = msgArgs["popSize"].toScalar(int),          // population size
              state = msgArgs["state"].toScalar(int);               // rng state

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "gname: %? aname %? wname: %? numSamples %i replace %i hasWeights %i isDom %i dtype %? popSize %? state %i"
                         .format(gName, aName, wName, numSamples, replace, hasWeights, isDom, type2str(array_dtype), popSize, state));


        // TODO this doesn't support multi-dim because it calls out to sampleDomWeighted, which results in the following error
        // RandArray.chpl:138: In function 'sampleDomWeighted':
        // RandArray.chpl:154: error: isSorted() requires 1-D array
        param array_nd: int = 1;

        // do we still need sampleDomWeighted? or is weight sample in our lowest supported version
        // there's probably a way to flatten the weights but it's not immediately obvious how to go about that,
        // so I'm leaving that for future work
        proc weightedIdxHelper() throws {
            var generatorEntry = st[gName]: borrowed GeneratorSymEntry(real);
            ref rng = generatorEntry.generator;
            if state != 1 then rng.skipTo(state-1);

            const weights = (st[wName]: borrowed SymEntry(real, array_nd)).a;
            return sampleDomWeighted(rng, numSamples, weights, replace);
        }

        proc idxHelper() throws {
            var generatorEntry = st[gName]: borrowed GeneratorSymEntry(int);
            ref rng = generatorEntry.generator;
            if state != 1 then rng.skipTo(state-1);

            const choiceDom = {0..<popSize};
            return rng.sample(choiceDom, numSamples, replace);
        }

        // I had to break these 2 helpers out into seprate functions since they have different types for generatorEntry
        // const choiceIdx = if hasWeights then weightedIdxHelper() else idxHelper();

        // TODO originally choiceIdx was declared before this if-else but chapel got mad about
        // it potentially going out of scope? figure out what that's all about
        if isDom {
            const choiceIdx = if hasWeights then weightedIdxHelper() else idxHelper();
            return st.insert(createSymEntry(choiceIdx));
        }
        else {
            const choiceIdx = if hasWeights then weightedIdxHelper() else idxHelper();
            var choiceArr: [makeDistDom(numSamples)] array_dtype;
            const arrEntry = st[aName]: borrowed SymEntry(array_dtype, array_nd);
            const myArr = arrEntry.a;

            forall (c, idx) in zip(choiceArr, choiceIdx) with (var agg = newSrcAggregator(array_dtype)) {
                agg.copy(c, myArr[idx]);
            }

            return st.insert(createSymEntry(choiceArr));
        }
    }

    inline proc logisticGenerator(mu: real, scale: real, ref rs) {
        var U = rs.next(0, 1);

        while U <= 0.0 {
            /* Reject U == 0.0 and call again to get next value */
            U = rs.next(0, 1);
        }
        return mu + scale * log(U / (1.0 - U));
    }

    proc logisticGeneratorMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const name = msgArgs["name"],
              isSingleMu = msgArgs["is_single_mu"].toScalar(bool),
              muStr = msgArgs["mu"].toScalar(string),
              isSingleScale = msgArgs["is_single_scale"].toScalar(bool),
              scaleStr = msgArgs["scale"].toScalar(string),
              size = msgArgs["size"].toScalar(int),
              hasSeed = msgArgs["has_seed"].toScalar(bool),
              state = msgArgs["state"].toScalar(int);

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(real);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        var logisticArr = makeDistArray(size, real);
        const mu = new scalarOrArray(muStr, !isSingleMu, st),
              scale = new scalarOrArray(scaleStr, !isSingleScale, st);

        uniformStreamPerElem(logisticArr, rng, GenerationFunction.LogisticGenerator, hasSeed, mu=mu, scale=scale);
        return st.insert(createSymEntry(logisticArr));
    }

    @arkouda.instantiateAndRegister
    proc permutation(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        const name = msgArgs["name"],
              xName = msgArgs["x"],
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              size = msgArgs["size"].toScalar(int),
              state = msgArgs["state"].toScalar(int),
              isDomPerm = msgArgs["isDomPerm"].toScalar(bool);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? dtype: %? state %i isDomPerm %?".format(name, shape, type2str(array_dtype), state, isDomPerm));

        // we need the int generator in order for permute(domain) to work correctly
        var intGeneratorEntry = st[name]: borrowed GeneratorSymEntry(int);
        ref intRng = intGeneratorEntry.generator;

        if state != 1 then intRng.skipTo(state-1);
        const permutedDom = makeDistDom(size);

        // same error about scoping:
        // error: Reference to scoped variable idx reachable after lifetime ends
        // consider scope of permutedIdx
        if isDomPerm {
            const permutedIdx = intRng.permute(permutedDom);

            const permutedEntry = createSymEntry(permutedIdx);
            return st.insert(permutedEntry);
        }
        else {
            const permutedIdx = intRng.permute(permutedDom);

            // permute requires that the stream's eltType is coercible to the array/domain's idxType,
            // so we use permute(dom) and use that to gather the permuted vals
            const arrEntry = st[xName]: SymEntry(array_dtype, array_nd);
            ref myArr = arrEntry.a;

            var permutedArr: [myArr.domain] array_dtype;


            forall (idx, arrIdx) in zip(permutedIdx, 0..) with (var agg = newSrcAggregator(array_dtype)) {
                // slightly concerned about remote-to-remote aggregation
                agg.copy(permutedArr[permutedArr.domain.orderToIndex(arrIdx)], myArr[myArr.domain.orderToIndex(idx)]);
            }

            const permutedEntry = createSymEntry(permutedArr);
            return st.insert(permutedEntry);
        }
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
        const name = msgArgs["name"],                                       // generator name
              isSingleLam = msgArgs["is_single_lambda"].toScalar(bool),     // boolean indicating if lambda is a single value or array
              lamStr = msgArgs["lam"].toScalar(string),                     // lambda for poisson distribution
              size = msgArgs["size"].toScalar(int),                         // number of values to be generated
              hasSeed = msgArgs["has_seed"].toScalar(bool),                 // boolean indicating if the generator has a seed
              state = msgArgs["state"].toScalar(int);                       // rng state


        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "name: %? size %i hasSeed %? isSingleLam %? lamStr %? state %i".format(name, size, hasSeed, isSingleLam, lamStr, state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(real);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        var poissonArr = makeDistArray(size, int);
        const lam = new scalarOrArray(lamStr, !isSingleLam, st);

        uniformStreamPerElem(poissonArr, rng, GenerationFunction.PoissonGenerator, hasSeed, lam=lam);
        return st.insert(createSymEntry(poissonArr));
    }

    @arkouda.instantiateAndRegister
    proc shuffle(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
        do return shuffleHelp(cmd, msgArgs, st, array_dtype, array_nd);

    proc shuffleHelp(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws 
        where array_nd == 1
    {
        const name = msgArgs["name"],
              xName = msgArgs["x"].toScalar(string),
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              state = msgArgs["state"].toScalar(int);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? dtype: %? state %i".format(name, shape, type2str(array_dtype), state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(int);
        ref rng = generatorEntry.generator;

        if state != 1 then rng.skipTo(state-1);

        const arrEntry = st[xName]: SymEntry(array_dtype, array_nd);
        ref myArr = arrEntry.a;
        rng.shuffle(myArr);
        return MsgTuple.success();
    }

    proc shuffleHelp(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws 
        where array_nd != 1
    {
        const name = msgArgs["name"],
              xName = msgArgs["x"].toScalar(string),
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              state = msgArgs["state"].toScalar(int);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? dtype: %? state %i".format(name, shape, type2str(array_dtype), state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(int);
        ref rng = generatorEntry.generator;

        if state != 1 then rng.skipTo(state-1);

        const arrEntry = st[xName]: SymEntry(array_dtype, array_nd);
        ref myArr = arrEntry.a;

        // stolen; will be in chpl 2.2 so it can be deleted then
        // Fisher-Yates shuffle
        const d = myArr.domain;
        for i in 0..#d.size by -1 {
            const ki = rng.next(0:int, i:int),
                k = d.orderToIndex(ki),
                j = d.orderToIndex(i);

            myArr[k] <=> myArr[j];
        }

        return MsgTuple.success();
    }

    use CommandMap;
    registerFunction("logisticGenerator", logisticGeneratorMsg, getModuleName());
    registerFunction("segmentedSample", segmentedSampleMsg, getModuleName());
    registerFunction("poissonGenerator", poissonGeneratorMsg, getModuleName());
}
