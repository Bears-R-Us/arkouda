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
    use PrivateDist;
    use RandArray;
    use RandUtil;
    use ArkoudaSortCompat;
    use CommAggregation;

    use AryUtil ; // to get indexToOrder
    use Repartition;
    use ZigguratConstants;
    use BitOps;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    import BigInteger;

    use List;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const randLogger = new Logger(logLevel, logChannel);

    /*
        The delGeneratorMsg allows the deletion of random number generator objects.
        This was implemented so that in the event that the global rng is reseeded
        multiple times, we won't build up a never-ending set of generatorEntrys and
        RandomStreams.
     */

    @chplcheck.ignore("UnusedFormal")
    proc delGeneratorMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const name = msgArgs.getValueOf("name");
        var generatorEntry = st(name);
        var repMsg = "deleted " +  generatorEntry.name;
        st.deleteEntry(generatorEntry.name);
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return MsgTuple.success();
    }

    /*
    parse, execute, and respond to randint message
    uniform int in half-open interval [min,max)

    :arg reqMsg: message to process (contains cmd,aMin,aMax,len,dtype)
    */

    @arkouda.instantiateAndRegister
    proc randint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
                type array_dtype, param array_nd: int): MsgTuple throws
        where array_dtype != BigInteger.bigint
    {
        const shape = msgArgs["shape"].toScalarTuple(int, array_nd),
                seed  = msgArgs["seed"].toScalar(int);

        type T = array_dtype;
        param isInt  = isIntegralType(T);
        param isBool = isBoolType(T);

        var len = 1;
        for s in shape do len *= s;
        overMemLimit(len);

        var t = new stopwatch();
        t.start();

        var e = createSymEntry((...shape), array_dtype);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "alloc time = %? sec".format(t.elapsed()));
        t.restart();

        if isBool {
            // ----- BOOL: no bounded fill; decide via integer bounds -----
            const li = msgArgs["low"].toScalar(int);
            const hi = msgArgs["high"].toScalar(int);

            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "cmd: %s len: %i dtype: %s aMin: %? aMax: %?".format(
                            cmd, len, type2str(array_dtype), li, hi));

            // [0,1) -> all False, [1,2) -> all True, [0,2) -> random bools
            if li == 0 && hi == 1 {
                e.a = false;
            } else if li == 1 && hi == 2 {
                e.a = true;
            } else {
            if seed == -1 then
                fillRandom(e.a);         // unbounded boolean fill
            else
                fillRandom(e.a, seed);   // seeded unbounded boolean fill
            }

        } else {
            // ----- NON-BOOL: bounded fill; Chapel is inclusive on the upper end -----
            const lowT     : T = msgArgs["low"].toScalar(T);
            const highRawT : T = msgArgs["high"].toScalar(T);
            const highT    : T = if isInt then (highRawT - (1:T)) else highRawT;

            randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "cmd: %s len: %i dtype: %s aMin: %? aMax: %?".format(
                            cmd, len, type2str(array_dtype), lowT, highT));

            if seed == -1 then
                fillRandom(e.a, lowT, highT);
            else
                fillRandom(e.a, lowT, highT, seed);
        }

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "compute time = %i sec".format(t.elapsed()));

        return st.insert(e);
    }

    @arkouda.instantiateAndRegister
    @chplcheck.ignore("UnusedFormal")
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
    @chplcheck.ignore("UnusedFormal")
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


    // frivolousMsg is a demonstration of random number generation that's independent
    // of the number of locales.

    @arkouda.instantiateAndRegister
    @chplcheck.ignore("UnusedFormal")
    proc frivolousMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd: int): MsgTuple throws
    {
        const name = msgArgs["name"],
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              state = msgArgs["state"].toScalar(int),
              seed  = msgArgs["seed"].toScalar(int);
    
        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? state %i seed %i".format(name, shape, state, seed));

        var frivolousEntry = createSymEntry((...shape), int);
        var ord = new orderer(shape);
        coforall loc in Locales do on loc {
            var relative_start = if state != 1 then state else 1; 
            forall entry in frivolousEntry.a.localSubdomain() { // I'd rather the rng was in
                var rng = new randomStream(int, seed);          // a with in the forall, but
                var spot = relative_start + ord.indexToOrder(entry) - 1; // it wasn't working
                rng.skipTo(spot);
                frivolousEntry.a[entry] = rng.next();
            }
        }
        return st.insert(frivolousEntry);
    }



    @arkouda.instantiateAndRegister
    @chplcheck.ignore("UnusedFormal")
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
            if sign & 0x1 {
                x = -x;
            }
            if rabs < ki_double[idx] {
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
            if idx == 0 {
                // first rectangular slice

                // this was written as an infinite inner loop in numpy, but we want to avoid that possibility
                //  since we don't have a probability of success, we'll just loop at most 10**6 times
                var innerCount = 0;
                while innerCount <= 10**6 {
                    const xx = -ziggurat_nor_inv_r * log1p(-realRng.next());
                    const yy = -log1p(-realRng.next());
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 then -(ziggurat_nor_r + xx) else ziggurat_nor_r + xx;
                    }
                    innerCount += 1;
                }
            } else {
                if ((fi_double[idx - 1] - fi_double[idx]) * realRng.next() + fi_double[idx]) < exp(-0.5 * x * x) {
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

    @chplcheck.ignore("UnusedFormal")
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


    @chplcheck.ignore("UnusedFormal")
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

    @chplcheck.ignore("UnusedFormal")
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

    @chplcheck.ignore("UnusedFormal")
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

    inline proc gammaGenerator(kArg: real, ref rs) throws {
        if kArg == 1.0 {
            return standardExponentialInvCDF(1, rs)[0];
        }
        else if kArg == 0.0 {
            return 0.0;
        }
        else if kArg < 1.0 {
            var count = 0;
            while count < 10000 {
                var U = rs.next(0, 1);
                var V = standardExponentialInvCDF(1, rs)[0];
                if U <= (1.0 - kArg) {
                    var X = U ** (1.0 / kArg);
                    if X <= V {
                        return X;
                    }
                }
                else {
                    var Y = -log((1.0 - U) / kArg);
                    var X = (1.0 - kArg + kArg * Y) ** (1.0 / kArg);
                    if X <= (V + Y) {
                        return X;
                    }
                }
                count+= 1;
            }
            return -1.0;  // we failed 10000 times in a row which should practically never happen
        }
        else {
            var b = kArg - 1/3.0;
            var c = 1/sqrt(9.0 * b);
            var count = 0;
            while count < 10000 {
                var V = -1.0;
                var X = 0.0;
                while V <= 0 {
                    X = standardNormBoxMuller(1, rs)[0];
                    V = 1.0 + c * X;
                }
                V = V * V * V;
                var U = rs.next(0, 1);
                if U < 1.0 - 0.0331 * (X * X) * (X * X) {
                    return b * V;
                }
                if log(U) < 0.5 * X * X + b * (1.0 - V + log(V)) {
                    return b * V;
                }
            }
            return -1.0;  // we failed 10000 times in a row which should practically never happen
        }
    }

    @arkouda.instantiateAndRegister
    @chplcheck.ignore("UnusedFormal")
    proc standardGamma(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param array_nd): MsgTuple throws {
        const name = msgArgs["name"],
              shape = msgArgs["size"].toScalarTuple(int, array_nd),
              isSingleK = msgArgs["is_single_k"].toScalar(bool),
              kStr = msgArgs["k_arg"].toScalar(string),
              hasSeed = msgArgs["has_seed"].toScalar(bool),
              state = msgArgs["state"].toScalar(int);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? k %s state %i".format(name, shape, kStr, state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(real);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);
        //state used to be shape
        var gammaArr = makeDistArray((...shape), real);
        const kArg = new scalarOrArray(kStr, !isSingleK, st);
        uniformStreamPerElem(gammaArr, rng, GenerationFunction.GammaGenerator, hasSeed, kArg=kArg);
        return st.insert(createSymEntry(gammaArr));
    }

    @chplcheck.ignore("UnusedFormal")
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
    @chplcheck.ignore("UnusedFormal")
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

    @chplcheck.ignore("UnusedFormal")
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
    @chplcheck.ignore("UnusedFormal")
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

    @chplcheck.ignore("UnusedFormal")
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
    proc shuffle(
        cmd: string,
        msgArgs: borrowed MessageArgs,
        st: borrowed SymTab,
        type array_dtype,
        param array_nd: int
    ): MsgTuple throws {
        const method = msgArgs["method"].toScalar(string);
        if method == "mergeshuffle" {
            return mergeShuffleHelp(cmd, msgArgs, st, array_dtype, array_nd);
        } else if method == "fisheryates" {
            return shuffleHelp(cmd, msgArgs, st, array_dtype, array_nd);
        } else if method == "feistel" {
            return feistelShuffleHelp(cmd, msgArgs, st, array_dtype, array_nd);
        } else {
            const errorMsg = "Error: Invalid method for shuffle.  " +
                            "Allowed values: fisheryates, mergeshuffle, feistel";
            return MsgTuple.error(errorMsg);
        }
    }

    @chplcheck.ignore("UnusedFormal")
    proc shuffleHelp(
        cmd: string, 
        msgArgs: borrowed MessageArgs, 
        st: borrowed SymTab, 
        type array_dtype, 
        param array_nd: int
        ): MsgTuple throws 
        where array_nd == 1
    {
        const name = msgArgs["name"],
              xName = msgArgs["x"].toScalar(string),
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              state = msgArgs["state"].toScalar(int);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? dtype: %? state %i".format(
                                    name, shape, type2str(array_dtype), state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(int);
        ref rng = generatorEntry.generator;

        if state != 1 then rng.skipTo(state-1);

        const arrEntry = st[xName]: SymEntry(array_dtype, array_nd);
        ref myArr = arrEntry.a;
        rng.shuffle(myArr);
        return MsgTuple.success();
    }

    @chplcheck.ignore("UnusedFormal")
    proc shuffleHelp(
        cmd: string, 
        msgArgs: borrowed MessageArgs, 
        st: borrowed SymTab, 
        type array_dtype, 
        param array_nd: int
        ): MsgTuple throws 
        where array_nd != 1
    {
        const name = msgArgs["name"],
              xName = msgArgs["x"].toScalar(string),
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              state = msgArgs["state"].toScalar(int);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? dtype: %? state %i".format(
                                    name, shape, type2str(array_dtype), state));

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

    @chplcheck.ignore("UnusedFormal")
    proc mergeShuffleHelp(
        cmd: string, 
        msgArgs: borrowed MessageArgs, 
        st: borrowed SymTab, 
        type array_dtype, 
        param array_nd: int
        ): MsgTuple throws 
        where array_nd == 1
    {
        const name = msgArgs["name"],
              xName = msgArgs["x"].toScalar(string),
              shape = msgArgs["shape"].toScalarTuple(int, array_nd),
              state = msgArgs["state"].toScalar(int);

        randLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "name: %? shape %? dtype: %? state %i".format(
                                    name, shape, type2str(array_dtype), state));

        var generatorEntry = st[name]: borrowed GeneratorSymEntry(int);
        ref rng = generatorEntry.generator;

        if state != 1 then rng.skipTo(state-1);
        const generatorSeed = (rng.next() * 2**62):int;

        const arrEntry = st[xName]: SymEntry(array_dtype, array_nd);
        ref myArr = arrEntry.a;
        mergeShuffle(myArr, generatorSeed);
        return MsgTuple.success();
    }

    @chplcheck.ignore("UnusedFormal")
    proc mergeShuffleHelp(
        cmd: string, 
        msgArgs: borrowed MessageArgs, 
        st: borrowed SymTab, 
        type array_dtype, 
        param array_nd: int
        ): MsgTuple throws 
        where array_nd != 1
    {
        return new MsgTuple("mergeShuffle does not support " +
        "arrays of dimension > 1.",MsgType.ERROR);
    }

    /*
        Perform an in-place distributed merge shuffle on the array `x`.

        This procedure produces a globally shuffled permutation of `x` across all locales.
        It uses a two-phase hierarchical strategy:
        
        1. Each locale independently shuffles its local data using a recursive
        Fisher-Yates-based algorithm (`shuffleLocales`).
        
        2. Then, across multiple rounds, pairs of shuffled locale blocks are merged
        probabilistically using a randomized merge process (`merge`), simulating
        the structure of a parallel merge sort. Each merge pass doubles the size
        of shuffled blocks until the entire dataset is randomized.

        All randomness is seeded and deterministic to ensure reproducibility across runs.

        Parameters
        ----------
        x : [] ?t, `ref`
            The distributed array to shuffle in place. The element type may be any type
            that supports swapping (`<=>`) and indexing.
        
        generatorSeed : int
            A global seed used to generate deterministic random streams for the shuffle.
            Random streams for real-valued and integer-valued randomness are derived
            deterministically from this seed.

        Raises
        ------
        May raise errors internally from subroutines such as `shuffleLocales` or `merge`
        if domain bounds or RNG behavior are invalid.

        Notes
        -----
        - This shuffle is communication-efficient: locale-local shuffling is done first,
        and only necessary data is exchanged via coordinated merges.
        - The shuffle is reproducible: running it multiple times with the same input
        and `generatorSeed` will yield the same result.
        - The array `x` is modified in place; no copy is created.

        See Also
        --------
        shuffleLocales : Locally shuffles array data without cross-locale communication.
        merge : Merges two chunks probabilistically using a fixed threshold ratio.
    */
    proc mergeShuffle(ref x: [], generatorSeed: int) throws {
        const numRounds = log2(numLocales) + 1;
        const domainLows = getDomainLows(x);
        const domainHighs = getDomainHighs(x);
        var rngs = getRandomStreams(generatorSeed, real);
        var intRngs = getRandomStreams(generatorSeed + numLocales, int);

        shuffleLocales(x, intRngs);

        for m in 0..#numRounds {
            const maxLocalesPerPrevChunk = 2**m;
            const numNewChunks = (numLocales - 1) / 
                            (2 * maxLocalesPerPrevChunk) + 1;

            forall i in 0..#numNewChunks with (ref x){
                const (startLocale, endLocale, startLocale2, endLocale2) = 
                    getMergePair(i, maxLocalesPerPrevChunk, numLocales);
                if  endLocale < startLocale2 {
                        const start = domainLows[startLocale];
                        const size1 = domainHighs[endLocale] - 
                                    domainLows[startLocale] + 1;
                        const size2 = domainHighs[endLocale2] - 
                                    domainLows[startLocale2] + 1;
                        if size1 <= 0 || size2 <= 0 then continue;

                        merge(x, start, size1, size2, rngs, intRngs);
                }
            }
        }
    }

    /*
        Compute a pair of adjacent locale ranges to merge during a shuffle round.

        Given an index `i` into the set of merge chunks, this function calculates two
        contiguous ranges of locale IDs to be merged in the current round of
        `mergeShuffle`. Each range contains at most `width` locales. The calculation
        ensures that locale indices do not exceed `numLocales - 1`.

        Parameters
        ----------
        i : int
            The merge chunk index within the current round.

        width : int
            The maximum number of locales in each range being merged.

        numLocales : int
            The total number of locales available in the system.

        Returns
        -------
        (int, int, int, int)
            A tuple representing:
            - start index of the first locale range (a1),
            - end index of the first locale range (a2),
            - start index of the second locale range (b1),
            - end index of the second locale range (b2).

        Notes
        -----
        This function is used to coordinate merging of adjacent locale segments in
        `mergeShuffle`. The ranges are always calculated safely to stay within bounds,
        even near the upper edge of the locale index space.
    */
    proc getMergePair(i, width, numLocales): (int, int, int, int) {
        const a1 = 2 * i * width;
        const a2 = min(a1 + width - 1, numLocales - 1);
        const b1 = min(a2 + 1, numLocales - 1);
        const b2 = min(b1 + width - 1, numLocales - 1);
        return (a1, a2, b1, b2);
    }


    /*  
        Shuffles each locale of the array independently.
        There should be no communication between locales for this step.
    */
    @chplcheck.ignore("UnusedFormal")
    proc shuffleLocales(
            ref x: [], 
            ref rngs: [] randomStream(int),
            const maxFisherYatesPower = 6,
            const minSize = 10
        ) throws {
        coforall loc in Locales with (ref x) do on loc{
            var randStreamInt = rngs[here.id];
            var seed = randStreamInt.next();

            const localLower = x.localSubdomain().low;
            const localUpper = x.localSubdomain().high;
            const size = localUpper - localLower + 1;

            const smallestChunkSize = max(size/(2**maxFisherYatesPower), 
                            min(minSize, size)); //  Hardcoded minimum size
            const numChunks = ceil(size:real / smallestChunkSize:real):int;

            // Ensure that the starting index of the last chunk does not exceed the local array bounds.
            assert(localLower + (numChunks -1) * smallestChunkSize <= localUpper);

            forall i in 0..#numChunks with (ref x, const seed){
                const taskSeed = seed + i;
                const low = localLower + i * smallestChunkSize;
                const high = min(localUpper, low + smallestChunkSize - 1);
                var rng = new randomStream(int, seed=taskSeed);
                fisherYatesOnChunk(x, low..high, high, rng);

            }

            seed += numChunks;

            const numRounds = log2(numChunks) + 1;

            for m in 0..#(numRounds) {
                const prevChunkSize = smallestChunkSize * 2**m;
                const newChunkSize = 2 * prevChunkSize;
                const numNewChunks = (size - 1)/newChunkSize + 1;

                forall i in 0..#(numNewChunks) with (
                                ref x, 
                                const localLower, 
                                const localUpper, 
                                const newChunkSize, 
                                const prevChunkSize, 
                                const seed){

                    const start = localLower + i * newChunkSize;
                    const size1 = min(localUpper - start + 1, prevChunkSize);
                    const size2 = min(localUpper - start - size1 + 1, 
                                prevChunkSize);   

                    const taskSeed = seed + i;

                    var rng = new randomStream(real, seed=taskSeed);
                    var rngInt = new randomStream(int, seed=taskSeed + 1);  

                    // figure out which slice of [start..endIdx] 
                    // lives on this locale
                    const sub = x.localSubdomain();
                    const localLo = max(start, sub.low);
                    const localHi = min(start + size1 + size2 - 1, sub.high);

                    mergeLocalChunk(x, 
                                localLo..localHi, 
                                start, 
                                size1, 
                                size2, 
                                rng, 
                                rngInt);

                }
                seed += numNewChunks;
            }
        }
    } 

    /*
        Shuffle each locale's portion of the array independently using a hierarchical strategy.

        This procedure performs a reproducible, in-place shuffle of the portion of `x`
        local to each Chapel locale. It avoids any inter-locale communication. The shuffle
        is accomplished in two phases:

        1. **Chunk-wise Fisher-Yates shuffle**: The local array is divided into small chunks,
        each of which is independently shuffled using a seeded Fisher-Yates algorithm.
        
        2. **Hierarchical merge shuffle**: Pairs of adjacent shuffled chunks are merged
        using a probabilistic in-place merge process. This merge phase is applied
        recursively, doubling the chunk size in each round until the full local segment
        is shuffled.

        Parameters
        ----------
        x : [] ?t, `ref`
            The distributed array to shuffle. Each locale will only access and modify
            its own local portion.

        rngs : [] randomStream(int), `ref`
            A distributed array of integer-based random streams, one per locale.
            These streams are used to seed local substreams for chunk shuffling and merging.

        maxFisherYatesPower : int, `param`, optional
            Controls the maximum number of subdivisions used in the initial chunking phase.
            The number of chunks is roughly `2 ** maxFisherYatesPower`. Default is 6.

        minSize : int, `param`, optional
            The minimum allowable chunk size during initial subdivision.
            Prevents excessive fragmentation. Default is 10.

        Raises
        ------
        May throw errors from `fisherYatesOnChunk` or `mergeLocalChunk` if local bounds
        are invalid or array access is out of bounds.

        Notes
        -----
        - This function ensures reproducible shuffling by deriving all seeds from
        the input RNG streams.
        - No communication occurs between locales; this is suitable for large
        distributed arrays when global shuffle is not immediately required.
        - The array `x` is modified in place.

        See Also
        --------
        fisherYatesOnChunk : Shuffles a single range using Fisher-Yates.
        mergeLocalChunk : Probabilistically merges two adjacent local chunks.
        mergeShuffle : Performs a global distributed shuffle using this routine.
    */
    proc shuffleRange(
            ref x: [], 
            swapChunk: range(int),
            const bound: int, 
            ref rngs: [] randomStream(int)
        ) throws {

        var localesByLow = [loc in Locales] (loc, x.localSubdomain(loc).low);
        sort(localesByLow, new byLowKey());

        for (loc, _) in localesByLow {  
            on loc {
                //  Find the intersection of swapChunk with local subdomain
                const myDom = x.localSubdomain();
                const localLower = max(myDom.low, swapChunk.first);
                const localUpper = min(myDom.high, swapChunk.last);

                if localLower <= localUpper {
                    fisherYatesOnChunk(x, localLower..localUpper, bound, rngs[here.id]);
                }
            }
        }
    }

    /*
        Perform a constrained partial Fisher-Yates shuffle on a slice of the array.

        This procedure shuffles the elements of `x` over the index range `swapChunk`
        using a variant of the Fisher-Yates algorithm. Each element `i` in `swapChunk`
        may be swapped with another randomly selected index `idx` such that
        `idx ∈ [i, bound]` if `bound ≥ i`, or `idx ∈ [bound, i]` if `bound < i`.

        The `bound` must lie outside the interior of `swapChunk` to avoid biased shuffling
        or out-of-bounds behavior.

        Parameters
        ----------
        x : [] ?t, `ref`
            The array to shuffle, where each element must be indexable and swappable
            with `<=>`. Only the indices in `swapChunk` may be modified.

        swapChunk : range(int)
            The contiguous range of indices within `x` to shuffle.

        bound : int
            The index that sets the upper or lower limit of the swap target range
            for each `i` in `swapChunk`. It must lie outside the interior of `swapChunk`.  
            If bound is swapChunk.last, then this is forwards Fisher-Yates, 
            and if bound is swapChunk.first, then this is reverse Fisher-Yates, 
            although values further outside the range are also allowed.

        rng : randomStream(int), `ref`
            The integer-based random stream used to choose swap targets.
            Must be locale-safe and reproducibly seeded.

        Returns
        -------
        MsgTuple
            A success or error message depending on whether the shuffle completes
            successfully. Errors include invalid bounds or out-of-range access.

        Raises
        ------
        May throw from `rng.next()` or array indexing if internal bounds checks fail.

        Notes
        -----
        - This function is used during distributed or hierarchical shuffling to partially
        shuffle a block of data while preserving certain order boundaries.
        - It performs all swaps in-place and operates only within a single locale.
        - A `bound` that falls inside `swapChunk` (excluding its endpoints) is disallowed
        and results in an error.

        See Also
        --------
        mergeLocalChunk : Combines shuffling with probabilistic merging.
        shuffleLocales : Uses this function to shuffle initial chunks.
    */
    proc fisherYatesOnChunk(
            ref x: [],
            swapChunk: range(int),
            const bound: int,
            ref rng: randomStream(int)
        ): MsgTuple throws {

        if (bound > swapChunk.first) && (bound < swapChunk.last){
            const errorMsg = "Invalid bound=" + bound:string +
                    ", must not be inside swapChunk=[" +
                    swapChunk.first:string + "," + swapChunk.last:string + "]";
            randLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return MsgTuple.error(errorMsg);
        }

        for i in swapChunk {
            const lo = min(i, bound), hi = max(i, bound);
            const idx = rng.next(lo, hi);
            if i != idx {
                x[i] <=> x[idx];
            }
        }
        return MsgTuple.success("finished fisherYates on range.");
    }

    /*
        Randomly merge two shuffled runs in-place over a local chunk of data.

        This procedure merges two adjacent shuffled subarrays of `x`, starting at `start`
        with lengths `len1` and `len2`, using a probabilistic selection mechanism.
        The merge is performed only over the specified `localChunk`, which should
        correspond to the portion of the data owned by a specific locale.

        At each index `i` in `localChunk`, the merge selects an element from the first run
        with probability `len1 / (len1 + len2)`, and from the second run otherwise.
        If an element from the second run is selected, it is swapped into position `i`
        from index `j`, which tracks the next available element in the second run.

        After the merge completes (or the second run is exhausted), the remaining unmerged
        elements in `[j..endIdx]` are partially shuffled using a constrained
        Fisher-Yates shuffle (`fisherYatesOnChunk`) to ensure uniformity.

        Parameters
        ----------
        x : [] ?t, `ref`
            The array to merge and shuffle in-place.

        localChunk : range(int)
            The local slice of indices on which this locale should perform the merge.

        start : int
            The starting index of the first run.

        len1 : int
            The number of elements in the first run, starting at `start`.

        len2 : int
            The number of elements in the second run, which immediately follows the first.

        rng : randomStream(real), `ref`
            A real-valued random stream used to probabilistically select which run to take
            an element from during the merge phase.

        rngInt : randomStream(int), `ref`
            An integer-valued random stream used for the constrained shuffle of the
            remaining tail elements.

        Raises
        ------
        May throw from random stream calls or from subroutine `fisherYatesOnChunk`.

        Notes
        -----
        - This is a core subroutine in the distributed merge shuffle process.
        - Only the indices in `localChunk` are modified, and the function assumes
        the runs lie fully within this locale's portion of the array.
        - The final constrained shuffle ensures statistical fairness for any unmerged
        tail elements that were not deterministically assigned.

        See Also
        --------
        fisherYatesOnChunk : Performs constrained in-place shuffle of remaining tail.
        shuffleLocales : Coordinates multiple mergeLocalChunk calls per locale.
        mergeShuffle : Full distributed merge-based shuffle using this function.
    */
    proc mergeLocalChunk(
        ref x:    [],
        localChunk: range(int),
        start:    int,
        len1:     int,
        len2:     int,
        ref rng: randomStream(real),       
        ref rngInt: randomStream(int)  
        ) throws {
        const endIdx    = start + len1 + len2 - 1;
        const threshold = len1: real / (len1 + len2);

        // merge: run i from localLo..localHi, 
        // swapping in elements from the second run
        var j = start + len1;
        var final_i = localChunk.last;
        for i in localChunk {
            if i == j || j > endIdx{
                final_i = i;
                break;
            }

            if rng.next() < threshold {
                // pick from first run → leave x[i] alone
            } else {
                // pick from second run → swap it into position i
                x[i] <=> x[j];
                j += 1;
            }
        }
        fisherYatesOnChunk(x, final_i..endIdx, start, rngInt);
    }


    // use AtomicObjects;    // for atomic(bool)


    /*
        Comparator for sorting locale-index pairs by their integer key.

        This comparator is used to sort tuples of the form `(locale, int)` based
        on the integer component (typically the low index of a locale's local
        subdomain). It is used to enforce a deterministic ordering of locales
        in distributed algorithms, such as merge-based shuffling.

        The `key` function extracts the integer part of the tuple to use
        as the sort key.

        Returns
        -------
        int
            The integer component of the `(locale, int)` tuple, used for comparison.

        See Also
        --------
        mergeShuffle : Uses this comparator to sort locales by subdomain low index.
    */
    record byLowKey: keyComparator {
        proc key(p:(locale,int)): int { return p[1]; }
    }

    /*
        Generate a distributed array of random streams, one per locale.

        This function returns a distributed array of `randomStream` objects,
        one for each locale in `PrivateSpace`. Each stream is seeded deterministically
        from the base `seed0`, using the locale’s `here.id` to ensure uniqueness.

        The stream type can be specified using the optional type parameter `t`
        (e.g., `real`, `int`, `bool`), which controls the type of values the stream
        will produce.

        Parameters
        ----------
        seed0 : int
            The base seed from which all per-locale seeds are derived.

        t : type, optional
            The element type of the random stream. Defaults to `real`.

        Returns
        -------
        [] randomStream(t)
            A distributed array of random streams of type `t`, indexed over `PrivateSpace`,
            with one stream per locale.

        Notes
        -----
        - This function ensures reproducibility and parallel safety by using
        per-locale seed offsets (`seed0 + here.id`).
        - The returned array can be used in parallel computations (e.g. shuffle or merge)
        to guarantee deterministic results across locales.

        See Also
        --------
        shuffleLocales : Uses streams of type `int` for chunk-level shuffling.
        mergeShuffle : Uses both `real` and `int` random streams for probabilistic merging.
    */
    proc getRandomStreams(seed0: int, type t = real){
        return forall PrivateSpace do new randomStream(t, seed=seed0 + here.id);
    }

    /*
        Perform a distributed, probabilistic in-place merge of two adjacent runs.

        This function merges two shuffled runs within the global array `x` using a
        randomized selection process. The merge is performed across multiple locales,
        but each locale only processes its local chunk of data.

        For each index `i` in the merge range, an element is selected from the first run
        with probability `len1 / (len1 + len2)`, and from the second run otherwise.
        If the element comes from the second run, it is swapped into place from index `j`.
        Once the second run is exhausted or the merge completes, the remaining unmerged
        elements are partially shuffled using `shuffleRange` to ensure uniformity.

        Parameters
        ----------
        x : [] ?t, `ref`
            The distributed array containing the two runs to merge in-place.

        start : int
            The starting index of the first run.

        len1 : int
            The length of the first run, starting at `start`.

        len2 : int
            The length of the second run, which immediately follows the first.

        realRNGs : [] randomStream(real), `ref`
            Distributed array of real-valued random streams, one per locale, used to
            probabilistically choose between the two runs during merging.

        intRNGs : [] randomStream(int), `ref`
            Distributed array of integer-valued random streams used by `shuffleRange`
            to shuffle any remaining tail elements after the main merge.

        Raises
        ------
        May throw from random stream usage or from internal bounds checks during access
        or swapping.

        Notes
        -----
        - The merge is coordinated in sorted order of locale subdomains to ensure
        deterministic results.
        - Only the portion of the array owned by each locale is touched by that locale.
        - The `shuffleRange` call at the end ensures that remaining unmerged elements
        are randomized in a statistically fair way.

        See Also
        --------
        shuffleRange : Shuffles the tail of the merge range after second run is exhausted.
        mergeShuffle : Top-level distributed shuffle algorithm that uses this function.
        mergeLocalChunk : A localized, single-locale version of this merge logic.
    */
    proc merge(
        ref x:       [],
        start:       int,
        len1:        int,
        len2:        int,
        ref realRNGs:[] randomStream(real),
        ref intRNGs :[] randomStream(int)
        ) throws {
        const endIdx    = start + len1 + len2 - 1;
        const threshold = len1:real / (len1 + len2);

        var i = start, j = start + len1;
        var finished = false;

        // sort by each locale’s low index
        var byLow = [loc in Locales] (loc, x.localSubdomain(loc).low);
        sort(byLow, new byLowKey());

        for (loc, lowIdx) in byLow {
            if finished then break;

            const sub = x.localSubdomain(loc);
            const localLo = max(start, lowIdx);
            const localHi = min(endIdx, sub.high);
            if localLo > localHi then continue;

            on loc {
                var rng = realRNGs[here.id];
                var localI = i;
                var localJ = j;
                const localEndIdx = endIdx;
                var localDone = false;

                for k in localLo..localHi {
                    localI = k;    
                    if rng.next() < threshold {
                        if localI == localJ {
                            localDone = true;
                            break;
                        }
                    } else {
                        if localJ > localEndIdx {
                            localDone = true;
                            break;
                        }
                        x[localI] <=> x[localJ];
                        localJ += 1;
                    }
                }

                // Write back updates explicitly
                i = localI;
                j = localJ;
                if localDone {
                    finished = true;
                }
            }
        }

        // tail‐shuffle
        shuffleRange(x, i..endIdx, start, intRNGs);
    }

    /*
        Retrieve the lowest index of each locale's local subdomain.

        This function computes a distributed array containing the lowest (starting)
        index of the portion of the global array `x` that resides on each locale.
        The result is indexed by locale ID.

        Parameters
        ----------
        x : [] ?t, `ref`
            The distributed array whose local subdomain bounds are being queried.

        Returns
        -------
        [] int
            A one-dimensional array of length `numLocales`, where each element
            contains the `.low` bound of the local subdomain for that locale.

        Notes
        -----
        - This function is used to construct merge ranges or shuffle boundaries
        based on data ownership.
        - It assumes that the distribution of `x` is aligned with `Locales`.
        - The operation is parallelized using `coforall` to ensure efficiency.

        See Also
        --------
        getDomainHighs : Returns the highest index of each locale’s subdomain.
        mergeShuffle : Uses this function to coordinate merge boundaries across locales.
    */
    @chplcheck.ignore("UnusedFormal")
    proc getDomainLows(ref x: []): [] int {
        var domainLows: [0..#numLocales] int;
        coforall loc in Locales with (ref x) do on loc {
            domainLows[here.id] = x.localSubdomain(loc=here).low;
        }
        return domainLows;
    }

    /*
        Retrieve the highest index of each locale's local subdomain.

        This function computes a distributed array containing the highest (ending)
        index of the portion of the global array `x` that resides on each locale.
        The result is indexed by locale ID.

        Parameters
        ----------
        x : [] ?t, `ref`
            The distributed array whose local subdomain upper bounds are being queried.

        Returns
        -------
        [] int
            A one-dimensional array of length `numLocales`, where each element
            contains the `.high` bound of the local subdomain for the corresponding locale.

        Notes
        -----
        - This function is useful for constructing distributed merge ranges,
        especially in operations like `mergeShuffle`.
        - The bounds are retrieved in parallel using `coforall` over all locales.
        - Assumes the array `x` is distributed over the default `Locales` space.

        See Also
        --------
        getDomainLows : Returns the lowest index of each locale’s subdomain.
        mergeShuffle : Uses both low and high bounds to define merge block sizes.
    */
    @chplcheck.ignore("UnusedFormal")
    proc getDomainHighs(ref x: []): [] int {
        var domainHighs: [0..#numLocales] int;
        coforall loc in Locales with (ref x) do on loc {
            domainHighs[here.id] = x.localSubdomain(loc=here).high;
        } 
        return domainHighs;
    }

    @chplcheck.ignore("UnusedFormal")
    proc feistelShuffleHelp(
        cmd: string,
        msgArgs: borrowed MessageArgs,
        st: borrowed SymTab,
        type array_dtype,
        param array_nd: int
    ): MsgTuple throws where array_nd == 1
    {
        const name   = msgArgs["name"],
              xName  = msgArgs["x"].toScalar(string),
              shape  = msgArgs["shape"].toScalarTuple(int, array_nd),
              state  = msgArgs["state"].toScalar(int);

        const rounds = if msgArgs.contains("feistel_rounds")
                        then msgArgs["feistel_rounds"].toScalar(int)
                        else 16;

        randLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                        "name: %? shape %? dtype: %? state %i rounds %i".format(
                            name, shape, type2str(array_dtype), state, rounds));

        // RNG hookup
        var generatorEntry = st[name]: borrowed GeneratorSymEntry(int);
        ref rng = generatorEntry.generator;
        if state != 1 then rng.skipTo(state-1);

        // Key: client-supplied or derived
        var key: uint(64);
        if msgArgs.contains("feistel_key") {
            key = msgArgs["feistel_key"].toScalar(uint(64));
        } else {
            key = rng.next():uint(64);
            if key == 0:uint(64) then key = 0x9E37_79B9_7F4A_7C15:uint(64);
        }

        // Target array & domain
        const arrEntry = st[xName]: SymEntry(array_dtype, array_nd);
        ref a = arrEntry.a;

        const d = a.domain;
        const N = d.size;
        if N <= 1 then return MsgTuple.success();

        // Base of the 1-D global index range
        const idxRange = d.dim(0);
        const base = idxRange.low;

        // ------------------------
        // Per-locale index ranges to find owners of destination indices
        // ------------------------
        const flatLocRanges = [loc in Locales] d.localSubdomain(loc).dim(0);

        // ------------------------
        // Build *per-sender* aligned lists:
        //   myDestLocales : list(int)         -- destination locale id
        //   myDestIdx     : list(int)         -- destination global index
        //   myVals        : list(array_dtype) -- payloads aligned 1:1 with myDestIdx
        // Gather them into sender-indexed arrays for the repartitioner.
        // ------------------------
        var allDestLocales : [PrivateSpace] list(int);
        var allDestIdx     : [PrivateSpace] list(int);
        var allVals        : [PrivateSpace] list(array_dtype);

        coforall loc in Locales with (ref allDestLocales, ref allDestIdx, ref allVals) do on loc {

            const flatLocRangesHere = flatLocRanges;  // copies ranges to 'here' once
            inline proc ownerOfIndex(gIdx: int): int {
                for (rr, i) in zip(flatLocRangesHere, 0..<numLocales) do
                if rr.contains(gIdx) then return i;
                return numLocales-1;
            }

            const ld = d.localSubdomain();
            if ! ld.isEmpty() {

                const m = ld.size;

                // Preallocate simple arrays to avoid concurrent list appends inside forall
                var myDestLocalesArr: [0..#m] int;
                var myDestIdxArr    : [0..#m] int;
                var myValsArr       : [0..#m] array_dtype;

                // Local bulk read of payloads in the same iteration order
                myValsArr = a[ld];

                // Compute Feistel destinations for each locally-owned element
                // Iterate in lock-step over 0..#m and local indices
                forall (i, gIdx) in zip(0..#m, ld) with
                    (const n=N, const k=key, const r=rounds, const b=base) {
                    const ordHere = gIdx - b;                   // 0-based ordinal ∈ [0,N)
                    const ordDest = permuteIndexFeistel(ordHere, n, k, r);
                    const destIdx = b + ordDest;                // back to global index space
                    myDestIdxArr[i]     = destIdx;
                    myDestLocalesArr[i] = ownerOfIndex(destIdx);
                }

                // Stash sender-local lists in the global arrays (one slot per sender locale)
                allDestLocales[here.id] = new list(myDestLocalesArr);
                allDestIdx    [here.id] = new list(myDestIdxArr);
                allVals       [here.id] = new list(myValsArr);
            } else {
                allDestLocales[here.id] = new list(int);
                allDestIdx[here.id] = new list(int);
                allVals[here.id] = new list(array_dtype);
            }
        }

        // ------------------------
        // Repartition twice with the *same* routing:
        //   1) destination indices (int)
        //   2) payload values (array_dtype)
        // Returns [PrivateSpace] list(eltType), indexed by **receiver** locale ID.
        // ------------------------
        const recvIdx  = repartitionByLocale(int,         allDestLocales, allDestIdx);
        const recvVals = repartitionByLocale(array_dtype, allDestLocales, allVals);

        // ------------------------
        // Local writes: zip indices with payloads and store into local portion of 'a'
        // ------------------------
        coforall loc in Locales with (ref a) do on loc {
            const myIdxs  = recvIdx [here.id];
            const myValsL = recvVals[here.id];
            // Alignment guaranteed because both repartitions used identical routing
            forall (dstIdx, val) in zip(myIdxs.these(), myValsL.these()) {
                a[dstIdx] = val;
            }
        }

        return MsgTuple.success();
    }

    @chplcheck.ignore("UnusedFormal")
    proc feistelShuffleHelp(
        cmd: string,
        msgArgs: borrowed MessageArgs,
        st: borrowed SymTab,
        type array_dtype,
        param array_nd: int
    ): MsgTuple throws where array_nd != 1
    {
        return new MsgTuple("feistel shuffle does not support arrays of dimension > 1.",
                            MsgType.ERROR);
    }

    // ---- constants & bit helpers ----
    const MASK64: uint(64) = 0xFFFF_FFFF_FFFF_FFFF:uint(64);
    const GOLDEN: uint(64) = 0x9E37_79B9_7F4A_7C15:uint(64);

    inline proc rotl64(x: uint(64), r: int): uint(64) {
        return rotl(x, r & 63);
    }

    inline proc bitMask(nbits: int): uint(64) {
        if nbits <= 0 then return 0:uint(64);
        if nbits >= 64 then return MASK64;
        return (1:uint(64) << nbits:uint(64)) - 1:uint(64);
    }

    // Number of bits to represent N
    inline proc bitsFor(N: int): int {
        const m = if N > 0 then N - 1 else 0;
        var nbits = 0, v = m;
        while v > 0 do { v >>= 1; nbits += 1; }
        return max(1, nbits);
    }

    inline proc roundKey(master: uint(64), i: int): uint(64) {
        return rotl64(master ^ (GOLDEN * (i + 1):uint(64)), 7*i + 13);
    }

    // Expand a 64-bit seed to 'wantBits' (<=64 here).
    // prf = Pseudorandom function, not meant to be invertible, just to provide
    // some randomization bits.
    proc prfBits(seed0: uint(64), wantBits: int, tweak0: uint(64)): uint(64) {
        var acc: uint(64) = 0;
        var produced = 0;
        var v = seed0 ^ (GOLDEN ^ tweak0);
        var tweak = tweak0;

        while produced < wantBits {
            v = (v * GOLDEN) & MASK64;
            v ^= (v >> 29) ^ (v >> 43);
            v = rotl64(v, 17) ^ (tweak * 0xA5A5_A5A5_A5A5_A5A5:uint(64));
            const take = min(64, wantBits - produced);
            acc |= (v & bitMask(take)) << produced:uint(64);
            produced += take;
            tweak = (tweak * 0x2545_F491_4F6C_DD1D:uint(64)) & MASK64;
        }
        return acc & bitMask(wantBits);
    }

    inline proc F(R: uint(64), outBits: int, k: uint(64), roundIdx: int): uint(64) {
        return prfBits(k ^ R ^ rotl64(k, roundIdx), outBits, roundIdx:uint(64));
    }

    proc feistelEnc(x: uint(64), nbits: int, key: uint(64), rounds: int): uint(64) {
        const m = bitMask(nbits);
        var X = x & m;

        const rbits = nbits / 2;
        const lbits = nbits - rbits;
        var R = X & bitMask(rbits);
        var L = X >> rbits;

        var lb = lbits, rb = rbits;
        for i in 0..#rounds {
            const k = roundKey(key, i);
            const fout = F(R, lb, k, i) & bitMask(lb);
            const newL = R;
            const newR = (L ^ fout) & bitMask(rb);
            L = newL; R = newR;
            const tmp = lb; lb = rb; rb = tmp;  // swap widths
        }

        return ((R << rbits:uint(64)) | L) & m;
    }

    proc permuteIndexFeistel(i: int, N: int, key: uint(64), rounds: int): int {
        const nb = bitsFor(N);
        // Easier to work with even number of bits
        const nbits = if nb & 1 == 0 then nb else nb + 1;
        // TODO: look into unbalanced Feistel
        var x: uint(64) = i:uint(64);
        while true {
            const y = feistelEnc(x, nbits, key, rounds);
            if y:int < N then return y:int;
            x = y;  // cycle-walk if out of range
        }
        return 0;
    }

    use CommandMap;
    registerFunction("logisticGenerator", logisticGeneratorMsg, getModuleName());
    registerFunction("segmentedSample", segmentedSampleMsg, getModuleName());
    registerFunction("poissonGenerator", poissonGeneratorMsg, getModuleName());
    registerFunction("delGenerator", delGeneratorMsg, getModuleName());
}
