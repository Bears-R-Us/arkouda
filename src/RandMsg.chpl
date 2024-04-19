module RandMsg
{
    use ServerConfig;
    
    use ArkoudaTimeCompat as Time;
    use Math only;
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

    proc sampleWeightsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const pn = Reflection.getRoutineName(),
              gName = msgArgs.getValueOf("gName"),              // generator name
              aName = msgArgs.getValueOf("aName"),              // values array name
              wName = msgArgs.getValueOf("wName"),              // weights array name
              n = msgArgs.get("n").getIntValue(),               // number of samples
              replace = msgArgs.get("replace").getBoolValue(),  // sample with replacement?
              state = msgArgs.get("state").getIntValue(),       // rng state
              rname = st.nextName();

        randLogger.debug(getModuleName(),pn,getLineNumber(),
                         "gname: %? aname %? wname: %? n %i replace %i state %i rname %?"
                         .doFormat(gName, aName, wName, n, replace, state, rname));

        var aGEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(aName, st),
            wGEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(wName, st);

        proc sampleHelper(type t): MsgTuple throws {
            const aE = toSymEntry(aGEnt, t),
                  wE = toSymEntry(wGEnt, real); // weights are always real

            var generatorEntry: borrowed GeneratorSymEntry(real) = toGeneratorSymEntry(st.lookup(gName), real);
            ref rng = generatorEntry.generator;
            if state != 1 then rng.skipTo(state-1);

            const s = randSampleWeights(rng, aE.a, wE.a, n, replace);
            st.addEntry(rname, createSymEntry(s));

            const repMsg = "created " + st.attrib(rname);
            randLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select aGEnt.dtype {
            when DType.Int64 do return sampleHelper(int);
            when DType.UInt64 do return sampleHelper(uint);
            when DType.Float64 do return sampleHelper(real);
            when DType.Bool do return sampleHelper(bool);
            otherwise {
                const errorMsg = "Unhandled data type %s".doFormat(dtype2str(aGEnt.dtype));
                randLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
                return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
            }
        }

    }

    use CommandMap;
    registerFunction("randomNormal", randomNormalMsg, getModuleName());
    registerFunction("createGenerator", createGeneratorMsg, getModuleName());
    registerFunction("uniformGenerator", uniformGeneratorMsg, getModuleName());
    registerFunction("permutation", permutationMsg, getModuleName());
    registerFunction("shuffle", shuffleMsg, getModuleName());
    registerFunction("sampleWeights", sampleWeightsMsg, getModuleName());
}
