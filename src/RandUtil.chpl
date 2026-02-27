module RandUtil {
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use Random;
    use RandMsg;
    use CommAggregation;

    const minPerStream = 256; // minimum number of elements per random stream

    record scalarOrArray {
        var isArray: bool;
        var sym;
        var val: real;

        proc init() {
            this.isArray = false;
            this.sym = try! makeDistArray([0.0]);
            val = 0.0;
        }

        proc init(scalarOrArrayString: string, isArray: bool, st: borrowed SymTab) {
            this.isArray = isArray;
            if isArray {
                try! st.checkTable(scalarOrArrayString);
                this.sym = try! toSymEntry(getGenericTypedArrayEntry(scalarOrArrayString, st),real).a;
            }
            else {
                this.sym = try! makeDistArray([0.0]);
                val = try! scalarOrArrayString:real;
            }
        }

        proc this(idx): real {
            return if isArray then this.sym[idx] else this.val;
        }
    }

    enum GenerationFunction {
        ExponentialGenerator,
        GammaGenerator,
        LogisticGenerator,
        NormalGenerator,
        PoissonGenerator,
    }

    // TODO how to update this to handle randArr being a multi-dim array??
    // I thought to just do the same randArr[randArr.domain.orderToIndex(i)] trick
    // but im not sure how randArr.localSubdomain() will differ with multi-dim
    proc uniformStreamPerElem(ref randArr: [?D] ?t, ref rng, param function: GenerationFunction, hasSeed: bool,
                                                                const lam: scalarOrArray(?) = new scalarOrArray(),
                                                                const mu: scalarOrArray(?) = new scalarOrArray(),
                                                                const scale: scalarOrArray(?) = new scalarOrArray(),
                                                                const kArg: scalarOrArray(?) = new scalarOrArray()) throws 
        where D.rank == 1 {
            // use a fixed number of elements per stream instead of relying on number of locales or numTasksPerLoc because these
            // can vary from run to run / machine to mahchine. And it's important for the same seed to give the same results
            use Time;
            var next: rng.eltType;
            if hasSeed {
                next = rng.next();
            }
            else{
                const seed =  timeSinceEpoch().totalMicroseconds();
                var randStream0 = new randomStream(rng.eltType, seed);
                next = randStream0.next();
            }
            const generatorSeed = (next * 2**62):int, elemsPerStream = max(minPerStream, 2**(2 * ceil(log10(D.size)):int));

            // using nested coforalls over locales and tasks so we know how to generate taskSeed
            coforall loc in Locales do on loc {
                const locSubDom = randArr.localSubdomain(),
                    offset = if loc.id == 0 then 0 else elemsPerStream - (locSubDom.low % elemsPerStream);

                // skip if all the values were pulled to previous locale
                if offset <= locSubDom.high {
                    // we take the ceil in chunk calculation because if elemsPerStream doesn't evenly divide along locale boundaries, the remainder is pulled to the previous locale
                    const chunksAlreadyDone = if loc.id == 0 then 0 else ceil((locSubDom.low + 1) / elemsPerStream:real):int,  // number of chunks handled by previous locales
                        thisLocsNumChunks = ceil((locSubDom.high + 1 - (locSubDom.low + offset)) / elemsPerStream:real):int;  // number of chunks this locale needs to handle

                    coforall streamID in 0..<thisLocsNumChunks {
                        const taskSeed = generatorSeed + chunksAlreadyDone + streamID,  // initial seed offset by other locales threads plus current thread id
                            startIdx = (streamID * elemsPerStream) + locSubDom.low + offset,
                            stopIdx = min(startIdx + elemsPerStream - 1, randArr.domain.high);  // continue past end of localSubDomain to read full block to avoid seed sharing

                        var realRS = new randomStream(real, taskSeed),
                            uintRS = new randomStream(uint, taskSeed);

                        if stopIdx >= locSubDom.high {
                            // we are on the last chunk on a locale, so create a copy aggregator since we
                            // steal the remainder of the chunk that overflows onto the following locale
                            var agg = newDstAggregator(t);
                            for i in startIdx..stopIdx {
                                select function {
                                    when GenerationFunction.ExponentialGenerator {
                                        agg.copy(randArr[i], standardExponentialZig(realRS, uintRS));
                                    }
                                    when GenerationFunction.GammaGenerator {
                                        const x = gammaGenerator(kArg[i], realRS);
                                        agg.copy(randArr[i], gammaGenerator(kArg[i], realRS));
                                    }
                                    when GenerationFunction.LogisticGenerator {
                                        agg.copy(randArr[i], logisticGenerator(mu[i], scale[i], realRS));
                                    }
                                    when GenerationFunction.NormalGenerator {
                                        agg.copy(randArr[i], standardNormZig(realRS, uintRS));
                                    }
                                    when GenerationFunction.PoissonGenerator {
                                        agg.copy(randArr[i], poissonGenerator(lam[i], realRS));
                                    }
                                    otherwise {
                                        compilerError("Unrecognized generation function");
                                    }
                                }
                            }
                            // manually flush the aggregtor when we exit the scope
                            // I thought this would happen automatically, so I'm not convinced
                            // this is necessary. But radixSortLSD does it, so who am I to argue
                            agg.flush();
                        }
                        else {
                            for i in startIdx..stopIdx {
                                select function {
                                    when GenerationFunction.ExponentialGenerator {
                                        randArr[i] = standardExponentialZig(realRS, uintRS);
                                    }
                                    when GenerationFunction.GammaGenerator {
                                        const x = gammaGenerator(kArg[i], realRS);
                                        randArr[i] = gammaGenerator(kArg[i], realRS);
                                    }
                                    when GenerationFunction.LogisticGenerator {
                                        randArr[i] = logisticGenerator(mu[i], scale[i], realRS);
                                    }
                                    when GenerationFunction.NormalGenerator {
                                        randArr[i] = standardNormZig(realRS, uintRS);
                                    }
                                    when GenerationFunction.PoissonGenerator {
                                        randArr[i] = poissonGenerator(lam[i], realRS);
                                    }
                                    otherwise {
                                        compilerError("Unrecognized generation function");
                                    }
                                }
                            }
                        }
                    }  // coforall over randomStreams created
                }
            }  // coforall over locales
    }
    proc uniformStreamPerElem(ref randArr: [?D] ?t, ref rng, param function: GenerationFunction, hasSeed: bool,
                                                                const lam: scalarOrArray(?) = new scalarOrArray(),
                                                                const mu: scalarOrArray(?) = new scalarOrArray(),
                                                                const scale: scalarOrArray(?) = new scalarOrArray(),
                                                                const kArg: scalarOrArray(?) = new scalarOrArray()) throws 
        where D.rank > 1 {
            throw new Error ("uniformStreamPerElem does not support multidimensional arrays.");
    }
}
