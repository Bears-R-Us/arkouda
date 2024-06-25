module RandUtil {
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use RandMsg;
    use ArkoudaRandomCompat;

    const minPerStream = 256; // minimum number of elements per random stream

    record scalarOrArray {
        var isArray: bool;
        var sym;  // TODO figure out type hint here to avoid generic
        var val: real;

        proc init(scalarOrArrayString: string, isArray: bool, st: borrowed SymTab) {
            // I'm not sure if there's a good way to remove these try!
            this.isArray = isArray;
            if isArray {
                try! st.checkTable(scalarOrArrayString);
                this.sym = try! toSymEntry(getGenericTypedArrayEntry(scalarOrArrayString, st),real).a;
            }
            else {
                // prob not the smartest way of doing this
                // we just want to avoid unnecessarily creating a large array
                this.sym = try! makeDistArray([0.0]);
                val = try! scalarOrArrayString:real;
            }
        }

        proc this(idx): real {
            return if isArray then this.sym[idx] else this.val;
        }
    }

    enum GenerationFunction {
      PoissonGenerator,
    }

    proc uniformStreamPerElem(ref randArr: [?D] ?t, param function: GenerationFunction, hasSeed: bool, const lam: scalarOrArray(?), ref rng) throws {
        if hasSeed {
            // use a fixed number of elements per stream instead of relying on number of locales or numTasksPerLoc because these
            // can vary from run to run / machine to mahchine. And it's important for the same seed to give the same results
            const generatorSeed = (rng.next() * 2**62):int,
                elemsPerStream = max(minPerStream, 2**(2 * ceil(log10(D.size)):int));

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

                        var rs = new randomStream(real, taskSeed);
                        for i in startIdx..stopIdx {
                            select function {
                                // TODO look into adding copy aggregation looking here
                                when GenerationFunction.PoissonGenerator {
                                    randArr[i] = poissonGenerator(lam[i], rs);
                                }
                                otherwise {
                                    compilerError("Unrecognized generation function");
                                }
                            }
                        }
                    }
                }
            }
        }
        else {  // non-seeded case, we can just use task private variables for our random streams
            forall (rv, i) in zip(randArr, randArr.domain) with (var rs = new randomStream(real)) {
                select function {
                    when GenerationFunction.PoissonGenerator {
                        rv = poissonGenerator(lam[i], rs);
                    }
                    otherwise {
                        compilerError("Unrecognized generation function");
                    }
                }
            }
        }
    }
}
