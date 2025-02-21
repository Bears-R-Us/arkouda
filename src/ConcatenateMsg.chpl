module ConcatenateMsg
{
    use ServerConfig;

    use Time;
    use Math only;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use BigInteger;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;
    use CommAggregation;
    use PrivateDist;
    
    use AryUtil;
    
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const cmLogger = new Logger(logLevel, logChannel);

    use CommandMap;

    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html#array_api.concat
    /* Concatenate a list of arrays together
       to form one array
    */
    @arkouda.instantiateAndRegister(prefix='concatenate')
    proc concatenateMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const nArrays = msgArgs["n"].toScalar(int),
              names = msgArgs["names"].toScalarArray(string, nArrays),
              axis = msgArgs["axis"].getPositiveIntValue(array_nd);

        // Retrieve the arrays from the symbol table
        const eIns = for n in names do st[n]: borrowed SymEntry(array_dtype, array_nd),
              shapes = [i in 0..<nArrays] eIns[i].tupShape,
              (valid, shapeOut, offsets) = concatenatedShape(shapes, axis, array_nd);
        var eOut = createSymEntry((...shapeOut), array_dtype);

        if !valid {
            const errMsg = "All arrays must have the same shape except in the concatenation axis.";
            cmLogger.error(getModuleName(), pn, getLineNumber(), errMsg);
            return MsgTuple.error(errMsg);
        } else {
            forall (arrIdx, arr) in zip(eIns.domain, eIns) {
                forall idx in arr.a.domain with (
                    var agg = newDstAggregator(array_dtype),
                    const imap = new concatIndexMapper(array_nd, axis, offsets[arrIdx])
                ) do
                    agg.copy(eOut.a[imap(if array_nd == 1 then (idx,) else idx)], arr.a[idx]);
            }
            return st.insert(eOut);
        }
    }

    // Record to map indices for concatenation
    record concatIndexMapper {
        param ndOut: int;
        const axis: int;
        const offset: int;

        proc this(idx: ndOut*int): ndOut*int {
            var ret: ndOut*int;
            for param i in 0..<ndOut do ret[i] = idx[i];
            ret[axis] += offset; // Adjust index by offset
            return ret;
        }
    }

    // Function to validate shapes and determine output shape
    private proc concatenatedShape(shapes: [?d] ?t, axis: int, param N: int): (bool, N*int, N*int)
        where isHomogeneousTuple(t)
    {
        var shapeOut: N*int,
            offsets: N*int,
            firstShape = shapes[0];

        // Ensure all shapes match except for the concatenation axis
        for i in 1..d.last {
            for param j in 0..<N {
                if j != axis && shapes[i][j] != firstShape[j] {
                    return (false, shapeOut, offsets);
                }
            }
        }

        // Compute output shape
        shapeOut = firstShape;
        shapeOut[axis] = + reduce [i in 0..<d.size] shapes[i][axis]; // Sum sizes along axis

        // Compute offsets
        offsets[0] = 0;
        for i in 1..d.last {
            offsets[i] = offsets[i - 1] + shapes[i - 1][axis]; // Cumulative sum along axis
        }
        
        return (true, shapeOut, offsets);
    }

}
