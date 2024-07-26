module IndexingMsg
{
    use ServerConfig;
    use ServerErrorStrings;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use AryUtil;
    use IOUtils;

    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    use CommAggregation;

    use FileIO;
    use List;
    use BigInteger;


    use Map;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const imLogger = new Logger(logLevel, logChannel);

    proc arrayViewMixedIndexMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var ndim = msgArgs.get("ndim").getIntValue();
        const pdaName = msgArgs.getValueOf("base");
        const indexDimName = msgArgs.getValueOf("index_dim");
        const dimProdName = msgArgs.getValueOf("dim_prod");
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                    "%s %s %i %s %s".format(cmd, pdaName, ndim, dimProdName, msgArgs.getValueOf("coords")));

        var dimProd: borrowed GenSymEntry = getGenericTypedArrayEntry(dimProdName, st);
        var dimProdEntry = toSymEntry(dimProd, int);

        var indexDim: borrowed GenSymEntry = getGenericTypedArrayEntry(indexDimName, st);
        var indexDimEntry = toSymEntry(indexDim, int);
        ref dims = indexDimEntry.a;

        // String array containing the type of the following value at even indicies then the value ex. ["int", "7", "slice", "(0,5,-1)", "pdarray", "id_4"]
        var typeCoords: [0..#(ndim*2)] string = msgArgs.get("coords").getList(ndim*2);

        var scaledCoords = makeDistArray(+ reduce dims, int);
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * dims.size);
        var offsets = (+ scan dims) - dims;

        forall i in 0..#typeCoords.size by 2 {
            select typeCoords[i] {
                when "int" {
                    scaledCoords[offsets[i/2]] = typeCoords[i+1]:int * dimProdEntry.a[i/2];
                }
                when "slice" {
                    var (start, stop, stride) = parseJson(typeCoords[i+1], 3*int);
                    var slice: range(strides=strideKind.any) = convertSlice(start, stop, stride);
                    var scaled: [0..#slice.size] int = slice * dimProdEntry.a[i/2];
                    for j in 0..#slice.size {
                        scaledCoords[offsets[i/2]+j] = scaled[j];
                    }
                }
                // Advanced indexing not yet supported
                // when "pdarray" {
                //     // TODO if bool array convert to int array by doing arange(len)[bool_array]
                //     var arrName: string = typeCoords[i+1];
                //     var indArr: borrowed GenSymEntry = getGenericTypedArrayEntry(arrName, st);
                //     var indArrEntry = toSymEntry(indArr, int);
                //     var scaledArray = indArrEntry.a * dimProdEntry.a[i/2];
                //     // var localizedArray = new lowLevelLocalizingSlice(scaledArray, offsets[i/2]..#indArrEntry.a.size);
                //     forall (j, s) in zip(indArrEntry.a.domain, scaledArray) with (var DstAgg = newDstAggregator(int)) {
                //         DstAgg.copy(scaledCoords[offsets[i/2]+j], s);
                //     }
                // }
            }
        }

        // create full index list
        // get next symbol name
        var indiciesName = st.nextName();
        var indicies = st.addEntry(indiciesName, * reduce dims, int);

        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "rname = %s".format(indiciesName));

        // avoid dividing by 0
        // if any dim is 0 we return an empty list
        if & reduce (dims!=0) {
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * dims.size);
            var dim_prod = (* scan(dims)) / dims;

            recursiveIndexCalc(0,0,0);
            proc recursiveIndexCalc(depth: int, ind:int, sum:int) throws {
                for j in 0..#dims[depth] {
                    imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "depth = %i".format(depth));
                    imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "j = %i".format(j));
                    imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "SUM: sum + scaledCoords[offsets[depth]+j] = %i".format(sum + scaledCoords[offsets[depth]+j]));
                    imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "IND: ind + j*dim_prod[depth] = %i".format(ind+(j*dim_prod[depth])));

                    if depth == ndim-1 then indicies.a[ind+(j*dim_prod[depth])] = sum+scaledCoords[offsets[depth]+j];
                    else recursiveIndexCalc(depth+1, ind+(j*dim_prod[depth]), sum+scaledCoords[offsets[depth]+j]);
                }
            }
        }

        var arrParam = msgArgs.get("base");
        arrParam.setKey("array");
        var idxParam = new ParameterObj("idx", indiciesName, "int");
        var subArgs = new MessageArgs(new list([arrParam, idxParam]));
        return pdarrayIndexMsg(cmd, subArgs, st);
    }

    /* arrayViewIntIndex "av[int_list]" response to __getitem__(int_list) where av is an ArrayView */
    @arkouda.instantiateAndRegister()
    proc arrayViewIntIndex(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const pdaName = msgArgs.getValueOf("base");
        const dimProdName = msgArgs.getValueOf("dim_prod");
        const coordsName = msgArgs.getValueOf("coords");
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "%s %s %s %s".format(cmd, pdaName, dimProdName, coordsName));

        var dimProd: borrowed GenSymEntry = getGenericTypedArrayEntry(dimProdName, st);
        var dimProdEntry = toSymEntry(dimProd, int);
        var coords: borrowed GenSymEntry = getGenericTypedArrayEntry(coordsName, st);

        // multi-dim to 1D address calculation
        // (dimProd and coords are reversed on python side to account for row_major vs column_major)
        select (coords.dtype) {
            when (DType.Int64) {
                var coordsEntry = toSymEntry(coords, int);
                var idx = + reduce (dimProdEntry.a * coordsEntry.a);

                var a_array_sym = st[msgArgs['base']]: SymEntry(array_dtype, 1);
                return MsgTuple.fromScalar(a_array_sym.a[idx]);
            }
            when (DType.UInt64) {
                var coordsEntry = toSymEntry(coords, uint);
                var idx = + reduce (dimProdEntry.a: uint * coordsEntry.a);

                var a_array_sym = st[msgArgs['base']]: SymEntry(array_dtype, 1);
                return MsgTuple.fromScalar(a_array_sym.a[idx:int]);
            }
            otherwise {
                 var errorMsg = notImplementedError(pn, "("+dtype2str(coords.dtype)+")");
                 imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                 return new MsgTuple(errorMsg, MsgType.ERROR);
             }
        }
    }

    /* arrayViewIntIndexAssignMsg "av[int_list]=value" response to __getitem__(int_list) where av is an ArrayView */
    proc arrayViewIntIndexAssignMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const pdaName = msgArgs.getValueOf("base");
        const dimProdName = msgArgs.getValueOf("dim_prod");
        const coordsName = msgArgs.getValueOf("coords");
        const dtypeStr = msgArgs.getValueOf("dtype");
        var value = msgArgs.getValueOf("value");
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "%s %s %s %s".format(cmd, pdaName, dimProdName, coordsName));

        var dimProd: borrowed GenSymEntry = getGenericTypedArrayEntry(dimProdName, st);
        var dimProdEntry = toSymEntry(dimProd, int);
        var coords: borrowed GenSymEntry = getGenericTypedArrayEntry(coordsName, st);

        var arrParam = msgArgs.get("base");
        arrParam.setKey("array");
        var idxParam = new ParameterObj("idx", "", "");

        // multi-dim to 1D address calculation
        // (dimProd and coords are reversed on python side to account for row_major vs column_major)
        select (coords.dtype) {
            when (DType.Int64) {
                var coordsEntry = toSymEntry(coords, int);
                var idx = + reduce (dimProdEntry.a * coordsEntry.a);
                idxParam.setVal(idx:string);
                var subArgs = new MessageArgs(new list([arrParam, msgArgs.get("value"), msgArgs.get("dtype"), idxParam]));
                return setIntIndexToValueMsg(cmd, subArgs, st, 1);
            }
            when (DType.UInt64) {
                var coordsEntry = toSymEntry(coords, uint);
                var idx = + reduce (dimProdEntry.a: uint * coordsEntry.a);
                idxParam.setVal(idx:string);
                var subArgs = new MessageArgs(new list([arrParam, msgArgs.get("value"), msgArgs.get("dtype"), idxParam]));
                return setIntIndexToValueMsg(cmd, subArgs, st, 1);
            }
            otherwise {
                 var errorMsg = notImplementedError(pn, "("+dtype2str(coords.dtype)+")");
                 imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                 return new MsgTuple(errorMsg, MsgType.ERROR);
             }
        }
    }

    @arkouda.registerCommand("[int]")
    proc intIndex(const ref array: [?d] ?t, idx: d.rank*int): t {
        return array[idx];
    }

    /* convert python slice to chapel slice */
    proc convertSlice(start: int, stop: int, stride: int): range(strides=strideKind.any) {
        var slice: range(strides=strideKind.any);
        // backwards iteration with negative stride
        if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
        // forward iteration with positive stride
        else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
        // BAD FORM start < stop and stride is negative
        else {slice = 1..0;}
        return slice;
    }

    @arkouda.registerCommand("[slice]")
    proc sliceIndex(const ref array: [?d] ?t, starts: d.rank*int, stops: d.rank*int, strides: d.rank*int): [] t throws {
        var rngs: d.rank*range(strides=strideKind.any),
            outSizes: d.rank*int;
        for param dim in 0..<d.rank {
            rngs[dim] = convertSlice(starts[dim], stops[dim], strides[dim]);
            outSizes[dim] = rngs[dim].size;
        }
        const sliceDom = {(...rngs)};
        var arraySlice = makeDistArray((...outSizes), t);

        forall (elt,j) in zip(arraySlice, sliceDom) with (var agg = newSrcAggregator(t)) do
            agg.copy(elt,array[j]);

        return arraySlice;
    }

    /*
        Index into an array using one or more 1D arrays of indices.

        Each index array corresponds to a particular dimension of the array being indexed.
        Dimensions without an index array are indexed using all elements along that dimension.
        For example, a 3D array could be indexed by 1, 2, or 3 1D arrays of indices.

        Only one dimension in the output array corresponds to the index arrays. The other output
        dimensions corresponding to the index arrays will be singleton dimensions (squeeze should be
        called on the output array to remove them).

        Example: a 5x6x7 array indexed by:
         * rank 0: [1, 2, 3, 4]
         * rank 2: [3, 4, 5, 6]

        Would return a new 4x6x1 array, where the 1st and 3rd dimensions are indexed by
        (1, 3), (2, 4), (3, 5), (4, 6), and the 2nd dimension is indexed by all of 0..<6.
    */
    @arkouda.instantiateAndRegister(prefix="[pdarray]")
    proc multiPDArrayIndex(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_a, type array_dtype_idx, param array_nd: int): MsgTuple throws
        where array_dtype_idx == int || array_dtype_idx == uint
    {
        const a = st[msgArgs["array"]]: SymEntry(array_dtype_a, array_nd),
              nIndexArrays = msgArgs["nIdxArrays"].toScalar(int),
              names = msgArgs["idx"].toScalarArray(string, nIndexArrays),
              idxDims = msgArgs["idxDims"].toScalarArray(int, names.size);

        const idxArrays = for n in names do st[n]: borrowed SymEntry(array_dtype_idx, 1);

        const outSize = idxArrays[0].size;
        for i in 1..<nIndexArrays do
            if idxArrays[i].size != outSize then
                MsgTuple.error("All index arrays must have the same length");

        const (valid, outRankIdx, outShape) = multiIndexShape(a.tupShape, idxDims, outSize);

        if valid {
            var ret = makeDistArray((...outShape), array_dtype_a);
            forall i in ret.domain with (
                var agg = newSrcAggregator(array_dtype_a),
                in idxDims // small array: create a per-task copy
            ) {
                const idx = if array_nd == 1 then (i,) else i,
                      iIdxArray: int = idx[outRankIdx];

                var inIdx = if array_nd == 1 then (i,) else i;

                for (rank, j) in zip(idxDims, 0..<nIndexArrays) do
                    inIdx[rank] = idxArrays[j].a[iIdxArray]:int;

                agg.copy(ret[idx], a.a[inIdx]);
            }
            return st.insert(new shared SymEntry(ret));
        } else {
            return MsgTuple.error("Invalid index dimensions: %? for %iD array".format(idxDims, array_nd));
        }
    }

    proc multiPDArrayIndex(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_a, type array_dtype_idx, param array_nd: int): MsgTuple throws
        where array_dtype_idx != int && array_dtype_idx != uint
    {
        return MsgTuple.error("Invalid index type: %s; must be 'int' or 'uint'".format(type2str(array_dtype_idx)));
    }

    private proc multiIndexShape(inShape: ?N*int, idxDims: [?d] int, outSize: int): (bool, int, N*int) {
        var minShape: N*int = inShape,
            firstRank = -1;

        if d.size > N then return (false, firstRank, minShape);

        var isSet: N*bool,
            first = true;

        for rank in idxDims {
            if rank < 0 || rank >=N then return (false, firstRank, minShape); // invalid rank index
            if isSet[rank] then return (false, firstRank, minShape); // duplicate rank index

            if first {
                minShape[rank] = outSize;
                first = false;
                firstRank = rank;
            } else {
                minShape[rank] = 1;
            }
            isSet[rank] = true;
        }

        return (true, firstRank, minShape);
    }

    private proc getGenericEntries(names: [?d] string, st: borrowed SymTab): [] borrowed GenSymEntry throws {
        var gEnts: [d] borrowed GenSymEntry?;
        for (i, name) in zip(d, names) do gEnts[i] = getGenericTypedArrayEntry(name, st);
        const ret = [i in d] gEnts[i]!;
        return ret;
    }

    /* pdarrayIndex "a[pdarray]" response to __getitem__(pdarray) */
    proc pdarrayIndexMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        const iname = msgArgs.getValueOf("idx");

        // get next symbol name
        var rname = st.nextName();

        var gX: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gIV: borrowed GenSymEntry = getGenericTypedArrayEntry(iname, st);
        
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "cmd: %s name: %s gX: %? gIV: %?".format(
                                           cmd, name, st.attrib(name), st.attrib(iname)));       

        // gather indexing by integer index vector
        proc ivInt64Helper(type XType): MsgTuple throws {
            var e = toSymEntry(gX,XType);
            var iv = toSymEntry(gIV,int);
            if (e.size == 0) && (iv.size == 0) {
                var a = st.addEntry(rname, 0, XType);
                var repMsg = "created " + st.attrib(rname);
                imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            if ivMin < 0 {
                var errorMsg = "Error: %s: OOBindex %i < 0".format(pn,ivMin);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);               
            }
            if ivMax >= e.size {
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);             
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            var a = st.addEntry(rname, iv.size, XType);
            //[i in iv.a.domain] a.a[i] = e.a[iv.a[i]]; // bounds check iv[i] against e.a.domain?
            ref a2 = e.a;
            ref iva = iv.a;
            ref aa = a.a;
            forall (a1,idx) in zip(aa,iva) with (var agg = newSrcAggregator(XType)) {
              agg.copy(a1,a2[idx]);
            }
            a.max_bits = e.max_bits;
            var repMsg =  "created " + st.attrib(rname);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        // gather indexing by integer index vector
        proc ivUInt64Helper(type XType): MsgTuple throws {
            var e = toSymEntry(gX,XType);
            var iv = toSymEntry(gIV,uint);
            if (e.size == 0) && (iv.size == 0) {
                var a = st.addEntry(rname, 0, XType);
                var repMsg = "created " + st.attrib(rname);
                imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            if ivMin < 0 {
                var errorMsg = "Error: %s: OOBindex %i < 0".format(pn,ivMin);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);               
            }
            if ivMax >= e.size {
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);             
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            var a = st.addEntry(rname, iv.size, XType);
            //[i in iv.a.domain] a.a[i] = e.a[iv.a[i]]; // bounds check iv[i] against e.a.domain?
            ref a2 = e.a;
            ref iva = iv.a;
            ref aa = a.a;
            forall (a1,idx) in zip(aa,iva) with (var agg = newSrcAggregator(XType)) {
              agg.copy(a1,a2[idx:int]);
            }
            a.max_bits = e.max_bits;
            var repMsg =  "created " + st.attrib(rname);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        
        // compression boolean indexing by bool index vector
        proc ivBoolHelper(type XType): MsgTuple throws {
            var e = toSymEntry(gX,XType);
            var truth = toSymEntry(gIV,bool);
            if (e.size == 0) && (truth.size == 0) {
                var a = st.addEntry(rname, 0, XType);
                var repMsg = "created " + st.attrib(rname);
                imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * truth.size);
            var iv: [truth.a.domain] int = (+ scan truth.a);
            var pop = iv[iv.size-1];
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                              "pop = %? last-scan = %?".format(pop,iv[iv.size-1]));

            var a = st.addEntry(rname, pop, XType);
            //[i in e.a.domain] if (truth.a[i] == true) {a.a[iv[i]-1] = e.a[i];}// iv[i]-1 for zero base index
            const ref ead = e.a.domain;
            ref ea = e.a;
            ref trutha = truth.a;
            ref aa = a.a;
            forall (i, eai) in zip(ead, ea) with (var agg = newDstAggregator(XType)) {
              if (trutha[i] == true) {
                agg.copy(aa[iv[i]-1], eai);
              }
            }
            a.max_bits = e.max_bits;

            var repMsg = "created " + st.attrib(rname);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        
        select(gX.dtype, gIV.dtype) {
            when (DType.Int64, DType.Int64) {
                return ivInt64Helper(int);
            }
            when (DType.UInt64, DType.Int64) {
                return ivInt64Helper(uint);
            }
            when (DType.Int64, DType.UInt64) {
                return ivUInt64Helper(int);
            }
            when (DType.UInt64, DType.UInt64) {
                return ivUInt64Helper(uint);
            }
            when (DType.Int64, DType.Bool) {
                return ivBoolHelper(int);
            }
            when (DType.UInt64, DType.Bool) {
                return ivBoolHelper(uint);
            }
            when (DType.Float64, DType.Int64) {
                return ivInt64Helper(real);
            }
            when (DType.Float64, DType.UInt64) {
                return ivUInt64Helper(real);
            }
            when (DType.Float64, DType.Bool) {
                return ivBoolHelper(real);
            }
            when (DType.Bool, DType.Int64) {
                return ivInt64Helper(bool);
            }
            when (DType.Bool, DType.UInt64) {
                return ivUInt64Helper(bool);
            }
            when (DType.Bool, DType.Bool) {
                return ivBoolHelper(bool);
            }
            when (DType.BigInt, DType.Int64) {
                return ivInt64Helper(bigint);
            }
            when (DType.BigInt, DType.UInt64) {
                return ivUInt64Helper(bigint);
            }
            when (DType.BigInt, DType.Bool) {
                return ivBoolHelper(bigint);
            }
            otherwise {
                var errorMsg = notImplementedError(pn,
                                       "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+")");
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    /* setIntIndexToValue "a[int] = value" response to __setitem__(int, value) */
    @arkouda.registerND(cmd_prefix="[int]=val-")
    proc setIntIndexToValueMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array"),
              idx = msgArgs.get("idx").getTuple(nd),
              dtype = str2dtype(msgArgs.getValueOf("dtype")),
              valueArg = msgArgs.get("value");

        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                               "%s %s %? %s %s".format(cmd, name, idx, dtype2str(dtype), valueArg.getValue()));

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        proc setValue(type arrType, type valType): MsgTuple throws {
            var e = toSymEntry(gEnt, arrType, nd);
            const val = valueArg.toScalar(valType);
            e.a[(...idx)] = val:arrType;

            const repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        proc setBigintValue(type valType): MsgTuple throws {
            var e = toSymEntry(gEnt, bigint, nd),
                val = valueArg.toScalar(valType):bigint;
            if e.max_bits != -1 {
                var max_size = 1:bigint;
                max_size <<= e.max_bits;
                max_size -= 1;
                val &= max_size;
            }
            e.a[(...idx)] = val;

            const repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) do return setValue(int, int);
            when (DType.Int64, DType.UInt64) do return setValue(int, uint);
            when (DType.Int64, DType.Float64) do return setValue(int, real);
            when (DType.Int64, DType.Bool) do return setValue(int, bool);
            when (DType.UInt64, DType.Int64) do return setValue(uint, int);
            when (DType.UInt64, DType.UInt64) do return setValue(uint, uint);
            when (DType.UInt64, DType.Float64) do return setValue(uint, real);
            when (DType.UInt64, DType.Bool) do return setValue(uint, bool);
            when (DType.Float64, DType.Int64) do return setValue(real, int);
            when (DType.Float64, DType.UInt64) do return setValue(real, uint);
            when (DType.Float64, DType.Float64) do return setValue(real, real);
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real, nd);
                e.a[(...idx)] = if valueArg.getBoolValue() then 1.0 else 0.0;

                const repMsg = "%s success".format(pn);
                imLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when (DType.Bool, DType.Int64) do return setValue(bool, int);
            when (DType.Bool, DType.UInt64) do return setValue(bool, uint);
            when (DType.Bool, DType.Float64) do return setValue(bool, real);
            when (DType.Bool, DType.Bool) do return setValue(bool, bool);
            when (DType.BigInt, DType.BigInt) do return setBigintValue(bigint);
            when (DType.BigInt, DType.Int64) do return setBigintValue(int);
            when (DType.BigInt, DType.UInt64) do return setBigintValue(uint);
            when (DType.BigInt, DType.Bool) do return setBigintValue(bool);
            otherwise {
                const errorMsg = notImplementedError(pn,
                                    "("+dtype2str(gEnt.dtype)+","+dtype2str(dtype)+")");
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    /* setPdarrayIndexToValue "a[pdarray] = value" response to __setitem__(pdarray, value) */
    proc setPdarrayIndexToValueMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        const iname = msgArgs.getValueOf("idx");
        var gX: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gIV: borrowed GenSymEntry = getGenericTypedArrayEntry(iname, st);
        const dtype = if gX.dtype == DType.BigInt then DType.BigInt else str2dtype(msgArgs.getValueOf("dtype"));
        var value = msgArgs.getValueOf("value");
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "cmd: %s gX: %s gIV: %s value: %s".format(cmd,st.attrib(name),
                                        st.attrib(iname),value));

        // scatter indexing by integer index vector
        proc ivInt64Helper(type Xtype, type dtype): MsgTuple throws {
            var e = toSymEntry(gX,Xtype);
            var iv = toSymEntry(gIV,int);
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            if ivMin < 0 {
                var errorMsg = "Error: %s: OOBindex %i < 0".format(pn,ivMin);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            if ivMax >= e.size {
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            if isBool(dtype) {
                value = value.replace("True","true"); // chapel to python bool
                value = value.replace("False","false"); // chapel to python bool
            }
            var val = try! value:dtype;
            // [i in iv.a] e.a[i] = val;
            ref iva = iv.a;
            ref ea = e.a;
            forall i in iva with (var agg = newDstAggregator(dtype), var locVal = val) {
              agg.copy(ea[i],locVal);
            }
            var repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        // scatter indexing by unsigned integer index vector
        proc ivUInt64Helper(type Xtype, type dtype): MsgTuple throws {
            var e = toSymEntry(gX,Xtype);
            var iv = toSymEntry(gIV,uint);
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            if ivMin < 0 {
                var errorMsg = "Error: %s: OOBindex %i < 0".format(pn,ivMin);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            if ivMax >= e.size {
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            if isBool(dtype) {
                value = value.replace("True","true"); // chapel to python bool
                value = value.replace("False","false"); // chapel to python bool
            }
            var val = try! value:dtype;
            // [i in iv.a] e.a[i] = val;
            ref iva = iv.a;
            ref ea = e.a;
            forall i in iva with (var agg = newDstAggregator(dtype), var locVal = val) {
              agg.copy(ea[i:int],locVal);
            }
            var repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        // expansion boolean indexing by bool index vector
        proc ivBoolHelper(type Xtype, type dtype): MsgTuple throws {
            var e = toSymEntry(gX,Xtype);
            var truth = toSymEntry(gIV,bool);
            if (e.size != truth.size) {
                var errorMsg = "Error: %s: bool iv must be same size %i != %i".format(pn,e.size,
                                                                                    truth.size);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            if isBool(dtype) {
                value = value.replace("True","true"); // chapel to python bool
                value = value.replace("False","false"); // chapel to python bool
            }
            var val = try! value:dtype;
            const ref ead = e.a.domain;
            ref ea = e.a;
            ref trutha = truth.a;
            forall i in ead with (var agg = newDstAggregator(dtype), var locVal = val) {
              if (trutha[i]) {
                agg.copy(ea[i],locVal);
              }
            }

            var repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select(gX.dtype, gIV.dtype, dtype) {
            when (DType.Int64, DType.Int64, DType.Int64) {
                return ivInt64Helper(int, int);
            }
            when (DType.Int64, DType.UInt64, DType.Int64) {
                return ivUInt64Helper(int, int);
            }
            when (DType.Int64, DType.Bool, DType.Int64) {
                return ivBoolHelper(int, int);
            }
            when (DType.UInt64, DType.Int64, DType.UInt64) {
                return ivInt64Helper(uint, uint);
            }
            when (DType.UInt64, DType.UInt64, DType.UInt64) {
                return ivUInt64Helper(uint, uint);
            }
            when (DType.UInt64, DType.Bool, DType.UInt64) {
                return ivBoolHelper(uint, uint);
            }
            when (DType.Float64, DType.Int64, DType.Float64) {
                return ivInt64Helper(real, real);
            }
            when (DType.Float64, DType.UInt64, DType.Float64) {
                return ivUInt64Helper(real, real);
            }
            when (DType.Float64, DType.Bool, DType.Float64) {
                return ivBoolHelper(real, real);
            }
            when (DType.Bool, DType.Int64, DType.Bool) {
                return ivInt64Helper(bool, bool);
            }
            when (DType.Bool, DType.UInt64, DType.Bool) {
                return ivUInt64Helper(bool, bool);
            }
            when (DType.Bool, DType.Bool, DType.Bool) {
                return ivBoolHelper(bool, bool);
            }
            when (DType.BigInt, DType.Int64, DType.BigInt) {
                return ivInt64Helper(bigint, bigint);
            }
            when (DType.BigInt, DType.UInt64, DType.BigInt) {
                return ivUInt64Helper(bigint, bigint);
            }
            when (DType.BigInt, DType.Bool, DType.BigInt) {
                return ivBoolHelper(bigint, bigint);
            }
            otherwise {
                var errorMsg = notImplementedError(pn,
                      "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+","+dtype2str(dtype)+")");
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    /* setPdarrayIndexToPdarray "a[pdarray] = pdarray" response to __setitem__(pdarray, pdarray) */
    proc setPdarrayIndexToPdarrayMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        const iname = msgArgs.getValueOf("idx");
        const yname = msgArgs.getValueOf("value");

        var gX: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gIV: borrowed GenSymEntry = getGenericTypedArrayEntry(iname, st);
        var gY: borrowed GenSymEntry = getGenericTypedArrayEntry(yname, st);

        if logLevel == LogLevel.DEBUG {
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "cmd: %s gX: %? gIV: %? gY: %?".format(
                                              cmd, st.attrib(name), st.attrib(iname),
                                              st.attrib(yname)));
        }

        // add check for IV to be dtype of int64 or bool

        // scatter indexing by an integer index vector
        proc ivInt64Helper(type t): MsgTuple throws {
            // add check to make sure IV and Y are same size
            if (gIV.size != gY.size) {
                var errorMsg = "Error: %s: size mismatch %i %i".format(pn,gIV.size,gY.size);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            var iv = toSymEntry(gIV,int);
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            var y = toSymEntry(gY,t);
            if ivMin < 0 {
                var errorMsg = "Error: %s: OOBindex %i < 0".format(pn,ivMin);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            if ivMax >= gX.size {
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMax,gX.size-1);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);           
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            ref iva = iv.a;
            ref ya = y.a;
            if gX.dtype == DType.BigInt {
                var e = toSymEntry(gX, bigint);
                // NOTE y.etype will never be real when gX.dtype is bigint, but the compiler doesn't know that
                var tmp = if y.etype == bigint then ya else if y.etype == real then ya:int:bigint else ya:bigint;
                ref ea = e.a;
                forall (i,v) in zip(iva,tmp) with (var agg = newDstAggregator(bigint)) {
                    agg.copy(ea[i],v);
                }
            }
            else {
                var e = toSymEntry(gX,t);
                ref ea = e.a;
                forall (i,v) in zip(iva,ya) with (var agg = newDstAggregator(t)) {
                    agg.copy(ea[i],v);
                }
            }
            var repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        // scatter indexing by unsigned integer index vector
        proc ivUInt64Helper(type t): MsgTuple throws {
            // add check to make sure IV and Y are same size
            if (gIV.size != gY.size) {
                var errorMsg = "Error: %s: size mismatch %i %i".format(pn,gIV.size,gY.size);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            var iv = toSymEntry(gIV,uint);
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            var y = toSymEntry(gY,t);
            if ivMin < 0 {
                var errorMsg = "Error: %s: OOBindex %i < 0".format(pn,ivMin);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            if ivMax >= gX.size {
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMax,gX.size-1);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);           
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            ref iva = iv.a;
            ref ya = y.a;
            if gX.dtype == DType.BigInt {
                var e = toSymEntry(gX, bigint);
                // NOTE y.etype will never be real when gX.dtype is bigint, but the compiler doesn't know that
                var tmp = if y.etype == bigint then ya else if y.etype == real then ya:int:bigint else ya:bigint;
                ref ea = e.a;
                forall (i,v) in zip(iva,tmp) with (var agg = newDstAggregator(bigint)) {
                    agg.copy(ea[i:int],v);
                }
            }
            else {
                var e = toSymEntry(gX,t);
                ref ea = e.a;
                forall (i,v) in zip(iva,ya) with (var agg = newDstAggregator(t)) {
                    agg.copy(ea[i:int],v);
                }
            }
            var repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        // expansion indexing by a bool index vector
        proc ivBoolHelper(type t): MsgTuple throws {
            // add check to make sure IV and Y are same size
            if (gIV.size != gX.size) {
                var errorMsg = "Error: %s: size mismatch %i %i".format(pn,gIV.size,gX.size);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            var truth = toSymEntry(gIV,bool);
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * truth.size);
            var iv: [truth.a.domain] int = (+ scan truth.a);
            var pop = iv[iv.size-1];
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                        "pop = %? last-scan = %?".format(pop,iv[iv.size-1]));
            var y = toSymEntry(gY,t);
            if (y.size != pop) {
                var errorMsg = "Error: %s: pop size mismatch %i %i".format(pn,pop,y.size);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            ref ya = y.a;
            ref trutha = truth.a;
            if gX.dtype == DType.BigInt {
                var e = toSymEntry(gX, bigint);
                // NOTE y.etype will never be real when gX.dtype is bigint, but the compiler doesn't know that
                var tmp = if y.etype == bigint then ya else if y.etype == real then ya:int:bigint else ya:bigint;
                ref ea = e.a;
                const ref ead = ea.domain;
                forall (eai, i) in zip(ea, ead) with (var agg = newSrcAggregator(bigint)) {
                    if trutha[i] {
                        agg.copy(eai,tmp[iv[i]-1]);
                    }
                }
            }
            else {
                var e = toSymEntry(gX,t);
                const ref ead = e.a.domain;
                ref ea = e.a;
                forall (eai, i) in zip(ea, ead) with (var agg = newSrcAggregator(t)) {
                    if trutha[i] {
                        agg.copy(eai,ya[iv[i]-1]);
                    }
                }
            }
            var repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select(gX.dtype, gIV.dtype, gY.dtype) {
            when (DType.Int64, DType.Int64, DType.Int64) {
                return ivInt64Helper(int);
            }
            when (DType.Int64, DType.UInt64, DType.Int64) {
                return ivUInt64Helper(int);
            }
            when (DType.Int64, DType.Bool, DType.Int64) {
                return ivBoolHelper(int);
            }
            when (DType.UInt64, DType.Int64, DType.UInt64) {
                return ivInt64Helper(uint);
            }
            when (DType.UInt64, DType.UInt64, DType.UInt64) {
                return ivUInt64Helper(uint);
            }
            when (DType.UInt64, DType.Bool, DType.UInt64) {
                return ivBoolHelper(uint);
            }
            when (DType.Float64, DType.Int64, DType.Float64) {
                return ivInt64Helper(real);
            }
            when (DType.Float64, DType.UInt64, DType.Float64) {
                return ivUInt64Helper(real);
            }
            when (DType.Float64, DType.Bool, DType.Float64) {
                return  ivBoolHelper(real);
            }
            when (DType.Bool, DType.Int64, DType.Bool) {
                return ivInt64Helper(bool);
            }
            when (DType.Bool, DType.UInt64, DType.Bool) {
                return ivUInt64Helper(bool);
            }
            when (DType.Bool, DType.Bool, DType.Bool) {
                return ivBoolHelper(bool);
            }
            when (DType.BigInt, DType.Int64, DType.BigInt) {
                return ivInt64Helper(bigint);
            }
            when (DType.BigInt, DType.Int64, DType.Int64) {
                return ivInt64Helper(int);
            }
            when (DType.BigInt, DType.Int64, DType.UInt64) {
                return ivInt64Helper(uint);
            }
            when (DType.BigInt, DType.UInt64, DType.BigInt) {
                return ivUInt64Helper(bigint);
            }
            when (DType.BigInt, DType.UInt64, DType.Int64) {
                return ivUInt64Helper(int);
            }
            when (DType.BigInt, DType.UInt64, DType.UInt64) {
                return ivUInt64Helper(uint);
            }
            when (DType.BigInt, DType.Bool, DType.BigInt) {
                return ivBoolHelper(bigint);
            }
            when (DType.BigInt, DType.Bool, DType.Int64) {
                return ivBoolHelper(int);
            }
            when (DType.BigInt, DType.Bool, DType.UInt64) {
                return ivBoolHelper(uint);
            }
            otherwise {
                var errorMsg = notImplementedError(pn,
                     "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+","+dtype2str(gY.dtype)+")");
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);                     
            }
        }
    }

    /* setSliceIndexToValue "a[slice] = value" response to __setitem__(slice, value) */
    @arkouda.registerND(cmd_prefix="[slice]=val-")
    proc setSliceIndexToValueMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        if nd == 1 then return setSliceIndexToValue1DFast(cmd, msgArgs, st);

        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("array"),
              starts = msgArgs.get("starts").getTuple(nd),
              stops = msgArgs.get("stops").getTuple(nd),
              strides = msgArgs.get("strides").getTuple(nd),
              dtype = str2dtype(msgArgs.getValueOf("dtype")),
              valueArg = msgArgs.get("value");

        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "%s %s %? %? %? %s %s".format(cmd, name, starts, stops, strides,
                                  dtype2str(dtype), valueArg.getValue()));

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        var sliceRanges: nd * range(strides=strideKind.any);
        for param dim in 0..<nd do
            sliceRanges[dim] = convertSlice(starts[dim], stops[dim], strides[dim]);
        const sliceDom = {(...sliceRanges)};

        proc sliceAssign(type arrType, type valType): MsgTuple throws {
            var e = toSymEntry(gEnt, arrType, nd);
            const value = valueArg.toScalar(valType);
            e.a[sliceDom] = value:arrType;

            const repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        proc sliceAssignBigint(type valType): MsgTuple throws {
            var e = toSymEntry(gEnt, bigint, nd),
                value = valueArg.toScalar(valType):bigint;
            if e.max_bits != -1 {
                var max_size = 1:bigint;
                max_size <<= e.max_bits;
                max_size -= 1;
                value &= max_size;
            }
            e.a[sliceDom] = value;

            const repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) do return sliceAssign(int, int);
            when (DType.Int64, DType.UInt64) do return sliceAssign(int, uint);
            when (DType.Int64, DType.Float64) do return sliceAssign(int, real);
            when (DType.Int64, DType.Bool) do return sliceAssign(int, bool);
            when (DType.UInt64, DType.Int64) do return sliceAssign(uint, int);
            when (DType.UInt64, DType.UInt64) do return sliceAssign(uint, uint);
            when (DType.UInt64, DType.Float64) do return sliceAssign(uint, real);
            when (DType.UInt64, DType.Bool) do return sliceAssign(uint, bool);
            when (DType.Float64, DType.Int64) do return sliceAssign(real, int);
            when (DType.Float64, DType.UInt64) do return sliceAssign(real, uint);
            when (DType.Float64, DType.Float64) do return sliceAssign(real, real);
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real, nd);
                e.a[sliceDom] = if valueArg.getBoolValue() then 1.0 else 0.0;

                const repMsg = "%s success".format(pn);
                imLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when (DType.Bool, DType.Int64) do return sliceAssign(bool, int);
            when (DType.Bool, DType.UInt64) do return sliceAssign(bool, uint);
            when (DType.Bool, DType.Float64) do return sliceAssign(bool, real);
            when (DType.Bool, DType.Bool) do return sliceAssign(bool, bool);
            when (DType.BigInt, DType.BigInt) do return sliceAssignBigint(bigint);
            when (DType.BigInt, DType.Int64) do return sliceAssignBigint(int);
            when (DType.BigInt, DType.UInt64) do return sliceAssignBigint(uint);
            when (DType.BigInt, DType.Bool) do return sliceAssignBigint(bool);
            otherwise {
                const errorMsg = notImplementedError(pn,
                                        "("+dtype2str(gEnt.dtype)+","+dtype2str(dtype)+")");
                imLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    proc setSliceIndexToValue1DFast(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        const start = msgArgs.get("start").getIntValue();
        const stop = msgArgs.get("stop").getIntValue();
        const stride = msgArgs.get("stride").getIntValue();
        const dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var slice: range(strides=strideKind.any) = convertSlice(start, stop, stride);
        var value = msgArgs.get("value");

        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "%s %s %i %i %i %s %s".format(cmd, name, start, stop, stride, 
                                  dtype2str(dtype), value.getValue()));
        
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        
        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var val = value.getIntValue();
                e.a[slice] = val;
            }
            when (DType.Int64, DType.UInt64) {
                var e = toSymEntry(gEnt,int);
                var val = value.getUIntValue();
                e.a[slice] = val:int;
            }
            when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val = value.getRealValue();
                e.a[slice] = val:int;
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                var val = value.getBoolValue();
                e.a[slice] = val:int;
            }
            when (DType.UInt64, DType.Int64) {
                var e = toSymEntry(gEnt,uint);
                var val = value.getIntValue();
                e.a[slice] = val:uint;
            }
            when (DType.UInt64, DType.UInt64) {
                var e = toSymEntry(gEnt,uint);
                var val = value.getUIntValue();
                e.a[slice] = val:uint;
            }
            when (DType.UInt64, DType.Float64) {
                var e = toSymEntry(gEnt,uint);
                var val = value.getRealValue();
                e.a[slice] = val:uint;
            }
            when (DType.UInt64, DType.Bool) {
                var e = toSymEntry(gEnt,uint);
                var val = value.getBoolValue();
                e.a[slice] = val:uint;
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real);
                var val = value.getIntValue();
                e.a[slice] = val;
            }
            when (DType.Float64, DType.UInt64) {
                var e = toSymEntry(gEnt,real);
                var val = value.getUIntValue();
                e.a[slice] = val:real;
            }
            when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var val = value.getRealValue();
                e.a[slice] = val;
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real);
                var b = value.getBoolValue();
                var val:real;
                if b {val = 1.0;} else {val = 0.0;}
                e.a[slice] = val;
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool);
                var val = value.getIntValue();
                e.a[slice] = val:bool;
            }
            when (DType.Bool, DType.UInt64) {
                var e = toSymEntry(gEnt,bool);
                var val = value.getUIntValue();
                e.a[slice] = val:bool;
            }
            when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,bool);
                var val = value.getRealValue();
                e.a[slice] = val:bool;
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                var val = value.getBoolValue();
                e.a[slice] = val;
            }
            when (DType.BigInt, DType.BigInt) {
                var e = toSymEntry(gEnt,bigint);
                var val = value.getBigIntValue();
                if e.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= e.max_bits;
                    max_size -= 1;
                    val &= max_size;
                }
                e.a[slice] = val;
             }
            when (DType.BigInt, DType.Int64) {
                var e = toSymEntry(gEnt,bigint);
                var val = value.getIntValue():bigint;
                if e.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= e.max_bits;
                    max_size -= 1;
                    val &= max_size;
                }
                e.a[slice] = val;
             }
            when (DType.BigInt, DType.UInt64) {
                var e = toSymEntry(gEnt,bigint);
                var val = value.getUIntValue():bigint;
                if e.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= e.max_bits;
                    max_size -= 1;
                    val &= max_size;
                }
                e.a[slice] = val;
             }
            when (DType.BigInt, DType.Bool) {
                var e = toSymEntry(gEnt,bigint);
                var val = value.getBoolValue():bigint;
                if e.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= e.max_bits;
                    max_size -= 1;
                    val &= max_size;
                }
                e.a[slice] = val;
             }
            otherwise {
                var errorMsg = notImplementedError(pn,
                                        "("+dtype2str(gEnt.dtype)+","+dtype2str(dtype)+")");
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);                                         
            }
        }

        repMsg = "%s success".format(pn);
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    }

    @arkouda.registerND(cmd_prefix="[slice]=pdarray-")
    proc setSliceIndexToPdarrayMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        // take simplified path for 1D case
        if nd == 1 then return setSliceIndexToPdarrayMsg1D(cmd, msgArgs, st);

        param pn = Reflection.getRoutineName();
        const starts = msgArgs.get("starts").getTuple(nd),
              stops = msgArgs.get("stops").getTuple(nd),
              strides = msgArgs.get("strides").getTuple(nd),
              name = msgArgs.getValueOf("array"),
              yname = msgArgs.getValueOf("value");

        var sliceRanges: nd * range(strides=strideKind.any);
        for param dim in 0..<nd do
            sliceRanges[dim] = convertSlice(starts[dim], stops[dim], strides[dim]);
        const sliceDom = {(...sliceRanges)};

        imLogger.debug(getModuleName(),pn,getLineNumber(),
                       "%s into: '%s' over domain: '%?' from: %s''"
                       .format(cmd, name, sliceDom, yname));

        var gX: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st),
            gY: borrowed GenSymEntry = getGenericTypedArrayEntry(yname, st);

        proc sliceAssignHelper(type xt, type yt, param adjustMaxSize=false): MsgTuple throws {
            // note 'value'/'y' needs to be expanded to match 'array'/'x's rank before
            // calling this command
            var ex = toSymEntry(gX,xt,nd);
            const ey = toSymEntry(gY,yt,nd);

            // ensure the slice assignment is valid
            for dim in 0..<nd {
                if ey.tupShape[dim] != sliceDom.dim[dim].size {
                    const errMsg = "shape of slice does not match array in dimension %i".format(dim) +
                                    " (%i != %i)".format(ey.tupShape[dim], sliceDom.dim[dim].size);
                    imLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
                    return new MsgTuple(errMsg, MsgType.ERROR);
                }
                if sliceDom.dim[dim].low < 0 || sliceDom.dim[dim].high > ex.tupShape[dim] {
                    const errMsg = "slice indices out of bounds in dimension %i".format(dim) +
                                   " (%i..%i not in 0..<%i)".format(sliceDom.dim[dim].low,
                                                                sliceDom.dim[dim].high, ex.tupShape[dim]);
                    imLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
                    return new MsgTuple(errMsg, MsgType.ERROR);
                }
            }

            // adjust y's max size for bigint arrays
            if adjustMaxSize {
                var ya = ey.a:bigint;
                if ex.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= ex.max_bits;
                    max_size -= 1;
                    forall y in ya with (const local_max_size = max_size) {
                        y &= local_max_size;
                    }
                }

                ex.a[sliceDom] = ya;
            } else {
                // otherwise, just assign the values
                ex.a[sliceDom] = ey.a:xt;
            }

            const repMsg = "%s success".format(pn);
            imLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select (gX.dtype, gY.dtype) {
            when (DType.Int64, DType.Int64) do return sliceAssignHelper(int, int);
            when (DType.Int64, DType.UInt64) do return sliceAssignHelper(int, uint);
            when (DType.Int64, DType.Float64) do return sliceAssignHelper(int, real);
            when (DType.Int64, DType.Bool) do return sliceAssignHelper(int, bool);
            when (DType.UInt64, DType.Int64) do return sliceAssignHelper(uint, int);
            when (DType.UInt64, DType.UInt64) do return sliceAssignHelper(uint, uint);
            when (DType.UInt64, DType.Float64) do return sliceAssignHelper(uint, real);
            when (DType.UInt64, DType.Bool) do return sliceAssignHelper(uint, bool);
            when (DType.Float64, DType.Int64) do return sliceAssignHelper(real, int);
            when (DType.Float64, DType.UInt64) do return sliceAssignHelper(real, uint);
            when (DType.Float64, DType.Float64) do return sliceAssignHelper(real, real);
            when (DType.Float64, DType.Bool) do return sliceAssignHelper(real, bool);
            when (DType.Bool, DType.Int64) do return sliceAssignHelper(bool, int);
            when (DType.Bool, DType.UInt64) do return sliceAssignHelper(bool, uint);
            when (DType.Bool, DType.Float64) do return sliceAssignHelper(bool, real);
            when (DType.Bool, DType.Bool) do return sliceAssignHelper(bool, bool);
            when (DType.BigInt, DType.BigInt) do return sliceAssignHelper(bigint, bigint, true);
            when (DType.BigInt, DType.Int64) do return sliceAssignHelper(bigint, int, true);
            when (DType.BigInt, DType.UInt64) do return sliceAssignHelper(bigint, uint, true);
            when (DType.BigInt, DType.Bool) do return sliceAssignHelper(bigint, bool, true);
            otherwise {
                const errorMsg = notImplementedError(pn,
                                        "("+dtype2str(gX.dtype)+","+dtype2str(gY.dtype)+")");
                imLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    /* setSliceIndexToPdarray "a[slice] = pdarray" response to __setitem__(slice, pdarray) */
    proc setSliceIndexToPdarrayMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const start = msgArgs.get("starts").getIntValue();
        const stop = msgArgs.get("stops").getIntValue();
        const stride = msgArgs.get("strides").getIntValue();
        var slice: range(strides=strideKind.any);

        const name = msgArgs.getValueOf("array");
        const yname = msgArgs.getValueOf("value");

        // convert python slice to chapel slice
        // backwards iteration with negative stride
        if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
        // forward iteration with positive stride
        else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
        // BAD FORM start < stop and stride is negative
        else {slice = 1..0;}

        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                        "%s %s %i %i %i %s".format(cmd, name, start, stop, stride, yname));

        var gX: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gY: borrowed GenSymEntry = getGenericTypedArrayEntry(yname, st);

        // add check to make sure IV and Y are same size
        if (slice.size != gY.size) {      
            var errorMsg = "%s: size mismatch %i %i".format(pn,slice.size, gY.size);
            imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);        
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }

        select (gX.dtype, gY.dtype) {
            when (DType.Int64, DType.Int64) {
                var x = toSymEntry(gX,int);
                var y = toSymEntry(gY,int);
                x.a[slice] = y.a;
            }
            when (DType.Int64, DType.UInt64) {
                var x = toSymEntry(gX,int);
                var y = toSymEntry(gY,uint);
                x.a[slice] = y.a:int;
            }
            when (DType.Int64, DType.Float64) {
                var x = toSymEntry(gX,int);
                var y = toSymEntry(gY,real);
                x.a[slice] = y.a:int;
            }
            when (DType.Int64, DType.Bool) {
                var x = toSymEntry(gX,int);
                var y = toSymEntry(gY,bool);
                x.a[slice] = y.a:int;
            }
            when (DType.UInt64, DType.Int64) {
                var x = toSymEntry(gX,uint);
                var y = toSymEntry(gY,int);
                x.a[slice] = y.a:uint;
            }
            when (DType.UInt64, DType.UInt64) {
                var x = toSymEntry(gX,uint);
                var y = toSymEntry(gY,uint);
                x.a[slice] = y.a:uint;
            }
            when (DType.UInt64, DType.Float64) {
                var x = toSymEntry(gX,uint);
                var y = toSymEntry(gY,real);
                x.a[slice] = y.a:uint;
            }
            when (DType.UInt64, DType.Bool) {
                var x = toSymEntry(gX,uint);
                var y = toSymEntry(gY,bool);
                x.a[slice] = y.a:uint;
            }
            when (DType.Float64, DType.Int64) {
                var x = toSymEntry(gX,real);
                var y = toSymEntry(gY,int);
                x.a[slice] = y.a:real;
            }
            when (DType.Float64, DType.UInt64) {
                var x = toSymEntry(gX,real);
                var y = toSymEntry(gY,uint);
                x.a[slice] = y.a:real;
            }
            when (DType.Float64, DType.Float64) {
                var x = toSymEntry(gX,real);
                var y = toSymEntry(gY,real);
                x.a[slice] = y.a;
            }
            when (DType.Float64, DType.Bool) {
                var x = toSymEntry(gX,real);
                var y = toSymEntry(gY,bool);
                x.a[slice] = y.a:real;
            }
            when (DType.Bool, DType.Int64) {
                var x = toSymEntry(gX,bool);
                var y = toSymEntry(gY,int);
                x.a[slice] = y.a:bool;
            }
            when (DType.Bool, DType.UInt64) {
                var x = toSymEntry(gX,bool);
                var y = toSymEntry(gY,uint);
                x.a[slice] = y.a:bool;
            }
            when (DType.Bool, DType.Float64) {
                var x = toSymEntry(gX,bool);
                var y = toSymEntry(gY,real);
                x.a[slice] = y.a:bool;
            }
            when (DType.Bool, DType.Bool) {
                var x = toSymEntry(gX,bool);
                var y = toSymEntry(gY,bool);
                x.a[slice] = y.a;
            }
            when (DType.BigInt, DType.BigInt) {
                var x = toSymEntry(gX,bigint);
                var y = toSymEntry(gY,bigint);
                if x.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= x.max_bits;
                    max_size -= 1;
                    forall y in y.a with (var local_max_size = max_size) {
                        y &= local_max_size;
                    }
                }
                x.a[slice] = y.a;
             }
            when (DType.BigInt, DType.Int64) {
                var x = toSymEntry(gX,bigint);
                var y = toSymEntry(gY,int);
                var ya = y.a:bigint;
                if x.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= x.max_bits;
                    max_size -= 1;
                    forall y in ya with (var local_max_size = max_size) {
                        y &= local_max_size;
                    }
                }
                x.a[slice] = ya;
             }
            when (DType.BigInt, DType.UInt64) {
                var x = toSymEntry(gX,bigint);
                var y = toSymEntry(gY,uint);
                var ya = y.a:bigint;
                if x.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= x.max_bits;
                    max_size -= 1;
                    forall y in ya with (var local_max_size = max_size) {
                        y &= local_max_size;
                    }
                }
                x.a[slice] = ya;
             }
            when (DType.BigInt, DType.Bool) {
                var x = toSymEntry(gX,bigint);
                var y = toSymEntry(gY,bool);
                // TODO change once we can cast directly from bool to bigint
                var ya = y.a:int:bigint;
                if x.max_bits != -1 {
                    var max_size = 1:bigint;
                    max_size <<= x.max_bits;
                    max_size -= 1;
                    forall y in ya with (var local_max_size = max_size) {
                        y &= local_max_size;
                    }
                }
                x.a[slice] = ya;
             }
            otherwise {
                var errorMsg = notImplementedError(pn,
                                     "("+dtype2str(gX.dtype)+","+dtype2str(gY.dtype)+")");
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        repMsg = "%s success".format(pn);
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    @arkouda.registerND
    proc takeAlongAxisMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const name = msgArgs.getValueOf("x"),
              idxName = msgArgs.getValueOf("indices"),
              axis = msgArgs.get("axis").getIntValue();

        const rname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st),
            gIEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(idxName, st);

        proc doIndex(type eltType, type idxType, param doCast: bool): MsgTuple throws {
            var x = toSymEntry(gEnt, eltType, nd),
                idx = toSymEntry(gIEnt, idxType, 1),
                y = st.addEntry(rname, (...x.tupShape), eltType);

            if x.tupShape[axis] != idx.size {
                const errMsg = "Error: %s: index array length (%i) does not match x's length (%i) along the provided axis (%i)"
                    .format(pn, idx.size, x.tupShape[axis], axis);
                imLogger.error(getModuleName(),pn,getLineNumber(),errMsg);
                return new MsgTuple(errMsg, MsgType.ERROR);
            }

            const minIdx = min reduce idx.a,
                  maxIdx = max reduce idx.a;

            if minIdx < 0 || maxIdx >= x.a.shape(axis) {
                const errMsg = "Error: %s: index array contains out-of-bounds indices (%i, %i) along the provided axis (%i)"
                    .format(pn, minIdx, maxIdx, axis);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errMsg);
                return new MsgTuple(errMsg, MsgType.ERROR);
            }

            ref xa = x.a;
            ref ya = y.a;
            ref idxa = idx.a;

            if nd == 1 {
                forall i in idx.a.domain with (var agg = newSrcAggregator(eltType)) {
                    if doCast
                        then agg.copy(ya[i], xa[idxa[i]:int]);
                        else agg.copy(ya[i], xa[idxa[i]]);
                }
            } else {
                for sliceIdx in domOffAxis(x.a.domain, axis) {
                    forall i in idx.a.domain with (var agg = newSrcAggregator(eltType)) {
                        var yIdx = sliceIdx,
                            xIdx = sliceIdx;
                        yIdx[axis] = i;
                        xIdx[axis] = if doCast then idxa[i]:int else idxa[i];
                        agg.copy(ya[yIdx], xa[xIdx]);
                    }
                }
            }
            y.max_bits = x.max_bits;

            const repMsg = "created " + st.attrib(rname);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        select (gEnt.dtype, gIEnt.dtype) {
            when (DType.Int64, DType.Int64) do return doIndex(int, int, false);
            when (DType.Int64, DType.UInt64) do return doIndex(int, uint, true);
            when (DType.UInt64, DType.Int64) do return doIndex(uint, int, false);
            when (DType.UInt64, DType.UInt64) do return doIndex(uint, uint, true);
            when (DType.Float64, DType.Int64) do return doIndex(real, int, false);
            when (DType.Float64, DType.UInt64) do return doIndex(real, uint, true);
            when (DType.Bool, DType.Int64) do return doIndex(bool, int, false);
            when (DType.Bool, DType.UInt64) do return doIndex(bool, uint, true);
            when (DType.BigInt, DType.Int64) do return doIndex(bigint, int, false);
            when (DType.BigInt, DType.UInt64) do return doIndex(bigint, uint, true);
            otherwise {
                const errMsg = notImplementedError(Reflection.getRoutineName(),
                    "("+dtype2str(gEnt.dtype)+","+dtype2str(gIEnt.dtype)+")");
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errMsg);
                return new MsgTuple(errMsg, MsgType.ERROR);
            }
        }
    }

    use CommandMap;
    registerFunction("arrayViewMixedIndex", arrayViewMixedIndexMsg, getModuleName());
    registerFunction("arrayViewIntIndexAssign", arrayViewIntIndexAssignMsg, getModuleName());
    registerFunction("[pdarray]", pdarrayIndexMsg, getModuleName());
    registerFunction("[pdarray]=val", setPdarrayIndexToValueMsg, getModuleName());
    registerFunction("[pdarray]=pdarray", setPdarrayIndexToPdarrayMsg, getModuleName());
}
