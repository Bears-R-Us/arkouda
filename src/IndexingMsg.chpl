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
    proc sliceIndex(const ref array: [?d] ?t, starts: d.rank*int, stops: d.rank*int, strides: d.rank*int, max_bits: int): SymEntry(t, d.rank) throws {
        var rngs: d.rank*range(strides=strideKind.any),
            outSizes: d.rank*int;
        for param dim in 0..<d.rank {
            rngs[dim] = convertSlice(starts[dim], stops[dim], strides[dim]);
            outSizes[dim] = rngs[dim].size;
        }

        const sliceDom = makeDistDom({(...rngs)});
        var arraySlice = makeDistArray((...outSizes), t);

        forall (elt,j) in zip(arraySlice, sliceDom) with (var agg = newSrcAggregator(t)) {
            agg.copy(elt,array[j]);
        }

        return new shared SymEntry(arraySlice, max_bits=max_bits);
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
            return st.insert(new shared SymEntry(ret, max_bits=a.max_bits));
        } else {
            return MsgTuple.error("Invalid index dimensions: %? for %iD array".format(idxDims, array_nd));
        }
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

    @arkouda.registerCommand("[int]=val")
    proc setIndexToValue(ref array: [?d] ?t, idx: d.rank*int, value: t, max_bits: int) {
        if t == bigint {
            var val_mb = value;
            if max_bits != -1 {
                var max_size = 1:bigint;
                max_size <<= max_bits;
                max_size -= 1;
                val_mb &= max_size;
            }
            array[idx] = val_mb;
        } else {
            array[idx] = value;
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

    @arkouda.registerCommand("[slice]=val")
    proc setSliceIndexToValue(ref array: [?d] ?t, starts: d.rank*int, stops: d.rank*int, strides: d.rank*int, value: t, max_bits: int) throws {
        var rngs: d.rank*range(strides=strideKind.any);
        for param dim in 0..<d.rank do
            rngs[dim] = convertSlice(starts[dim], stops[dim], strides[dim]);
        const sliceDom = makeDistDom({(...rngs)});

        if t == bigint {
            var val_mb = value;
            if max_bits != -1 {
                var max_size = 1:bigint;
                max_size <<= max_bits;
                max_size -= 1;
                val_mb &= max_size;
            }
            array[sliceDom] = val_mb;
        } else {
            array[sliceDom] = value;
        }
    }

    @arkouda.instantiateAndRegister(prefix="[slice]=pdarray")
    proc setSliceIndexToPdarray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
        type array_dtype_a,
        type array_dtype_b,
        param array_nd: int
    ): MsgTuple throws {
        var a = st[msgArgs["array"]]: SymEntry(array_dtype_a, array_nd);
        const starts = msgArgs["starts"].getTuple(array_nd),
              stops = msgArgs["stops"].getTuple(array_nd),
              strides = msgArgs["strides"].getTuple(array_nd),
              b = st[msgArgs["value"]]: SymEntry(array_dtype_b, array_nd);

        var sliceRanges: array_nd * range(strides=strideKind.any);
        for param dim in 0..<array_nd do
            sliceRanges[dim] = convertSlice(starts[dim], stops[dim], strides[dim]);
        const sliceDom = makeDistDom({(...sliceRanges)});

        if array_dtype_a == bigint {
            var bb = b.a:bigint;
            if a.max_bits != -1 {
                var max_size = 1:bigint;
                max_size <<= a.max_bits;
                max_size -= 1;
                forall x in bb with (const local_max_size = max_size) do
                    x &= local_max_size;
            }
            a.a[sliceDom] = bb;
        } else {
            a.a[sliceDom] = b.a:array_dtype_a;
        }

        return MsgTuple.success();
    }

    proc setSliceIndexToPdarray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
        type array_dtype_a,
        type array_dtype_b,
        param array_nd: int
    ): MsgTuple throws
        where array_dtype_a != bigint && array_dtype_b == bigint
    {
        return MsgTuple.error("Cannot assign bigint pdarray to slice of non-bigint pdarray");
    }

    proc setSliceIndexToPdarray(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
        type array_dtype_a,
        type array_dtype_b,
        param array_nd: int
    ): MsgTuple throws
        where array_dtype_a == bigint && isRealType(array_dtype_b)
    {
        return MsgTuple.error("Cannot assign float pdarray to slice of bigint pdarray");
    }

    @arkouda.instantiateAndRegister
    proc takeAlongAxis(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
        type array_dtype_x,
        type array_dtype_idx,
        param array_nd: int
    ): MsgTuple throws
        where isIntegralType(array_dtype_idx)
    {
        const x = st[msgArgs['x']]: SymEntry(array_dtype_x, array_nd),
              idx = st[msgArgs['indices']]: SymEntry(array_dtype_idx, 1),
              axis = msgArgs['axis'].getPositiveIntValue(array_nd);

        var y = makeDistArray((...x.tupShape), array_dtype_x);

        if x.tupShape[axis] != idx.size {
            return MsgTuple.error("index array length does not match x's length along the provided axis");
        }

        const minIdx = min reduce idx.a,
              maxIdx = max reduce idx.a;

        if minIdx < 0 || maxIdx >= x.a.shape(axis) {
            return MsgTuple.error(
                "index array contains out-of-bounds indices (%i, %i) along axis %i (0,%i)".format(minIdx, maxIdx, axis, x.a.shape(axis)-1)
            );
        }

        ref xa = x.a;
        ref idxa = idx.a;

        if array_nd == 1 {
            forall i in idx.a.domain with (var agg = newSrcAggregator(array_dtype_x)) {
                agg.copy(y[i], xa[idxa[i]:int]);
            }
        } else {
            for sliceIdx in domOffAxis(x.a.domain, axis) {
                forall i in idx.a.domain with (var agg = newSrcAggregator(array_dtype_x)) {
                    var yIdx = sliceIdx,
                        xIdx = sliceIdx;
                    yIdx[axis] = i;
                    xIdx[axis] = idxa[i]:int;
                    agg.copy(y[yIdx], xa[xIdx]);
                }
            }
        }

        return st.insert(new shared SymEntry(y, x.max_bits));
    }

    use CommandMap;
    registerFunction("arrayViewMixedIndex", arrayViewMixedIndexMsg, getModuleName());
    registerFunction("[pdarray]", pdarrayIndexMsg, getModuleName());
    registerFunction("[pdarray]=val", setPdarrayIndexToValueMsg, getModuleName());
    registerFunction("[pdarray]=pdarray", setPdarrayIndexToPdarrayMsg, getModuleName());
}
