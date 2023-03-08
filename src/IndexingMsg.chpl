module IndexingMsg
{
    use ServerConfig;
    use ServerErrorStrings;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    use CommAggregation;
    use BigInteger;

    use FileIO;
    use List;

    use Map;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const imLogger = new Logger(logLevel, logChannel);

    proc jsonToTuple(json: string, type t) throws {
        var f = openmem(); defer { ensureClose(f); }
        var w = f.writer();
        w.write(json);
        w.close();
        var r = f.reader();
        var tup: t;
        r.readf("%jt", tup);
        r.close();
        return tup;
    }

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

        var scaledCoords: [makeDistDom(+ reduce dims)] int;
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * dims.size);
        var offsets = (+ scan dims) - dims;

        forall i in 0..#typeCoords.size by 2 {
            select typeCoords[i] {
                when "int" {
                    scaledCoords[offsets[i/2]] = typeCoords[i+1]:int * dimProdEntry.a[i/2];
                }
                when "slice" {
                    var (start, stop, stride) = jsonToTuple(typeCoords[i+1], 3*int);
                    var slice: range(stridable=true) = convertSlice(start, stop, stride);
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
        var idxParam = new ParameterObj("idx", indiciesName, ObjectType.PDARRAY, "int");
        var subArgs = new MessageArgs(new list([arrParam, idxParam]));
        return pdarrayIndexMsg(cmd, subArgs, st);
    }

    /* arrayViewIntIndexMsg "av[int_list]" response to __getitem__(int_list) where av is an ArrayView */
    proc arrayViewIntIndexMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        const pdaName = msgArgs.getValueOf("base");
        const dimProdName = msgArgs.getValueOf("dim_prod");
        const coordsName = msgArgs.getValueOf("coords");
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "%s %s %s %s".format(cmd, pdaName, dimProdName, coordsName));

        var dimProd: borrowed GenSymEntry = getGenericTypedArrayEntry(dimProdName, st);
        var dimProdEntry = toSymEntry(dimProd, int);
        var coords: borrowed GenSymEntry = getGenericTypedArrayEntry(coordsName, st);

        var arrParam = msgArgs.get("base");
        arrParam.setKey("array");
        var idxParam = new ParameterObj("idx", "", ObjectType.VALUE, "");

        // multi-dim to 1D address calculation
        // (dimProd and coords are reversed on python side to account for row_major vs column_major)
        select (coords.dtype) {
            when (DType.Int64) {
                var coordsEntry = toSymEntry(coords, int);
                var idx = + reduce (dimProdEntry.a * coordsEntry.a);
                idxParam.setVal(idx:string);
                idxParam.setDType("int");
                var subArgs = new MessageArgs(new list([arrParam, idxParam]));
                return intIndexMsg(cmd, subArgs, st);
            }
            when (DType.UInt64) {
                var coordsEntry = toSymEntry(coords, uint);
                var idx = + reduce (dimProdEntry.a: uint * coordsEntry.a);
                idxParam.setVal(idx:string);
                idxParam.setDType("uint");
                var subArgs = new MessageArgs(new list([arrParam, idxParam]));
                return intIndexMsg(cmd, subArgs, st);
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
        var idxParam = new ParameterObj("idx", "", ObjectType.VALUE, "");

        // multi-dim to 1D address calculation
        // (dimProd and coords are reversed on python side to account for row_major vs column_major)
        select (coords.dtype) {
            when (DType.Int64) {
                var coordsEntry = toSymEntry(coords, int);
                var idx = + reduce (dimProdEntry.a * coordsEntry.a);
                idxParam.setVal(idx:string);
                idxParam.setDType("int");
                var subArgs = new MessageArgs(new list([arrParam, msgArgs.get("value"), msgArgs.get("dtype"), idxParam]));
                return setIntIndexToValueMsg(cmd, subArgs, st);
            }
            when (DType.UInt64) {
                var coordsEntry = toSymEntry(coords, uint);
                var idx = + reduce (dimProdEntry.a: uint * coordsEntry.a);
                idxParam.setVal(idx:string);
                idxParam.setDType("uint");
                var subArgs = new MessageArgs(new list([arrParam, msgArgs.get("value"), msgArgs.get("dtype"), idxParam]));
                return setIntIndexToValueMsg(cmd, subArgs, st);
            }
            otherwise {
                 var errorMsg = notImplementedError(pn, "("+dtype2str(coords.dtype)+")");
                 imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                 return new MsgTuple(errorMsg, MsgType.ERROR);
             }
        }
    }

    /* intIndex "a[int]" response to __getitem__(int) */
    proc intIndexMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var idx = msgArgs.get("idx").getIntValue();
        const name = msgArgs.getValueOf("array");
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                    "%s %s %i".format(cmd, name, idx));
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
         
        select (gEnt.dtype) {
             when (DType.Int64) {
                 var e = toSymEntry(gEnt, int);
                 repMsg = "item %s %t".format(dtype2str(e.dtype),e.a[idx]);

                 imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                 return new MsgTuple(repMsg, MsgType.NORMAL);  
             }
             when (DType.UInt64) {
               var e = toSymEntry(gEnt, uint);
                 repMsg = "item %s %t".format(dtype2str(e.dtype),e.a[idx]);

                 imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                 return new MsgTuple(repMsg, MsgType.NORMAL);  
             }
             when (DType.Float64) {
                 var e = toSymEntry(gEnt,real);
                 repMsg = "item %s %.17r".format(dtype2str(e.dtype),e.a[idx]);

                 imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                 return new MsgTuple(repMsg, MsgType.NORMAL); 
             }
             when (DType.Bool) {
                 var e = toSymEntry(gEnt,bool);
                 repMsg = "item %s %t".format(dtype2str(e.dtype),e.a[idx]);
                 repMsg = repMsg.replace("true","True"); // chapel to python bool
                 repMsg = repMsg.replace("false","False"); // chapel to python bool

                 imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                 return new MsgTuple(repMsg, MsgType.NORMAL); 
             }
             when (DType.BigInt) {
                 var e = toSymEntry(gEnt,bigint);
                 repMsg = "item %s %t".format(dtype2str(e.dtype),e.a[idx]);
                 imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                 return new MsgTuple(repMsg, MsgType.NORMAL);
             }
             otherwise {
                 var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
                 imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                 return new MsgTuple(errorMsg, MsgType.ERROR);                
             }
         }
    }

    /* convert python slice to chapel slice */
    proc convertSlice(start: int, stop: int, stride: int): range(stridable=true) {
        var slice: range(stridable=true);
        // backwards iteration with negative stride
        if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
        // forward iteration with positive stride
        else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
        // BAD FORM start < stop and stride is negative
        else {slice = 1..0;}
        return slice;
    }

    /* sliceIndex "a[slice]" response to __getitem__(slice) */
    proc sliceIndexMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const start = msgArgs.get("start").getIntValue();
        const stop = msgArgs.get("stop").getIntValue();
        const stride = msgArgs.get("stride").getIntValue();
        var slice: range(stridable=true) = convertSlice(start, stop, stride);

        // get next symbol name
        var rname = st.nextName();
        const name = msgArgs.getValueOf("array");
        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s pdarray to slice: %s start: %i stop: %i stride: %i slice: %t new name: %s".format(
                       cmd, st.attrib(name), start, stop, stride, slice, rname));

        proc sliceHelper(type t) throws {
            var e = toSymEntry(gEnt,t);
            var a = st.addEntry(rname, slice.size, t);
            ref ea = e.a;
            ref aa = a.a;
            forall (elt,j) in zip(aa, slice) with (var agg = newSrcAggregator(t)) {
              agg.copy(elt,ea[j]);
            }
            var repMsg = "created " + st.attrib(rname);
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        
        select(gEnt.dtype) {
            when (DType.Int64) {
                return sliceHelper(int);
            }
            when (DType.UInt64) {
                return sliceHelper(uint);
            }
            when (DType.Float64) {
                return sliceHelper(real);
            }
            when (DType.Bool) {
                return sliceHelper(bool);
            }
            when (DType.BigInt) {
                return sliceHelper(bigint);
            }
            otherwise {
                var errorMsg = notImplementedError(pn,dtype2str(gEnt.dtype));
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);              
            }
        }
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
                                           "cmd: %s name: %s gX: %t gIV: %t".format(
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
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMin,e.size-1);
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
                var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,ivMin,e.size-1);
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
                                              "pop = %t last-scan = %t".format(pop,iv[iv.size-1]));

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
    proc setIntIndexToValueMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        const idx = msgArgs.get("idx").getIntValue();
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var valueArg = msgArgs.get("value");
        
        imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                               "%s %s %i %s %s".format(cmd, name, idx, dtype2str(dtype), valueArg.getValue()));

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        select (gEnt.dtype, dtype) {
             when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var val = valueArg.getIntValue();
                e.a[idx] = val;
             }
             when (DType.Int64, DType.UInt64) {
                var e = toSymEntry(gEnt,int);
                var val = valueArg.getUIntValue();
                e.a[idx] = val:int;
             }
             when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val = valueArg.getRealValue();
                e.a[idx] = val:int;
             }
             when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                var val = valueArg.getBoolValue();
                e.a[idx] = val:int;
             }
             when (DType.UInt64, DType.Int64) {
                var e = toSymEntry(gEnt,uint);
                var val = valueArg.getIntValue();
                e.a[idx] = val:uint;
             }
             when (DType.UInt64, DType.UInt64) {
                var e = toSymEntry(gEnt,uint);
                var val = valueArg.getUIntValue();
                e.a[idx] = val;
             }
             when (DType.UInt64, DType.Float64) {
                var e = toSymEntry(gEnt,uint);
                var val = valueArg.getRealValue();
                e.a[idx] = val:uint;
             }
             when (DType.UInt64, DType.Bool) {
                var e = toSymEntry(gEnt,uint);
                var val = valueArg.getBoolValue();
                e.a[idx] = val:uint;
             }
             when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real);
                var val = valueArg.getIntValue();
                e.a[idx] = val;
             }
             when (DType.Float64, DType.UInt64) {
                var e = toSymEntry(gEnt,real);
                var val = valueArg.getUIntValue();
                e.a[idx] = val:real;
             }
             when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var val = valueArg.getRealValue();
                e.a[idx] = val;
             }
             when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real);
                var b = valueArg.getBoolValue();
                var val:real;
                if b {val = 1.0;} else {val = 0.0;}
                e.a[idx] = val;
             }
             when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool);
                var val = valueArg.getIntValue();
                e.a[idx] = val:bool;
             }
             when (DType.Bool, DType.UInt64) {
                var e = toSymEntry(gEnt,bool);
                var val = valueArg.getUIntValue();
                e.a[idx] = val:bool;
             }
             when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,bool);
                var val = valueArg.getRealValue();
                e.a[idx] = val:bool;
             }
             when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                var val = valueArg.getBoolValue();
                e.a[idx] = val;
             }
            when (DType.BigInt, DType.BigInt) {
                var e = toSymEntry(gEnt,bigint);
                var val = valueArg.getBigIntValue();
                if e.max_bits != -1 {
                    val.mod(val, e.max_bits);
                }
                e.a[idx] = val;
             }
            when (DType.BigInt, DType.Int64) {
                var e = toSymEntry(gEnt,bigint);
                var val = valueArg.getIntValue():bigint;
                if e.max_bits != -1 {
                    val.mod(val, e.max_bits);
                }
                e.a[idx] = val;
             }
            when (DType.BigInt, DType.UInt64) {
                var e = toSymEntry(gEnt,bigint);
                var val = valueArg.getUIntValue():bigint;
                if e.max_bits != -1 {
                    val.mod(val, e.max_bits);
                }
                e.a[idx] = val;
             }
            when (DType.BigInt, DType.Bool) {
                var e = toSymEntry(gEnt,bigint);
                // TODO change once we can cast directly from bool to bigint
                var val = valueArg.getBoolValue():int:bigint;
                if e.max_bits != -1 {
                    val.mod(val, e.max_bits);
                }
                e.a[idx] = val;
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
            forall i in iva with (var agg = newDstAggregator(dtype)) {
              agg.copy(ea[i],val);
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
            forall i in iva with (var agg = newDstAggregator(dtype)) {
              agg.copy(ea[i:int],val);
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
            forall i in ead with (var agg = newDstAggregator(dtype)) {
              if (trutha[i] == true) {
                agg.copy(ea[i],val);
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
                                             "cmd: %s gX: %t gIV: %t gY: %t".format(
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
                var tmp = if y.etype == bigint then ya else if (y.etype == bool || y.etype == real) then ya:int:bigint else ya:bigint;
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
                var tmp = if y.etype == bigint then ya else if (y.etype == bool || y.etype == real) then ya:int:bigint else ya:bigint;
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
            
            var e = toSymEntry(gX, bigint);
            var truth = toSymEntry(gIV,bool);
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * truth.size);
            var iv: [truth.a.domain] int = (+ scan truth.a);
            var pop = iv[iv.size-1];
            imLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                        "pop = %t last-scan = %t".format(pop,iv[iv.size-1]));
            var y = toSymEntry(gY,t);
            if (y.size != pop) {
                var errorMsg = "Error: %s: pop size mismatch %i %i".format(pn,pop,y.size);
                imLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg,MsgType.ERROR);
            }
            ref ya = y.a;
            if gX.dtype == DType.BigInt {
                // NOTE y.etype will never be real when gX.dtype is bigint, but the compiler doesn't know that
                var tmp = if y.etype == bigint then ya else if (y.etype == bool || y.etype == real) then ya:int:bigint else ya:bigint;
                ref ea = e.a;
                const ref ead = ea.domain;
                ref trutha = truth.a;
                forall (eai, i) in zip(ea, ead) with (var agg = newSrcAggregator(bigint)) {
                    if trutha[i] {
                        agg.copy(eai,tmp[iv[i]-1]);
                    }
                }
            }
            else {
                var e = toSymEntry(gX,t);
                var truth = toSymEntry(gIV,bool);
                const ref ead = e.a.domain;
                ref ea = e.a;
                ref trutha = truth.a;
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
    proc setSliceIndexToValueMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const name = msgArgs.getValueOf("array");
        const start = msgArgs.get("start").getIntValue();
        const stop = msgArgs.get("stop").getIntValue();
        const stride = msgArgs.get("stride").getIntValue();
        const dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var slice: range(stridable=true) = convertSlice(start, stop, stride);
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
                val.mod(val, e.max_bits);
                }
                e.a[slice] = val;
             }
            when (DType.BigInt, DType.Int64) {
                var e = toSymEntry(gEnt,bigint);
                var val = value.getIntValue():bigint;
                if e.max_bits != -1 {
                val.mod(val, e.max_bits);
                }
                e.a[slice] = val;
             }
            when (DType.BigInt, DType.UInt64) {
                var e = toSymEntry(gEnt,bigint);
                var val = value.getUIntValue():bigint;
                if e.max_bits != -1 {
                val.mod(val, e.max_bits);
                }
                e.a[slice] = val;
             }
            when (DType.BigInt, DType.Bool) {
                var e = toSymEntry(gEnt,bigint);
                // TODO change once we can cast directly from bool to bigint
                var val = value.getBoolValue():int:bigint;
                if e.max_bits != -1 {
                val.mod(val, e.max_bits);
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
    
    /* setSliceIndexToPdarray "a[slice] = pdarray" response to __setitem__(slice, pdarray) */
    proc setSliceIndexToPdarrayMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        const start = msgArgs.get("start").getIntValue();
        const stop = msgArgs.get("stop").getIntValue();
        const stride = msgArgs.get("stride").getIntValue();
        var slice: range(stridable=true);

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
                    y.a.mod(y.a, x.max_bits);
                }
                x.a[slice] = y.a;
             }
            when (DType.BigInt, DType.Int64) {
                var x = toSymEntry(gX,bigint);
                var y = toSymEntry(gY,int);
                var ya = y.a:bigint;
                if x.max_bits != -1 {
                    ya.mod(ya, x.max_bits);
                }
                x.a[slice] = ya;
             }
            when (DType.BigInt, DType.UInt64) {
                var x = toSymEntry(gX,bigint);
                var y = toSymEntry(gY,uint);
                var ya = y.a:bigint;
                if x.max_bits != -1 {
                    ya.mod(ya, x.max_bits);
                }
                x.a[slice] = ya;
             }
            when (DType.BigInt, DType.Bool) {
                var x = toSymEntry(gX,bigint);
                var y = toSymEntry(gY,bool);
                // TODO change once we can cast directly from bool to bigint
                var ya = y.a:int:bigint;
                if x.max_bits != -1 {
                    ya.mod(ya, x.max_bits);
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

    use CommandMap;
    registerFunction("arrayViewIntIndex", arrayViewIntIndexMsg, getModuleName());
    registerFunction("arrayViewMixedIndex", arrayViewMixedIndexMsg, getModuleName());
    registerFunction("arrayViewIntIndexAssign", arrayViewIntIndexAssignMsg, getModuleName());
    registerFunction("[int]", intIndexMsg, getModuleName());
    registerFunction("[slice]", sliceIndexMsg, getModuleName());
    registerFunction("[pdarray]", pdarrayIndexMsg, getModuleName());
    registerFunction("[int]=val", setIntIndexToValueMsg, getModuleName());
    registerFunction("[pdarray]=val", setPdarrayIndexToValueMsg, getModuleName());
    registerFunction("[pdarray]=pdarray", setPdarrayIndexToPdarrayMsg, getModuleName());
    registerFunction("[slice]=val", setSliceIndexToValueMsg, getModuleName());
    registerFunction("[slice]=pdarray", setSliceIndexToPdarrayMsg, getModuleName());
}
