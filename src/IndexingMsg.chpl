module IndexingMsg
{
    use ServerConfig;
    use ServerErrorStrings;

    use Reflection only;

    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    use CommAggregation;

    /* intIndex "a[int]" response to __getitem__(int) */
    proc intIndexMsg(cmd: string, payload: bytes, st: borrowed SymTab):string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, idxStr) = payload.decode().splitMsgToTuple(2);
        var idx = try! idxStr:int;
        if v {try! writeln("%s %s %i".format(cmd, name, idx));try! stdout.flush();}

         var gEnt: borrowed GenSymEntry = st.lookup(name);
         
         select (gEnt.dtype) {
             when (DType.Int64) {
                 var e = toSymEntry(gEnt, int);
                 return try! "item %s %t".format(dtype2str(e.dtype),e.a[idx]);
             }
             when (DType.Float64) {
                 var e = toSymEntry(gEnt,real);
                 return try! "item %s %.17r".format(dtype2str(e.dtype),e.a[idx]);
             }
             when (DType.Bool) {
                 var e = toSymEntry(gEnt,bool);
                 var s = try! "item %s %t".format(dtype2str(e.dtype),e.a[idx]);
                 s = s.replace("true","True"); // chapel to python bool
                 s = s.replace("false","False"); // chapel to python bool
                 return s;
             }
             otherwise {return notImplementedError(pn,dtype2str(gEnt.dtype));}
         }
    }

    /* sliceIndex "a[slice]" response to __getitem__(slice) */
    proc sliceIndexMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (name, startStr, stopStr, strideStr)
              = payload.decode().splitMsgToTuple(4); // split request into fields
        var start = try! startStr:int;
        var stop = try! stopStr:int;
        var stride = try! strideStr:int;
        var slice: range(stridable=true);

        // convert python slice to chapel slice
        // backwards iteration with negative stride
        if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
        // forward iteration with positive stride
        else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
        // BAD FORM start < stop and stride is negative
        else {slice = 1..0;}

        // get next symbol name
        var rname = st.nextName();

        if v {try! writeln("%s %s %i %i %i : %t , %s".format(cmd, name, start, stop, stride, slice, rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);

        proc sliceHelper(type t) throws {
            var e = toSymEntry(gEnt,t);
            var a = st.addEntry(rname, slice.size, t);
            ref ea = e.a;
            ref aa = a.a;
            forall (elt,j) in zip(aa, slice) with (var agg = newSrcAggregator(t)) {
              agg.copy(elt,ea[j]);
            }
            return try! "created " + st.attrib(rname);
        }
        
        select(gEnt.dtype) {
            when (DType.Int64) {
                return sliceHelper(int);
            }
            when (DType.Float64) {
                return sliceHelper(real);
            }
            when (DType.Bool) {
                return sliceHelper(bool);
            }
            otherwise {return notImplementedError(pn,dtype2str(gEnt.dtype));}
        }
    }

    /* pdarrayIndex "a[pdarray]" response to __getitem__(pdarray) */
    proc pdarrayIndexMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, iname) = payload.decode().splitMsgToTuple(2);

        // get next symbol name
        var rname = st.nextName();

        if v {try! writeln("%s %s %s : %s".format(cmd, name, iname, rname));try! stdout.flush();}

        var gX: borrowed GenSymEntry = st.lookup(name);
        var gIV: borrowed GenSymEntry = st.lookup(iname);

        // gather indexing by integer index vector
        proc ivInt64Helper(type XType) throws {
            var e = toSymEntry(gX,XType);
            var iv = toSymEntry(gIV,int);
            if (e.size == 0) && (iv.size == 0) {
                var a = st.addEntry(rname, 0, XType);
                return try! "created " + st.attrib(rname);
            }
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
            if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMin,e.size-1);}
            var a = st.addEntry(rname, iv.size, XType);
            //[i in iv.aD] a.a[i] = e.a[iv.a[i]]; // bounds check iv[i] against e.aD?
            ref a2 = e.a;
            ref iva = iv.a;
            ref aa = a.a;
            forall (a1,idx) in zip(aa,iva) with (var agg = newSrcAggregator(XType)) {
              agg.copy(a1,a2[idx]);
            }
            
            return try! "created " + st.attrib(rname);
        }

        // compression boolean indexing by bool index vector
        proc ivBoolHelper(type XType) throws {
            var e = toSymEntry(gX,XType);
            var truth = toSymEntry(gIV,bool);
            if (e.size == 0) && (truth.size == 0) {
                var a = st.addEntry(rname, 0, XType);
                return try! "created " + st.attrib(rname);
            }
            var iv: [truth.aD] int = (+ scan truth.a);
            var pop = iv[iv.size-1];
            if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}
            var a = st.addEntry(rname, pop, XType);
            //[i in e.aD] if (truth.a[i] == true) {a.a[iv[i]-1] = e.a[i];}// iv[i]-1 for zero base index
            ref ead = e.aD;
            ref ea = e.a;
            ref trutha = truth.a;
            ref aa = a.a;
            forall (i, eai) in zip(ead, ea) with (var agg = newDstAggregator(XType)) {
              if (trutha[i] == true) {
                agg.copy(aa[iv[i]-1], eai);
              }
            }
            return try! "created " + st.attrib(rname);
        }
        
        select(gX.dtype, gIV.dtype) {
            when (DType.Int64, DType.Int64) {
                return ivInt64Helper(int);
            }
            when (DType.Int64, DType.Bool) {
                return ivBoolHelper(int);
            }
            when (DType.Float64, DType.Int64) {
                return ivInt64Helper(real);
            }
            when (DType.Float64, DType.Bool) {
                return ivBoolHelper(real);
            }
            when (DType.Bool, DType.Int64) {
                return ivInt64Helper(bool);
           }
            when (DType.Bool, DType.Bool) {
                return ivBoolHelper(bool);
            }
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+")");}
        }
    }

    /* setIntIndexToValue "a[int] = value" response to __setitem__(int, value) */
    proc setIntIndexToValueMsg(cmd: string, payload: bytes, st: borrowed SymTab):string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, idxStr, dtypeStr, value) = payload.decode().splitMsgToTuple(4);
        var idx = try! idxStr:int;
        var dtype = str2dtype(dtypeStr);
        if v {try! writeln("%s %s %i %s %s".format(cmd, name, idx, dtype2str(dtype), value));try! stdout.flush();}

         var gEnt: borrowed GenSymEntry = st.lookup(name);

         select (gEnt.dtype, dtype) {
             when (DType.Int64, DType.Int64) {
                 var e = toSymEntry(gEnt,int);
                 var val = try! value:int;
                 e.a[idx] = val;
             }
             when (DType.Int64, DType.Float64) {
                 var e = toSymEntry(gEnt,int);
                 var val = try! value:real;
                 e.a[idx] = val:int;
             }
             when (DType.Int64, DType.Bool) {
                 var e = toSymEntry(gEnt,int);
                 value = value.replace("True","true");// chapel to python bool
                 value = value.replace("False","false");// chapel to python bool
                 var val = try! value:bool;
                 e.a[idx] = val:int;
             }
             when (DType.Float64, DType.Int64) {
                 var e = toSymEntry(gEnt,real);
                 var val = try! value:int;
                 e.a[idx] = val;
             }
             when (DType.Float64, DType.Float64) {
                 var e = toSymEntry(gEnt,real);
                 var val = try! value:real;
                 e.a[idx] = val;
             }
             when (DType.Float64, DType.Bool) {
                 var e = toSymEntry(gEnt,real);
                 value = value.replace("True","true");// chapel to python bool
                 value = value.replace("False","false");// chapel to python bool
                 var b = try! value:bool;
                 var val:real;
                 if b {val = 1.0;} else {val = 0.0;}
                 e.a[idx] = val;
             }
             when (DType.Bool, DType.Int64) {
                 var e = toSymEntry(gEnt,bool);
                 var val = try! value:int;
                 e.a[idx] = val:bool;
             }
             when (DType.Bool, DType.Float64) {
                 var e = toSymEntry(gEnt,bool);
                 var val = try! value:real;
                 e.a[idx] = val:bool;
             }
             when (DType.Bool, DType.Bool) {
                 var e = toSymEntry(gEnt,bool);
                 value = value.replace("True","true");// chapel to python bool
                 value = value.replace("False","false");// chapel to python bool
                 var val = try! value:bool;
                 e.a[idx] = val;
             }
             otherwise {return notImplementedError(pn,
                                                   "("+dtype2str(gEnt.dtype)+","+dtype2str(dtype)+")");}
         }
         return try! "%s success".format(pn);
    }

    /* setPdarrayIndexToValue "a[pdarray] = value" response to __setitem__(pdarray, value) */
    proc setPdarrayIndexToValueMsg(cmd: string, payload: bytes, st: borrowed SymTab):string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, iname, dtypeStr, value) = payload.decode().splitMsgToTuple(4);
        var dtype = str2dtype(dtypeStr);

        if v {try! writeln("%s %s %s %s %s".format(cmd, name, iname, dtype2str(dtype), value));try! stdout.flush();}

        var gX: borrowed GenSymEntry = st.lookup(name);
        var gIV: borrowed GenSymEntry = st.lookup(iname);

        // scatter indexing by integer index vector
        proc ivInt64Helper(type Xtype, type dtype): string {
            var e = toSymEntry(gX,Xtype);
            var iv = toSymEntry(gIV,int);
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
            if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
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
            return try! "%s success".format(pn);
        }

        // expansion boolean indexing by bool index vector
        proc ivBoolHelper(type Xtype, type dtype): string throws {
            var e = toSymEntry(gX,Xtype);
            var truth = toSymEntry(gIV,bool);
            if (e.size != truth.size) {return try! "Error: %s: bool iv must be same size %i != %i".format(pn,e.size,truth.size);}
            if isBool(dtype) {
                value = value.replace("True","true"); // chapel to python bool
                value = value.replace("False","false"); // chapel to python bool
            }
            var val = try! value:dtype;
            ref ead = e.aD;
            ref ea = e.a;
            ref trutha = truth.a;
            forall i in ead with (var agg = newDstAggregator(dtype)) {
              if (trutha[i] == true) {
                agg.copy(ea[i],val);
              }
            }
            return try! "%s success".format(pn);
        }
        
        select(gX.dtype, gIV.dtype, dtype) {
            when (DType.Int64, DType.Int64, DType.Int64) {
              return ivInt64Helper(int, int);
            }
            when (DType.Int64, DType.Bool, DType.Int64) {
              return ivBoolHelper(int, int);
            }
            when (DType.Float64, DType.Int64, DType.Float64) {
              return ivInt64Helper(real, real);
            }
            when (DType.Float64, DType.Bool, DType.Float64) {
              return ivBoolHelper(real, real);
            }
            when (DType.Bool, DType.Int64, DType.Bool) {
              return ivInt64Helper(bool, bool);
            }
            when (DType.Bool, DType.Bool, DType.Bool) {
              return ivBoolHelper(bool, bool);
            }
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+","+dtype2str(dtype)+")");}
        }
    }

    /* setPdarrayIndexToPdarray "a[pdarray] = pdarray" response to __setitem__(pdarray, pdarray) */
    proc setPdarrayIndexToPdarrayMsg(cmd: string, payload: bytes, st: borrowed SymTab):string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, iname, yname) = payload.decode().splitMsgToTuple(3);

        if v {try! writeln("%s %s %s %s".format(cmd, name, iname, yname));try! stdout.flush();}

        var gX: borrowed GenSymEntry = st.lookup(name);
        var gIV: borrowed GenSymEntry = st.lookup(iname);
        var gY: borrowed GenSymEntry = st.lookup(yname);

        // add check for IV to be dtype of int64 or bool

        // scatter indexing by an integer index vector
        proc ivInt64Helper(type t) {
            // add check to make syre IV and Y are same size
            if (gIV.size != gY.size) {return try! "Error: %s: size mismatch %i %i".format(pn,gIV.size,gY.size);}
            var e = toSymEntry(gX,t);
            var iv = toSymEntry(gIV,int);
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            var y = toSymEntry(gY,t);
            if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
            if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
            //[(i,v) in zip(iv.a,y.a)] e.a[i] = v;
            ref iva = iv.a;
            ref ya = y.a;
            ref ea = e.a;
            forall (i,v) in zip(iva,ya) with (var agg = newDstAggregator(t)) {
              agg.copy(ea[i],v);
            }
            return try! "%s success".format(pn);
        }

        // expansion indexing by a bool index vector
        proc ivBoolHelper(type t) {
            // add check to make syre IV and Y are same size
            if (gIV.size != gX.size) {return try! "Error: %s: size mismatch %i %i".format(pn,gIV.size,gX.size);}
            var e = toSymEntry(gX,t);
            var truth = toSymEntry(gIV,bool);
            var iv: [truth.aD] int = (+ scan truth.a);
            var pop = iv[iv.size-1];
            if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}
            var y = toSymEntry(gY,t);
            if (y.size != pop) {return try! "Error: %s: pop size mismatch %i %i".format(pn,pop,y.size);}
            ref ya = y.a;
            ref ead = e.aD;
            ref ea = e.a;
            ref trutha = truth.a;
            forall (eai, i) in zip(ea, ead) with (var agg = newSrcAggregator(t)) {
              if (trutha[i] == true) {
                agg.copy(eai,ya[iv[i]-1]);
              }
            }
            return try! "%s success".format(pn);
        }

        select(gX.dtype, gIV.dtype, gY.dtype) {
            when (DType.Int64, DType.Int64, DType.Int64) {
                return ivInt64Helper(int);
            }
            when (DType.Int64, DType.Bool, DType.Int64) {
                return ivBoolHelper(int);
            }
            when (DType.Float64, DType.Int64, DType.Float64) {
                return ivInt64Helper(real);
            }
            when (DType.Float64, DType.Bool, DType.Float64) {
                return ivBoolHelper(real);
            }
            when (DType.Bool, DType.Int64, DType.Bool) {
                return ivInt64Helper(bool);
            }
            when (DType.Bool, DType.Bool, DType.Bool) {
                return ivBoolHelper(bool);
            }
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+","+dtype2str(gY.dtype)+")");}
        }
    }

    /* setSliceIndexToValue "a[slice] = value" response to __setitem__(slice, value) */
    proc setSliceIndexToValueMsg(cmd: string, payload: bytes, st: borrowed SymTab):string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (name, startStr, stopStr, strideStr, dtypeStr, value)
              = payload.decode().splitMsgToTuple(6); // split request into fields
        var start = try! startStr:int;
        var stop = try! stopStr:int;
        var stride = try! strideStr:int;
        var dtype = str2dtype(dtypeStr);
        var slice: range(stridable=true);

        // convert python slice to chapel slice
        // backwards iteration with negative stride
        if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
        // forward iteration with positive stride
        else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
        // BAD FORM start < stop and stride is negative
        else {slice = 1..0;}

        if v {try! writeln("%s %s %i %i %i %s %s".format(cmd, name, start, stop, stride, dtype2str(dtype), value));try! stdout.flush();}
        
        var gEnt: borrowed GenSymEntry = st.lookup(name);

        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var val = try! value:int;
                e.a[slice] = val;
            }
            when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val = try! value:real;
                e.a[slice] = val:int;
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                value = value.replace("True","true");// chapel to python bool
                value = value.replace("False","false");// chapel to python bool
                var val = try! value:bool;
                e.a[slice] = val:int;
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real);
                var val = try! value:int;
                e.a[slice] = val;
            }
            when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var val = try! value:real;
                e.a[slice] = val;
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real);
                value = value.replace("True","true");// chapel to python bool
                value = value.replace("False","false");// chapel to python bool
                var b = try! value:bool;
                var val:real;
                if b {val = 1.0;} else {val = 0.0;}
                e.a[slice] = val;
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool);
                var val = try! value:int;
                e.a[slice] = val:bool;
            }
            when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,bool);
                var val = try! value:real;
                e.a[slice] = val:bool;
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                value = value.replace("True","true");// chapel to python bool
                value = value.replace("False","false");// chapel to python bool
                var val = try! value:bool;
                e.a[slice] = val;
            }
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gEnt.dtype)+","+dtype2str(dtype)+")");}
        }
        return try! "%s success".format(pn); 
    }
    
    /* setSliceIndexToPdarray "a[slice] = pdarray" response to __setitem__(slice, pdarray) */
    proc setSliceIndexToPdarrayMsg(cmd: string, payload: bytes, st: borrowed SymTab):string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (name, startStr, stopStr, strideStr, yname)
              = payload.decode().splitMsgToTuple(5); // split request into fields
        var start = try! startStr:int;
        var stop = try! stopStr:int;
        var stride = try! strideStr:int;
        var slice: range(stridable=true);

        // convert python slice to chapel slice
        // backwards iteration with negative stride
        if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
        // forward iteration with positive stride
        else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
        // BAD FORM start < stop and stride is negative
        else {slice = 1..0;}

        if v {try! writeln("%s %s %i %i %i %s".format(cmd, name, start, stop, stride, yname));try! stdout.flush();}

        var gX: borrowed GenSymEntry = st.lookup(name);
        var gY: borrowed GenSymEntry = st.lookup(yname);

        // add check to make syre IV and Y are same size
        if (slice.size != gY.size) {return try! "Error: %s: size mismatch %i %i".format(pn,slice.size, gY.size);}

        select (gX.dtype, gY.dtype) {
            when (DType.Int64, DType.Int64) {
                var x = toSymEntry(gX,int);
                var y = toSymEntry(gY,int);
                x.a[slice] = y.a;
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
            when (DType.Float64, DType.Int64) {
                var x = toSymEntry(gX,real);
                var y = toSymEntry(gY,int);
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
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gX.dtype)+","+dtype2str(gY.dtype)+")");}
        }
        return try! "%s success".format(pn);
    }
    
}
