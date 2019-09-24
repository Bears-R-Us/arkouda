module IndexingMsg
{
    use ServerConfig;
    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    // experimental
    use UnorderedCopy;

    /* intIndex "a[int]" response to __getitem__(int) */
    proc intIndexMsg(reqMsg: string, st: borrowed SymTab):string {
        var pn = "intIndex";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var idx = try! fields[3]:int;
        if v {try! writeln("%s %s %i".format(cmd, name, idx));try! stdout.flush();}

         var gEnt: borrowed GenSymEntry = st.lookup(name);
         if (gEnt == nil) {return unknownSymbolError(pn,name);}
         
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
    proc sliceIndexMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "sliceIndex";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var start = try! fields[3]:int;
        var stop = try! fields[4]:int;
        var stride = try! fields[5]:int;
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
        if (gEnt == nil) {return unknownSymbolError(pn,name);}

        proc sliceHelper(type t) {
          var e = toSymEntry(gEnt,t);
          var aD = makeDistDom(slice.size);
          var a = makeDistArray(slice.size, t);
          ref ea = e.a;
          [(elt,j) in zip(a, slice)] elt = ea[j];
          //[(elt,j) in zip(a, slice)] unorderedCopy(elt,ea[j]);
          st.addEntry(rname, new shared SymEntry(a));
        }

        select(gEnt.dtype) {
            when (DType.Int64) {
                sliceHelper(int);
            }
            when (DType.Float64) {
                sliceHelper(real);
            }
            when (DType.Bool) {
                sliceHelper(bool);
            }
            otherwise {return notImplementedError(pn,dtype2str(gEnt.dtype));}
        }
        return try! "created " + st.attrib(rname);
    }

    /* pdarrayIndex "a[pdarray]" response to __getitem__(pdarray) */
    proc pdarrayIndexMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "pdarrayIndex";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var iname = fields[3];

        // get next symbol name
        var rname = st.nextName();

        if v {try! writeln("%s %s %s : %s".format(cmd, name, iname, rname));try! stdout.flush();}

        var gX: borrowed GenSymEntry = st.lookup(name);
        if (gX == nil) {return unknownSymbolError(pn,name);}
        var gIV: borrowed GenSymEntry = st.lookup(iname);
        if (gIV == nil) {return unknownSymbolError(pn,iname);}

        /* proc ivInt64Helper(type XType) { */
        /*     var e = toSymEntry(gX,XType); */
        /*     var iv = toSymEntry(gIV,int); */
        /*     var ivMin = min reduce iv.a; */
        /*     var ivMax = max reduce iv.a; */
        /*     if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);} */
        /*     if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMin,e.size-1);} */
        /*     var a: [iv.aD] XType; */
        /*     //[i in iv.aD] a[i] = e.a[iv.a[i]]; // bounds check iv[i] against e.aD? */
        /*     ref a2 = e.a; */
        /*     ref iva = iv.a; */
        /*     [(a1,idx) in zip(a,iva)] unorderedCopy(a1,a2[idx]); // bounds check iv[i] against e.aD? */
        /*     st.addEntry(rname, new shared SymEntry(a)); */
        /* } */
        
        /* proc ivBoolHelper(type XType) { */
        /*     var e = toSymEntry(gX,XType); */
        /*     var truth = toSymEntry(gIV,bool); */
        /*     var iv: [truth.aD] int = (+ scan truth.a); */
        /*     var pop = iv[iv.size-1]; */
        /*     if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();} */
        /*     var a = makeDistArray(pop, XType); */
        /*     //[i in e.aD] if (truth.a[i] == true) {a[iv[i]-1] = e.a[i];}// iv[i]-1 for zero base index */
        /*     ref ead = e.aD; */
        /*     ref ea = e.a; */
        /*     ref trutha = truth.a; */
        /*     [i in ead] if (trutha[i] == true) {unorderedCopy(a[iv[i]-1], ea[i]);}// iv[i]-1 for zero base index */
        /*     st.addEntry(rname, new shared SymEntry(a)); */
        /* } */
        
        select(gX.dtype, gIV.dtype) {
            when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gX,int);
                var iv = toSymEntry(gIV,int);
                var ivMin = min reduce iv.a;
                var ivMax = max reduce iv.a;
                if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
                if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
                var a: [iv.aD] int;
                //[i in iv.aD] a[i] = e.a[iv.a[i]]; // bounds check iv[i] against e.aD?
                ref a2 = e.a;
                ref iva = iv.a;
                [(a1,idx) in zip(a,iva)] unorderedCopy(a1,a2[idx]); // bounds check iv[i] against e.aD?
                st.addEntry(rname, new shared SymEntry(a));
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gX,int);
                var truth = toSymEntry(gIV,bool);
                var iv: [truth.aD] int = (+ scan truth.a);
                var pop = iv[iv.size-1];
                if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}
                var a = makeDistArray(pop, int);
                //[i in e.aD] if (truth.a[i] == true) {a[iv[i]-1] = e.a[i];}// iv[i]-1 for zero base index
                ref ead = e.aD;
                ref ea = e.a;
                ref trutha = truth.a;
                [i in ead] if (trutha[i] == true) {unorderedCopy(a[iv[i]-1], ea[i]);}// iv[i]-1 for zero base index
                st.addEntry(rname, new shared SymEntry(a));
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gX,real);
                var iv = toSymEntry(gIV,int);
                var ivMin = min reduce iv.a;
                var ivMax = max reduce iv.a;
                if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
                if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
                var a: [iv.aD] real;
                //[i in iv.aD] a[i] = e.a[iv.a[i]]; // bounds check iv[i] against e.aD?
                ref a2 = e.a;
                ref iva = iv.a;
                [(a1,idx) in zip(a,iva)] unorderedCopy(a1,a2[idx]); // bounds check iv[i] against e.aD?
                st.addEntry(rname, new shared SymEntry(a));
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gX,real);
                var truth = toSymEntry(gIV,bool);
                var iv: [truth.aD] int = (+ scan truth.a);
                var pop = iv[iv.size-1];
                if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}
                var a = makeDistArray(pop, real);
                //[i in e.aD] if (truth.a[i] == true) {a[iv[i]-1] = e.a[i];}// iv[i]-1 for zero base index
                ref ead = e.aD;
                ref ea = e.a;
                ref trutha = truth.a;
                [i in ead] if (trutha[i] == true) {unorderedCopy(a[iv[i]-1], ea[i]);}// iv[i]-1 for zero base index
                st.addEntry(rname, new shared SymEntry(a));
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gX,bool);
                var iv = toSymEntry(gIV,int);
                var ivMin = min reduce iv.a;
                var ivMax = max reduce iv.a;
                if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
                if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
                var a: [iv.aD] bool;
                //[i in iv.aD] a[i] = e.a[iv.a[i]];// bounds check iv[i] against e.aD?
                ref a2 = e.a;
                ref iva = iv.a;
                [(a1,idx) in zip(a,iva)] a1 = a2[idx]; // bounds check iv[i] against e.aD?
                //[(a1,idx) in zip(a,iva)] unorderedCopy(a1,a2[idx]); // bounds check iv[i] against e.aD?
                st.addEntry(rname, new shared SymEntry(a));
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gX,bool);
                var truth = toSymEntry(gIV,bool);
                var iv: [truth.aD] int = (+ scan truth.a);
                var pop = iv[iv.size-1];
                if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}
                var a = makeDistArray(pop, bool);
                //[i in e.aD] if (truth.a[i] == true) {a[iv[i]-1] = e.a[i];}// iv[i]-1 for zero base index
                ref ead = e.aD;
                ref ea = e.a;
                ref trutha = truth.a;
                //[i in e.aD] if (truth.a[i] == true) {unorderedCopy(a[iv[i]-1], e.a[i]);}// iv[i]-1 for zero base index
                [i in ead] if (trutha[i] == true) {a[iv[i]-1] = ea[i];}// iv[i]-1 for zero base index
                st.addEntry(rname, new shared SymEntry(a));
            }
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+")");}
        }
        return try! "created " + st.attrib(rname);
    }

    /* setIntIndexToValue "a[int] = value" response to __setitem__(int, value) */
    proc setIntIndexToValueMsg(reqMsg: string, st: borrowed SymTab):string {
        var pn = "setIntIndexToValue";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var idx = try! fields[3]:int;
        var dtype = str2dtype(fields[4]);
        var value = fields[5];
        if v {try! writeln("%s %s %i %s %s".format(cmd, name, idx, dtype2str(dtype), value));try! stdout.flush();}

         var gEnt: borrowed GenSymEntry = st.lookup(name);
         if (gEnt == nil) {return unknownSymbolError(pn,name);}

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
    proc setPdarrayIndexToValueMsg(reqMsg: string, st: borrowed SymTab):string {
        var pn = "setPdarrayIndexToValue";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var iname = fields[3];
        var dtype = str2dtype(fields[4]);
        var value = fields[5];

        if v {try! writeln("%s %s %s %s %s".format(cmd, name, iname, dtype2str(dtype), value));try! stdout.flush();}

        var gX: borrowed GenSymEntry = st.lookup(name);
        if (gX == nil) {return unknownSymbolError(pn,name);}
        var gIV: borrowed GenSymEntry = st.lookup(iname);
        if (gIV == nil) {return unknownSymbolError(pn,iname);}

        proc idxToValHelper(type Xtype, type IVtype, type dtype): string {
            var e = toSymEntry(gX,Xtype);
            var iv = toSymEntry(gIV,IVtype);
            var ivMin = min reduce iv.a;
            var ivMax = max reduce iv.a;
            if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
            if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
            if isBool(dtype) {
                value = value.replace("True","true"); // chapel to python bool
                value = value.replace("False","false"); // chapel to python bool
            }
            var val = try! value:dtype;
            [i in iv.a] e.a[i] = val;
            return try! "%s success".format(pn);
        }
        
        // add check for IV to be dtype of int64 or bool

        select(gX.dtype, gIV.dtype, dtype) {
            when (DType.Int64, DType.Int64, DType.Int64) {
              return idxToValHelper(int, int, int);
            }
            when (DType.Float64, DType.Int64, DType.Float64) {
              return idxToValHelper(real, int, real);
            }
            when (DType.Bool, DType.Int64, DType.Bool) {
              return idxToValHelper(bool, int, bool);
            }
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+","+dtype2str(dtype)+")");}
        }
    }

    /* setPdarrayIndexToPdarray "a[pdarray] = pdarray" response to __setitem__(pdarray, pdarray) */
    proc setPdarrayIndexToPdarrayMsg(reqMsg: string, st: borrowed SymTab):string {
        var pn = "setPdarrayIndexToPdarray";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var iname = fields[3];
        var yname = fields[4];

        if v {try! writeln("%s %s %s %s".format(cmd, name, iname, yname));try! stdout.flush();}

        var gX: borrowed GenSymEntry = st.lookup(name);
        if (gX == nil) {return unknownSymbolError(pn,name);}
        var gIV: borrowed GenSymEntry = st.lookup(iname);
        if (gIV == nil) {return unknownSymbolError(pn,iname);}
        var gY: borrowed GenSymEntry = st.lookup(yname);
        if (gY == nil) {return unknownSymbolError(pn,yname);}

        // add check to make syre IV and Y are same size
        if (gIV.size != gY.size) {return try! "Error: %s: size mismatch %i %i".format(pn,gIV.size, gY.size);}
        // add check for IV to be dtype of int64 or bool
        
        select(gX.dtype, gIV.dtype, gY.dtype) {
            when (DType.Int64, DType.Int64, DType.Int64) {
                var e = toSymEntry(gX,int);
                var iv = toSymEntry(gIV,int);
                var ivMin = min reduce iv.a;
                var ivMax = max reduce iv.a;
                var y = toSymEntry(gY,int);
                if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
                if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
                //[(i,v) in zip(iv.a,y.a)] e.a[i] = v;
                [(i,v) in zip(iv.a,y.a)] unorderedCopy(e.a[i],v);
            }
            when (DType.Float64, DType.Int64, DType.Float64) {
                var e = toSymEntry(gX,real);
                var iv = toSymEntry(gIV,int);
                var ivMin = min reduce iv.a;
                var ivMax = max reduce iv.a;
                var y = toSymEntry(gY,real);
                if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
                if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
                //[(i,v) in zip(iv.a,y.a)] e.a[i] = v;
                [(i,v) in zip(iv.a,y.a)] unorderedCopy(e.a[i],v);
            }
            when (DType.Bool, DType.Int64, DType.Bool) {
                var e = toSymEntry(gX,bool);
                var iv = toSymEntry(gIV,int);
                var ivMin = min reduce iv.a;
                var ivMax = max reduce iv.a;
                var y = toSymEntry(gY,bool);
                if ivMin < 0 {return try! "Error: %s: OOBindex %i < 0".format(pn,ivMin);}
                if ivMax >= e.size {return try! "Error: %s: OOBindex %i > %i".format(pn,ivMax,e.size-1);}
                //[(i,v) in zip(iv.a,y.a)] e.a[i] = v;
                [(i,v) in zip(iv.a,y.a)] unorderedCopy(e.a[i],v);
            }
            otherwise {return notImplementedError(pn,
                                                  "("+dtype2str(gX.dtype)+","+dtype2str(gIV.dtype)+","+dtype2str(gY.dtype)+")");}
        }
        return try! "%s success".format(pn);
    }

    /* setSliceIndexToValue "a[slice] = value" response to __setitem__(slice, value) */
    proc setSliceIndexToValueMsg(reqMsg: string, st: borrowed SymTab):string {
        var pn = "setSliceIndexToValue";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var start = try! fields[3]:int;
        var stop = try! fields[4]:int;
        var stride = try! fields[5]:int;
        var dtype = str2dtype(fields[6]);
        var value = fields[7];
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
        if (gEnt == nil) {return unknownSymbolError(pn,name);}

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
    proc setSliceIndexToPdarrayMsg(reqMsg: string, st: borrowed SymTab):string {
        var pn = "setSliceIndexToPdarray";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var start = try! fields[3]:int;
        var stop = try! fields[4]:int;
        var stride = try! fields[5]:int;
        var yname = fields[6];
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
        if (gX == nil) {return unknownSymbolError(pn,name);}
        var gY: borrowed GenSymEntry = st.lookup(yname);
        if (gY == nil) {return unknownSymbolError(pn,yname);}

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
