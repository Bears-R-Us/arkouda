module ReductionMsg
{
    use ServerConfig;

    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use PerLocaleHelper;

    use AryUtil;
    use PrivateDist;

    const lBins = 2**25 * numLocales;
      
    // these functions take an array and produce a scalar
    // parse and respond to reduction message
    // scalar = reductionop(vector)
    proc reductionMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var reductionop = fields[2];
        var name = fields[3];
        if v {try! writeln("%s %s %s".format(cmd,reductionop,name));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("reduction",name);}
       
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum != 0 {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var sum = + reduce e.a;
                        var val = sum:string;
                        return try! "int64 %i".format(val);
                    }
                    when "prod" {
		      var prod = * reduce e.a:real;
                        var val = prod:string;
                        return try! "int64 %i".format(val);
                    }
		    when "min" {
		        var minVal = min reduce e.a;
			var val = minVal:string;
			return try! "int64 %i".format(val);
		    }
		    when "max" {
		        var maxVal = max reduce e.a;
			var val = maxVal:string;
			return try! "int64 %i".format(val);
		    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        var val = minLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        var val = maxLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "is_sorted" {
                        ref ea = e.a;
                        var sorted = isSorted(ea);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
		    when "is_locally_sorted" {
		      var locSorted: [LocaleSpace] bool;
		      coforall loc in Locales {
			on loc {
			  ref myA = e.a[e.a.localSubdomain()];
			  locSorted[here.id] = isSorted(myA);
			}
		      }
		      var val: string;
		      if (& reduce locSorted) {val = "True";} else {val = "False";}
		      return try! "bool %s".format(val);
		    }
                    otherwise {return notImplementedError("reduction",reductionop,gEnt.dtype);}
                }
            }
            when (DType.Float64) {
                var e = toSymEntry(gEnt,real);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum != 0.0 {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var sum = + reduce (e.a != 0.0);
                        if sum == e.aD.size {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var sum = + reduce e.a;
                        var val = sum:string;
                        return try! "float64 %.17r".format(val);
                    }
                    when "prod" {
                        var prod = * reduce e.a;
                        var val = prod:string;
                        return try! "float64 %.17r".format(val);
                    }
		    when "min" {
                        var minVal = min reduce e.a;
                        var val = minVal:string;
                        return try! "float64 %.17r".format(val);
                    }
		    when "max" {
                        var maxVal = max reduce e.a;
                        var val = maxVal:string;
                        return try! "float64 %.17r".format(val);
                    }
                    when "argmin" {
                        var (minVal, minLoc) = minloc reduce zip(e.a,e.aD);
                        var val = minLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "argmax" {
                        var (maxVal, maxLoc) = maxloc reduce zip(e.a,e.aD);
                        var val = maxLoc:string;
                        return try! "int64 %i".format(val);
                    }
                    when "is_sorted" {
                        var sorted = isSorted(e.a);
                        var val:string;
                        if sorted {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    otherwise {return notImplementedError("reduction",reductionop,gEnt.dtype);}
                }
            }
            when (DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                select reductionop
                {
                    when "any" {
                        var val:string;
                        var any = | reduce e.a;
                        if any {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "all" {
                        var val:string;
                        var all = & reduce e.a;
                        if all {val = "True";} else {val = "False";}
                        return try! "bool %s".format(val);
                    }
                    when "sum" {
                        var sum = + reduce e.a:int;
                        var val = sum:string;
                        return try! "int64 %i".format(val);
                    }
                    when "prod" {
                        var prod = * reduce e.a:int;
                        var val = prod:string;
                        return try! "int64 %i".format(val);
                    }
		    when "min" {
		        var val:string;
			if (& reduce e.a) { val = "True"; } else { val = "False"; }
			return try! "bool %s".format(val);
		    }
		    when "max" {
		        var val:string;
			if (| reduce e.a) { val = "True"; } else { val = "False"; }
			return try! "bool %s".format(val);
		    }
                    otherwise {return notImplementedError("reduction",reductionop,gEnt.dtype);}
                }
            }
            otherwise {return unrecognizedTypeError("reduction", dtype2str(gEnt.dtype));}
        }
    }

    proc countReductionMsg(reqMsg: string, st: borrowed SymTab): string {
      // reqMsg: segmentedReduction values segments operator
      var fields = reqMsg.split();
      var cmd = fields[1];
      var segments_name = fields[2]; // segment offsets
      var size = try! fields[3]:int;
      var rname = st.nextName();
      if v {try! writeln("%s %s %s".format(cmd,segments_name, size));try! stdout.flush();}

      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      if (gSeg == nil) {return unknownSymbolError("segmentedReduction",segments_name);}
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      var counts = segCount(segments.a, size);
      st.addEntry(rname, new shared SymEntry(counts));
      return try! "created " + st.attrib(rname);
    }

    proc segCount(segments:[?D] int, upper: int):[D] int {
      var counts:[D] int;
      forall (c, low, i) in zip(counts, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = upper - 1;
	}
	c = high - low + 1;
      }
      return counts;
    }
    
    proc countLocalRdxMsg(reqMsg: string, st: borrowed SymTab): string {
      // reqMsg: countLocalRdx segments
      // segments.size = numLocales * numKeys
      var fields = reqMsg.split();
      var cmd = fields[1];
      var segments_name = fields[2]; // segment offsets
      var size = try! fields[3]:int; // size of original keys array
      var rname = st.nextName();
      if v {try! writeln("%s %s %s".format(cmd,segments_name, size));try! stdout.flush();}

      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      if (gSeg == nil) {return unknownSymbolError("segmentedReduction",segments_name);}
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      var counts = perLocCount(segments.a, size);
      st.addEntry(rname, new shared SymEntry(counts));
      return try! "created " + st.attrib(rname);
    }

    proc perLocCount(segments:[?D] int, size: int): [] int {
      var origD = makeDistDom(size);
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var localCounts: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
	on loc {
	  localCounts[here.id] = segCount(segments.localSlice[D.localSubdomain()],
					  origD.localSubdomain().high + 1);
	}
      }
      var counts: [keyDom] int = + reduce [i in PrivateSpace] localCounts[i];
      return counts;
    }


    proc segmentedReductionMsg(reqMsg: string, st: borrowed SymTab): string {
      // reqMsg: segmentedReduction values segments operator
      var fields = reqMsg.split();
      var cmd = fields[1];
      var keys_name = fields[2];
      var values_name = fields[3];   // segmented array of values to be reduced
      var segments_name = fields[4]; // segment offsets
      var operator = fields[5];      // reduction operator
      var rname = st.nextName();
      if v {try! writeln("%s %s %s %s %s".format(cmd,keys_name,values_name,segments_name,operator));try! stdout.flush();}
      var gKey: borrowed GenSymEntry = st.lookup(keys_name);
      if (gKey == nil) {return unknownSymbolError("segmentedReduction", keys_name);}
      if (gKey.dtype != DType.Int64) {return unrecognizedTypeError("segmentedLocalRdx", dtype2str(gKey.dtype));}
      var keys = toSymEntry(gKey, int);
      var gVal: borrowed GenSymEntry = st.lookup(values_name);
      if (gVal == nil) {return unknownSymbolError("segmentedReduction",values_name);}
      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      if (gSeg == nil) {return unknownSymbolError("segmentedReduction",segments_name);}
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      select (gVal.dtype) {
      when (DType.Int64) {
	var values = toSymEntry(gVal, int);
	select operator {
	  when "sum" {
	    var res = segSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "prod" {
	    var res = segProduct(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var res = segMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "min" {
	    var res = segMin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = segMax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var (vals, locs) = segArgmin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(locs));
	  }
	  when "argmax" {
	    var (vals, locs) = segArgmax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(locs));
	  }
	  when "nunique" {
	    var res = segNumUnique(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedReduction",operator,gVal.dtype);}
	  }
      }
      when (DType.Float64) {
	var values = toSymEntry(gVal, real);
	select operator {
	  when "sum" {
	    var res = segSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "prod" {
	    var res = segProduct(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var res = segMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "min" {
	    var res = segMin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = segMax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var (vals, locs) = segArgmin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(locs));
	  }
	  when "argmax" {
	    var (vals, locs) = segArgmax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(locs));
	  }
	  otherwise {return notImplementedError("segmentedReduction",operator,gVal.dtype);}
	  }
      }
      when (DType.Bool) {
	var values = toSymEntry(gVal, bool);
	select operator {
	  when "sum" {
	    var res = segSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "any" {
	    var res = segAny(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "all" {
	    var res = segAll(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var res = segMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedReduction",operator,gVal.dtype);}
	  }
      }
      otherwise {return unrecognizedTypeError("segmentedReduction", dtype2str(gVal.dtype));}
      }
      return try! "created " + st.attrib(rname);
    }

    proc segmentedLocalRdxMsg(reqMsg: string, st: borrowed SymTab): string {
      // reqMsg: segmentedReduction keys values segments operator
      var fields = reqMsg.split();
      var cmd = fields[1];
      var keys_name = fields[2];
      var values_name = fields[3];   // segmented array of values to be reduced
      var segments_name = fields[4]; // segment offsets
      var operator = fields[5];      // reduction operator
      var rname = st.nextName();
      if v {try! writeln("%s %s %s %s".format(cmd,keys_name,values_name,segments_name,operator));try! stdout.flush();}

      var gKey: borrowed GenSymEntry = st.lookup(keys_name);
      if (gKey == nil) {return unknownSymbolError("segmentedLocalRdx",keys_name);}
      if (gKey.dtype != DType.Int64) {return unrecognizedTypeError("segmentedLocalRdx", dtype2str(gKey.dtype));}
      var keys = toSymEntry(gKey, int);
      var gVal: borrowed GenSymEntry = st.lookup(values_name);
      if (gVal == nil) {return unknownSymbolError("segmentedLocalRdx",values_name);}
      var gSeg: borrowed GenSymEntry = st.lookup(segments_name);
      if (gSeg == nil) {return unknownSymbolError("segmentedLocalRdx",segments_name);}
      var segments = toSymEntry(gSeg, int);
      if (segments == nil) {return "Error: array of segment offsets must be int dtype";}
      select (gVal.dtype) {
      when (DType.Int64) {
	var values = toSymEntry(gVal, int);
	select operator {
	  when "sum" {
	    var res = perLocSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "prod" {
	    var res = perLocProduct(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var res = perLocMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "min" {
	    var res = perLocMin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = perLocMax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var res = perLocArgmin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var res = perLocArgmax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "nunique" {
	    var res = perLocNumUnique(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedLocalRdx",operator,gVal.dtype);}
	  }
      }
      when (DType.Float64) {
	var values = toSymEntry(gVal, real);
	select operator {
	  when "sum" {
	    var res = perLocSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "prod" {
	    var res = perLocProduct(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var res = perLocMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "min" {
	    var res = perLocMin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = perLocMax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var res = perLocArgmin(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var res = perLocArgmax(keys.a, values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedLocalRdx",operator,gVal.dtype);}
	  }
      }
      when (DType.Bool) {
	var values = toSymEntry(gVal, bool);
	select operator {
	  when "sum" {
	    var res = perLocSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "any" {
	    var res = perLocAny(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "all" {
	    var res = perLocAll(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var res = perLocMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedLocalRdx",operator,gVal.dtype);}
	  }
      }
      otherwise {return unrecognizedTypeError("segmentedLocalRdx", dtype2str(gVal.dtype));}
      }
      return try! "created " + st.attrib(rname);
    }
	  
    /* Segmented Reductions of the form: seg<Op>(values:[] t, segments: [] int)
       Use <segments> as the boundary indices to divide <values> into chunks, 
       and then reduce over each chunk uisng the operator <Op>. The return array 
       of reduced values is the same size as <segments>.
     */
    proc segSum(values:[] ?t, segments:[?D] int): [D] t {
      var res: [D] t;
      var cumsum = + scan values;
      // Iterate over segments
      forall (i, r) in zip(D, res) {
	// Find the segment boundaries
	var vl: t, vr: t;
	if (i == D.low) {
	  vl = 0;
	} else {
	  vl = cumsum[segments[i] - 1];
	}
	if (i == D.high) {
	  vr = cumsum[values.domain.high];
	} else {
	  vr = cumsum[segments[i+1] -1];
	}
	r = vr - vl;
      }
      return res;
    }


    /* Per-Locale Segmented Reductions have the same form as segmented reductions:
       perLoc<Op>(values:[] t, segments: [] int)
       However, in this case <segments> has length <numSegments>*<numLocales> and
       stores the segment boundaries for each locale's chunk of <values>. These
       reductions perform two stages: a local reduction (implemented via a call
       to seg<Op> on the local slice of values) and a global reduction of the 
       local results. The return is the same as seg<Op>: one reduced value per segment.
    */
    proc perLocSum(values:[] ?t, segments:[?D] int): [] t {
      // Infer the number of keys from size of <segments>
      var numKeys:int = segments.size / numLocales;
      // Make the distributed domain of the final result
      var keyDom = makeDistDom(numKeys);
      // Local reductions stored in a PrivateDist
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      coforall loc in Locales {
	on loc {
	  // Each locale reduces its local slice of <values>
	  perLocVals[here.id] = segSum(values.localSlice[values.localSubdomain()],
				       segments.localSlice[D.localSubdomain()]);
	}
      }
      // The global result is a distributed array, computed as a vector reduction over local results
      var res:[keyDom] t = + reduce [i in PrivateSpace] perLocVals[i];
      return res;
    }

    proc segSum(values:[] bool, segments:[?D] int): [D] int {
      var res: [D] int;
      var cumsum = + scan values;
      // Iterate over segments
      forall (i, r) in zip(D, res) {
	// Find the values to the left of the segment boundaries
	var vl: int, vr: int;
	if (i == D.low) {
	  vl = 0;
	} else {
	  vl = cumsum[segments[i] - 1];
	}
	if (i == D.high) {
	  vr = cumsum[values.domain.high];
	} else {
	  vr = cumsum[segments[i+1] -1];
	}
	r = vr - vl;
      }
      return res;
    }

    proc perLocSum(values:[] bool, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
	on loc {
	  perLocVals[here.id] = segSum(values.localSlice[values.localSubdomain()],
				       segments.localSlice[D.localSubdomain()]);
	}
      }
      var res:[keyDom] int = + reduce [i in PrivateSpace] perLocVals[i];
      return res;
    }
    
    proc segProduct(values:[], segments:[?D] int): [D] real {
      var logs = Math.log(values:real);
      return Math.exp(segSum(logs, segments));
      /* var res: [D] real = 1; */
      /* var cumprod = * scan values; */
      /* // Iterate over segments */
      /* forall (i, r) in zip(D, res) { */
      /* 	// Find the segment boundaries */
      /* 	var vl: real, vr: real; */
      /* 	if (i == D.low) { */
      /* 	  vl = 1.0; */
      /* 	} else { */
      /* 	  vl = cumprod[segments[i] - 1]; */
      /* 	} */
      /* 	if (i == D.high) { */
      /* 	  vr = cumprod[values.domain.high]; */
      /* 	} else { */
      /* 	  vr = cumprod[segments[i+1] -1]; */
      /* 	} */
      /* 	r = vr / vl; */
      /* } */
      /* return res; */
    }

    proc perLocProduct(values:[] ?t, segments:[?D] int): [] real {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] real;
      coforall loc in Locales {
	on loc {
	  perLocVals[here.id] = segProduct(values.localSlice[values.localSubdomain()],
					   segments.localSlice[D.localSubdomain()]);
	}
      }
      var res: [keyDom] real = * reduce [i in PrivateSpace] perLocVals[i];
      return res;
    }
    
    proc segMean(values:[] ?t, segments:[?D] int): [D] real {
      var res: [D] real;
      var cumsum = + scan values;
      forall (i, r) in zip(D, res) {
	// Find the values to the left of the segment boundaries
	var vl: t, vr: t;
	var j: int;
	if (i == D.low) {
	  vl = 0;
	} else {
	  vl = cumsum[segments[i] - 1];
	}
	if (i == D.high) {
	  j = values.domain.high;
	} else {
	  j = segments[i+1] - 1;
	}
	vr = cumsum[j];
	r = (vr - vl):real / (j - i + 1):real;
      }
      return res;
    }

    proc perLocMean(values:[] ?t, segments:[?D] int): [] real {
      var numKeys:int = segments.size / numLocales;
      var keyCounts = perLocCount(segments, values.size);
      var res = perLocSum(values, segments);
      return res:real / keyCounts:real;
    }

    proc segMin(keys:[] int, values:[] ?t, segments:[?D] int): [D] t {
      var res: [D] t = max(t);
      var kv = [(k, v) in zip(keys, values)] (-k, v);
      var cummin = min scan kv;
      forall (i, r, low) in zip(D, res, segments) {
	var vi: int;
	if (i < D.high) {
	  vi = segments[i+1] - 1;
	} else {
	  vi = values.domain.high;
	}
	if (vi >= low) {
	  r = cummin[vi][2];
	}
      }
      return res;
    }
    
    proc perLocMin(keys:[] int, values:[] ?t, segments:[?D] int): [] t {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      coforall loc in Locales {
	on loc {
	  perLocVals[here.id] = segMin(keys.localSlice[keys.localSubdomain()],
				       values.localSlice[values.localSubdomain()],
				       segments.localSlice[D.localSubdomain()]);
	}
      }
      var res: [keyDom] t;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
	r = min reduce [i in PrivateSpace] perLocVals[i][keyInd];
      }
      return res;
    }    

    proc segMax(keys:[] int, values:[] ?t, segments:[?D] int): [D] t {
      var res: [D] t = min(t);
      var kv = [(k, v) in zip(keys, values)] (k, v);
      var cummax = max scan kv;
      forall (i, r, low) in zip(D, res, segments) {
	var vi: int;
	if (i < D.high) {
	  vi = segments[i+1] - 1;
	} else {
	  vi = values.domain.high;
	}
	if (vi >= low) {
	  r = cummax[vi][2];
	}
      }
      return res;
    }

    proc perLocMax(keys:[] int, values:[] ?t, segments:[?D] int): [] t {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      coforall loc in Locales {
	on loc {
	  perLocVals[here.id] = segMax(keys.localSlice[keys.localSubdomain()],
				       values.localSlice[values.localSubdomain()],
				       segments.localSlice[D.localSubdomain()]);
	}
      }
      var res: [keyDom] t;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
	r = max reduce [i in PrivateSpace] perLocVals[i][keyInd];
      }    
      return res;
    }
    
    proc segArgmin(keys:[] int, values:[] ?t, segments:[?D] int): ([D] t, [D] int) {
      var kvi = [(k, v, i) in zip(keys, values, keys.domain)] ((k, v), i);
      var cummin = minloc scan kvi;
      var locs: [D] int;
      var vals: [D] t = max(t);
      forall (l, v, low, i) in zip(locs, vals, segments, D) {
	var vi: int;
	if (i < D.high) {
	  vi = segments[i+1] - 1;
	} else {
	  vi = values.domain.high;
	}
	if (vi >= low) {
	  v = cummin[vi][1][2];
	  l = cummin[vi][2];
	}
      }
      return (vals, locs);
    }

    proc perLocArgmin(keys:[] int, values:[] ?t, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      var perLocLocs: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
	on loc {
	  (perLocVals[here.id], perLocLocs[here.id]) = segArgmin(keys.localSlice[keys.localSubdomain()],
								 values.localSlice[values.localSubdomain()],
								 segments.localSlice[D.localSubdomain()]);
	}
      }
      var res: [keyDom] int;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
	var val: t;
	(val, r) = minloc reduce zip([i in PrivateSpace] perLocVals[i][keyInd],
				     [i in PrivateSpace] perLocLocs[i][keyInd]);
      }
      return res;
    }
    
    proc segArgmax(keys:[] int, values:[] ?t, segments:[?D] int): ([D] t, [D] int) {
      var kvi = [(k, v, i) in zip(keys, values, keys.domain)] ((k, v), i);
      var cummax = maxloc scan kvi;
      var locs: [D] int;
      var vals: [D] t = min(t);
      forall (l, v, low, i) in zip(locs, vals, segments, D) {
	var vi: int;
	if (i < D.high) {
	  vi = segments[i+1] - 1;
	} else {
	  vi = values.domain.high;
	}
	if (vi >= low) {
	  v = cummax[vi][1][2];
	  l = cummax[vi][2];
	}
      }
      return (vals, locs);
    }

    proc perLocArgmax(keys:[] int, values:[] ?t, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] t;
      var perLocLocs: [PrivateSpace] [0..#numKeys] int;
      coforall loc in Locales {
	on loc {
	  (perLocVals[here.id], perLocLocs[here.id]) = segArgmax(keys.localSlice[keys.localSubdomain()],
								 values.localSlice[values.localSubdomain()],
								 segments.localSlice[D.localSubdomain()]);
	}
      }
      var res: [keyDom] int;
      forall (r, keyInd) in zip(res, 0..#numKeys) {
	var val: t;
	(val, r) = maxloc reduce zip([i in PrivateSpace] perLocVals[i][keyInd],
				     [i in PrivateSpace] perLocLocs[i][keyInd]);
      }
      return res;
    }
    
    proc segAny(values:[] bool, segments:[?D] int): [D] bool {
      var res: [D] bool;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = || reduce values[low..high];
      }
      return res;
    }

    proc perLocAny(values:[] bool, segments:[?D] int): [] bool {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] bool;
      coforall loc in Locales {
	on loc {
	  perLocVals[here.id] = segAny(values.localSlice[values.localSubdomain()],
				       segments.localSlice[D.localSubdomain()]);
	}
      }
      var res: [keyDom] bool;
      for (r, keyInd) in zip(res, 0..#numKeys) {
	r = || reduce [i in PrivateSpace] perLocVals[i][keyInd];
      }
      return res;
    }
    
    proc segAll(values:[] bool, segments:[?D] int): [D] bool {
      var res: [D] bool;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = && reduce values[low..high];
      }
      return res;
    }

    proc perLocAll(values:[] bool, segments:[?D] int): [] bool {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var perLocVals: [PrivateSpace] [0..#numKeys] bool;
      coforall loc in Locales {
	on loc {
	  perLocVals[here.id] = segAll(values.localSlice[values.localSubdomain()],
				       segments.localSlice[D.localSubdomain()]);
	}
      }
      var res: [keyDom] bool;
      for (r, keyInd) in zip(res, 0..#numKeys) {
	r = && reduce [i in PrivateSpace] perLocVals[i][keyInd];
      }    
      return res;
    }
    
    proc segNumUnique(values:[] int, segments:[?D] int): [D] int {
      var res: [D] int;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	var unique: domain(int);
	var domLock$:sync bool = true;
	forall v in values[low..high] with (ref unique, ref domLock$) {
	  if !unique.contains(v) {
	    domLock$;
	    if !unique.contains(v) {
	      unique += v;
	    }
	    domLock$ = true;
	  }
	}
	r = unique.size;
      }
      return res;
    }

    proc perLocNumUnique(values:[] int, segments:[?D] int): [] int {
      var minVal = min reduce values;
      var valRange = (max reduce values) - minVal + 1;
      var numKeys:int = segments.size / numLocales;
      if (numKeys*valRange <= lBins) {
	if v {try! writeln("bins %i <= %i; using perLocNumUniqueHist".format(numKeys*valRange, lBins));}
	return perLocNumUniqueHist(values, segments, minVal, valRange, numKeys);
      } else {
	if v {try! writeln("bins %i > %i; using perLocNumUniqueAssoc".format(numKeys*valRange, lBins));}
	return perLocNumUniqueAssoc(values, segments, numKeys);
      }
    }

    /* proc perLocNumUnique(values:[] int, segments:[?D] int, numKeys: int): [] int { */
    /*   // First get all per-locale sets of unique values */
    /*   var localUnique: [PrivateSpace] [0..#numKeys] domain(int); */
    /*   coforall loc in Locales { */
    /* 	on loc { */
    /* 	  var myD = D.localSubdomain(); */
    /* 	  // Loop over keys */
    /* 	  forall (i, low, myU) in zip(myD, segments.localSlice[myD], localUnique[here.id]) { */
    /* 	    // Find segment boundaries */
    /* 	    var high: int; */
    /* 	    if (i < myD.high) { */
    /* 	      high = segments[i+1] - 1; */
    /* 	    } else { */
    /* 	      high = values.localSubdomain().high; */
    /* 	    } */
    /* 	    // Aggregate segment's values into a set */
    /* 	    var domLock$:sync bool = true; */
    /* 	    forall v in values.localSlice[low..high] with (ref myU, ref domLock$) { */
    /* 	      // This outer check is not logically necessary, but it reduces unnecesary acquisition of the lock, which saves a lot of time when keys are dense */
    /* 	      if !myU.contains(v) { */
    /* 		domLock$; */
    /* 		if !myU.contains(v) { */
    /* 		  myU += v; */
    /* 		} */
    /* 		domLock$ = true; */
    /* 	      } */
    /* 	    } */
    /* 	  } */
    /* 	} */
    /*   } */
    /*   // Union the local sets */
    /*   var keyDom = makeDistDom(numKeys); */
    /*   var globalUnique: [keyDom] domain(int) = + reduce [i in PrivateSpace] localUnique[i]; */
    /*   var res = [i in keyDom] globalUnique[i].size; */
    /*   return res; */
    /* } */

    proc perLocNumUniqueHist(values: [] int, segments: [?D] int, minVal: int, valRange: int, numKeys: int): [] int {
      var valDom = makeDistDom(numKeys*valRange);
      var globalValFlags: [valDom] bool;
      coforall loc in Locales {
	on loc {
	  var myD = D.localSubdomain();
	  forall (i, low) in zip(myD, segments.localSlice[myD]) {
	    var high: int;
	    if (i < myD.high) {
	      high = segments[i+1] - 1;
	    } else {
	      high = values.localSubdomain().high;
	    }
	    if (high >= low) {
	      var perm: [0..#(high-low+1)] int;
	      ref myVals = values.localSlice[low..high];
	      var myMin = min reduce myVals;
	      var myRange = (max reduce myVals) - myMin + 1;
	      localHistArgSort(perm, myVals, myMin, myRange);
	      var sorted: [low..high] int;
	      [(s, idx) in zip(sorted, perm)] unorderedCopy(s, myVals[idx]);
	      var (mySegs, myUvals) = segsAndUkeysFromSortedArray(sorted);
	      var keyInd = i - myD.low;
	      forall v in myUvals with (ref globalValFlags) {
		// Does not need to be atomic
		globalValFlags[keyInd*valRange + v - minVal] = true;
	      }
	    }	
	  }
	}
      }
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] int;
      forall (keyInd, r) in zip(keyDom, res) {
	r = + reduce globalValFlags[keyInd*valRange..#valRange];
      }
      return res;
    }

    proc perLocNumUniqueAssoc(values: [] int, segments: [?D] int, numKeys: int): [] int {
      var localUvals: [PrivateSpace] [0..#numKeys] domain(int, parSafe=false);
      coforall loc in Locales {
	on loc {
	  var myD = D.localSubdomain();
	  forall (i, low) in zip(myD, segments.localSlice[myD]) {
	    var high: int;
	    if (i < myD.high) {
	      high = segments[i+1] - 1;
	    } else {
	      high = values.localSubdomain().high;
	    }
	    if (high >= low) {
	      var perm: [0..#(high-low+1)] int;
	      ref myVals = values.localSlice[low..high];
	      var myMin = min reduce myVals;
	      var myRange = (max reduce myVals) - myMin + 1;
	      localHistArgSort(perm, myVals, myMin, myRange);
	      var sorted: [low..high] int;
	      [(s, idx) in zip(sorted, perm)] unorderedCopy(s, myVals[idx]);
	      var (mySegs, myUvals) = segsAndUkeysFromSortedArray(sorted);
	      var keyInd = i - myD.low;
	      forall v in myUvals {
		localUvals[here.id][keyInd] += v;
	      }
	    }	
	  }
	}
      }
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] int;
      forall (keyInd, r) in zip(keyDom, res) {
	r = (+ reduce [i in PrivateSpace] localUvals[i][keyInd]).size;
      }
      return res;
    }
}

