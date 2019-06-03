module ReductionMsg
{
    use ServerConfig;

    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
      
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
                        var prod = * reduce e.a;
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

    proc segCount(segments:[?D] int, size: int):[D] int {
      var counts:[D] int;
      forall (c, low, i) in zip(counts, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = size - 1;
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
      var origSubdomains = makeDistDom(size);
      var numKeys:int = segments.size / numLocales;
      var localCounts: [D] int;
      coforall loc in Locales {
	on loc {
	  var myHigh = origSubdomains.localSubdomain().high;
	  var myDom = segments.localSubdomain();
	  ref mySeg = segments[myDom];
	  forall (i, low, c) in zip(myDom, mySeg, localCounts[myDom]) {
	    var high: int;
	    if (i < myDom.high) {
	      high = mySeg[i+1] - 1;
	    } else {
	      high = myHigh;
	    }
	    c = high - low + 1;
	  }
	}
      }
      var counts = makeDistArray(numKeys, int);
      forall i in counts.domain {
	counts[i] = + reduce localCounts[i.. by numKeys];
      }
      return counts;
    }


    proc segmentedReductionMsg(reqMsg: string, st: borrowed SymTab): string {
      // reqMsg: segmentedReduction values segments operator
      var fields = reqMsg.split();
      var cmd = fields[1];
      var values_name = fields[2];   // segmented array of values to be reduced
      var segments_name = fields[3]; // segment offsets
      var operator = fields[4];      // reduction operator
      var rname = st.nextName();
      if v {try! writeln("%s %s %s %s".format(cmd,values_name,segments_name,operator));try! stdout.flush();}

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
	    var (count, res) = segSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "prod" {
	    var (count, res) = segProduct(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var (count, res) = segMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "min" {
	    var (count, res) = segMin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var (count, res) = segMax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var (count, res) = segArgmin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var (count, res) = segArgmax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
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
	    var (count, res) = segSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "prod" {
	    var (count, res) = segProduct(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var (count, res) = segMean(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "min" {
	    var (count, res) = segMin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var (count, res) = segMax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var (count, res) = segArgmin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var (count, res) = segArgmax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedReduction",operator,gVal.dtype);}
	  }
      }
      when (DType.Bool) {
	var values = toSymEntry(gVal, bool);
	select operator {
	  when "sum" {
	    var (count, res) = segSum(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "any" {
	    var (count, res) = segAny(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "all" {
	    var (count, res) = segAll(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "mean" {
	    var (count, res) = segMean(values.a, segments.a);
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
      // reqMsg: segmentedReduction values segments operator
      var fields = reqMsg.split();
      var cmd = fields[1];
      var values_name = fields[2];   // segmented array of values to be reduced
      var segments_name = fields[3]; // segment offsets
      var operator = fields[4];      // reduction operator
      var rname = st.nextName();
      if v {try! writeln("%s %s %s %s".format(cmd,values_name,segments_name,operator));try! stdout.flush();}

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
	    var res = perLocMin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = perLocMax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var res = perLocArgmin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var res = perLocArgmax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "nunique" {
	    var res = perLocNumUnique(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedReduction",operator,gVal.dtype);}
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
	    var res = perLocMin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = perLocMax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var res = perLocArgmin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var res = perLocArgmax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  otherwise {return notImplementedError("segmentedReduction",operator,gVal.dtype);}
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
	  otherwise {return notImplementedError("segmentedReduction",operator,gVal.dtype);}
	  }
      }
      otherwise {return unrecognizedTypeError("segmentedReduction", dtype2str(gVal.dtype));}
      }
      return try! "created " + st.attrib(rname);
    }
	  

    proc segSum(values:[] ?t, segments:[?D] int): ([D] int, [D] t) {
      var res: [D] t;
      var count: [D] int;
      forall (r, c, low, i) in zip(res, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	r = + reduce values[low..high];
      }
      return (count, res);
    }

    proc perLocSum(values:[] ?t, segments:[?D] int): [] t {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] atomic t;
      coforall loc in Locales {
	on loc {
	  var (myCounts, myVals) = segSum(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, v, i) in zip(myCounts, myVals, 0..#numKeys) {
	    if (c > 0) {
	      res[i].add(v);
	    }
	  }
	}
      }
      return [v in res] v.read();
    }

    proc segSum(values:[] bool, segments:[?D] int): ([D] int, [D] int) {
      var count: [D] int;
      var res: [D] int;
      forall (r, c, low, i) in zip(res, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	r = + reduce (values[low..high]:int);
      }
      return (count, res);
    }

    proc perLocSum(values:[] bool, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] atomic int;
      coforall loc in Locales {
	on loc {
	  var (myCounts, myVals) = segSum(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, v, i) in zip(myCounts, myVals, 0..#numKeys) {
	    if (c > 0) {
	      res[i].add(v);
	    }
	  }
	}
      }
      return [v in res] v.read();
    }
    
    proc segProduct(values:[], segments:[?D] int): ([D] int, [D] real) {
      var count: [D] int;
      var res: [D] real = 1;
      forall (r, c, low, i) in zip(res, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	r = * reduce values[low..high]:real;
      }
      return (count, res);
    }

    proc perLocProduct(values:[] ?t, segments:[?D] int): [] real {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] real = 1;
      var lock$: [keyDom] sync bool = [i in keyDom] true;
      coforall loc in Locales {
	on loc {
	  var (myCounts, myVals) = segProduct(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, v, i) in zip(myCounts, myVals, 0..#numKeys) {
	    if (c > 0) {
	      lock$[i];
	      res[i] *= v;
	      lock$[i] = true;
	    }
	  }
	}
      }
      return res;
    }
    
    proc segMean(values:[] ?t, segments:[?D] int): [D] real {
      var res: [D] real;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = (+ reduce values[low..high]):real / (high - low + 1):real;
      }
      return res;
    }

    proc perLocMean(values:[] ?t, segments:[?D] int): [] real {
      var numKeys:int = segments.size / numLocales;
      var keyCounts = perLocCount(segments, values.size);
      var res = perLocSum(values, segments);
      return res:real / keyCounts:real;
    }

    proc segMin(values:[] ?t, segments:[?D] int): ([D] int, [D] t) {
      var count: [D] int;
      var res: [D] t;
      forall (r, c, low, i) in zip(res, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	r = min reduce values[low..high];
      }
      return (count, res);
    }
    
    proc perLocMin(values:[] ?t, segments:[?D] int): [] t {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var lock$: [keyDom] sync bool = [i in keyDom] true;
      var res: [keyDom] t = max(t);
      coforall loc in Locales {
	on loc {
	  var (myCounts, myVals) = segMin(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, v, i) in zip(myCounts, myVals, 0..#numKeys) {
	    if (c > 0) {
	      lock$[i];
	      if (v < res[i]) {
		res[i] = v;
	      }
	      lock$[i] = true;
	    }
	  }
	}
      }
      return res;
    }    

    proc segMax(values:[] ?t, segments:[?D] int): ([D] int, [D] t) {
      var count: [D] int;
      var res: [D] t;
      forall (r, c, low, i) in zip(res, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	r = max reduce values[low..high];
      }
      return (count, res);
    }

    proc perLocMax(values:[] ?t, segments:[?D] int): [] t {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var lock$: [keyDom] sync bool = [i in keyDom] true;
      var res: [keyDom] t = min(t);
      coforall loc in Locales {
	on loc {
	  var (myCounts, myVals) = segMax(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, v, i) in zip(myCounts, myVals, 0..#numKeys) {
	    if (c > 0) {
	      lock$[i];
	      if (v > res[i]) {
		res[i] = v;
	      }
	      lock$[i] = true;
	    }
	  }
	}
      }
      return res;
    }
    
    proc segArgmin(values:[] ?t, segments:[?D] int): ([D] int, [D] int, [D] t) {
      var count: [D] int;
      var locs: [D] int;
      var vals: [D] int;
      forall (l, v, c, low, i) in zip(locs, vals, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	if (high < low) {
	  v = max(t);
	  l = -1; // no values in this segment, so return a sentinel index
	} else {
	  var segment: subdomain(values.domain) = values.domain[low..high];
	  var (minVal, minInd) = minloc reduce zip(values[segment],segment);
	  v = minVal;
	  l = minInd;
	}
      }
      return (count, locs, vals);
    }

    proc perLocArgmin(values:[] ?t, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var lock$: [keyDom] sync bool = [i in keyDom] true;
      var locs: [keyDom] int;
      var vals: [keyDom] t = max(t);
      coforall loc in Locales {
	on loc {
	  var (myCounts, myLocs, myVals) = segArgmin(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, l, v, i) in zip(myCounts, myLocs, myVals, 0..#numKeys) {
	    if (c > 0) {
	      lock$[i];
	      if (v < vals[i]) {
		vals[i] = v;
		locs[i] = l;
	      } else if (v == vals[i]) && (l < locs[i]) {
		locs[i] = l;
	      }
	      lock$[i] = true;
	    }
	  }
	}
      }
      return locs;
    }
    
    proc segArgmax(values:[] ?t, segments:[?D] int): ([D] int, [D] int, [D] t) {
      var count: [D] int;
      var locs: [D] int;
      var vals: [D] int;
      forall (l, v, c, low, i) in zip(locs, vals, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	if (high < low) {
	  v = min(t);
	  l = -1; // no values in this segment, so return a sentinel index
	} else {
	  var segment: subdomain(values.domain) = values.domain[low..high];
	  var (maxVal, maxInd) = maxloc reduce zip(values[segment],segment);
	  v = maxVal;
	  l = maxInd;
	}
      }
      return (count, locs, vals);
    }

    proc perLocArgmax(values:[] ?t, segments:[?D] int): [] int {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var lock$: [keyDom] sync bool = [i in keyDom] true;
      var locs: [keyDom] int;
      var vals: [keyDom] t = min(t);
      coforall loc in Locales {
	on loc {
	  var (myCounts, myLocs, myVals) = segArgmax(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, l, v, i) in zip(myCounts, myLocs, myVals, 0..#numKeys) {
	    if (c > 0) {
	      lock$[i];
	      if (v > vals[i]) {
		vals[i] = v;
		locs[i] = l;
	      } else if (v == vals[i]) && (l < locs[i]) {
		locs[i] = l;
	      }
	      lock$[i] = true;
	    }
	  }
	}
      }
      return locs;
    }
    
    proc segAny(values:[] bool, segments:[?D] int): ([D] int, [D] bool) {
      var count: [D] int;
      var res: [D] bool;
      forall (r, c, low, i) in zip(res, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	r = || reduce values[low..high];
      }
      return (count, res);
    }

    proc perLocAny(values:[] bool, segments:[?D] int): [] bool {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] bool;
      coforall loc in Locales {
	on loc {
	  var (myCounts, myVals) = segAny(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, v, i) in zip(myCounts, myVals, 0..#numKeys) {
	    if (c > 0) && v {
	      // Does not need to be atomic, because race conditions will still produce the correct answer
	      res[i] = true;
	    }
	  }
	}
      }
      return res;
    }
    
    proc segAll(values:[] bool, segments:[?D] int): [D] bool {
      var count: [D] int;
      var res: [D] bool;
      forall (r, c, low, i) in zip(res, count, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	c = high - low + 1;
	r = && reduce values[low..high];
      }
      return (count, res);
    }

    proc perLocAll(values:[] bool, segments:[?D] int): [] bool {
      var numKeys:int = segments.size / numLocales;
      var keyDom = makeDistDom(numKeys);
      var res: [keyDom] bool = true;
      coforall loc in Locales {
	on loc {
	  var (myCounts, myVals) = segAll(values[values.localSubdomain()], segments[D.localSubdomain()]);
	  forall (c, v, i) in zip(myCounts, myVals, 0..#numKeys) {
	    if (c > 0) && !v {
	      // Does not need to be atomic, because race conditions will still produce the correct answer
	      res[i] = false;
	    }
	  }
	}
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
      // First get all per-locale sets of unique values
      var localUnique: [D] domain(int);
      coforall loc in Locales {
	on loc {
	  var myD = D.localSubdomain();
	  forall (i, low, myU) in zip(myD, segments[myD], localUnique[D]) {
	    var high: int;
	    if (i < myD.high) {
	      high = segments[i+1] - 1;
	    } else {
	      high = values.localSubdomain().high;
	    }
	    var domLock$:sync bool = true;
	    forall v in values[values.localSubdomain()][low..high] with (ref myU, ref domLock$) {
	      if !myU.contains(v) {
		domLock$;
		if !myU.contains(v) {
		  myU += v; // will this propagate, or do we need a ref intent?
		}
		domLock$ = true;
	      }
	    }
	  }
	}
      }
      // For each key, union all sets of unique values across locales
      var numKeys:int = segments.size / numLocales;
      var res = makeDistArray(numKeys, int);
      forall i in res.domain {
	var globalUnique: domain(int); // set of all unique values for key i
	// Does not make sense to do this in parallel
	for locU in localUnique[i.. by numKeys] {
	  globalUnique += locU;
	}
	res[i] = globalUnique.size;
      }
      return res;
    }
}

