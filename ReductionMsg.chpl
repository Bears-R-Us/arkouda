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
      var counts:[segments.aD] int;
      forall (c, low, i) in zip(counts, segments.a, segments.aD) {
	var high: int;
	if (i < segments.aD.high) {
	  high = segments.a[i+1] - 1;
	} else {
	  high = size - 1;
	}
	c = high - low + 1;
      }
      st.addEntry(rname, new shared SymEntry(counts));
      return try! "created " + st.attrib(rname);
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
	    var res = segMin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = segMax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var res = segArgmin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var res = segArgmax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "num_unique" {
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
	    var res = segMin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "max" {
	    var res = segMax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmin" {
	    var res = segArgmin(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
	  }
	  when "argmax" {
	    var res = segArgmax(values.a, segments.a);
	    st.addEntry(rname, new shared SymEntry(res));
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

    proc segSum(values:[] ?t, segments:[?D] int): [D] t {
      var res: [D] t;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = + reduce values[low..high];
      }
      return res;
    }

    proc segSum(values:[] bool, segments:[?D] int): [D] int {
      var res: [D] int;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = + reduce (values[low..high]:int);
      }
      return res;
    }

    proc segProduct(values:[] ?t, segments:[?D] int): [D] t {
      var res: [D] t;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = * reduce values[low..high];
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

    proc segMin(values:[] ?t, segments:[?D] int): [D] t {
      var res: [D] t;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = min reduce values[low..high];
      }
      return res;
    }

    proc segMax(values:[] ?t, segments:[?D] int): [D] t {
      var res: [D] t;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	r = max reduce values[low..high];
      }
      return res;
    }

    proc segArgmin(values:[] ?t, segments:[?D] int): [D] int {
      var res: [D] int;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	var segment: subdomain(values.domain) = values.domain[low..high];
	var (minVal, minInd) = minloc reduce zip(values[segment],segment);
	r = minInd;
      }
      return res;
    }

    proc segArgmax(values:[] ?t, segments:[?D] int): [D] int {
      var res: [D] int;
      forall (r, low, i) in zip(res, segments, D) {
	var high: int;
	if (i < D.high) {
	  high = segments[i+1] - 1;
	} else {
	  high = values.domain.high;
	}
	var segment: subdomain(values.domain) = values.domain[low..high];
	var (maxVal, maxInd) = maxloc reduce zip(values[segment],segment);
	r = maxInd;
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
      
}

