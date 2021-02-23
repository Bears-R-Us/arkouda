module JoinEqWithDTMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Sort only;
    use Reflection;
    use Errors;
    use Logging;
    use Message;
    use PrivateDist;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use AryUtil;

    param TRUE_DT = 0;
    param ABS_DT = 1;
    param POS_DT = 2;
    
    const jeLogger = new Logger();
  
    if v {
        jeLogger.level = LogLevel.DEBUG;
    } else {
        jeLogger.level = LogLevel.INFO;
    }
    
    // operator overloads so + reduce and + scan can work on atomic int arrays
    proc +(x: atomic int, y: atomic int) {
        return x.read() + y.read();
    }
    proc +=(X: [?D] int, Y: [D] atomic int) {
        [i in D] {X[i] += Y[i].read();}
    }

    /*
      Stole this code from Bill's Merge.chpl until we merge the PR's
      This version does not throw exceptions
      Given a *sorted*, zero-up array, use binary search to find the index of the first element 
      that is greater than or equal to a target.
     */
    proc binarySearch(a: [?D] int, x: int): int {
        var l = 0;
        var r = a.size - 1;
        while l <= r {
            var mid = l + (r - l + 1)/2;
            if a[mid] < x {
                l = mid + 1;
            } else if a[mid] > x {
                r = mid - 1;
            } else { // this[mid] == s
                // Find the *first* match
                while (mid >= 0) && (a[mid] == x) {
                    mid -= 1;
                }
                return mid + 1;
            } 
        }
        return l;
    }

    // find matching value in ukeys and return found flag and range for segment
    proc findMatch(v: int, seg: [?segD] int, ukeys: [segD] int, perm: [?aD] int): (bool, range) {

        // default is not found and zero length segment
        var found = false;
        var segRange: range = 1..0;

        // binary search unique keys to find segment number
        // returns index of >= value
        var segNum = binarySearch(ukeys, v);

        // if v was greater than all ukeys segNum is greater than all segments
        if (segNum <= seg.domain.high) {
            // found!
            if (v == ukeys[segNum]) {
                found = true;
                // last segment?
                if (segNum == seg.domain.high) {
                    segRange = (seg[segNum])..(perm.domain.high);
                }
                // otherwise
                else {
                    segRange = (seg[segNum])..(seg[segNum+1] -1);
                }
            }
        }
        
        return (found, segRange);
    }

    // core join on equality with delta time predicate logic 
    proc joinEqWithDT(a1: [?a1D] int,
                      seg: [?segD] int,  ukeys: [segD] int, perm: [?a2D] int,
                      t1: [a1D] int, t2: [a2D] int, dt: int, pred: int,
                      resLimitPerLocale: int) {

        try! jeLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                            "resLimitPerLocale = %t".format(resLimitPerLocale));
        
        // allocate result arrays per locale
        var locResI: [PrivateSpace] [0..#resLimitPerLocale] int;
        var locResJ: [PrivateSpace] [0..#resLimitPerLocale] int;

        // atomic result counter per locale
        var resCounters: [PrivateSpace] atomic int;
        // actual number of results per locale
        var locNumResults: [PrivateSpace] int;
        
        coforall loc in Locales {
            on loc {
                forall i in a1.localSubdomain() {
                    // more space in result list???
                    if (resCounters[here.id].read() < resLimitPerLocale) {
                        // find matching value(unique key in g2) and
                        // return found flag and a range for the segment of that value(unique key)
                        var (found, j_seg) = findMatch(a1[i], seg, ukeys, perm);
                        try! jeLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "a1: %t found: %t j_seq: %t".format(a1[i], found, j_seg));
                        // if there is a matching value in ukeys
                        if (found) {
                            var t1_i = t1[i];
                            // all j's come from the original a2 array
                            // so all the values from perm over the segment for the value
                            for j in perm[j_seg] {
                                var addResFlag = false;
                                var t2_j = t2[j];
                                // which predicate to use
                                select pred {
                                        // absolute difference predicate
                                        when ABS_DT {
                                            if (t1_i <= t2_j) {
                                                addResFlag = ((t2_j - t1_i) <= dt);
                                            } else {
                                                addResFlag = ((t1_i - t2_j) <= dt);
                                            }
                                        }
                                        // positive difference predicate
                                        when POS_DT {
                                            if (t1_i <= t2_j) {
                                                addResFlag = ((t2_j - t1_i) <= dt);
                                            }
                                            else {
                                                addResFlag = false;
                                            }
                                        }
                                        // always true predicate
                                        when TRUE_DT {
                                            addResFlag = true;
                                        }
                                        otherwise {
                                            try! jeLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "OOPS! bad predicate number!"); 
                                        }
                                    }
                                // add result to list if predicate was true
                                if addResFlag {
                                    var pos = resCounters[here.id].fetchAdd(1);
                                    // add result if there is still room in the result list
                                    if (pos < resLimitPerLocale) {
                                        locResI[here.id][pos] = i;
                                        locResJ[here.id][pos] = j;
                                    }
                                    // there is no room left in list then break out of for loop
                                    else { break; }
                                }
                                // if there is no room left in list then break out of for loop
                                if (resCounters[here.id].read() >= resLimitPerLocale) { break; }
                            } // for j
                        } // if (found)
                    } // if more space in result list
                } // forall i
                // set locNumResults to correct value
                if (resCounters[here.id].read() > resLimitPerLocale) {
                    locNumResults[here.id] = resLimitPerLocale;
                }
                else {
                    locNumResults[here.id] = resCounters[here.id].read();
                }
            } // on loc
        } // coforall loc

        // +scan for all the local result ends
        // last value should be total results
        var resEnds: [PrivateSpace] int = + scan locNumResults;
        var numResults: int = resEnds[resEnds.domain.high];

        try! jeLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "result ends: %t".format(resEnds));
        try! jeLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "numResults = %i".format(numResults));
        
        // allocate result arrays
        var resI = makeDistArray(numResults, int);
        var resJ = makeDistArray(numResults, int);

        // move results per locale to result arrays
        coforall loc in Locales {
            on loc {
                // construct start and end for array assignment
                var gEnd: int = resEnds[here.id] - 1;
                var gStart: int = resEnds[here.id] - locNumResults[here.id];
                try! jeLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "start: %t end: %t".format(gStart,gEnd));
                // copy local results into global results
                resI[{gStart..gEnd}] = locResI[here.id][{0..#locNumResults[here.id]}];
                resJ[{gStart..gEnd}] = locResJ[here.id][{0..#locNumResults[here.id]}];                
            }
        }
        
        // return result arrays
        return (resI, resJ);
    }
    
    /* 
       joinEqWithDT is a specialized inner-join on equality between
       two integer arrays where the time-window predicate is also true.
       a1: is the first array
       g2: is a GroupBy of the second array a2
       (seg,ukeys,perm) are derived from a2 by doing a GroupBy
       t1: time stamp array corresponding to a1
       t2: time stamp array corresponding to a2
       dt: is the delta time
       pred: is the dt-predicate ("absDT","posDT","trueDT")
       resLimit: is how many answers can you tolerate ;-)
    */
    proc joinEqWithDTMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (a1_name, g2Seg_name, g2Ukeys_name, g2Perm_name, t1_name,
             t2_name, dtStr, predStr, resLimitStr) = payload.splitMsgToTuple(9);
        var dt = try! dtStr:int;
        var pred = predStr:int;
        var resLimit = try! resLimitStr:int;
        
        // get next symbol names for results
        var resI_name = st.nextName();
        var resJ_name = st.nextName();
        
        jeLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "cmd: %s array1: %s g2seg2: %s g2ukeys: %s g2perm: %s t1name: %s t2name: %s dt: %t pred: %t resLimit: %t resI: %s resJ: %s".format(
                                            cmd, a1_name, g2Seg_name, g2Ukeys_name, 
                                            g2Perm_name, t1_name, t2_name, dt, 
                                            pred, resLimit, resI_name, resJ_name));
        
        // check and throw if over memory limit
        overMemLimit(resLimit*6*8);
        
        // lookup arguments and check types
        // !!!!! check for DType.Int64 on all of these !!!!!
        // !!!!! check matching length on some arguments !!!!!
        var a1Ent: borrowed GenSymEntry = st.lookup(a1_name);
        if (a1Ent.dtype != DType.Int64) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, dtype2str(a1Ent.dtype)),
                        lineNumber = getLineNumber(),
                        routineName = getRoutineName(),
                        moduleName = getModuleName(),
                        errorClass="ErrorWithContext");
        }
        var a1 = toSymEntry(a1Ent, int);
        
        var g2SegEnt: borrowed GenSymEntry = st.lookup(g2Seg_name);
        if (g2SegEnt.dtype != DType.Int64) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, dtype2str(g2SegEnt.dtype)),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext");
        }
        var g2Seg = toSymEntry(g2SegEnt, int);
        
        var g2UkeysEnt: borrowed GenSymEntry = st.lookup(g2Ukeys_name);
        if (g2UkeysEnt.dtype != DType.Int64) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, dtype2str(g2UkeysEnt.dtype)),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext");
        }
        else if (g2UkeysEnt.size != g2SegEnt.size) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, "ukeys and seg must be same size"),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext"
                        );
        }
        var g2Ukeys = toSymEntry(g2UkeysEnt, int);
        
        var g2PermEnt: borrowed GenSymEntry = st.lookup(g2Perm_name);
        if (g2PermEnt.dtype != DType.Int64) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, dtype2str(g2PermEnt.dtype)),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext"
                        );
        }
        var g2Perm = toSymEntry(g2PermEnt, int);
        
        var t1Ent: borrowed GenSymEntry = st.lookup(t1_name);
        if (t1Ent.dtype != DType.Int64) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, dtype2str(t1Ent.dtype)),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext"
                        );
        }
        else if (t1Ent.size != a1Ent.size) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, "a1 and t1 must be same size"),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext"
                        );
        }
        var t1 = toSymEntry(t1Ent, int);
        
        var t2Ent: borrowed GenSymEntry = st.lookup(t2_name);
        if (t2Ent.dtype != DType.Int64) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, dtype2str(t2Ent.dtype)),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext"
                        );
        }
        else if (t2Ent.size != g2PermEnt.size) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, "a2 and t2 must be same size"),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext"
                        );
        }
        var t2 = toSymEntry(t2Ent, int);

        if (pred < 0) || (pred > 2) {
            throw getErrorWithContext(
                        msg=incompatibleArgumentsError(pn, "bad predicate number"),
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="ErrorWithContext"
                        );
        }

        var resLimitPerLocale: int = resLimit / numLocales;

        // call the join and return the result arrays
        var (resI, resJ) = joinEqWithDT(a1.a,
                                        g2Seg.a, g2Ukeys.a, g2Perm.a, // derived from a2
                                        t1.a, t2.a, dt, pred, resLimitPerLocale);
        
        // puth results in the symbol table
        st.addEntry(resI_name, new shared SymEntry(resI));
        st.addEntry(resJ_name, new shared SymEntry(resJ));
        
        repMsg = "created " + st.attrib(resI_name) + " +created " + st.attrib(resJ_name);
        jeLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }// end joinEqWithDTMsg()
    
}// end module JoinEqWithDTMsg
