module Utils {
    // Chapel modules.
    use IO;
    use Map;
    use Sort;
    use List;
    use Set;

    // Arachne modules.
    use GraphArray;

    // Arkouda modules. 
    use Logging;
    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;
    use ServerConfig;
    use ArgSortMsg;
    use AryUtil;

    // Allow graphs to be printed server-side? Defaulted to false. MUST BE MANUALLY CHANGED.
    // TODO: make this a param instead of a set variable?
    var debug_print = false; 

    // Server message logger. 
    private config const logLevel = LogLevel.DEBUG;
    const smLogger = new Logger(logLevel);
    private var outMsg:string;

    /** 
    * Helper procedure to parse ranges and return the locale(s) we must write to.
    * 
    * val: value for which we want to find the locale that owns it. 
    * 
    * returns: array of the locale names. */
    proc find_locs(val:int, graph:SegGraph) throws {
        var ranges = toSymEntry(graph.getComp("RANGES"), (int,locale)).a;
        var locs = new list(locale);
        for i in 1..numLocales - 1 {
            if (val >= ranges[i-1][0] && val <= ranges[i][0]) {
                locs.pushBack(ranges[i-1][1]);
            }
            if (i == numLocales - 1) {
                if val >= ranges[i][0] {
                    locs.pushBack(ranges[i][1]);
                }
            }
        }
        return locs.toArray();
    }

    /**
    * Binary search for a given key, original version by Dr. Du.
    *
    * ary: int array
    * l: low index bound
    * h: high index bound
    * key: value we are searching for
    *
    * returns: index if key is found, -1 if not found
    */
    proc bin_search_v(ref ary: [?D] int, l: int, h: int, key: int): int throws {
        if ( (l < D.lowBound) || (h > D.highBound) || (l < 0)) {
            return -1;
        }
        if ( (l > h) || ((l == h) &&  (ary[l] != key))) {
            return -1;
        }
        if (ary[l] == key) {
            return l;
        }
        if (ary[h] == key) {
            return h;
        }
        
        var m = (l + h) / 2: int;
        
        if ((m == l) ) {
            return -1;
        }
        if (ary[m] == key ) {
            return m;
        } else {
            if (ary[m] < key) {
                return bin_search_v(ary, m+1, h, key);
            }
            else {
                return bin_search_v(ary, l, m-1, key);
            }
        }
    }// end bin_search_v

    /**
    * Non-recursive, distributed-memory binary search for a given key.
    *
    * arr: int array
    * key: value we are searching for
    *
    * returns: index if key is found, -1 if not found
    */
    proc bin_search(arr: [?D] int, key: int, lo: int, hi: int, comparator:?rec=defaultComparator): int throws {
        var found:int = -1; // index of found key, -1 if not found.
        coforall loc in Locales with (ref found) do on loc {
            var start_loc:bool, end_loc:bool, mid_loc:bool, skip_loc:bool;
            var l:int, h:int, local_lo:int, local_hi:int;
            local_lo = arr.localSubdomain().lowBound;
            local_hi = arr.localSubdomain().highBound;

            // Check to see if loc is either a starting locale or an end locale.
            if arr.localSubdomain().contains(lo) then start_loc = true;
            else if arr.localSubdomain().contains(hi) then end_loc = true;
            else if !start_loc && !end_loc && local_lo > lo && local_hi < hi then mid_loc = true;
            else skip_loc = true;

            if !skip_loc {
                // Start the search from the actual lo index stored on start_loc.
                if start_loc {
                    l = if arr.localSubdomain().lowBound < lo then lo
                        else arr.localSubdomain().lowBound;
                } else l = arr.localSubdomain().lowBound;
                
                // End the search from the actual hi index stored on end_loc.
                if end_loc {
                    h = if arr.localSubdomain().highBound > hi then hi
                        else arr.localSubdomain().highBound;
                } else h = arr.localSubdomain().highBound;

                // Actual binary search steps. 
                while(l <= h) {
                    if arr[l] == key {found = l; break;}
                    if arr[h] == key {found = h; break;}
                    
                    const m = (l + h) / 2 : int;

                    if m == l then break;
                    if arr[m] == key {found = m; break;}
                    
                    if chpl_compare(key, arr[m], comparator=comparator) > 0 then l = m + 1;
                    else h = m - 1;
                }
            }
        } // end of coforall
        return found;
    }// end bin_search

    /**
    * Print graph data structure server-side to visualize the raw array data.
    *
    * G: graph we want to print out. 
    *
    * returns: message back to Python.
    */
    proc print_graph_serverside(G: borrowed SegGraph) throws {
        for comp in Component {
            var curr_comp = comp:string;
            if G.hasComp(curr_comp) {
                select curr_comp {
                    when "RELATIONSHIPS", "NODE_LABELS" {
                        var X = toSymEntry(G.getComp(comp:string), list(string, parSafe=true)).a;
                        writeln(comp:string, " = ", X);
                    }
                    when "NODE_PROPS", "EDGE_PROPS" {
                        var X = toSymEntry(G.getComp(comp:string), list((string,string), parSafe=true)).a;
                        writeln(comp:string, " = ", X);
                    }
                    when "EDGE_WEIGHT", "EDGE_WEIGHT_R" {
                        var X = toSymEntry(G.getComp(comp:string), real).a;
                        writeln(comp:string, " = ", X);
                    }
                    when "NODE_MAP_R" {
                        var X = toSymEntryAD(G.getComp(comp:string)).a;
                        writeln(comp:string, " = ", X);
                    }
                    otherwise {
                        var X = toSymEntry(G.getComp(comp:string), int).a;
                        writeln(comp:string, " = ", X);
                    }
                }
            }
        }
    } // end of print_graph_serverside
}