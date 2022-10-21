module GroupBy {
    use AryUtil;
    use CTypes;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerConfig;
    use Reflection;
    use Logging;
    use ServerErrors;
    use UniqueMsg;
    use CommAggregation;
    use Map;

    private config const logLevel = ServerConfig.logLevel;
    const gbLogger = new Logger(logLevel);

    proc getGroupBy(name: string, st: borrowed SymTab): owned GroupBy throws {
        var abstractEntry = st.lookup(name);
        if !abstractEntry.isAssignableTo(SymbolEntryType.GroupBySymEntry) {
            var errorMsg = "Error: Unhandled SymbolEntryType %s".format(abstractEntry.entryType);
            gbLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        var entry: GroupBySymEntry = abstractEntry: borrowed GroupBySymEntry();
        return new owned GroupBy(name, entry);
    }

    proc getGroupBy(keyCount: int, keys: [] string, keyTypes: [] string, assumeSorted: bool, st: borrowed SymTab): owned GroupBy throws {
        var keyNamesEntry = new shared SymEntry(keys);
        var keyTypesEntry = new shared SymEntry(keyTypes);
        var (permEntry, segmentsEntry) = uniqueAndCount(keyCount, keys, keyTypes, assumeSorted, st);

        var uniqueKeyInds = new shared SymEntry(segmentsEntry.size, int);
        if (segmentsEntry.size > 0) {
          // Avoid initializing aggregators if empty array
          ref perm = permEntry.a;
          ref segs = segmentsEntry.a;
          ref inds = uniqueKeyInds.a;
          forall (i, s) in zip(inds, segs) with (var agg = newSrcAggregator(int)) {
            agg.copy(i, perm[s]);
          }
        }

        // compute itemsize
        var itemsize: int;
        forall n in keyNamesEntry.a with (max reduce itemsize){
            var entry: borrowed GenSymEntry = getGenericTypedArrayEntry(n, st);
            itemsize = entry.itemsize;
        }

        var gbEntry = new shared GroupBySymEntry(keyNamesEntry, keyTypesEntry, segmentsEntry, permEntry, uniqueKeyInds, itemsize);
        var name = st.nextName();
        st.addEntry(name, gbEntry);
        return getGroupBy(name, st);
    }

    class GroupBy {
        var name: string;
        var composite: borrowed GroupBySymEntry;

        var keyNames: shared SymEntry(string);
        var keyTypes: shared SymEntry(string);
        var segments: shared SymEntry(int);
        var permutation: shared SymEntry(int);
        var uniqueKeyIndexes: shared SymEntry(int);

        var length: int;
        var ngroups: int;
        var nkeys: int;

        proc init(entryName: string, entry:borrowed GroupBySymEntry) {
            name = entryName;
            composite = entry;

            keyNames = composite.keyNamesEntry;
            keyTypes = composite.keyTypesEntry;
            segments = composite.segmentsEntry;
            permutation = composite.permEntry;
            uniqueKeyIndexes = composite.ukIndEntry;

            length = permutation.size;
            ngroups = segments.size;
            nkeys = keyNames.size;


            //For initial implementation, UniqueKeyIndexes returned to compute UniqueKeys client side
        }

        proc getSegmentsName(st: borrowed SymTab): string throws {
            var sname = st.nextName();
            st.addEntry(sname, segments);
            return sname;
        }

        proc getPermutationName(st: borrowed SymTab): string throws {
            var pname = st.nextName();
            st.addEntry(pname, permutation);
            return pname;
        }

        proc getUniqueKeyIndexName(st: borrowed SymTab): string throws {
            var ukname = st.nextName();
            st.addEntry(ukname, uniqueKeyIndexes);
            return ukname;
        }

        proc fillReturnMap(ref rm: map(string, string), st: borrowed SymTab) throws {
            rm.add("groupby", "created " + st.attrib(this.name));
            rm.add("segments", "created " + st.attrib(this.getSegmentsName(st)));
            rm.add("permutation", "created " + st.attrib(this.getPermutationName(st)));
            rm.add("uniqueKeyIdx", "created " + st.attrib(this.getUniqueKeyIndexName(st)));
        }
    }
}