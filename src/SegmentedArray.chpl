module SegmentedArray {
    use AryUtil;
    use CTypes;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerConfig;
    use Reflection;
    use Logging;
    use ServerErrors;
    use List;

    private config const logLevel = ServerConfig.logLevel;
    const saLogger = new Logger(logLevel);

    proc getSegArray(name: string, st: borrowed SymTab, type eltType): owned SegArray throws {
        var abstractEntry = st.lookup(name);
        if !abstractEntry.isAssignableTo(SymbolEntryType.SegArraySymEntry) {
            var errorMsg = "Error: Unhandled SymbolEntryType %s".format(abstractEntry.entryType);
            saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        var entry: SegArraySymEntry = abstractEntry: borrowed SegArraySymEntry(eltType);
        return new owned SegArray(name, entry, eltType);
    }

    /*
    * This version of the getSegArray method takes segments and values arrays as
    * inputs, generates the SymEntry objects for each and passes the
    * offset and value SymTab lookup names to the alternate init method
    */
    proc getSegArray(segments: [] int, values: [] ?t, st: borrowed SymTab): owned SegArray throws {
        var segmentsEntry = new shared SymEntry(segments);
        var valuesEntry = new shared SymEntry(values);
        var segEntry = new shared SegArraySymEntry(segmentsEntry, valuesEntry, t);
        var name = st.nextName();
        st.addEntry(name, segEntry);
        return getSegArray(name, st, segEntry.etype);
    }

    class SegArray {
        var name: string;

        var composite: borrowed SegArraySymEntry;

        var segments: shared SymEntry(int);
        var values;
        var size: int;
        var nBytes: int;

        proc init(entryName:string, entry:borrowed SegArraySymEntry, type eType) {
            name = entryName;
            composite = entry;
            segments = composite.segmentsEntry: shared SymEntry(int);
            values = composite.valuesEntry: shared SymEntry(eType);
            
            size = segments.size;
            nBytes = values.size;

            // Note - groupby remaining client side because groupby does not have server side object
        }

        proc getLengths() {
            // Format the same as in SegmentedString
            //Note that this logic may need to move into init when everything move to prevent recompute
            var lengths: [segments.aD] int;
            if (size == 0) {
                return lengths;
            }
            ref sa = segments.a;
            const low = segments.aD.low;
            const high = segments.aD.high;
            forall (i, s, l) in zip(segments.aD, sa, lengths) {
                if (i == high) {
                    l = values.size - s;
                } else {
                    l = sa[i+1] - s;
                }
            }
            return lengths;
        }

        proc getNonEmpty() throws {
            return getLengths() > 0;
        }

        proc getNonEmptyCount() throws {
            var non_empty = getNonEmpty();
            return + reduce non_empty:int;
        }
    }
}