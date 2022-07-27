module SegmentedArray {
    use AryUtil;
    use CTypes;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerConfig;
    use Reflection;
    use Logging;
    use ServerErrors;
    use CommAggregation;
    use Time only Timer, getCurrentTime;
    use Map;

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
        var lengths: shared SymEntry(int);
        var size: int;
        var nBytes: int;

        proc init(entryName:string, entry:borrowed SegArraySymEntry, type eType) {
            name = entryName;
            composite = entry;
            segments = composite.segmentsEntry: shared SymEntry(int);
            values = composite.valuesEntry: shared SymEntry(eType);
            lengths = composite.lengthsEntry: shared SymEntry(int);

            size = segments.size;
            nBytes = values.size;

            // Note - groupby remaining client side because groupby does not have server side object
        }

        /* Retrieve one string from the array */
        proc this(idx: ?t) throws where t == int || t == uint {
            if (idx < segments.a.domain.low) || (idx > segments.a.domain.high) {
                throw new owned OutOfBoundsError();
            }
            // Start index of the segment
            var start: int = segments.a[idx:int];
            // end index
            var end: int = if idx:int == segments.a.domain.high then values.size-1 else segments.a[idx:int+1]-1;
            return values.a[start..end];
        }

        proc this(const slice: range(stridable=false)) throws {
            if (slice.low < segments.a.domain.low) || (slice.high > segments.a.domain.high) {
                saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                "Slice is out of bounds");
                throw new owned OutOfBoundsError();
            }
            // Early return for zero-length result
            if (size == 0) || (slice.size == 0) {
                return (makeDistArray(0, int), makeDistArray(0, values.etype));
            }

            ref sa = segments.a;
            var start = sa[slice.low];
            // end index
            var end: int = if slice.high == segments.a.domain.high then values.size-1 else sa[slice.high:int+1]-1;

            // Segment offsets of the new slice
            var newSegs = makeDistArray(slice.size, int);
            forall (i, ns) in zip(newSegs.domain, newSegs) with (var agg = newSrcAggregator(int)) {
                agg.copy(ns, sa[slice.low + i]);
            }

            // re-zero segments
            newSegs -= start;
            
            var newVals = makeDistArray(end - start + 1, values.etype);
            ref va = values.a;
            forall (i, nv) in zip(newVals.domain, newVals) with (var agg = newSrcAggregator(values.etype)) {
                agg.copy(nv, va[start + i]);
            }
            return (newSegs, newVals);
        }

        /* Gather segments by index. Returns arrays for the segments and values.*/
        proc this(iv: [?D] ?t) throws where t == int || t == uint {
            use ChplConfig;

            // Early return for zero-length result
            if (D.size == 0) {
                return (makeDistArray(0, int), makeDistArray(0, values.etype));
            }
            // Check all indices within bounds
            var ivMin = min reduce iv;
            var ivMax = max reduce iv;
            if (ivMin < 0) || (ivMax >= segments.size) {
                saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                    "Array out of bounds");
                throw new owned OutOfBoundsError();
            }
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                    "Computing lengths and offsets");
            var t1 = getCurrentTime();
            ref oa = segments.a;
            const low = segments.a.domain.low, high = segments.a.domain.high;

            // Gather the right and left boundaries of the indexed strings
            // NOTE: cannot compute lengths inside forall because agg.copy will
            // experience race condition with loop-private variable
            var right: [D] int, left: [D] int;
            forall (r, l, idx) in zip(right, left, iv) with (var agg = newSrcAggregator(int)) {
                if (idx == high) {
                    agg.copy(r, values.size);
                } else {
                    agg.copy(r, oa[idx:int+1]);
                }
                agg.copy(l, oa[idx:int]);
            }

            // Lengths of segments
            var gatheredLengths: [D] int = right - left;
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * gatheredLengths.size);
            // The returned offsets are the 0-up cumulative lengths
            var gatheredOffsets = (+ scan gatheredLengths);
            // The total number of bytes in the gathered strings
            var rtn = gatheredOffsets[D.high];
            gatheredOffsets -= gatheredLengths;
            
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                        "aggregation in %i seconds".format(getCurrentTime() - t1));
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Copying values");
            if logLevel == LogLevel.DEBUG {
                t1 = getCurrentTime();
            }
            var gatheredVals = makeDistArray(rtn, values.etype);
            if CHPL_COMM != 'none' {
                // Compute the src index for each byte in gatheredVals
                /* For performance, we will do this with a scan, so first we need an array
                with the difference in index between the current and previous byte. For
                the interior of a segment, this is just one, but at the segment boundary,
                it is the difference between the src offset of the current segment ("left")
                and the src index of the last byte in the previous segment (right - 1).
                */
                var srcIdx = makeDistArray(rtn, int);
                srcIdx = 1;
                var diffs: [D] int;
                diffs[D.low] = left[D.low]; // first offset is not affected by scan

                forall idx in D {
                    if idx!=0 {
                        diffs[idx] = left[idx] - (right[idx-1]-1);
                    }
                }
                // Set srcIdx to diffs at segment boundaries
                forall (go, d) in zip(gatheredOffsets, diffs) with (var agg = newDstAggregator(int)) {
                    agg.copy(srcIdx[go], d);
                }
                // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
                overMemLimit(numBytes(int) * srcIdx.size);
                srcIdx = + scan srcIdx;
                // Now srcIdx has a dst-local copy of the source index and vals can be efficiently gathered
                ref va = values.a;
                forall (v, si) in zip(gatheredVals, srcIdx) with (var agg = newSrcAggregator(values.etype)) {
                    agg.copy(v, va[si]);
                }
            } else {
                ref va = values.a;
                // Copy string data to gathered result
                forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, iv) {
                    for pos in 0..#gl {
                        gatheredVals[go+pos] = va[oa[idx:int]+pos];
                    }
                }
            }
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                    "Gathered offsets and vals in %i seconds".format(
                                                getCurrentTime() -t1));
            return (gatheredOffsets, gatheredVals);
        }

        /* Logical indexing (compress) of SegArray. */
        proc this(iv: [?D] bool) throws {
            // Index vector must be same domain as array
            if (D != segments.a.domain) {
                saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                                "Array out of bounds");
                throw new owned OutOfBoundsError();
            }
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                        "Computing lengths and offsets");

            ref oa = segments.a;
            const low = segments.a.domain.low, high = segments.a.domain.high;
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * iv.size);
            // Calculate the destination indices
            var steps = + scan iv;
            var newSize = steps[high];
            steps -= iv;
            // Early return for zero-length result
            if (newSize == 0) {
                return (makeDistArray(0, int), makeDistArray(0, values.etype));
            }
            var segInds = makeDistArray(newSize, int);
            forall (t, dst, idx) in zip(iv, steps, D) with (var agg = newDstAggregator(int)) {
                if t {
                    agg.copy(segInds[dst], idx);
                }
            }
            return this[segInds];
        }

        proc getNonEmpty() throws {
            return lengths.a > 0;
        }

        proc getNonEmptyCount() throws {
            var non_empty = getNonEmpty();
            return + reduce non_empty:int;
        }

        proc getComponentName(obj: SymEntry, st: borrowed SymTab): string throws {
            // early out if name exists
            if obj.name != "" {
                return obj.name;
            }
            var rname = st.nextName();
            st.addEntry(rname, obj);
            return obj.name;
        }

        proc fillReturnMap(ref rm: map(string, string), st: borrowed SymTab) throws {
            rm.add("segarray", "created " + st.attrib(this.name));
            rm.add("values", "created " + st.attrib(this.getComponentName(this.values, st)));
            rm.add("segments", "created " + st.attrib(this.getComponentName(this.segments, st)));
            rm.add("lengths", "created " + st.attrib(this.getComponentName(this.lengths, st)));
        }
    }
}
