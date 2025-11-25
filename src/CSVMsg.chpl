module CSVMsg {
    use CommAggregation;
    use GenSymIO;
    use List;
    use Reflection;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use NumPyDType;
    use ServerConfig;
    use ServerErrors;
    use ServerErrorStrings;
    use SegmentedString;
    use FileSystem;
    use Sort;
    use FileIO;
    use Set;
    use IOUtils;

    const CSV_HEADER_OPEN = "**HEADER**";
    const CSV_HEADER_CLOSE = "*/HEADER/*";
    const LINE_DELIM = "\n"; // currently assumed all files are newline delimited.

    private config const logLevel = ServerConfig.logLevel;
    const csvLogger = new Logger(logLevel);

    // Future Work (TODO)
    //  - write to single file
    //  - Custom Line Delimiters
    //  - Handle CSV without column names

    proc lsCSVMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var filename = msgArgs.getValueOf("filename");
        if filename.isEmpty() {
            var errorMsg = "Filename was Empty";
            csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // If the filename represents a glob pattern, retrieve the locale 0 filename
        if isGlobPattern(filename) {
            // Attempt to interpret filename as a glob expression and ls the first result
            const tmp = glob(filename);
            csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "glob-expanded filename: %s to size: %i files".format(filename, tmp.size));

            if tmp.size <= 0 {
                var errorMsg = "Cannot retrieve filename from glob expression %s, check file name or format".format(filename);
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }

            // Set filename to globbed filename corresponding to locale 0
            filename = tmp[tmp.domain.first];
        }

        // Check to see if the file exists. If not, return an error message
        if !exists(filename) {
            var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filename);
            csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg,MsgType.ERROR);
        }

        // open file and determine if header exists.
        var idx = 0;
        var reader = openReader(filename);
        var line = readCSVRecord(reader);
        var hasHeader = false;
        if line == CSV_HEADER_OPEN {
            hasHeader=true;
            // The first three lines are headers we don't care about
            // Advance through them without reading them in
            for param i in 1..<3 {
                try {reader.advanceThrough(b'\n');}
                catch {
                    throw getErrorWithContext(
                        msg="This CSV file is missing header values when headers were detected.",
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IOError");
                }
            }
        }

        const col_delim: string = msgArgs.getValueOf("col_delim");
        // If the CSV File has headers, then we haven't actually read the column names yet
        // so read the next line
        if hasHeader then line = readCSVRecord(reader);
        var column_names: list(string);
        for field in parseCSVRecord(line, col_delim) {
            column_names.pushBack(field.strip());
        }
        const column_names_array: [0..<column_names.size] string = column_names.toArray();
        return new MsgTuple(formatJson(column_names_array), MsgType.NORMAL);

    }

    proc prepFiles(filename: string, overwrite: bool, A) throws {
        const (prefix,extension) = getFileMetadata(filename);

        const targetSize: int = A.targetLocales().size;
        // TODO maybe make filenames distributed? since we are accessing it within a coforall
        // and do forall (i,f) in zip(filesnames.domain, filenames)
        var filenames: [0..#targetSize] string;
        forall i in 0..#targetSize {
            // TODO think about inlining generateFilename (and potential impact on other IO methods)
            filenames[i] = generateFilename(prefix, extension, i);
        }

        csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Identified %i files for provided name, %s".format(filenames.size, filename));

        const matchingFilenames = glob("%s_LOCALE*%s".format(prefix, extension));
        const filesExist: bool = matchingFilenames.size > 0;

        if !overwrite && filesExist {
            throw getErrorWithContext(
                msg="Filenames for the provided name exist. Overwrite must be set to true in order to save with the name %s".format(filename),
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass="InvalidArgumentError");
        }
        else {
            if overwrite {
                coforall loc in A.targetLocales() do on loc {
                    const fn = filenames[loc.id].localize();
                    const existList = glob(fn);
                    if existList.size == 1 {
                        remove(fn);
                        csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Overwriting CSV File, %s".format(fn));
                    }
                }
            }
        }
        return filenames;
    }

    proc getLocalDomain(GSE: GenSymEntry) throws {
        select GSE.dtype {
            when DType.Int64 {
                var e = toSymEntry(GSE, int);
                return e.a.localSubdomain();
            }
            when DType.UInt64 {
                var e = toSymEntry(GSE, uint);
                return e.a.localSubdomain();
            }
            when DType.Float64 {
                var e = toSymEntry(GSE, real);
                return e.a.localSubdomain();
            }
            when DType.Bool {
                var e = toSymEntry(GSE, bool);
                return e.a.localSubdomain();
            }
            when DType.Strings {
                var e = GSE: borrowed SegStringSymEntry;
                var ss = new SegString("", e);
                return ss.offsets.a.localSubdomain();
            }
            otherwise {
                throw getErrorWithContext(
                    msg="Invalid DType Found, %s".format(dtype2str(GSE.dtype)),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="DataTypeError");
            }
        }
    }

    proc writeCSVMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const filename = msgArgs.getValueOf("filename");
        const ndsets = msgArgs.get("num_dsets").getIntValue();
        const datasets = msgArgs.get("datasets").getList(ndsets);
        const dtypes = msgArgs.get("dtypes").getList(ndsets);
        const col_names = msgArgs.get("col_names").getList(ndsets);
        const col_delim = msgArgs.getValueOf("col_delim");
        const row_count = msgArgs.get("row_count").getIntValue();
        const overwrite = msgArgs.get("overwrite").getBoolValue();

        // access the first SymEntry -> all SymEntries should have same locality due to them being required to be the same size.
        var gse = toGenSymEntry(st[datasets[0]]);
        var filenames;
        select gse.dtype {
            when DType.Int64 {
                var e = toSymEntry(gse, int);
                filenames = prepFiles(filename, overwrite, e.a);
            }
            when DType.UInt64 {
                var e = toSymEntry(gse, uint);
                filenames = prepFiles(filename, overwrite, e.a);
            }
            when DType.Float64 {
                var e = toSymEntry(gse, real);
                filenames = prepFiles(filename, overwrite, e.a);
            }
            when DType.Bool {
                var e = toSymEntry(gse, bool);
                filenames = prepFiles(filename, overwrite, e.a);
            }
            when DType.Strings {
                var e = gse: borrowed SegStringSymEntry;
                var ss = new SegString("", e);
                filenames = prepFiles(filename, overwrite, ss.offsets.a);
            }
            otherwise {
                throw getErrorWithContext(
                           msg="Invalid DType Found, %s".format(dtype2str(gse.dtype)),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(),
                           moduleName=getModuleName(),
                           errorClass="DataTypeError");
            }
        }

        // only use 90% of getMemLimit to give ourselves some buffer
        // these are each per locale measurements
        const memLim = (getMemLimit() * (.9)):uint;
        const memUsed = getMemUsed();

        // calculate the amount of memory required to write the csv file all in one go
        var memRequired: uint = 0;
        var numStrings = 0;
        for (cname, dt) in zip(datasets, dtypes) {
            if dt == "str" {
                var ss = new SegString("", toSegStringSymEntry(st[cname]));
                // strings vals are uint(8), so they're only one byte but overcount
                // because they're not evenly distributed across locales
                memRequired += (ss.values.a.size * 1.5):int;
                numStrings += 1;
            }
        }
        // assume pdarrays are 64 bits / 8 bytes (this is an overestimate for bools, but that's fine)
        memRequired += ((ndsets-numStrings) * row_count * 8);
        memRequired = (memRequired:real / numLocales):uint;

        // if we can put the entire df in mem (i.e. memLim > (memUsed + memRequired)), then we can do it all in one go.
        // Otherwise we can only put (memLim - memUsed) in memory at a time, so that's our batchSize
        // if memLim > memUsed then we use chunks no bigger than 5% of total memory (memLim is 90% so dividing by 18 gives us 5%)
        const numBatches = if memLim > memUsed then ceil(memRequired:real / (memLim - memUsed)):int else ceil(memRequired:real / (memLim / 18)):int;

        csvLogger.info(getModuleName(),getRoutineName(),getLineNumber(), "Start csv write with %i batches".format(numBatches));

        coforall loc in Locales do on loc {
            const localeFilename = filenames[loc.id];

            // create the file to write to
            const csvFile = open(localeFilename, ioMode.cw);
            const writer = csvFile.writer(locking=false);

            // write the header
            writer.write(CSV_HEADER_OPEN + LINE_DELIM);
            writer.write(",".join(dtypes) + LINE_DELIM);
            writer.write(CSV_HEADER_CLOSE + LINE_DELIM);
            writer.write(col_delim.join(col_names) + LINE_DELIM);

            // need to get local subdomain -- should be the same for each element due to sizes being same
            const localSubdomain = getLocalDomain(gse);
            const batchSize = (localSubdomain.size / numBatches):int;
            for N in 0..<numBatches {
                const batchSlice = if (N != (numBatches - 1)) then (N*batchSize + localSubdomain.first)..<((N+1)*batchSize + localSubdomain.first) else (N*batchSize + localSubdomain.first)..localSubdomain.last;
                var nativeStr: [batchSlice] string;
                for (i, cname) in zip(0..#ndsets, datasets) {
                    var col_gen: borrowed GenSymEntry = getGenericTypedArrayEntry(cname, st);
                    select col_gen.dtype {
                        when DType.Int64 {
                            var col = toSymEntry(col_gen, int);
                            nativeStr += [i in col.a.localSlice[batchSlice]] i:string;
                        }
                        when DType.UInt64 {
                            var col = toSymEntry(col_gen, uint);
                            nativeStr += [i in col.a.localSlice[batchSlice]] i:string;
                        }
                        when DType.Float64 {
                            var col = toSymEntry(col_gen, real);
                            nativeStr += [i in col.a.localSlice[batchSlice]] i:string;
                        }
                        when DType.Bool {
                            var col = toSymEntry(col_gen, bool);
                            nativeStr += [i in col.a.localSlice[batchSlice]] i:string;
                        }
                        when DType.Strings {
                            var segString:SegString = getSegString(cname, st);
                            nativeStr += try! [i in batchSlice] segString[i];
                        } otherwise {
                            throw getErrorWithContext(
                                    msg="Data Type %s cannot be written to CSV.".format(dtypes[i]),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="IOError"
                            );
                        }
                    }
                    if i != (ndsets - 1) {
                        nativeStr += col_delim;
                    }
                    else {
                        nativeStr += LINE_DELIM;
                    }
                }
                // TODO might be able to better than looping the elements
                //  (calling write directly on nativeStr gives spaces between elements)
                for s in nativeStr {
                    writer.write(s);
                }
            }
            writer.close();
            csvFile.close();
        }

        return new MsgTuple("CSV Data written successfully!", MsgType.NORMAL);
    }

    proc get_info(filename: string, datasets: [] string, col_delim: string) throws {
        // Verify that the file exists
        if !exists(filename) {
            throw getErrorWithContext(
                           msg="The file %s does not exist".format(filename),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(),
                           moduleName=getModuleName(),
                           errorClass="FileNotFoundError");
        }

        var reader = openReader(filename);
        var hasHeader = false;

        var line = readCSVRecord(reader);
        if line == CSV_HEADER_OPEN {
            hasHeader = true;
            // The first three lines are headers
            for param i in 1..<3 {
                try {
                    if i==1 {
                        line = readCSVRecord(reader);
                        // Line has the data types after this
                    } else {
                        // Advance through CSV_HEADER_CLOSE line
                        reader.advanceThrough(b'\n');
                    }
                }
                catch {
                    throw getErrorWithContext(
                        msg="This CSV file is missing header values when headers were detected.",
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IOError");
                }
            }
        }
        var column_names: string;
        if hasHeader then
            column_names = readCSVRecord(reader);
        else
            column_names = line;
        var columns: list(string);
        for field in parseCSVRecord(column_names, col_delim) {
            columns.pushBack(field.strip());
        }
        var file_dtypes: [0..<columns.size] string;
        if hasHeader {
            file_dtypes = line.split(",").strip(); // Line was already read above
        }
        else {
            file_dtypes = "str";
        }
        // get the row count - use fast quote-aware counting
        var (row_ct, hasQuotes) = countRowsAndQuotes(reader);

        reader.close();

        var dtypes: [0..#datasets.size] string;
        forall (i, dset) in zip(0..#datasets.size, datasets) {
            var idx: int;
            const col_exists = columns.find(dset, idx);
            csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Column: %s, Exists: ".format(dset)+formatJson(col_exists)+", IDX: %i".format(idx));
            if !col_exists {
                throw getErrorWithContext(
                    msg="The dataset %s was not found in %s".format(dset, filename),
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="DatasetNotFoundError");
            }
            dtypes[i] = file_dtypes[idx];
        }

        // get datatype of column
        return (row_ct, hasHeader, hasQuotes, new list(dtypes));
    }

    proc read_csv_pattern(ref A: [] ?t, filename: string, filedom: domain(1), colName: string, colDelim: string, hasHeaders: bool, lineOffset: int, allowErrors: bool, ref hadError: bool, const hasQuotes: bool) throws {
        // We do a check to see if the filedom even intersects with
        // A.localSubdomain before even opening the file
        // The implementation assumes a single local subdomain so we make
        // sure that is the case first
        if !A.hasSingleLocalSubdomain() then
            throw getErrorWithContext(
                msg="The array A must have a single local subdomain on locale %i".format(here.id),
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass="InvalidArgumentError");
        const intersection = domain_intersection(A.localSubdomain(), filedom);
        if(intersection.size == 0) {
            // There is nothing to be done for this file on this locale
            return;
        }

        var fr = openReader(filename);
        if(hasHeaders) {
            var line:string;
            // The first three lines are headers we don't care about
            // Advance through them without reading them in
            for param i in 0..<3 {
                try {fr.advanceThrough(b'\n');}
                catch {
                    throw getErrorWithContext(
                        msg="This CSV file is missing header values when headers were detected. This error should not arise.",
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass="IOError");
                }
            }
        }
        var colIdx = -1;
        // This next line will have the column header
        // Use proper CSV parsing to handle quoted delimiters
        const headerLine = readCSVRecord(fr);
        const headerFields = parseCSVRecord(headerLine, colDelim);
        for (column, columnIndex) in zip(headerFields, 0..) {
            if column == colName {
                colIdx = columnIndex;
                break;
            }
        }

        if(colIdx == -1) then
            throw getErrorWithContext(
                msg="The dataset %s was not found in %s".format(colName, filename),
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass="DatasetNotFoundError");

        // The same file might have data meant to be distributed across locales,
        // Therefore the 0th record in the file may not correspond to the 0th index of the array
        // So we skip over complete CSV records till we get the lower bound of the intersection
        // But the filedom may not start at 0, so we need to subtract that offset
        for 0..<(intersection.low-filedom.low) {
            try {
                advanceCSVRecord(fr);  // Skip complete CSV records, not just lines
            }
            catch {
                throw getErrorWithContext(
                    msg="This CSV file is missing records.",
                    lineNumber=getLineNumber(),
                    routineName=getRoutineName(),
                    moduleName=getModuleName(),
                    errorClass="IOError");
            }
        }

        var line = new csvLine(t, colIdx, colDelim, hasQuotes);
        for x in intersection {
            try {
                fr.read(line);
            } catch e: BadFormatError {
                // Handle errors
                if t!=string {
                    if allowErrors {
                        A[x] = min(t);
                        continue;
                    } else {
                        hadError |= true;
                    }
                } else {
                    A[x]="";
                }
            }
            // store the items from the csvLine in the array
            A[x] = line.item;
        }
    }

    use Regex;

    // Lightweight helper: advance through a CSV record without storing lines (for skip/count)
    // Returns: hasQuotes
    proc advanceCSVRecord(reader: fileReader(?)) : bool throws {
        var line: string;
        var hasQuotes = false;

        // Read the first line
        line = reader.readLine(stripNewline=true);

        // Quick check for quotes
        if line.find('"') != -1 {
            hasQuotes = true;
            var totalQuotes = line.count('"');

            // If odd number of quotes, keep reading until balanced
            while totalQuotes % 2 == 1 {
                try {
                    line = reader.readLine(stripNewline=true);
                    totalQuotes += line.count('"');
                } catch e: EofError {
                    break;
                } catch e: UnexpectedEofError {
                    break;
                }
            }
        }

        return hasQuotes;
    }

    // Full helper: read and store lines for complete CSV record (for readCSVRecord)
    proc readCSVRecordLines(reader: fileReader(?)) : list(string) throws {
        var lines: list(string);
        var line: string;

        // Read the first line
        line = reader.readLine(stripNewline=true);
        lines.pushBack(line);

        // Quick check for quotes
        if line.find('"') != -1 {
            var totalQuotes = line.count('"');

            // If odd number of quotes, keep reading until balanced
            while totalQuotes % 2 == 1 {
                try {
                    line = reader.readLine(stripNewline=true);
                    lines.pushBack(line);
                    totalQuotes += line.count('"');
                } catch e: EofError {
                    break;
                } catch e: UnexpectedEofError {
                    break;
                }
            }
        }

        return lines;
    }

    // Fast row counting with quote detection - much faster than readCSVRecord
    proc countRowsAndQuotes(reader: fileReader(?)) : (int, bool) throws {
        var rowCount = 0;
        var hasAnyQuotes = false;

        while true {
            var hasQuotes = try advanceCSVRecord(reader);
            if hasQuotes {
                hasAnyQuotes = true;
            }
            rowCount += 1;
        }

        return (rowCount, hasAnyQuotes);
    }

    // Read a complete CSV record from the reader, handling multi-line quoted fields
    proc readCSVRecord(reader: fileReader(?)) throws {
        var lines = try readCSVRecordLines(reader);

        // Return single line if no multi-line record
        if lines.size == 1 {
            return lines[0];
        } else {
            return "\n".join(lines.these());
        }
    }

    // Core CSV parsing engine that handles both single field extraction and full parsing
    proc parseCSVCore(csvRecord: string, colDelim: string, targetIdx: int = -1): (list((int, int)), string) throws {
        const recordLen = csvRecord.size;
        const delimLen = colDelim.size;
        var inQuotes = false;
        var fieldStart = 0;
        var currentFieldIdx = 0;
        var i = 0;
        var fieldBoundaries: list((int, int));
        var targetField = "";
        var foundTarget = false;

        while i < recordLen {
            var ch = csvRecord[i];

            if ch == "\"" {
                if inQuotes {
                    // Check if next character is also a quote (escaped quote)
                    if i + 1 < recordLen && csvRecord[i + 1] == "\"" {
                        // Escaped quote - skip both characters
                        i += 2;
                    } else {
                        // End of quoted field
                        inQuotes = false;
                        i += 1;
                    }
                } else {
                    // Start of quoted field
                    inQuotes = true;
                    i += 1;
                }
            } else if !inQuotes && delimLen == 1 && ch == colDelim {
                // Found single-character delimiter outside quotes
                if targetIdx != -1 && currentFieldIdx == targetIdx {
                    // Extract target field and return early
                    if fieldStart <= i - 1 {
                        var fieldSlice = csvRecord[fieldStart..i-1];
                        targetField = processField(fieldSlice);
                    } else {
                        targetField = "";
                    }
                    foundTarget = true;
                    return (fieldBoundaries, targetField);
                }

                // Store boundary for full parsing
                fieldBoundaries.pushBack((fieldStart, i - 1));
                fieldStart = i + 1;
                currentFieldIdx += 1;
                i += 1;
            } else if !inQuotes && delimLen > 1 && i + delimLen <= recordLen && csvRecord[i..#delimLen] == colDelim {
                // Found multi-character delimiter outside quotes
                if targetIdx != -1 && currentFieldIdx == targetIdx {
                    // Extract target field and return early
                    if fieldStart <= i - 1 {
                        var fieldSlice = csvRecord[fieldStart..i-1];
                        targetField = processField(fieldSlice);
                    } else {
                        targetField = "";
                    }
                    foundTarget = true;
                    return (fieldBoundaries, targetField);
                }

                // Store boundary for full parsing
                fieldBoundaries.pushBack((fieldStart, i - 1));
                fieldStart = i + delimLen;
                currentFieldIdx += 1;
                i += delimLen;
            } else {
                // Regular character - continue
                i += 1;
            }
        }

        // Handle the final field
        if targetIdx != -1 && currentFieldIdx == targetIdx {
            // Extract final target field
            if fieldStart <= recordLen - 1 {
                var fieldSlice = csvRecord[fieldStart..recordLen-1];
                targetField = processField(fieldSlice);
            } else {
                targetField = "";
            }
            foundTarget = true;
            return (fieldBoundaries, targetField);
        }

        // Add final field boundary for full parsing
        fieldBoundaries.pushBack((fieldStart, recordLen - 1));

        // Check if target field was found
        if targetIdx != -1 && !foundTarget {
            throw new BadFormatError("CSV record does not have enough fields (need field " + targetIdx:string + ")");
        }

        return (fieldBoundaries, targetField);
    }

    // Helper function to process a field and handle quote unescaping
    proc processField(in fieldSlice: string): string throws {

        // Handle quoted fields - remove outer quotes and unescape inner quotes
        if fieldSlice.size >= 2 && fieldSlice.startsWith("\"") && fieldSlice.endsWith("\"") {
            // Remove outer quotes
            fieldSlice = fieldSlice[1..<fieldSlice.size-1];

            // quote unescaping
            fieldSlice = fieldSlice.replace("\"\"", "\"");
        }

        return fieldSlice;
    }

    // Optimized function to extract a specific field by index (early exit)
    proc getFieldByIndex(csvRecord: string, colDelim: string, targetIdx: int): string throws {
        // Fast path: if no quotes, use simple split
        if csvRecord.find('"') == -1 {
            const fields = csvRecord.split(colDelim);
            if targetIdx >= fields.size {
                throw new BadFormatError("CSV record does not have enough fields (need field " + targetIdx:string + ")");
            }
            return fields[targetIdx];
        }

        // Slow path: use full parsing for quoted fields
        const (boundaries, targetField) = parseCSVCore(csvRecord, colDelim, targetIdx);
        return targetField;
    }

    // Helper function to find all field boundaries
    proc findFieldBoundaries(csvRecord: string, colDelim: string): list((int, int)) throws {
        const (boundaries, targetField) = parseCSVCore(csvRecord, colDelim, -1);
        return boundaries;
    }

    // Iterator to parse CSV fields from a complete CSV record string
    iter parseCSVRecord(csvRecord: string, colDelim: string) throws {
        // Phase 1: Find all field boundaries in a single pass
        const fieldBoundaries = findFieldBoundaries(csvRecord, colDelim);

        // Phase 2: Extract and process fields using efficient string operations
        for (start, end) in fieldBoundaries {
            if start <= end {
                const fieldSlice = csvRecord[start..end];
                yield processField(fieldSlice);
            } else {
                // Empty field
                yield "";
            }
        }
    }

    record csvLine: readDeserializable {

        type itemType;
        var item: itemType;
        const colIdx: int;
        const colDelim: string;
        const hasQuotes: bool;

        proc init(type itemType, colIdx: int, colDelim: string, hasQuotes: bool) throws {
            this.itemType = itemType;
            this.colIdx = colIdx;
            this.colDelim = colDelim;
            this.hasQuotes = hasQuotes;
        }

        proc ref deserialize(reader: fileReader(?), ref deserializer) throws {
            if !this.hasQuotes && itemType!=string {
                // Old implementation that didn't care about quotes
                // It was faster for non-string types
                // read the comma delimited items in a single line
                for 0..<colIdx {
                    reader.advanceThrough(this.colDelim:bytes);  // Skip over the columns we don't care about
                }
                var success = reader.read(this.item);
                if !success {
                    throw new BadFormatError("Cannot parse field at index " + this.colIdx:string + " as " + itemType:string);
                }
                // Advance through the rest of the line to prepare for next read
                reader.advanceThrough(b'\n');
            }
            else {
                // Try fast path first: no quotes and single line
                var line: string;
                try {
                    line = reader.readLine(stripNewline=true);
                } catch e: EofError {
                    throw e;
                }

                // Check if this line has quotes - if not, use fast path
                if line.find('"') == -1 {
                    const fields = line.split(this.colDelim);
                    if this.colIdx < fields.size {
                        const targetField = fields[this.colIdx];
                        try {
                            this.item = targetField:itemType;
                            return;
                        } catch {
                            throw new BadFormatError("Cannot parse field value '" + targetField + "' as " + itemType:string);
                        }
                    }
                }

                // If we're here: Fast path failed, need to handle quotes or multi-line
                // Check if we need to read more lines for multi-line quoted fields
                var totalQuotes = line.count("\"");
                if totalQuotes % 2 == 1 {
                    // Multi-line quoted field - need to read more
                    var lines: list(string);
                    lines.pushBack(line);

                    while totalQuotes % 2 == 1 {
                        try {
                            line = reader.readLine(stripNewline=true);
                            lines.pushBack(line);
                            totalQuotes += line.count("\"");
                        } catch e: EofError {
                            break;
                        } catch e: UnexpectedEofError {
                            break;
                        }
                    }

                    line = "\n".join(lines.these());
                }

                // Use complex parsing for quoted fields
                const targetField = getFieldByIndex(line, this.colDelim, this.colIdx);
                try {
                    this.item = targetField:itemType;
                } catch {
                    throw new BadFormatError("Cannot parse field value '" + targetField + "' as " + itemType:string);
                }
            }
        }
    }

    proc read_files_into_dist_array(ref A: [?D] ?t, dset: string, filenames: [] string, filedomains: [] domain(1), skips: set(string), hasHeaders: bool, col_delim: string, offsets: [] int, allowErrors: bool, const hasQuotes: [] bool) throws {
        var hadError = false;
        coforall loc in A.targetLocales() with (ref A, | reduce hadError) do on loc {
            // Create local copies of args
            const locFiles = filenames;
            const locFiledoms = filedomains;
            /* On this locale, find all files containing data that belongs in
                this locale's chunk of A */
            forall (filedom, filename, file_idx) in zip(locFiledoms, locFiles, 0..) with (ref A, | reduce hadError) {
                if (skips.contains(filename)) {
                    csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                             "File %s does not contain data for this dataset, skipping".format(filename));
                    continue;
                } else {
                    read_csv_pattern(A, filename, filedom, dset, col_delim, hasHeaders, offsets[file_idx], allowErrors, hadError, hasQuotes[file_idx]);
                }
            }
        }
        if hadError {
            throw getErrorWithContext(
                msg="This CSV is missing values. To read anyway and replace these with min(col_dtype), set allow_errors to True",
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass="IOError");
        }
    }

    proc generate_subdoms(filenames: [?D] string, row_counts: [D] int, validFiles: [D] bool) throws {
        var skips = new set(string);
        var offsets = (+ scan row_counts) - row_counts;
        var subdoms = D.tryCreateArray(domain(1)); // Non dist array
        for (i, fname, off, ct, vf) in zip(D, filenames, offsets, row_counts, validFiles) {
            if (!vf) {
                skips.add(fname);
                csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Adding invalid file to skips, %s".format(fname));
                continue;
            }
            subdoms[i] = {off..#ct};
        }
        return (subdoms, offsets, skips);
    }

    proc readTypedCSV(filenames: [] string, datasets: [?D] string, dtypes: list(string), row_counts: [] int, validFiles: [] bool, col_delim: string, allowErrors: bool, const hasQuotes: [] bool, st: borrowed SymTab): list((string, ObjType, string)) throws {
        // assumes the file has header since we were able to access type info
        var rtnData: list((string, ObjType, string));
        const record_count = + reduce row_counts;
        var (subdoms, offsets, skips) = generate_subdoms(filenames, row_counts, validFiles);

        for (i, dset) in zip(D, datasets) {
            var dtype = str2dtype(dtypes[i]);
            select dtype {
                when DType.Int64 {
                    var a = makeDistArray(record_count, int);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets, allowErrors, hasQuotes);
                    var entry = createSymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, ObjType.PDARRAY, rname));
                }
                when DType.UInt64 {
                    var a = makeDistArray(record_count, uint);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets, allowErrors, hasQuotes);
                    var entry = createSymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, ObjType.PDARRAY, rname));
                }
                when DType.Float64 {
                    var a = makeDistArray(record_count, real);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets, allowErrors, hasQuotes);
                    var entry = createSymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, ObjType.PDARRAY, rname));
                }
                when DType.Bool {
                    var a = makeDistArray(record_count, bool);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets, allowErrors, hasQuotes);
                    var entry = createSymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, ObjType.PDARRAY, rname));
                }
                when DType.Strings {
                    var a = makeDistArray(record_count, string);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets, allowErrors, hasQuotes);
                    var col_lens = makeDistArray(record_count, int);
                    forall (i, v) in zip(0..#a.size, a) {
                        var tmp_str = v + "\x00";
                        var vbytes = tmp_str.bytes();
                        col_lens[i] = vbytes.size;
                    }
                    var str_offsets = (+ scan col_lens) - col_lens;
                    var value_size: int = + reduce col_lens;
                    var data = makeDistArray(value_size, uint(8));
                    forall (i, v) in zip(0..#a.size, a) {
                        var tmp_str = v + "\x00";
                        var vbytes = tmp_str.bytes();
                        ref low = str_offsets[i];
                        data[low..#vbytes.size] = vbytes;
                    }
                    var ss = getSegString(str_offsets, data, st);
                    var rst = (dset, ObjType.STRINGS, "%s+%?".format(ss.name, ss.nBytes));
                    rtnData.pushBack(rst);
                }
                otherwise {
                    throw getErrorWithContext(
                                    msg="Data Type %s cannot be read into Arkouda.".format(dtypes[i]),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="IOError"
                            );
                }
            }
        }
        return rtnData;
    }

    proc readGenericCSV(filenames: [] string, datasets: [?D] string, row_counts: [] int, validFiles: [] bool, col_delim: string, allowErrors: bool, const hasQuotes: [] bool, st: borrowed SymTab): list((string, ObjType, string)) throws {
        // assumes the file does not have a header since we were not able to access type info
        var rtnData: list((string, ObjType, string));
        const record_count = + reduce row_counts;
        var (subdoms, offsets, skips) = generate_subdoms(filenames, row_counts, validFiles);

        for (i, dset) in zip(D, datasets) {
            var a = makeDistArray(record_count, string);
            read_files_into_dist_array(a, dset, filenames, subdoms, skips, false, col_delim, offsets, allowErrors, hasQuotes);
            var col_lens = makeDistArray(record_count, int);
            forall (i, v) in zip(0..#a.size, a) {
                var tmp_str = v + "\x00";
                var vbytes = tmp_str.bytes();
                col_lens[i] = vbytes.size;
            }
            var str_offsets = (+ scan col_lens) - col_lens;
            var value_size: int = + reduce col_lens;
            var data = makeDistArray(value_size, uint(8));
            forall (i, v) in zip(0..#a.size, a) {
                var tmp_str = v + "\x00";
                var vbytes = tmp_str.bytes();
                ref low = str_offsets[i];
                data[low..#vbytes.size] = vbytes;
            }
            var ss = getSegString(str_offsets, data, st);
            var rst = (dset, ObjType.STRINGS, "%s+%?".format(ss.name, ss.nBytes));
            rtnData.pushBack(rst);
        }
        return rtnData;
    }

    proc readCSVMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        const col_delim: string = msgArgs.getValueOf("col_delim");

        var allowErrors = msgArgs.get("allow_errors").getBoolValue();

        var nfiles = msgArgs.get("nfiles").getIntValue();
        var filelist: [0..#nfiles] string;
        try {
            filelist = msgArgs.get("filenames").getList(nfiles);
        } catch {
            // limit length of file names to 2000 chars
            var n: int = 1000;
            var jsonfiles = msgArgs.getValueOf("filenames");
            var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
            var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, files);
            csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // access the list of datasets user
        var ndsets = msgArgs.get("num_dsets").getIntValue();
        var dsetlist: [0..#ndsets] string;
        try {
            dsetlist = msgArgs.get("datasets").getList(ndsets);
        } catch {
            // limit length of dataset names to 2000 chars
            var n: int = 1000;
            var jsondsets = msgArgs.getValueOf("datasets");
            var dsets: string = if jsondsets.size > 2*n then jsondsets[0..#n]+'...'+jsondsets[jsondsets.size-n..#n] else jsondsets;
            var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".format(
                                                ndsets, dsets);
            csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var filedom = filelist.domain;
        var filenames: [filedom] string;

        if filelist.size == 1 {
            if filelist[0].strip().size == 0 {
                var errorMsg = "filelist was empty.";
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            var tmp = glob(filelist[0]);
            csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "glob expanded %s to %i files".format(filelist[0], tmp.size));
            if tmp.size == 0 {
                var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            // Glob returns filenames in weird order. Sort for consistency
            sort(tmp);
            filedom = tmp.domain;
            filenames = tmp;
        } else {
            // assumes that we are providing
            filenames = filelist;
        }

        var row_cts: [filedom] int;
        var data_types: list(list(string));
        var headers: [filedom] bool;
        var hasQuotes: [filedom] bool;
        var rtnData: list((string, ObjType, string));
        var fileErrors: list(string);
        var fileErrorCount:int = 0;
        var fileErrorMsg:string = "";
        var validFiles: [filedom] bool = true;
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                var dtypes: list(string);
                (row_cts[i], headers[i], hasQuotes[i], dtypes) = get_info(fname, dsetlist, col_delim);
                data_types.pushBack(dtypes);
            } catch e: FileNotFoundError {
                fileErrorMsg = "File %s not found".format(fname);
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e: PermissionError {
                fileErrorMsg = "Permission error %s opening %s".format(e.message(),fname);
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e: DatasetNotFoundError {
                fileErrorMsg = "1 or more Datasets not found in file %s".format(fname);
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e: SegStringError {
                fileErrorMsg = "SegmentedString error: %s".format(e.message());
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            } catch e : Error {
                fileErrorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                hadError = true;
                if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
            }

            if hadError {
                // Keep running total, but we'll only report back the first 10
                if fileErrorCount < 10 {
                    fileErrors.pushBack(fileErrorMsg.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip());
                }
                fileErrorCount += 1;
                validFiles[i] = false;
            }
        }

        const dtype = data_types[0];
        const rows = row_cts[0];
        const hasHeader = headers[0];
        for (isValid, fname, dt, rc, hh) in zip(validFiles, filenames, data_types, row_cts, headers) {
            if isValid {
                if (dtype != dt) {
                    var errorMsg = "Inconsistent dtypes in file %s".format(fname);
                    csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
                else if (hasHeader != hh){
                    var errorMsg = "Inconsistent file formatting. %s has no header.".format(fname);
                    csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
            csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "Verified all dtypes across files for file %s".format(fname));
        }

        var rtnMsg: string;
        try {
            if headers[0] {
                rtnData = readTypedCSV(filenames, dsetlist, data_types[0], row_cts, validFiles, col_delim, allowErrors, hasQuotes, st);
                rtnMsg = buildReadAllMsgJson(rtnData, allowErrors, fileErrorCount, fileErrors, st);
            }
            else {
                rtnData = readGenericCSV(filenames, dsetlist, row_cts, validFiles, col_delim, allowErrors, hasQuotes, st);
                rtnMsg = buildReadAllMsgJson(rtnData, allowErrors, fileErrorCount, fileErrors, st);
            }
        }
        catch e: Error {
            var errMsg = e.message();
            csvLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errMsg);
            return new MsgTuple(errMsg, MsgType.ERROR);
        }
        return new MsgTuple(rtnMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("readcsv", readCSVMsg, getModuleName());
    registerFunction("writecsv", writeCSVMsg, getModuleName());
    registerFunction("lscsv", lsCSVMsg, getModuleName());
}
