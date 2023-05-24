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

    use ArkoudaFileCompat;
    use ArkoudaArrayCompat;
    use ArkoudaListCompat;

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
            var tmp = glob(filename);
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
        var csvFile = open(filename, ioMode.r);
        var reader = csvFile.reader();
        var lines = reader.lines().strip();
        if lines[0] == CSV_HEADER_OPEN {
            idx = 3; // set to first line after header
        }

        var col_delim: string = msgArgs.getValueOf("col_delim");
        var column_names = lines[idx].split(col_delim);
        reader.close();
        csvFile.close();
        return new MsgTuple("%jt".format(column_names), MsgType.NORMAL);

    }

    proc prepFiles(filename: string, overwrite: bool, A) throws { 
        var prefix: string;
        var extension: string;
        (prefix,extension) = getFileMetadata(filename);

        var targetSize: int = A.targetLocales().size;
        var filenames: [0..#targetSize] string;
        forall i in 0..#targetSize {
            filenames[i] = generateFilename(prefix, extension, i);
        }

        csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Identified %i files for provided name, %s".format(filenames.size, filename));

        var matchingFilenames = glob("%s_LOCALE*%s".format(prefix, extension));
        var filesExist: bool = matchingFilenames.size > 0;

        if !overwrite && filesExist {
            throw getErrorWithContext(
                msg="Filenames for the provided name exist. Overwrite must be set to true in order to save with the name %s".format(filename),
                lineNumber=getLineNumber(),
                routineName=getRoutineName(), 
                moduleName=getModuleName(),
                errorClass="InvalidArgumentError");
        }
        else {
            coforall loc in A.targetLocales() do on loc {
                var fn = filenames[loc.id].localize();
                var existList = glob(fn);
                if overwrite && existList.size == 1 {
                    remove(fn);
                    csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Overwriting CSV File, %s".format(fn));
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
        var datasets = msgArgs.get("datasets").getList(ndsets);
        var dtypes = msgArgs.get("dtypes").getList(ndsets);
        var col_names = msgArgs.get("col_names").getList(ndsets);
        const col_delim = msgArgs.getValueOf("col_delim");
        const row_count = msgArgs.get("row_count").getIntValue();
        const overwrite = msgArgs.get("overwrite").getBoolValue();

        // access the first SymEntry -> all SymEntries should have same locality due to them being required to be the same size.
        var gse = toGenSymEntry(st.lookup(datasets[0]));
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

        coforall loc in Locales do on loc{
            const localeFilename = filenames[loc.id];
            
            // create the file to write to
            var csvFile = open(localeFilename, ioMode.cw);
            var writer = csvFile.writer();

            // write the header
            writer.write(CSV_HEADER_OPEN + LINE_DELIM);
            writer.write(",".join(dtypes) + LINE_DELIM);
            writer.write(CSV_HEADER_CLOSE + LINE_DELIM);

            // write the column names
            var column_header:[0..#ndsets] string;
            writer.write(col_delim.join(col_names) + LINE_DELIM);

            // need to get local subdomain -- should be the same for each element due to sizes being same
            var localSubdomain = getLocalDomain(gse);

            for r in localSubdomain {
                var row: [0..#ndsets] string;
                forall (i, cname) in zip(0..#ndsets, datasets) {
                    var col_gen: borrowed GenSymEntry = getGenericTypedArrayEntry(cname, st);
                    select col_gen.dtype {
                        when DType.Int64 {
                            var col = toSymEntry(col_gen, int);
                            row[i] = col.a[r]: string;
                        }
                        when DType.UInt64 {
                            var col = toSymEntry(col_gen, uint);
                            row[i] = col.a[r]: string;
                        }
                        when DType.Float64 {
                            var col = toSymEntry(col_gen, real);
                            row[i] = col.a[r]: string;
                        }
                        when DType.Bool {
                            var col = toSymEntry(col_gen, bool);
                            row[i] = col.a[r]: string;
                        }
                        when DType.Strings {
                            var segString:SegString = getSegString(cname, st);
                            row[i] = segString[r];
                            
                        } otherwise {
                            throw getErrorWithContext(
                                    msg="Data Type %s cannot be written to CSV.".format(dtypes[i]),
                                    lineNumber=getLineNumber(), 
                                    routineName=getRoutineName(), 
                                    moduleName=getModuleName(), 
                                    errorClass='IOError'
                            );
                        }
                    }
                }
                var write_row = col_delim.join(row) + LINE_DELIM;
                writer.write(write_row);
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

        var csvFile = open(filename, ioMode.r);
        var reader = csvFile.reader();
        var lines = reader.lines();
        var hasHeader = false;
        var dtype_idx = 0;
        var column_name_idx = 0;
        
        if lines[0] == CSV_HEADER_OPEN + "\n" {
            hasHeader = true;
            dtype_idx = 1;
            column_name_idx = 3;
        }
        
        var columns = lines[column_name_idx].split(col_delim).strip();
        var file_dtypes: [0..#columns.size] string;
        if dtype_idx > 0 {
            file_dtypes = lines[dtype_idx].split(",").strip();
        }
        else {
            file_dtypes = "str";
        }
        var data_start_offset = column_name_idx + 1;
        // get the row count
        var row_ct: int = lines.size - data_start_offset;

        reader.close();
        csvFile.close();

        var dtypes: [0..#datasets.size] string;
        forall (i, dset) in zip(0..#datasets.size, datasets) {
            var idx: int;
            var col_exists = columns.find(dset, idx);
            csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Column: %s, Exists: %jt, IDX: %i".format(dset, col_exists, idx));
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
        return (row_ct, hasHeader, new list(dtypes));
    }

    proc read_files_into_dist_array(A: [?D] ?t, dset: string, filenames: [] string, filedomains: [] domain(1), skips: set(string), hasHeaders: bool, col_delim: string, offsets: [] int) throws {

        coforall loc in A.targetLocales() do on loc {
            // Create local copies of args
            var locFiles = filenames;
            var locFiledoms = filedomains;
            /* On this locale, find all files containing data that belongs in
                this locale's chunk of A */
            for (filedom, filename, file_idx) in zip(locFiledoms, locFiles, 0..) {
                if (skips.contains(filename)) {
                    csvLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                             "File %s does not contain data for this dataset, skipping".format(filename));
                    continue;
                } else {
                    var csvFile = open(filename, ioMode.r);
                    var reader = csvFile.reader();
                    var lines = reader.lines().strip();
                    var data_offset = 1;
                    if hasHeaders {
                        data_offset  = 4;
                    }
                    // determine the index of the column.
                    var column_names = lines[data_offset-1].split(col_delim);
                    var colidx: int;
                    var colExists = column_names.find(dset, colidx);
                    if !colExists{
                        throw getErrorWithContext(
                            msg="The dataset %s was not found in %s".format(dset, filename),
                            lineNumber=getLineNumber(),
                            routineName=getRoutineName(), 
                            moduleName=getModuleName(),
                            errorClass="DatasetNotFoundError");
                    }
                    for locdom in A.localSubdomains() {
                        const intersection = domain_intersection(locdom, filedom);
                        if intersection.size > 0 {
                            forall x in intersection {
                                var row = lines[x-offsets[file_idx]+data_offset].split(col_delim);
                                A[x] = row[colidx]: t;
                            }
                        }
                    }
                    reader.close();
                    csvFile.close();
                }
            }
        }
    }

    proc generate_subdoms(filenames: [?D] string, row_counts: [D] int, validFiles: [D] bool) throws {
        var skips = new set(string);
        var offsets = (+ scan row_counts) - row_counts;
        var subdoms: [D] domain(1);
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

    proc readTypedCSV(filenames: [] string, datasets: [?D] string, dtypes: list(string), row_counts: [] int, validFiles: [] bool, col_delim: string, st: borrowed SymTab): list((string, string, string)) throws {
        // assumes the file has header since we were able to access type info
        var rtnData: list((string, string, string));
        var record_count = + reduce row_counts;
        var (subdoms, offsets, skips) = generate_subdoms(filenames, row_counts, validFiles);

        for (i, dset) in zip(D, datasets) {
            var dtype = str2dtype(dtypes[i]);
            select dtype {
                when DType.Int64 {
                    var a = makeDistArray(record_count, int);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets);
                    var entry = new shared SymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, "pdarray", rname));
                }
                when DType.UInt64 {
                    var a = makeDistArray(record_count, uint);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets);
                    var entry = new shared SymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, "pdarray", rname));
                }
                when DType.Float64 {
                    var a = makeDistArray(record_count, real);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets);
                    var entry = new shared SymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, "pdarray", rname));
                }
                when DType.Bool {
                    var a = makeDistArray(record_count, bool);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets);
                    var entry = new shared SymEntry(a);
                    var rname = st.nextName();
                    st.addEntry(rname, entry);
                    rtnData.pushBack((dset, "pdarray", rname));
                }
                when DType.Strings {
                    var a = makeDistArray(record_count, string);
                    read_files_into_dist_array(a, dset, filenames, subdoms, skips, true, col_delim, offsets);
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
                    var rst = (dset, "seg_string", "%s+%t".format(ss.name, ss.nBytes));
                    rtnData.pushBack(rst);
                }
                otherwise {
                    throw getErrorWithContext(
                                    msg="Data Type %s cannot be read into Arkouda.".format(dtypes[i]),
                                    lineNumber=getLineNumber(), 
                                    routineName=getRoutineName(), 
                                    moduleName=getModuleName(), 
                                    errorClass='IOError'
                            );
                }
            }
        }
        return rtnData;
    }

    proc readGenericCSV(filenames: [] string, datasets: [?D] string, row_counts: [] int, validFiles: [] bool, col_delim: string, st: borrowed SymTab): list((string, string, string)) throws {
        // assumes the file does not have a header since we were not able to access type info
        var rtnData: list((string, string, string));
        var record_count = + reduce row_counts;
        var (subdoms, offsets, skips) = generate_subdoms(filenames, row_counts, validFiles);

        for (i, dset) in zip(D, datasets) {
            var a = makeDistArray(record_count, string);
            read_files_into_dist_array(a, dset, filenames, subdoms, skips, false, col_delim, offsets);
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
            var rst = (dset, "seg_string", "%s+%t".format(ss.name, ss.nBytes));
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
        var rtnData: list((string, string, string));
        var fileErrors: list(string);
        var fileErrorCount:int = 0;
        var fileErrorMsg:string = "";
        var validFiles: [filedom] bool = true;
        for (i, fname) in zip(filedom, filenames) {
            var hadError = false;
            try {
                var dtypes: list(string);
                (row_cts[i], headers[i], dtypes) = get_info(fname, dsetlist, col_delim);
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
                if !allowErrors { return new MsgTuple(e.message(), MsgType.ERROR); }
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

        var dtype = data_types[0];
        var rows = row_cts[0];
        var hasHeader = headers[0];
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
        if headers[0] {
            rtnData = readTypedCSV(filenames, dsetlist, data_types[0], row_cts, validFiles, col_delim, st);
            rtnMsg = _buildReadAllMsgJson(rtnData, allowErrors, fileErrorCount, fileErrors, st);
        }
        else {
            rtnData = readGenericCSV(filenames, dsetlist, row_cts, validFiles, col_delim, st);
            rtnMsg = _buildReadAllMsgJson(rtnData, allowErrors, fileErrorCount, fileErrors, st);
        }
        
        return new MsgTuple(rtnMsg, MsgType.NORMAL);
    } 

    use CommandMap;
    registerFunction("readcsv", readCSVMsg, getModuleName());
    registerFunction("writecsv", writeCSVMsg, getModuleName());
    registerFunction("lscsv", lsCSVMsg, getModuleName());
}
