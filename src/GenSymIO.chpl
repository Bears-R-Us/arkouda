module GenSymIO {
    use HDF5;
    use Time only;
    use IO;
    use CPtr;
    use SysCTypes;
    use Path;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use FileSystem;
    use FileIO;
    use Sort;
    use CommAggregation;
    use NumPyDType;
    use List;
    use Map;
    use Set;
    use PrivateDist;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use ServerConfig;
    use Search;
    use IndexingMsg;

    require "c_helpers/help_h5ls.h", "c_helpers/help_h5ls.c";

    private config const logLevel = ServerConfig.logLevel;
    const gsLogger = new Logger(logLevel);

    config const GenSymIO_DEBUG = false;
    config const SEGARRAY_OFFSET_NAME = "segments";
    config const SEGARRAY_VALUE_NAME = "values";
    config const NULL_STRINGS_VALUE = 0:uint(8);
    config const TRUNCATE: int = 0;
    config const APPEND: int = 1;

    // Constants etc. related to intenral HDF5 file metadata
    const ARKOUDA_HDF5_FILE_METADATA_GROUP = "/_arkouda_metadata";
    const ARKOUDA_HDF5_ARKOUDA_VERSION_KEY = "arkouda_version"; // see ServerConfig.arkoudaVersion
    type ARKOUDA_HDF5_ARKOUDA_VERSION_TYPE = c_string;
    const ARKOUDA_HDF5_FILE_VERSION_KEY = "file_version";
    const ARKOUDA_HDF5_FILE_VERSION_VAL = 1.0:real(32);
    type ARKOUDA_HDF5_FILE_VERSION_TYPE = real(32);


    /*
     * Creates a pdarray server-side and returns the SymTab name used to
     * retrieve the pdarray from the SymTab.
     */
    proc arrayMsg(cmd: string, args: string, ref data: bytes, st: borrowed SymTab): MsgTuple throws {
        // Set up our return items
        var msgType = MsgType.NORMAL;
        var msg:string = "";
        var rname:string = "";

        var (dtypeBytes, sizeBytes, segStr) = args.splitMsgToTuple(" ", 3);
        var dtype = DType.UNDEF;
        var size:int;
        var asSegStr = false;
        
        try {
            dtype = str2dtype(dtypeBytes);
            size = sizeBytes:int;
            asSegStr = "seg_string=True" == segStr.strip();
        } catch {
            var errorMsg = "Error parsing/decoding either dtypeBytes or size";
            gsLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        overMemLimit(2*size);

        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "dtype: %t size: %i".format(dtype,size));

        proc bytesToSymEntry(size:int, type t, st: borrowed SymTab, ref data:bytes): string throws {
            var entry = new shared SymEntry(size, t);
            var localA = makeArrayFromPtr(data.c_str():c_void_ptr:c_ptr(t), size:uint);
            entry.a = localA;
            var name = st.nextName();
            st.addEntry(name, entry);
            return name;
        }

        if dtype == DType.Int64 {
            rname = bytesToSymEntry(size, int, st, data);
        } else if dtype == DType.Float64 {
            rname = bytesToSymEntry(size, real, st, data);
        } else if dtype == DType.Bool {
            rname = bytesToSymEntry(size, bool, st, data);
        } else if dtype == DType.UInt8 {
            rname = bytesToSymEntry(size, uint(8), st, data);
        } else {
            msg = "Unhandled data type %s".format(dtypeBytes);
            msgType = MsgType.ERROR;
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),msg);
        }

        if asSegStr {
            try {
                var values = toSymEntry(st.lookup(rname), uint(8));
                var offsets = segmentedCalcOffsets(values.a, values.aD);
                var oname = st.nextName();
                var offsetsEntry = new shared SymEntry(offsets);
                st.addEntry(oname, offsetsEntry);
                msg = "created " + st.attrib(oname) + "+created " + st.attrib(rname);
            } catch e: Error {
                msg = "Error creating offsets for SegString";
                msgType = MsgType.ERROR;
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),msg);
            }
        }

        if (MsgType.ERROR != msgType) {  // success condition
            // Set up return message indicating SymTab name corresponding to new pdarray
            // If we made a SegString or we encountered an error, the msg will already be populated
            if (msg.isEmpty()) {
                msg = "created " + st.attrib(rname);
            }
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),msg);
        }
        return new MsgTuple(msg, msgType);
    }

    /**
     * For creating the Strings/SegString object we can calculate the offsets array on the server
     * by finding the null terminators given the values/bytes array which should have already been
     * converted to uint8
     */
    proc segmentedCalcOffsets(values:[] uint(8), valuesDom:domain): [] int throws {
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Calculating offsets for SegString");
        var nb_locs = forall (i,v) in zip(valuesDom, values) do if v == NULL_STRINGS_VALUE then i+1;
        // We need to adjust nb_locs b/c offsets is really the starting position of each string
        // So allocated a new array of zeros and assign nb_locs offset by one
        var offsets: [nb_locs.domain] int;
        offsets[1..offsets.domain.high] = nb_locs[0..#nb_locs.domain.high];
        return offsets;
    }

    /*
     * Ensure the file is closed, disregard errors
     */
    private proc ensureClose(tmpf:file): bool {
        var success = true;
        try {
            tmpf.close();
        } catch {
            success = false;
        }
        return success;
    }

    /*
     * Outputs the pdarray as a Numpy ndarray in the form of a 
     * Chapel Bytes object
     */
    proc tondarrayMsg(cmd: string, payload: string, st: 
                                          borrowed SymTab): bytes throws {
        var arrayBytes: bytes;
        var entry = st.lookup(payload);
        overMemLimit(2*entry.size*entry.itemsize);

        proc distArrToBytes(A: [?D] ?eltType) {
            var ptr = c_malloc(eltType, D.size);
            var localA = makeArrayFromPtr(ptr, D.size:uint);
            localA = A;
            const size = D.size*c_sizeof(eltType):int;
            return createBytesWithOwnedBuffer(ptr:c_ptr(uint(8)), size, size);
        }

        if entry.dtype == DType.Int64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, int).a);
        } else if entry.dtype == DType.Float64 {
            arrayBytes = distArrToBytes(toSymEntry(entry, real).a);
        } else if entry.dtype == DType.Bool {
            arrayBytes = distArrToBytes(toSymEntry(entry, bool).a);
        } else if entry.dtype == DType.UInt8 {
            arrayBytes = distArrToBytes(toSymEntry(entry, uint(8)).a);
        } else {
            var errorMsg = "Error: Unhandled dtype %s".format(entry.dtype);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return errorMsg.encode(); // return as bytes
        }

       return arrayBytes;
    }

    /*
     * Converts the JSON array to a pdarray
     */
    proc jsonToPdArray(json: string, size: int) throws {
        var f = opentmp(); defer { ensureClose(f); }
        var w = f.writer();
        w.write(json);
        w.close();
        var r = f.reader(start=0);
        var array: [0..#size] string;
        r.readf("%jt", array);
        r.close();
        return array;
    }

    /*
     * Converts the JSON array to a integer pdarray
     */
    proc jsonToPdArrayInt(json: string, size: int) throws {
        var f = opentmp();
        var w = f.writer();
        w.write(json);
        w.close();
        var r = f.reader(start=0);
        var array: [0..#size] int;
        r.readf("%jt", array);
        r.close();
        f.close();
        return array;
    }


    /*
     * Indicates whether the filename represents a glob expression as opposed to
     * an specific filename
     */
    proc isGlobPattern(filename: string): bool throws {
        return filename.endsWith("*");
    }

    /*
     * Simulates the output of h5ls for top level datasets or groups
     * :returns: string formatted as json list
     * i.e. ["_arkouda_metadata", "pda1", "s1"]
     */
    proc lshdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        // reqMsg: "lshdf [<json_filename>]"
        var repMsg: string;
        var (jsonfile) = payload.splitMsgToTuple(1);

        // Retrieve filename from payload
        var filename: string;
        try {
            filename = jsonToPdArray(jsonfile, 1)[0];
            if filename.isEmpty() {
                throw new IllegalArgumentError("filename was empty");  // will be caught by catch block
            }
        } catch {
            var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(
                                     1, jsonfile);                                     
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);                                    
        }

        // If the filename represents a glob pattern, retrieve the locale 0 filename
        if isGlobPattern(filename) {
            // Attempt to interpret filename as a glob expression and ls the first result
            var tmp = glob(filename);
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "glob-expanded filename: %s to size: %i files".format(filename, tmp.size));

            if tmp.size <= 0 {
                var errorMsg = "Cannot retrieve filename from glob expression %s, check file name or format".format(filename);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            
            // Set filename to globbed filename corresponding to locale 0
            filename = tmp[tmp.domain.first];
        }
        
        // Check to see if the file exists. If not, return an error message
        if !exists(filename) {
            var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filename);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg,MsgType.ERROR);
        } 

        if !isHdf5File(filename) {
            var errorMsg = "File %s is not an HDF5 file".format(filename);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg,MsgType.ERROR);
        }
        
        try {

            var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
            defer { C_HDF5.H5Fclose(file_id); } // ensure file is closed
            repMsg = simulate_h5ls(file_id);
            var items = new list(repMsg.split(",")); // convert to json

            // TODO: There is a bug with json formatting of lists in Chapel 1.24.x fixed in 1.25
            //       See: https://github.com/chapel-lang/chapel/issues/18156
            //       Below works in 1.25, but until we are fully off of 1.24 we should format json manually for lists
            // repMsg = "%jt".format(items); // Chapel >= 1.25.0
            repMsg = "[";  // Manual json building Chapel <= 1.24.1
            var first = true;
            for i in items {
                if first {
                    first = false;
                } else {
                    repMsg += ",";
                }
                repMsg += '"' + i + '"';
            }
            repMsg += "]";
        } catch e : Error {
            var errorMsg = "Failed to process HDF5 file %t".format(e.message());
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    private extern proc c_get_HDF5_obj_type(loc_id:C_HDF5.hid_t, name:c_string, obj_type:c_ptr(C_HDF5.H5O_type_t)):C_HDF5.herr_t;
    private extern proc c_strlen(s:c_ptr(c_char)):size_t;
    private extern proc c_incrementCounter(data:c_void_ptr);
    private extern proc c_append_HDF5_fieldname(data:c_void_ptr, name:c_string);

    /**
     * Simulate h5ls call by using HDF5 API (top level datasets and groups only, not recursive)
     * This uses both internal call back functions as well as exter c functions defined above to
     * work with the HDF5 API and handle the the data objects it passes between calls as opaque void*
     * which can't be used directly in chapel code.
     */
    proc simulate_h5ls(fid:C_HDF5.hid_t):string throws {
        /** Note: I tried accessing a list inside my inner procs but it leads to segfaults.
         * It only works if the thing you are trying to access is a global.  This is some type
         * of strange interplay between C & chapel as straight chapel didn't cause problems.
         * var items = new list(string);  
         */

        /**
         * This is an H5Literate call-back function, c_helper funcs are used to process data in void*
         * this proc counts the number of of HDF5 groups/datasets under the root, non-recursive
         */
        proc _get_item_count(loc_id:C_HDF5.hid_t, name:c_void_ptr, info:c_void_ptr, data:c_void_ptr) {
            var obj_name = name:c_string;
            var obj_type:C_HDF5.H5O_type_t;
            var status:C_HDF5.H5O_type_t = c_get_HDF5_obj_type(loc_id, obj_name, c_ptrTo(obj_type));
            if (obj_type == C_HDF5.H5O_TYPE_GROUP || obj_type == C_HDF5.H5O_TYPE_DATASET) {
                c_incrementCounter(data);
            }
            return 0; // to continue iteration
        }

        /**
         * This is an H5Literate call-back function, c_helper funcs are used to process data in void*
         * this proc builds string of HDF5 group/dataset objects names under the root, non-recursive
         */
        proc _simulate_h5ls(loc_id:C_HDF5.hid_t, name:c_void_ptr, info:c_void_ptr, data:c_void_ptr) {
            var obj_name = name:c_string;
            var obj_type:C_HDF5.H5O_type_t;
            var status:C_HDF5.H5O_type_t = c_get_HDF5_obj_type(loc_id, obj_name, c_ptrTo(obj_type));
            if (obj_type == C_HDF5.H5O_TYPE_GROUP || obj_type == C_HDF5.H5O_TYPE_DATASET) {
                // items.append(obj_name:string); This doesn't work unless items is global
                c_append_HDF5_fieldname(data, obj_name);
            }
            return 0; // to continue iteration
        }
        
        var idx_p:C_HDF5.hsize_t; // This is the H5Literate index counter
        
        // First iteration to get the item count so we can ballpark the char* allocation
        var nfields:c_int = 0:c_int;
        C_HDF5.H5Literate(fid, C_HDF5.H5_INDEX_NAME, C_HDF5.H5_INDEX_NAME, idx_p, c_ptrTo(_get_item_count), c_ptrTo(nfields));
        
        // Allocate space for array of strings
        var c_field_names = c_calloc(c_char, 255 * nfields);
        idx_p = 0:C_HDF5.hsize_t; // reset our iteration counter
        C_HDF5.H5Literate(fid, C_HDF5.H5_INDEX_NAME, C_HDF5.H5_INDEX_NAME, idx_p, c_ptrTo(_simulate_h5ls), c_field_names);
        var pos = c_strlen(c_field_names):int;
        var items = createStringWithNewBuffer(c_field_names, pos, pos+1);
        c_free(c_field_names);
        return items;
    }

    /*
     * Utility proc to test casting a string to a specified type
     * :arg c: String to cast
     * :type c: string
     * 
     * :arg toType: the type to cast into
     * :type toType: type
     *
     * :returns: bool true if the cast was successful, false otherwise
     */
    proc checkCast(c:string, type toType): bool {
        try {
            var x:toType = c:toType;
            return true;
        } catch {
            return false;
        }
    }

    /* 
     * Reads all datasets from 1..n HDF5 files into an Arkouda symbol table. 
     */
    proc readAllHdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string;
        // May need a more robust delimiter then " | "
        var (strictFlag, ndsetsStr, nfilesStr, allowErrorsFlag, calcStringOffsetsFlag, arraysStr) = payload.splitMsgToTuple(6);
        var strictTypes: bool = true;
        if (strictFlag.toLower().strip() == "false") {
          strictTypes = false;
        }

        var allowErrors: bool = "true" == allowErrorsFlag.toLower(); // default is false
        if allowErrors {
            gsLogger.warn(getModuleName(), getRoutineName(), getLineNumber(), "Allowing file read errors");
        }

        var calcStringOffsets: bool = (calcStringOffsetsFlag.toLower() == "true"); // default is false
        if calcStringOffsets {
            gsLogger.warn(getModuleName(), getRoutineName(), getLineNumber(),
                "Calculating string array offsets instead of reading from HDF5");
        }

        // Test arg casting so we can send error message instead of failing
        if (!checkCast(ndsetsStr, int)) {
            var errMsg = "Number of datasets:`%s` could not be cast to an integer".format(ndsetsStr);
            gsLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errMsg);
            return new MsgTuple(errMsg, MsgType.ERROR);
        }
        if (!checkCast(nfilesStr, int)) {
            var errMsg = "Number of files:`%s` could not be cast to an integer".format(nfilesStr);
            gsLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errMsg);
            return new MsgTuple(errMsg, MsgType.ERROR);
        }

        var (jsondsets, jsonfiles) = arraysStr.splitMsgToTuple(" | ",2);
        var ndsets = ndsetsStr:int; // Error checked above
        var nfiles = nfilesStr:int; // Error checked above
        var dsetlist: [0..#ndsets] string;
        var filelist: [0..#nfiles] string;

        try {
            dsetlist = jsonToPdArray(jsondsets, ndsets);
        } catch {
            var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".format(
                                               ndsets, jsondsets);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        try {
            filelist = jsonToPdArray(jsonfiles, nfiles);
        } catch {
            var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var filedom = filelist.domain;
        var filenames: [filedom] string;

        if filelist.size == 1 {
            if filelist[0].strip().size == 0 {
                var errorMsg = "filelist was empty.";
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            var tmp = glob(filelist[0]);
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "glob expanded %s to %i files".format(filelist[0], tmp.size));
            if tmp.size == 0 {
                var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            // Glob returns filenames in weird order. Sort for consistency
            sort(tmp);
            filedom = tmp.domain;
            filenames = tmp;
        } else {
            filenames = filelist;
        }
        var segArrayFlags: [filedom] bool;
        var dclasses: [filedom] C_HDF5.hid_t;
        var bytesizes: [filedom] int;
        var signFlags: [filedom] bool;
        var rnames: list((string, string, string)); // tuple (dsetName, item type, id)
        var fileErrors: list(string);
        var fileErrorCount:int = 0;
        var fileErrorMsg:string = "";
        const AK_META_GROUP = ARKOUDA_HDF5_FILE_METADATA_GROUP(1..ARKOUDA_HDF5_FILE_METADATA_GROUP.size-1); // strip leading slash
        for dsetName in dsetlist do {
            if dsetName == AK_META_GROUP { // Always skip internal metadata group if present
                continue;
            }
            for (i, fname) in zip(filedom, filenames) {
                var hadError = false;
                try {
                    (segArrayFlags[i], dclasses[i], bytesizes[i], signFlags[i]) = get_dtype(fname, dsetName, calcStringOffsets);
                } catch e: FileNotFoundError {
                    fileErrorMsg = "File %s not found".format(fname);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: PermissionError {
                    fileErrorMsg = "Permission error %s opening %s".format(e.message(),fname);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: DatasetNotFoundError {
                    fileErrorMsg = "Dataset %s not found in file %s".format(dsetName,fname);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: NotHDF5FileError {
                    fileErrorMsg = "The file %s is not an HDF5 file: %s".format(fname,e.message());
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: SegArrayError {
                    fileErrorMsg = "SegmentedArray error: %s".format(e.message());
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e : Error {
                    fileErrorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                }

                if hadError {
                    // Keep running total, but we'll only report back the first 10
                    if fileErrorCount < 10 {
                        fileErrors.append(fileErrorMsg.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip());
                    }
                    fileErrorCount += 1;
                }
            }
            const isSegArray = segArrayFlags[filedom.first];
            const dataclass = dclasses[filedom.first];
            const bytesize = bytesizes[filedom.first];
            const isSigned = signFlags[filedom.first];
            for (name, sa, dc, bs, sf) in zip(filenames, segArrayFlags, dclasses, bytesizes, signFlags) {
              if ((sa != isSegArray) || (dc != dataclass)) {
                  var errorMsg = "Inconsistent dtype in dataset %s of file %s".format(dsetName, name);
                  gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                  return new MsgTuple(errorMsg, MsgType.ERROR);
              } else if (strictTypes && ((bs != bytesize) || (sf != isSigned))) {
                  var errorMsg = "Inconsistent precision or sign in dataset %s of file %s\nWith strictTypes, mixing of precision and signedness not allowed (set strictTypes=False to suppress)".format(dsetName, name);
                  gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                  return new MsgTuple(errorMsg, MsgType.ERROR);
              }
            }

            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Verified all dtypes across files for dataset %s".format(dsetName));
            var subdoms: [filedom] domain(1);
            var segSubdoms: [filedom] domain(1);
            var skips = new set(string);
            var len: int;
            var nSeg: int;
            try {
                if isSegArray {
                    if (!calcStringOffsets) {
                        (segSubdoms, nSeg, skips) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                    }
                    (subdoms, len, skips) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);
                } else {
                    (subdoms, len, skips) = get_subdoms(filenames, dsetName);
                }
            } catch e: HDF5RankError {
                var errorMsg = notImplementedError("readhdf", "Rank %i arrays".format(e.rank));
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            } catch e: Error {
                var errorMsg = "Other error in accessing dataset %s: %s".format(dsetName,e.message());
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }

            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Got subdomains and total length for dataset %s".format(dsetName));

            select (isSegArray, dataclass) {
                when (true, C_HDF5.H5T_INTEGER) {
                    if (bytesize != 1) || isSigned {
                        var errorMsg = "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(
                                                isSegArray, dataclass, bytesize, isSigned);
                        gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }

                    // Load the strings bytes/values first
                    var entryVal = new shared SymEntry(len, uint(8));
                    read_files_into_distributed_array(entryVal.a, subdoms, filenames, dsetName + "/" + SEGARRAY_VALUE_NAME, skips);
                    var valName = st.nextName();
                    st.addEntry(valName, entryVal);

                    // Either load or derive the offsets array based on user preference
                    var offsetsName = st.nextName();
                    if (calcStringOffsets || nSeg < 1) {
                        var offsetsArray = segmentedCalcOffsets(entryVal.a, entryVal.aD);
                        var offsetsEntry = new shared SymEntry(offsetsArray);
                        st.addEntry(offsetsName, offsetsEntry);
                    } else {
                        var offsetsEntry = new shared SymEntry(nSeg, int);
                        read_files_into_distributed_array(offsetsEntry.a, segSubdoms, filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME, skips);
                        fixupSegBoundaries(offsetsEntry.a, segSubdoms, subdoms);
                        st.addEntry(offsetsName, offsetsEntry);
                    }

                    rnames.append((dsetName, "seg_string", "%s+%s".format(offsetsName, valName)));
                }
                when (false, C_HDF5.H5T_INTEGER) {
                    var entryInt = new shared SymEntry(len, int);
                    gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "Initialized int entry for dataset %s".format(dsetName));

                    read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName, skips);
                    var rname = st.nextName();
                    
                    /*
                     * Since boolean pdarrays are saved to and read from HDF5 as ints, confirm whether this
                     * is actually a boolean dataset. If so, (1) convert the SymEntry pdarray to a boolean 
                     * pdarray, (2) create a new SymEntry of type bool, (3) set the SymEntry pdarray 
                     * reference to the bool pdarray, and (4) add the entry to the SymTable
                     */
                    if isBooleanDataset(filenames[0],dsetName) {
                        //var a_bool = entryInt.a:bool;
                        var entryBool = new shared SymEntry(len, bool);
                        entryBool.a = entryInt.a:bool;
                        st.addEntry(rname, entryBool);
                    } else {
                        // Not a boolean dataset, so add original SymEntry to SymTable
                        st.addEntry(rname, entryInt);
                    }
                    rnames.append((dsetName, "pdarray", rname));
                }
                when (false, C_HDF5.H5T_FLOAT) {
                    var entryReal = new shared SymEntry(len, real);
                    gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                      "Initialized float entry");
                    read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName, skips);
                    var rname = st.nextName();
                    st.addEntry(rname, entryReal);
                    rnames.append((dsetName, "pdarray", rname));
                }
                otherwise {
                    var errorMsg = "detected unhandled datatype: segmented? %t, class %i, size %i, " +
                                   "signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
        }

        if allowErrors && fileErrorCount > 0 {
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "allowErrors:true, fileErrorCount:%t".format(fileErrorCount));
        }
        repMsg = _buildReadAllHdfMsgJson(rnames, allowErrors, fileErrorCount, fileErrors, st);
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    /**
     * Construct json object to be returned from readAllHdfMsg
     * :arg rnames: List of (DataSetName, arkouda_type, id of SymEntry) for items read from HDF5 files
     * :type rnames: List of 3*string tuples
     *
     * :arg allowErrors: True if we allowed errors when reading files from HDF5
     * :type allowErros: bool
     *
     * :arg fileErrorCount: Number of files which threw errors when being read
     * :type fileErrorCount: int
     *
     * :arg fileErrors: List of the error messages when trying to read HDF5 files
     * :type fileErrors: list(string)
     *
     * :arg st: SymTab used to look up attributes of pdarray/seg_string ids
     * :type borrowed SymTab:
     *
     * :returns: response message string formatted in json
     *
     * Example
     *   {
     *       "items": [
     *           {
     *               "dataset_name": "int_tens_pdarray",
     *               "arkouda_type": "pdarray",
     *               "created": "created id_9 int64 1000 1 (1000) 8"
     *           }
     *       ],
     *       "allow_errors": "true",
     *       "file_error_count": "1",
     *       "file_errors": [
     *           "Permission error Operation not permitted (error msg) opening path/to/file"
     *       ]
     *   }
     *  Uses keys:  dataset_name, arkouda_type->[pdarray|seg_string], created->(legacy creation statement)
     */
    proc _buildReadAllHdfMsgJson(rnames:list(3*string), allowErrors:bool, fileErrorCount:int, fileErrors:list(string), st: borrowed SymTab): string throws {
        // TODO: Right now we're building the legacy "created ..." string so we'll stuff them in a single array of items
        // in the future we should begin to build out actual json objects of each pdarray as k:v pairs
        var items: list(string);
        for rname in rnames {
            var (dsetName, akType, id) = rname;
            var item = "{" + Q + "dataset_name"+ QCQ + dsetName + Q +
                       "," + Q + "arkouda_type" + QCQ + akType + Q;
            select (akType) {
                when ("pdarray") {
                    item +="," + Q + "created" + QCQ + "created " + st.attrib(id) + Q + "}";
                }
                when ("seg_string") {
                    var (segName, valName) = id.splitMsgToTuple("+", 2);
                    item += "," + Q + "created" + QCQ + "created " + st.attrib(segName) + "+created " + st.attrib(valName) + Q + "}";
                }
                otherwise {
                    item += "}";
                }
            }
            items.append(item);
        }

        // Now assemble the reply message
        var reply = "{" + Q + "items" + Q + ":[" + ",".join(items.these()) + "]";
        if allowErrors && !fileErrors.isEmpty() { // If configured, build the allowErrors portion
            reply += ",";
            reply += Q + "allow_errors" + QCQ + "true" + Q + ",";
            reply += Q + "file_error_count" + QCQ + fileErrorCount:string + Q + ",";
            reply += Q + "file_errors" + Q + ": [" + Q;
            reply += (Q +"," + Q).join(fileErrors.these()) + Q + "]";
        }
        reply += "}";
        return reply;
    }

    /**
     * inline proc to validate the range for our domain.
     * Valid domains must be increasing with the lower bound <= upper bound
     * :arg r: 1D domain
     * :type domain(1): one dimensional domain
     *
     * :returns: bool True iff the lower bound is less than or equal to upper bound
     */
    inline proc _isValidRange(r: domain(1)): bool {
        return r.low <= r.high;
    }

    proc fixupSegBoundaries(a: [?D] int, segSubdoms: [?fD] domain(1), valSubdoms: [fD] domain(1)) throws {
        var boundaries: [fD] int; // First index of each region that needs to be raised
        var diffs: [fD] int; // Amount each region must be raised over previous region
        forall (i, sd, vd, b) in zip(fD, segSubdoms, valSubdoms, boundaries) {
            // if we encounter a malformed subdomain i.e. {1..0} that means we encountered a file
            // that has no data for this SegString object, we can safely skip processing this file.
            if (_isValidRange(sd)) {
                b = sd.low; // Boundary is index of first segment in file
                // Height increase of next region is number of bytes in current region
                if (i < fD.high) {
                    diffs[i+1] = vd.size;
                }
            } else {
                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "fD:%t segments subdom:%t is malformed signaling no segment data in file, skipping".format(i, sd));
            }
        }
        // Insert height increases at region boundaries
        var sparseDiffs: [D] int;
        forall (b, d) in zip(boundaries, diffs) with (var agg = newDstAggregator(int)) {
            agg.copy(sparseDiffs[b], d);
        }
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * sparseDiffs.size);
        // Make plateaus from peaks
        var corrections = + scan sparseDiffs;
        // Raise the segment offsets by the plateaus
        a += corrections;
    }

    /* 
     * Retrieves the datatype of the dataset read from HDF5 
     */
    proc get_dtype(filename: string, dsetName: string, skipSegStringOffsets: bool = false) throws {
        const READABLE = (S_IRUSR | S_IRGRP | S_IROTH);

        if !exists(filename) {
            throw getErrorWithContext(
                           msg="The file %s does not exist".format(filename),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="FileNotFoundError");
        }
        
        /*
         * Checks to see if the file is indeed an HDF5 file. If there is a error
         * in opening file to check format, it is highly likely it is due to 
         * a permissions issue, so a PermissionError is thrown.
         */             
        if !isHdf5File(filename) {
            throw getErrorWithContext(
                           msg="%s is not an HDF5 file".format(filename),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="NotHDF5FileError");        
        }
        
        var file_id = C_HDF5.H5Fopen(filename.c_str(), 
                                         C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
                                         
        if file_id < 0 { // HF5open returns negative value on failure
            C_HDF5.H5Fclose(file_id);
            throw getErrorWithContext(
                           msg="in accessing %s HDF5 file content".format(filename),
                           lineNumber=getLineNumber(), 
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(), 
                           errorClass="HDF5FileFormatError");            
        }

        var dName = getReadDsetName(file_id, dsetName);

        if !C_HDF5.H5Lexists(file_id, dName.c_str(), C_HDF5.H5P_DEFAULT) {
            C_HDF5.H5Fclose(file_id);
            throw getErrorWithContext(
                 msg="The dataset %s does not exist in the file %s".format(dsetName, 
                                                filename),
                 lineNumber=getLineNumber(), 
                 routineName=getRoutineName(), 
                 moduleName=getModuleName(), 
                 errorClass='DatasetNotFoundError');
        }

        var dataclass: C_HDF5.H5T_class_t;
        var bytesize: int;
        var isSigned: bool;
        var isSegArray: bool;

        try {
            defer { // Close the file on exit
                C_HDF5.H5Fclose(file_id);
            }
            if isStringsDataset(file_id, dsetName) {
                if ( !skipSegStringOffsets ) {
                    var offsetDset = dsetName + "/" + SEGARRAY_OFFSET_NAME;
                    var (offsetClass, offsetByteSize, offsetSign) = 
                                            try get_dataset_info(file_id, offsetDset);
                    if (offsetClass != C_HDF5.H5T_INTEGER) {
                        throw getErrorWithContext(
                        msg="dataset %s has incorrect one or more sub-datasets" +
                        " %s %s".format(dsetName,SEGARRAY_OFFSET_NAME,SEGARRAY_VALUE_NAME), 
                        lineNumber=getLineNumber(),
                        routineName=getRoutineName(),
                        moduleName=getModuleName(),
                        errorClass='SegArrayError');                    
                    }
                }
                var valueDset = dsetName + "/" + SEGARRAY_VALUE_NAME;
                try (dataclass, bytesize, isSigned) = 
                                           try get_dataset_info(file_id, valueDset);
                isSegArray = true;
            } else if isBooleanDataset(file_id, dsetName) {
                var booleanDset = dsetName + "/" + "booleans";
                (dataclass, bytesize, isSigned) = get_dataset_info(file_id, booleanDset);
                isSegArray = false;            
            } else {
                (dataclass, bytesize, isSigned) = get_dataset_info(file_id, dsetName);
                isSegArray = false;
            }
        } catch e : Error {
            //:TODO: recommend revisiting this catch block 
            throw getErrorWithContext( 
                msg="in getting_dataset_info %s".format(e.message()), 
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass='Error');
        }
        return (isSegArray, dataclass, bytesize, isSigned);
    }


    /*
     * Returns boolean indicating whether the file is a valid HDF5 file.
     * Note: if the file cannot be opened due to permissions, throws
     * a PermissionError
     */
    proc isHdf5File(filename : string) : int throws {
        var isHdf5 = C_HDF5.H5Fis_hdf5(filename.c_str());
        
        if isHdf5 == 1 {
            return true;
        } else if isHdf5 == 0 {
            return false;
        }

        var errorMsg="%s cannot be opened to check if hdf5, \
                           check file permissions or format".format(filename);
        throw getErrorWithContext(
                       msg=errorMsg,
                       lineNumber=getLineNumber(),
                       routineName=getRoutineName(), 
                       moduleName=getModuleName(),
                       errorClass="PermissionError");      
    }

    /*
     * Returns a tuple containing the data type, data class, and a 
     * boolean indicating whether the datatype is signed for the 
     * supplied file id and dataset name.
     */
    proc get_dataset_info(file_id, dsetName) throws {
        var dset = C_HDF5.H5Dopen(file_id, dsetName.c_str(),
                                                   C_HDF5.H5P_DEFAULT);
        if (dset < 0) {
            throw getErrorWithContext( 
                msg="dataset %s does not exist".format(dsetName), 
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass='DatasetNotFoundError');
        }
        var datatype = C_HDF5.H5Dget_type(dset);
        var dataclass = C_HDF5.H5Tget_class(datatype);
        var bytesize = C_HDF5.H5Tget_size(datatype):int;
        var isSigned = (C_HDF5.H5Tget_sign(datatype) == C_HDF5.H5T_SGN_2);
        C_HDF5.H5Tclose(datatype);
        C_HDF5.H5Dclose(dset);
        return (dataclass, bytesize, isSigned);
    }

    class HDF5RankError: Error {
        var rank: int;
        var filename: string;
        var dsetName: string;
    }

    /*
     * Returns a boolean indicating whether the dataset is a Strings
     * dataset, checking if the values dataset is embedded within a 
     * group named after the dsetName.
     */
    proc isStringsDataset(file_id: int, dsetName: string): bool throws {
        var groupExists = -1;
        
        try {
            // Suppress HDF5 error message that's printed even with no error
            C_HDF5.H5Eset_auto1(nil, nil);
            groupExists = C_HDF5.H5Oexists_by_name(file_id, 
                  "/%s/values".format(dsetName).c_str(),C_HDF5.H5P_DEFAULT);
                
        } catch e: Error {
            /*
             * If there's an actual error, print it here. :TODO: revisit this
             * catch block after confirming the best way to handle HDF5 error
             */
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                    "checking if isStringsDataset %t".format(e.message())); 
        }

        return groupExists > -1;
    }

    /*
     * Returns a boolean indicating whether the dataset is a boolean
     * dataset, checking if the booleans dataset is embedded within a 
     * group named after the dsetName.
     */
    proc isBooleanDataset(file_id: int, dsetName: string): bool throws {
        var groupExists = -1;
        
        try {
            // Suppress HDF5 error message that's printed even with no error
            C_HDF5.H5Eset_auto1(nil, nil);
            groupExists = C_HDF5.H5Oexists_by_name(file_id, 
                  "/%s/booleans".format(dsetName).c_str(),C_HDF5.H5P_DEFAULT);
                
        } catch e: Error {
            /*
             * If there's an actual error, print it here. :TODO: revisit this
             * catch block after confirming the best way to handle HDF5 error
             */
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                      "checking if isBooleanDataset %t".format(e.message()));
        }

        return groupExists > -1;
    }

    /*
     * Overloaded method returns a boolean indicating whether the dataset is a
     * boolean dataset, checking if the booleans dataset is embedded within a 
     * group named after the dsetName. This implementation retrieves the file id
     * for a file name and invokes isBooleanDataset with file id.
     */
    proc isBooleanDataset(fileName: string, dsetName: string): bool throws {
        var fileId = C_HDF5.H5Fopen(fileName.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                           C_HDF5.H5P_DEFAULT);
        defer { // Close the file on exit
            C_HDF5.H5Fclose(fileId);
        }
        var boolDataset: bool;

        try {
            boolDataset = isBooleanDataset(fileId, dsetName);
        } catch e: Error {
            /*
             * If there's an actual error, print it here. :TODO: revisit this
             * catch block after confirming the best way to handle HDF5 error
             */
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                        "checking if isBooleanDataset %t with file %s".format(e.message()));
        }
        return boolDataset;
    }

    /*
     *  Get the subdomains of the distributed array represented by each file, 
     *  as well as the total length of the array. 
     */
    proc get_subdoms(filenames: [?FD] string, dsetName: string) throws {
        use SysCTypes;

        var lengths: [FD] int;
        var skips = new set(string); // Case where there is no data in the file for this dsetName
        for (i, filename) in zip(FD, filenames) {
            try {
                var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                           C_HDF5.H5P_DEFAULT);
                defer { // Close the file on exit
                    C_HDF5.H5Fclose(file_id);
                }

                var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now
                var dName = try! getReadDsetName(file_id, dsetName);

                // Read array length into dims[0]
                C_HDF5.HDF5_WAR.H5LTget_dataset_info_WAR(file_id, dName.c_str(), 
                                           c_ptrTo(dims), nil, nil);
                lengths[i] = dims[0]: int;
                if lengths[i] == 0 {
                    skips.add(filename);
                    gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Adding filename:%s to skips, dsetName:%s, dims[0]:%t".format(filename, dsetName, dims[0]));
                }

            } catch e: Error {
                throw getErrorWithContext(
                             msg="in getting dataset info %s".format(e.message()),
                             lineNumber=getLineNumber(), 
                             routineName=getRoutineName(), 
                             moduleName=getModuleName(), 
                             errorClass='WriteModeError'
                );
            }
        }
        // Compute subdomain of master array contained in each file
        var subdoms: [FD] domain(1);
        var offset = 0;
        for i in FD {
            subdoms[i] = {offset..#lengths[i]};
            offset += lengths[i];
        }
        return (subdoms, (+ reduce lengths), skips);
    }

    /* This function gets called when A is a BlockDist or DefaultRectangular array. */
    proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), 
                                                 filenames: [FD] string, dsetName: string, skips: set(string)) throws 
        where (MyDmap == Dmap.blockDist || MyDmap == Dmap.defaultRectangular) {
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "entry.a.targetLocales() = %t".format(A.targetLocales()));
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "Filedomains: %t".format(filedomains));
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "skips: %t".format(skips));

            coforall loc in A.targetLocales() do on loc {
                // Create local copies of args
                var locFiles = filenames;
                var locFiledoms = filedomains;
                /* On this locale, find all files containing data that belongs in
                 this locale's chunk of A */
                for (filedom, filename) in zip(locFiledoms, locFiles) {
                    var isopen = false;
                    var file_id: C_HDF5.hid_t;
                    var dataset: C_HDF5.hid_t;

                    if (skips.contains(filename)) {
                        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "File %s does not contain data for this dataset, skipping".format(filename));
                    } else {
                        // Look for overlap between A's local subdomains and this file
                        for locdom in A.localSubdomains() {
                            const intersection = domain_intersection(locdom, filedom);
                            if intersection.size > 0 {
                                // Only open the file once, even if it intersects with many local subdomains
                                if !isopen {
                                    file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                                                                            C_HDF5.H5P_DEFAULT);  
                                    var locDsetName = try! getReadDsetName(file_id,dsetName);
                                    try! dataset = C_HDF5.H5Dopen(file_id, locDsetName.c_str(), C_HDF5.H5P_DEFAULT);
                                    isopen = true;
                                }
                                // do A[intersection] = file[intersection - offset]
                                var dataspace = C_HDF5.H5Dget_space(dataset);
                                var dsetOffset = [(intersection.low - filedom.low): C_HDF5.hsize_t];
                                var dsetStride = [intersection.stride: C_HDF5.hsize_t];
                                var dsetCount = [intersection.size: C_HDF5.hsize_t];
                                C_HDF5.H5Sselect_hyperslab(dataspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(dsetOffset), 
                                                                c_ptrTo(dsetStride), c_ptrTo(dsetCount), nil);
                                var memOffset = [0: C_HDF5.hsize_t];
                                var memStride = [1: C_HDF5.hsize_t];
                                var memCount = [intersection.size: C_HDF5.hsize_t];
                                var memspace = C_HDF5.H5Screate_simple(1, c_ptrTo(memCount), nil);
                                C_HDF5.H5Sselect_hyperslab(memspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(memOffset), 
                                                                c_ptrTo(memStride), c_ptrTo(memCount), nil);

                                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                        "Locale %t intersection %t dataset slice %t".format(loc,intersection,
                                        (intersection.low - filedom.low, intersection.high - filedom.low)));

                                /*
                                * The fact that intersection is a subset of a local subdomain means
                                * there should be no communication in the read
                                */
                                local {
                                    C_HDF5.H5Dread(dataset, getHDF5Type(A.eltType), memspace, 
                                            dataspace, C_HDF5.H5P_DEFAULT, 
                                            c_ptrTo(A.localSlice(intersection)));
                                }
                                C_HDF5.H5Sclose(memspace);
                                C_HDF5.H5Sclose(dataspace);
                            }
                        }
                    }
                    if isopen {
                        C_HDF5.H5Dclose(dataset);
                        C_HDF5.H5Fclose(file_id);
                    }
                }
            }
        }

    /* This function is called when A is a CyclicDist array. */
    proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), 
                                           filenames: [FD] string, dsetName: string)
        where (MyDmap == Dmap.cyclicDist) {
            use CyclicDist;
            /*
             * Distribute filenames across locales, and ensure single-threaded
             * reads on each locale
             */
            var fileSpace: domain(1) dmapped Cyclic(startIdx=FD.low, dataParTasksPerLocale=1) = FD;
            forall fileind in fileSpace with (ref A) {
                var filedom: subdomain(A.domain) = filedomains[fileind];
                var filename = filenames[fileind];
                var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                                                       C_HDF5.H5P_DEFAULT);
                // TODO: use select_hyperslab to read directly into a strided slice of A
                // Read file into a temporary array and copy into the correct chunk of A
                var AA: [1..filedom.size] A.eltType;
                
                // Retrieve the dsetName that accounts for enclosing group, if applicable
                try! readHDF5Dataset(file_id, getReadDsetName(file_id, dsetName), AA);
                A[filedom] = AA;
                C_HDF5.H5Fclose(file_id);
           }
    }

    proc domain_intersection(d1: domain(1), d2: domain(1)) {
        var low = max(d1.low, d2.low);
        var high = min(d1.high, d2.high);
        if (d1.stride !=1) && (d2.stride != 1) {
            //TODO: change this to throw
            halt("At least one domain must have stride 1");
        }
        var stride = max(d1.stride, d2.stride);
        return {low..high by stride};
    }

    proc tohdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        var (arrayName, dsetName, modeStr, jsonfile, dataType, segsName, writeOffsetsFlag)= payload.splitMsgToTuple(7);
        var mode = try! modeStr: int;
        var filename: string;
        var entry = st.lookup(arrayName);
        var writeOffsets = "true" == writeOffsetsFlag.strip().toLower();

        try {
            filename = jsonToPdArray(jsonfile, 1)[0];
        } catch {
            var errorMsg = "Could not decode json filenames via tempfile " +
                                                    "(%i files: %s)".format(1, jsonfile);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var warnFlag: bool;

        try {
            select entry.dtype {
                when DType.Int64 {
                    var e = toSymEntry(entry, int);
                    warnFlag = write1DDistArray(filename, mode, dsetName, e.a, DType.Int64);
                }
                when DType.Float64 {
                    var e = toSymEntry(entry, real);
                    warnFlag = write1DDistArray(filename, mode, dsetName, e.a, DType.Float64);
                }
                when DType.Bool {
                    var e = toSymEntry(entry, bool);
                    warnFlag = write1DDistArray(filename, mode, dsetName, e.a, DType.Bool);
                }
                when DType.UInt8 {
                    /*
                     * Look up the values and segments arrays, both of which are needed to write
                     * uint8 arrays such as Strings out to external systems.
                     */
                    var e = toSymEntry(entry, uint(8));
                    var segsEntry = st.lookup(segsName);
                    var s_e = toSymEntry(segsEntry, int);
                    warnFlag = write1DDistStrings(filename, mode, dsetName, e.a, DType.UInt8, s_e.a, writeOffsets);
                } otherwise {
                    var errorMsg = unrecognizedTypeError("tohdf", dtype2str(entry.dtype));
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
        } catch e: FileNotFoundError {
              var errorMsg = "Unable to open %s for writing: %s".format(filename,e.message());
              gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
        } catch e: MismatchedAppendError {
              var errorMsg = "Mismatched append %s".format(e.message());
              gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
        } catch e: WriteModeError {
              var errorMsg = "Write mode error %s".format(e.message());
              gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
        } catch e: Error {
              var errorMsg = "problem writing to file %s".format(e);
              gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        if warnFlag {
             var warnMsg = "Warning: possibly overwriting existing files matching filename pattern";
             return new MsgTuple(warnMsg, MsgType.WARNING);
        } else {
            var repMsg = "wrote array to file";
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);            
        }
    }

    /*
     * Writes out the two pdarrays composing a Strings object to hdf5.
     */
    private proc write1DDistStrings(filename: string, mode: int, dsetName: string, A, 
                                                                array_type: DType, SA, writeOffsets:bool) throws {
        var prefix: string;
        var extension: string;
        var warnFlag: bool;

        var total = new Time.Timer();
        total.clear();
        total.start();
        
        (prefix,extension) = getFileMetadata(filename);
 
        // Generate the filenames based upon the number of targetLocales.
        var filenames = generateFilenames(prefix, extension, A);
        
        // Generate a list of matching filenames to test against. 
        var matchingFilenames = getMatchingFilenames(prefix, extension);
        
        // Create files with groups needed to persist values and segments pdarrays
        var group = getGroup(dsetName);
        warnFlag = processFilenames(filenames, matchingFilenames, mode, A, group);
        
        /*
         * The shuffleLeftIndices object, which is a globally-scoped PrivateSpace, 
         * contains indices for each locale that (1) specify the chars that can be 
         * shuffled left to complete the last string in the previous locale and (2)
         * are used to remove the corresponding chars from the current, donor locale.  
         *
         * The shuffleRightIndices PrivateSpace is used in the special case 
         * where the majority of a large string spanning two locales is the sole
         * string on a locale; in this case, each index specifies the chars that 
         * can be shuffled right to start the string completed in the next locale
         * and remove the corresponding chars from the current, donor locale 
         *
         * The isSingleString PrivateSpace indicates whether each locale contains
         * chars corresponding to one string/string segment; this occurs if 
         * (1) the char array contains no null uint(8) characters or (2) there is
         * only one null uint(8) char at the end of the string/string segment
         *
         * The endsWithCompleteString PrivateSpace indicates whether the values
         * array for each locale ends with complete string, meaning that the last
         * character in the local slice is a null uint(8) char.
         *
         * The charArraySize PrivateSpace contains the size of char local slice
         * corresponding to each locale.
         */
        var shuffleLeftIndices: [PrivateSpace] int;
        var shuffleRightIndices: [PrivateSpace] int;
        var isSingleString: [PrivateSpace] bool;
        var endsWithCompleteString: [PrivateSpace] bool;
        var charArraySize: [PrivateSpace] int;

        /*
         * Loop through all locales and set the shuffleLeftIndices, shuffleRightIndices,
         * isSingleString, endsWithCompleteString, and charArraySize PrivateSpaces
         */
        // initialize timer
        var t1 = new Time.Timer();
        t1.clear();
        t1.start();

        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) 
             with (ref shuffleLeftIndices, ref shuffleRightIndices, 
                   ref isSingleString, ref endsWithCompleteString, ref charArraySize) do on loc {
             generateStringsMetadata(idx, shuffleLeftIndices, shuffleRightIndices, 
                          isSingleString, endsWithCompleteString, charArraySize, A, SA);
        }

        t1.stop();  
        var elapsed = t1.elapsed();
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "Time for generating all values metadata: %.17r".format(elapsed));

        const SALength = SA.domain.size;
        /*
         * Iterate through each locale and (1) open the hdf5 file corresponding to the
         * locale (2) prepare char and segment lists to be written (3) write each
         * list as a Chapel array to the open hdf5 file and (4) close the hdf5 file
         */
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with 
                        (ref shuffleLeftIndices, ref shuffleRightIndices, 
                                                            ref charArraySize) do on loc {
                        
            /*
             * Generate metadata such as file name, file id, and dataset name
             * for each file to be written
             */
            const myFilename = filenames[idx];

            var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), 
                                       C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
            defer { // Close the file on exit
                C_HDF5.H5Fclose(myFileID);
            }
            const locDom = A.localSubdomain();
            var dims: [0..#1] C_HDF5.hsize_t;
            dims[0] = locDom.size: C_HDF5.hsize_t;
            var myDsetName = "/" + dsetName;

            use C_HDF5.HDF5_WAR;

            /*
             * Confirm if the Strings write is in append mode. If so, the Strings dataset 
             * is going to be appended to an hdf5 file as a set of values and segments 
             * arrays within a new group named after the dsetName. Consequently, need
             * to create the group within the existing hdf5 file.
             */
            if mode == APPEND {
                prepareGroup(myFileID, group);
            }

            if idx > SALength - 1  {
                // Case where num_elements < num_locales
                // We need to write a nil into this locale's file
                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "write1DDistStrings: num elements < num locales, locale %i, will get empty dataset".format(idx));
                var charList : list(uint(8)) = new list(uint(8));
                var segmentsList : list(int) = new list(int);
                writeStringsToHdf(myFileID, idx, group, charList, segmentsList, true);

            } else {
                /*
                 * Check for the possibility that a string in the current locale spans
                 * two neighboring locales by seeing if the final character in the local 
                 * slice is the null uint(8) character. If it is not, this means the last string 
                 * in the current locale (idx) spans the current AND next locale.
                 */
                if A.localSlice(locDom).back() != NULL_STRINGS_VALUE { 
                    /*
                     * Retrieve the chars array slice from this locale and populate the charList
                     * that will be updated per left and/or right shuffle operations until the 
                     * final char list is assembled
                     */ 
                    var charArray = A.localSlice(locDom);
                    var charList : list(uint(8)) = new list(charArray);

                    gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Locale %i does not end with null char, need left or right shuffle".format(idx));

                    /*
                     * If (1) this locale contains a single string/string segment (and therefore no
                     * leading slice or trailing slice), and (2) is not the first locale, retrieve
                     * the right shuffle chars from the previous locale, if applicable, to set the
                     * correct starting chars for the lone string/string segment on this locale.
                     *
                     * Note: if this is the first locale, there are no chars from a previous 
                     * locale to shuffle right, so this code block is not executed in this case.
                     */
                    if isSingleString[idx] && idx > 0 {
                        // Retrieve the shuffleRightIndex from the previous locale
                        var shuffleRightIndex = shuffleRightIndices[idx-1];
                        
                        if shuffleRightIndex > -1 {
                            /*
                            * There are 1..n chars to be shuffled right from the previous locale
                            * (idx-1) to complete the beginning of the one string assigned 
                            * to the current locale (idx). Accordingly, slice the right shuffle
                            * chars from the previous locale
                            */
                            var rightShuffleSlice : [shuffleRightIndex..charArraySize[idx-1]-1] uint(8);

                            on Locales[idx-1] {
                                const locDom = A.localSubdomain();
                                var localeArray = A.localSlice(locDom);
                                rightShuffleSlice = localeArray[shuffleRightIndex..localeArray.size-1];
                            }

                            /*
                            * Prepend the current locale charsList with the chars shuffled right from 
                            * the previous locale
                            */
                            charList.insert(0,rightShuffleSlice);

                            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            'right shuffle from locale %i into single string locale %i'.format(
                                                idx-1,idx));
                        } else {
                            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            'no right shuffle from locale %i into single string locale %i'.format(
                                                idx-1,idx));
                        }
                    }

                    /*
                     * Now that the start of the first string of the current locale (idx) is correct,
                     * shuffle chars to place a complete string at the end the current locale. 
                     *
                     * There are two possible scenarios to account for. First, the next locale 
                     * has a shuffleLeftIndex > -1. If so, the chars up to the shuffleLeftIndex 
                     * will be shuffled from the next locale (idx+1) to complete the last string 
                     * in the current locale (idx). In the second scenario, the next locale is 
                     * the last locale in the Arkouda cluster. If so, all of the chars 
                     * from the next locale are shuffled to the current locale.
                     */
                    var shuffleLeftSlice: [0..shuffleLeftIndices[idx+1]-2] uint(8);

                    if shuffleLeftIndices[idx+1] > -1 || isLastLocale(idx+1) {
                        on Locales[idx+1] {
                            const locDom = A.localSubdomain();

                            var localeArray = A.localSlice(locDom);
                            var shuffleLeftIndex = shuffleLeftIndices[here.id];
                            var localStart = locDom.first;
                            var localLeadingSliceIndex = localStart + shuffleLeftIndex -2;

                            shuffleLeftSlice = localeArray[localStart..localLeadingSliceIndex];    
                            charList.extend(shuffleLeftSlice);  
    
                            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),   
                            'shuffled left from locale %i to complete string in locale %i'.format(
                                            idx+1,idx));
                        }
                    } else {
                        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                 'no left shuffle from locale %i to locale %i'.format(idx+1,idx));
                    }

                    /* 
                    * To prepare for writing the charList to hdf5, do the following, if applicable:
                    * 1. Remove the characters shuffled left to the previous locale
                    * 2. Remove the characters shuffled right to the next locale
                    * 3. If (2) does not apply, add null uint(8) char to end of the charList
                    */
                    var shuffleLeftIndex = shuffleLeftIndices[idx]:int;
                    var shuffleRightIndex = shuffleRightIndices[idx]:int;

                    /*
                    * Verify if the current locale (idx) contains chars shuffled left to the previous
                    * locale (idx-1) by checking the shuffleLeftIndex, the number of strings in 
                    * the current locale, and whether the preceding locale ends with a complete
                    * string. If (1) the shuffleLeftIndex > -1, (2) this locale contains 2..n
                    * strings, and (3) the previous locale does not end with a complete string,
                    * this means the charList contains chars that were shuffled left to complete
                    * the last string in the previous locale (idx-1). If so, generate
                    * a new charList that has those values sliced out. 
                    */
                    if shuffleLeftIndex > -1 && !isSingleString[idx]
                                                        && !endsWithCompleteString[idx-1] {
                        /*
                        * Since the leading slice was used to complete the last string in
                        * the previous locale (idx-1), slice those chars from the charList
                        */
                        charList = new list(adjustForLeftShuffle(shuffleLeftIndex,charList));    

                        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                'adjusted locale %i for left shuffle to %i'.format(idx,idx-1)); 
                    } else {
                        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                'no left shuffle adjustment for locale %i'.format(idx));
                    }

                    /*
                    * Verify if the current locale contains chars shuffled right to the next
                    * locale because (1) the next locale only has one string/string segment
                    * and (2) the current locale's shuffleRightIndex > -1. If so, remove the
                    * chars starting with the shuffleRightIndex, which will place the null 
                    * uint(8) char at the end of the charList. Otherwise, manually add the
                    * null uint(8) char to the end of the charList.
                    */
                    if shuffleRightIndex > -1 && isSingleString[idx+1] {
                        // adjustForRightShuffle is inclusive but we need exclusive on the last char
                        charList = new list(adjustForRightShuffle(shuffleRightIndex-1, charList));
                        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "adjusted locale %i for right shuffle to locale %i".format(idx,idx+1));
                        
                    } else {
                        charList.append(NULL_STRINGS_VALUE);
                        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            'no adjustment for right shuffle from locale %i to locale %i'.format(
                                            idx,idx+1));        
                    }
                    
                    // Generate the segments list now that the char list is finalized
                    var segmentsList = if writeOffsets then generateFinalSegmentsList(charList, idx) else new list(int);
                
                    // Write the finalized valuesList and segmentsList to the hdf5 group
                    writeStringsToHdf(myFileID, idx, group, charList, segmentsList);
                } else {
                    /*
                    * The current local slice (idx) ends with the uint(8) null character,
                    * which is the value required to ensure correct read logic.
                    */
                    var charList : list(uint(8));

                    gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        'locale %i ends with null char, no left or right shuffle needed'.format(idx));

                    /*
                    * Check to see if the current locale (idx) slice contains 1..n chars that
                    * complete the last string in the previous (idx-1) locale.
                    */
                    var shuffleLeftIndex = shuffleLeftIndices[idx]:int;

                    if shuffleLeftIndex == -1 {
                        /*
                        * Since the shuffleLeftIndex is -1, the current local slice (idx) does 
                        * not contain chars from a string started in the previous locale (idx-1).
                        * Accordingly, initialize with the current locale slice.
                        */
                        charList = new list(A.localSlice(locDom));

                        /*
                        * If this locale (idx) ends with the null uint(8) char, check to see if 
                        * the shuffleRightIndex from the previous locale (idx-1) is > -1. If so,
                        * the chars following the shuffleRightIndex from the previous locale complete 
                        * the one string/string segment within the current locale.
                        */
                        if isSingleString[idx] && idx > 0 {
                            /*
                            * Get shuffleRightIndex from previous locale to see if the current locale
                            * charList needs to be prepended with chars shuffled from previous locale
                            */
                            var shuffleRightIndex = shuffleRightIndices[idx-1];

                            if shuffleRightIndex > -1 {
                                var shuffleRightSlice: [shuffleRightIndex..charArraySize[idx-1]-1] uint(8);
                                on Locales[idx-1] {
                                    const locDom = A.localSubdomain();  
                                    var localeArray = A.localSlice(locDom);
                                    shuffleRightSlice = localeArray[shuffleRightIndex..localeArray.size-1]; 
                                }
                                charList.insert(0,shuffleRightSlice);
                                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                    'inserted right shuffle slice from locale %i into locale %i'.format(
                                                idx-1,idx));
                            } else {
                                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),  
                                    'no right shuffle from locale %i inserted into locale %i'.format(
                                                idx-1,idx));                       
                            }
                        }

                        /*
                        * Account for the special case where the following is true about the
                        * current locale (idx):
                        *
                        * 1. This is the last locale in a multi-locale deployment
                        * 2. There is one partial string started in the previous locale
                        * 3. The previous locale has no trailing slice to complete the partial
                        *    string in the current locale
                        *
                        * In this very special case, (1) move the current locale (idx) chars to
                        * the previous locale (idx-1) and (2) clear out the current locale charList.
                        */                     
                        if numLocales > 1 && isLastLocale(idx) {
                            if !endsWithCompleteString[idx-1] && isSingleString[idx] 
                                                            && shuffleRightIndices[idx-1] == -1 {
                                charList.clear();
                                gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                    'cleared out last locale %i due to left shuffle to locale %i'.format(
                                            idx,idx-1));
                            }
                        }
                        
                        // Generate the segments list now that the char list is finalized
                        var segmentsList = if writeOffsets then generateFinalSegmentsList(charList, idx) else new list(int);
    
                        // Write the finalized valuesList and segmentsList to the hdf5 group
                        writeStringsToHdf(myFileID, idx, group, charList, segmentsList);
                    } else {
                        /*
                        * Check to see if previous locale (idx-1) ends with a null character.
                        * If not, then the left shuffle slice of this locale was used to complete
                        * the last string in the previous locale, so slice those chars from
                        * this locale and create a new, corresponding charList.
                        */
                        if !endsWithCompleteString(idx-1) {
                            var localStart = locDom.first;
                            var localLeadingSliceIndex = localStart + shuffleLeftIndex;
                            var leadingCharArray = adjustCharArrayForLeadingSlice(localLeadingSliceIndex, 
                                            A.localSlice(locDom),locDom.last);
                            charList = new list(leadingCharArray);  
                            gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                    'adjusted locale %i for left shuffle to locale %i'.format(
                                            idx,idx-1));
                        } else {
                            charList = new list(A.localSlice(locDom));
                            gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                    'no left shuffle from locale %i to locale %i'.format(
                                            idx,idx-1));
                        } 

                        // Generate the segments list now that the char list is finalized
                        var segmentsList = if writeOffsets then generateFinalSegmentsList(charList, idx) else new list(int);

                        // Write the finalized valuesList and segmentsList to the hdf5 group
                        writeStringsToHdf(myFileID, idx, group, charList, segmentsList);
                    }
                }
            }
        }
        total.stop();  
        gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                             "Completed write1DDistStrings in %.17r seconds".format(total.elapsed()));  
        return warnFlag;
    }

    /*
     * Writes the float, int, or bool pdarray out to hdf5
     */
    proc write1DDistArray(filename: string, mode: int, dsetName: string, A,
                                                                array_type: DType) throws {
        /* Output is 1 file per locale named <filename>_<loc>, and a dataset
        named <dsetName> is created in each one. If mode==1 (append) and the
        correct number of files already exists, then a new dataset named
        <dsetName> will be created in each. Strongly recommend only using
        append mode to write arrays with the same domain. */

        var prefix: string;
        var extension: string;
      
        (prefix,extension) = getFileMetadata(filename);

        // Generate the filenames based upon the number of targetLocales.
        var filenames = generateFilenames(prefix, extension, A);

        //Generate a list of matching filenames to test against. 
        var matchingFilenames = getMatchingFilenames(prefix, extension);

        var warnFlag = processFilenames(filenames, matchingFilenames, mode, A);

        /*
         * Iterate through each locale and (1) open the hdf5 file corresponding to the
         * locale (2) prepare pdarray(s) to be written (3) write pdarray(s) to open
         * hdf5 file and (4) close the hdf5 file
         */
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) do on loc {
            const myFilename = filenames[idx];

            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "%s exists? %t".format(myFilename, exists(myFilename)));

            var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), 
                                       C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
            defer { // Close the file on scope exit
                C_HDF5.H5Fclose(myFileID);
            }

            const locDom = A.localSubdomain();
            var dims: [0..#1] C_HDF5.hsize_t;
            dims[0] = locDom.size: C_HDF5.hsize_t;

            use C_HDF5.HDF5_WAR;

            var dType: C_HDF5.hid_t = getDataType(A);

            /*
             * Prepare the HDF5 group if the datatype requires the array to be written 
             * out to a group other than the top-level HDF5 group.
             */
            if isGroupedDataType(dType) {
                prepareGroup(fileId=myFileID, dsetName);
            }
            
            var myDsetName = getWriteDsetName(dType=dType, dsetName=dsetName);

            /*
             * Depending upon the datatype, write the local slice out to the top-level
             * or nested, named group within the hdf5 file corresponding to the locale.
             */
            if locDom.size <= 0 {
                H5LTmake_dataset_WAR(myFileID, myDsetName.c_str(), 1, c_ptrTo(dims), dType, nil);
            } else {
                H5LTmake_dataset_WAR(myFileID, myDsetName.c_str(), 1, c_ptrTo(dims), dType, c_ptrTo(A.localSlice(locDom)));
            }
        }
        return warnFlag;
    }
    
    /*
     * Returns a boolean indicating if the data type is written to an HDF5
     * group, which currently is C_HDF5.H5T_NATIVE_HBOOL.
     */
    proc isGroupedDataType(dType: C_HDF5.hid_t) : bool {
        return dType  == C_HDF5.H5T_NATIVE_HBOOL;
    }
    
    /*
     * Returns the HDF5 data type corresponding to the dataset, which delegates
     * to getHDF5Type for all datatypes supported by Chapel. For datatypes that
     * are not supported by Chapel, getDataType encapsulates logic to retrieve
     * the HDF5 data type.
     */
    proc getDataType(A) : C_HDF5.hid_t {
        var dType : C_HDF5.hid_t;
            
        if A.eltType == bool {
            return C_HDF5.H5T_NATIVE_HBOOL;
        } else {
            return getHDF5Type(A.eltType);
        }
    }
    
    /*
     * Retrieves the full dataset name including the group name, if applicable,
     * for the dataset to be written to HDF5.
     */
    proc getWriteDsetName(dType: C_HDF5.hid_t, 
                                    dsetName: string) : string throws {
        if dType == C_HDF5.H5T_NATIVE_HBOOL {
            return "/%s/booleans".format(dsetName);
        } else {
            return "/" + dsetName;
        }
    }

    /*
     * Retrieves the full dataset name including the group name, if applicable,
     * for the dataset to be read from HDF5.
     */
    proc getReadDsetName(file_id: int, dsetName: string) : string throws {
        if isBooleanDataset(file_id, dsetName) {
            return "%s/booleans".format(dsetName);
        } else {
            return dsetName;
        }
    }

    /*
     * Returns a tuple composed of a file prefix and extension to be used to
     * generate locale-specific filenames to be written to.
     */
    proc getFileMetadata(filename : string) {
        const fields = filename.split(".");
        var prefix: string;
        var extension: string;
 
        if fields.size == 1 || fields[fields.domain.high].count(pathSep) > 0 { 
            prefix = filename;
            extension = "";
        } else {
            prefix = ".".join(fields#(fields.size-1)); // take all but the last
            extension = "." + fields[fields.domain.high];
        }

        return (prefix,extension);
    }

    /*
     * Generates a list of filenames to be written to based upon a file prefix,
     * extension, and number of locales.
     */
    proc generateFilenames(prefix : string, extension : string, A) : [] string throws { 
        // Generate the filenames based upon the number of targetLocales.
        var filenames: [0..#A.targetLocales().size] string;
        for i in 0..#A.targetLocales().size {
            filenames[i] = generateFilename(prefix, extension, i);
        }   
        return filenames;
    }

    /*
     * Generates a file name composed of a prefix, which is a filename provided by
     * the user along with a file index and extension.
     */
    proc generateFilename(prefix : string, extension : string, idx : int) : string throws {
        var suffix = '%04i'.format(idx);
        return "%s_LOCALE%s%s".format(prefix, suffix, extension);
    }    

    /*
     * If APPEND mode, checks to see if the matchingFilenames matches the filenames
     * array and, if not, raises a MismatchedAppendError. If in TRUNCATE mode, creates
     * the files matching the filenames. If 1..n of the filenames exist, returns 
     * warning to the user that 1..n files were overwritten. Since a group name is 
     * passed in, and hdf5 group is created in the file(s).
     */
    proc processFilenames(filenames: [] string, matchingFilenames: [] string, mode: int, 
                                            A, group: string) throws {
      // if appending, make sure number of files hasn't changed and all are present
      var warnFlag: bool;
      
      /*
       * Generate a list of matching filenames to test against. If in 
       * APPEND mode, check to see if list of filenames to be written
       * to match the names of existing files corresponding to the dsetName.
       * if in TRUNCATE mode, see if there are any filenames that match, 
       * meaning that 1..n files will be overwritten.
       */
      if (mode == APPEND) {
          var allexist = true;
          var anyexist = false;
          
          for f in filenames {
              var result =  try! exists(f);
              allexist &= result;
              if result {
                  anyexist = true;
              }
          }

          /*
           * Check to see if any exist. If not, this means the user is attempting to append
           * to 1..n files that don't exist. In this situation, the user is alerted that
           * the dataset must be saved in TRUNCATE mode.
           */
          if !anyexist {
              throw getErrorWithContext(
                 msg="Cannot append a non-existent file, please save without mode='append'",
                 lineNumber=getLineNumber(), 
                 routineName=getRoutineName(), 
                 moduleName=getModuleName(), 
                 errorClass='WriteModeError'
              );
          }

          /*
           * Check if there is a mismatch between the number of files to be appended to and
           * the number of files actually on the file system. This typically happens when 
           * a file append is attempted where the number of locales between the file 
           * creates and updates changes.
           */
          if !allexist || (matchingFilenames.size != filenames.size) {
              throw getErrorWithContext(
                   msg="appending to existing files must be done with the same number " +
                      "of locales. Try saving with a different directory or filename prefix?",
                   lineNumber=getLineNumber(), 
                   routineName=getRoutineName(), 
                   moduleName=getModuleName(), 
                   errorClass='MismatchedAppendError'
              );
          }

      } else if mode == TRUNCATE { // if truncating, create new file per locale
          if matchingFilenames.size > 0 {
              warnFlag = true;
          } else {
              warnFlag = false;
          }

          coforall loc in A.targetLocales() do on loc {
              var file_id: C_HDF5.hid_t;

              gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "Creating or truncating file");

              file_id = C_HDF5.H5Fcreate(filenames[loc.id].localize().c_str(), C_HDF5.H5F_ACC_TRUNC,
                                                        C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
              defer { // Close file upon exiting scope
                  C_HDF5.H5Fclose(file_id);
              }

              // Prepare file versioning metadata
              addArkoudaHdf5VersioningMetadata(file_id);

              if (!group.isEmpty()) {
                  prepareGroup(file_id, group);
              }

              if file_id < 0 { // Negative file_id means error
                  throw getErrorWithContext(
                                    msg="The file %s does not exist".format(filenames[loc.id]),
                                    lineNumber=getLineNumber(), 
                                    routineName=getRoutineName(), 
                                    moduleName=getModuleName(), 
                                    errorClass='FileNotFoundError');
              }
           }
        } else {
            throw getErrorWithContext(
                                    msg="The mode %t is invalid".format(mode),
                                    lineNumber=getLineNumber(), 
                                    routineName=getRoutineName(), 
                                    moduleName=getModuleName(), 
                                    errorClass='IllegalArgumentError');
        }      
        return warnFlag;
    }

    /*
     * If APPEND mode, checks to see if the matchingFilenames matches the filenames
     * array and, if not, raises a MismatchedAppendError. If in TRUNCATE mode, creates
     * the files matching the filenames. If 1..n of the filenames exist, returns 
     * warning to the user that 1..n files were overwritten.
     */
    proc processFilenames(filenames: [] string, matchingFilenames: [] string, mode: int, A) throws {
        return processFilenames(filenames, matchingFilenames, mode, A, "");
    }
    
    /*
     * Generates an array of filenames to be matched in APPEND mode and to be
     * checked in TRUNCATE mode that will warn the user that 1..n files are
     * being overwritten.
     */
    proc getMatchingFilenames(prefix : string, extension : string) throws {
        return glob("%s_LOCALE*%s".format(prefix, extension));    
    }

    /*
     * Generates Strings metadata required to partition the corresponding string sequences
     * across 1..n locales via shuffle operations. The metadata includes (1) left and
     * right shuffle slice indices (2) flags indicating whether the locale char arrays
     * contain one string (3) if the char arrays end with a complete string and (4)
     * the length of each locale slice of the chars array (used for some array slice ops).
     */
    private proc generateStringsMetadata(idx : int, shuffleLeftIndices, 
                       shuffleRightIndices, isSingleString, endsWithCompleteString, 
                       charArraySize, A, SA) throws {
        on Locales[idx] {
            //Retrieve the chars and segs local slices (portions of arrays on this locale)
            const locDom = A.localSubdomain();
            const segsLocDom = SA.localSubdomain();
            const charArray = A.localSlice(locDom);
            const segsArray = SA.localSlice(segsLocDom);

            const totalSegs = SA.size;

            /**
             * There are a couple of cases here
             * 1. This locale doesn't actually have any data to serve because the size of the SegStrings is too small
             * 2. The number of segments is less the number of locales but we do have values/bytes in which case
             *    we need to shuffle all of it to the left.
             * 3. There is enough elements & data that everybody is going to save something, which is the normal case
             */
            if locDom.size == 0 && segsLocDom.size == 0 { // no data served, nothing to shuffle
                gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                    "Locale idx:%t, has segsLocDom.size && locDom.size of zero, this locale serves no data".format(idx));
                
                // We have nothing, so pretend we are first locale
                shuffleLeftIndices[idx] = -1;
                shuffleRightIndices[idx] = -1;
                isSingleString[idx] = false;
                endsWithCompleteString[idx] = false;
                charArraySize[idx] = 0;

            } else if locDom.size > 0 && (totalSegs < numLocales) && (idx >= totalSegs) { // num segs < num locales so move data left
                gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                    "Locale idx:%t has data, but totalSegs < numLocales, so shuffle all of it left".format(idx));
                charArraySize[idx] = charArray.size;
                shuffleLeftIndices[idx] = locDom.size; // should be all of the characters
                shuffleRightIndices[idx] = -1;
                isSingleString[idx] =  false;
                endsWithCompleteString[idx] = false;
                gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                    "Locale idx:%t shuffleLeft:%t".format(idx, locDom.size));

            } else {
                charArraySize[idx] = charArray.size;
                var leadingSliceSet = false;

                //Initialize both indices to -1 to indicate neither exists for locale
                shuffleLeftIndices[idx] = -1;
                shuffleRightIndices[idx] = -1;

                /*
                * Check if the last char is the null uint(8) char. If so, the last
                * string on the locale completes within the locale. Otherwise,
                * the last string spans to the next locale.
                */
                if charArray.back() == NULL_STRINGS_VALUE {
                    endsWithCompleteString[idx] = true;
                } else {
                    endsWithCompleteString[idx] = false;
                }
                
                // initialize the firstSeg and lastSeg variables
                var firstSeg = -1;
                var lastSeg = -1;

                /*
                * If the first locale (locale 0), the first segment is retrieved
                * via segsArray.front(), corresponding to 0. Otherwise, find the 
                * first occurrence of the null uint(8) char and the firstSeg is the
                * next non-null char. The lastSeg in all cases is the final segsArray
                * element retrieved via segsArray.back()
                */
                if idx == 0 {
                    firstSeg = segsArray.front();
                    lastSeg = segsArray.back();
                    gsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                    "Locale idx:%t firstSeg:%t, lastSeg:%t".format(idx, firstSeg, lastSeg));
                } else {
                    var (nullString,fSeg) = charArray.find(NULL_STRINGS_VALUE);
                    if nullString {
                        firstSeg = fSeg + 1;
                    }
                    lastSeg = segsArray.back();
                }

                /*
                * Normalize the first and last seg elements (make them zero-based) by
                * subtracting the char domain first index element. 
                */
                var normalize = 0;
                if idx > 0 {
                    normalize = locDom.first;
                }
        
                var adjFirstSeg = firstSeg - normalize;
                var adjLastSeg = lastSeg - normalize;
                                                    
                if adjFirstSeg == 0 {
                    shuffleLeftIndices[idx] = -1;
                } else {
                    shuffleLeftIndices[idx] = adjFirstSeg;
                }
                
                if !endsWithCompleteString[idx] {
                    shuffleRightIndices[idx] = adjLastSeg;
                } else {
                    shuffleRightIndices[idx] = -1;
                }
            
                if shuffleLeftIndices[idx] > -1 || shuffleRightIndices[idx] > -1 {
                    /*
                    * If either of the indices are > -1, this means there's 2..n null characters
                    * in the char array, which means the char array contains 2..n strings and/or
                    * string portions.
                    */   
                    isSingleString[idx] = false;
                } else {
                    /*
                    * Since there is neither a shuffleLeftIndex nor a shuffleRightIndex for
                    * this locale, this local contains a single, complete string.
                    */
                    isSingleString[idx] = true;
                }

                /*
                * For the special case of this being the first locale, set the shuffleLeftIndex
                * to -1 since there is no previous locale that has an incomplete string at the
                * end that will require chars sliced from locale 0 to complete. If there is one
                * null uint(8) char that is not at the end of the values array, this is the 
                * shuffleRightIndex for the first locale.
                */
                if idx == 0 {
                    if shuffleLeftIndices[idx] > -1 {
                        shuffleRightIndices[idx] = shuffleLeftIndices[idx];
                    }
                    shuffleLeftIndices[idx] = -1;
                    
                    // Special case we have only one segment, figure out if we're hosting extra characters
                    if firstSeg == 0 && lastSeg == 0 && shuffleRightIndices[idx] == 0 {
                        // We can't just look at the last character to see if it is null,
                        // we have to determine we HAVE a null char AND that it preceeds the last char.
                        var (found, foundLoc) = charArray.find(NULL_STRINGS_VALUE);
                        if (found && foundLoc != charArray.size - 1) {
                            shuffleRightIndices[idx] = foundLoc + 1; // This is the start position of the string to shuffle right
                        }
                    }
                }
                
                /*
                * For the special case of this being the last locale, set the shuffleRightIndex
                * to -1 since there is no next locale to shuffle a trailing slice to.
                */
                if isLastLocale(idx) {
                    shuffleRightIndices[idx] = -1;
                }
            }

        }
    }
    
    /*
     * Adjusts for the shuffling of a leading char sequence to the previous locale by 
     * slicing leading chars that compose a string started in the previous locale and 
     * returning a new char array.
     */
    private proc adjustCharArrayForLeadingSlice(sliceIndex, charArray, last) throws { 
        return charArray[sliceIndex..last]; 
    }    

    /*
     * Adjusts for the left shuffle of the leading char sequence from the current locale
     * to the previous locale by returning a slice containing chars from the shuffleLeftIndex
     * to the end of the charList.
     */
    private proc adjustForLeftShuffle(shuffleLeftIndex: int, charList) throws {
        return charList[shuffleLeftIndex..charList.size-1];
    }

    /* 
     * Adjusts for the right shuffle of the trailing char sequence from the current locale
     * to the next locale by returning a slice containing chars up to and including 
     * the rightShuffleIndex. 
     */
    private proc adjustForRightShuffle(shuffleRightIndex: int, 
                                               charsList: list(uint(8))) throws {        
        return charsList[0..shuffleRightIndex];
    }

    private proc generateFinalSegmentsList(charList : list(uint(8)), idx: int) throws {
        var segments: list(int);
        segments.append(0);

        for (value, i) in zip(charList, 0..charList.size-1) do {
            /*
             * If the char is the null uint(8) char, check to see if it is the 
             * last char. If not, added to the indices. If it is the last char,  
             * don't add, because it is the correct ending char for a Strings 
             * values array to be written to a locale.
             */ 
            if value == NULL_STRINGS_VALUE && i < charList.size-1 {
                segments.append(i+1);
            }
        }
        
        return segments;
    }

    /*
     * Returns the name of the hdf5 group corresponding to a dataset name.
     */
    private proc getGroup(dsetName : string) : string throws {
        var values = dsetName.split('/');
        if values.size < 1 {
            throw getErrorWithContext(
               msg="Strings dataset format must be {dset}/values, Booleans {dset}/booleans",
               lineNumber=getLineNumber(), 
               routineName=getRoutineName(), 
               moduleName=getModuleName(), 
               errorClass='IllegalArgumentError'
            );            
        } else {
            return values[0];
        }
    }

    /*
     * Creates an HDF5 Group named via the group parameter to store a grouped
     * object's data and metadata.
     * 
     * Note: The file corresponding to the fileId must be open prior to 
     * attempting the group create.
     */
    private proc prepareGroup(fileId: int, group: string) throws {
        var groupId:C_HDF5.hid_t = C_HDF5.H5Gcreate2(fileId, "/%s".format(group).c_str(),
              C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        C_HDF5.H5Gclose(groupId);
    }

    /**
     * Add internal metadata to HDF5 file
     * Group - /_arkouda_metadata (see ARKOUDA_HDF5_FILE_METADATA_GROUP)
     * Attrs - See constants for attribute names and associated values

     * :arg fileId: HDF5 H5Fopen identifer (hid_t)
     * :type int:
     *
     * This adds both the arkoudaVersion from ServerConfig as well as an internal file_version
     * In the future we may remove the file_version if the server version string proves sufficient.
     * Internal metadata related to the HDF5 API / capabilities / etc. can be added to this group.
     * Data specific metadata should be attached directly to the dataset / group itself.
     */
    proc addArkoudaHdf5VersioningMetadata(fileId:int) throws {
        // Note: can't write attributes to a closed group, easier to encapsulate here than call prepareGroup
        var metaGroupId:C_HDF5.hid_t = C_HDF5.H5Gcreate2(fileId,
                                                         ARKOUDA_HDF5_FILE_METADATA_GROUP.c_str(),
                                                         C_HDF5.H5P_DEFAULT,
                                                         C_HDF5.H5P_DEFAULT,
                                                         C_HDF5.H5P_DEFAULT);
        // Build the "file_version" attribute
        var attrSpaceId = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        var attrFileVersionType = getHDF5Type(ARKOUDA_HDF5_FILE_VERSION_TYPE);
        var attrId = C_HDF5.H5Acreate2(metaGroupId,
                          ARKOUDA_HDF5_FILE_VERSION_KEY.c_str(),
                          attrFileVersionType,
                          attrSpaceId,
                          C_HDF5.H5P_DEFAULT,
                          C_HDF5.H5P_DEFAULT);
        
        // H5Awrite requires a pointer and we have a const, so we need a variable ref we can turn into a pointer
        var fileVersion = ARKOUDA_HDF5_FILE_VERSION_VAL;
        C_HDF5.H5Awrite(attrId, attrFileVersionType, c_ptrTo(fileVersion));
        
        // release "file_version" HDF5 resources
        C_HDF5.H5Aclose(attrId);
        C_HDF5.H5Sclose(attrSpaceId);
        // getHDF5Type returns an immutable type so we don't / can't actually close this one.
        // C_HDF5.H5Tclose(attrFileVersionType);

        // Repeat for "ArkoudaVersion" which is a string
        // Need to allocate fixed size string type, docs say to copy from pre-defined type & modify
        // Chapel getHDF5Type only returns a variable length version for string/c_string
        var attrStringType = C_HDF5.H5Tcopy(C_HDF5.H5T_C_S1): C_HDF5.hid_t;
        C_HDF5.H5Tset_size(attrStringType, arkoudaVersion.size:uint(64) + 1); // ensure space for NULL terminator
        C_HDF5.H5Tset_strpad(attrStringType, C_HDF5.H5T_STR_NULLTERM);
        
        attrSpaceId = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        
        attrId = C_HDF5.H5Acreate2(metaGroupId,
                            ARKOUDA_HDF5_ARKOUDA_VERSION_KEY.c_str(),
                            attrStringType,
                            attrSpaceId,
                            C_HDF5.H5P_DEFAULT,
                            C_HDF5.H5P_DEFAULT);

        // For the value, we need to build a ptr to a char[]; c_string doesn't work because it is a const char*        
        var akVersion = c_calloc(c_char, arkoudaVersion.size);
        for (c, i) in zip(arkoudaVersion.codepoints(), 0..<arkoudaVersion.size) {
            akVersion[i] = c:c_char;
        }
        akVersion[arkoudaVersion.size] = 0:c_char; // ensure NULL termination

        C_HDF5.H5Awrite(attrId, attrStringType, akVersion);

        // release ArkoudaVersion HDF5 resources
        C_HDF5.H5Aclose(attrId);
        c_free(akVersion);
        C_HDF5.H5Sclose(attrSpaceId);
        C_HDF5.H5Tclose(attrStringType);

        // Release the group resource
        C_HDF5.H5Gclose(metaGroupId);
    }
    
    /*
     * Writes the values and segments lists to hdf5 within a group.
     */
    private proc writeStringsToHdf(fileId: int, idx: int, group: string, 
                              valuesList: list(uint(8)), segmentsList: list(int), writeNil:bool = false) throws {
        // initialize timer
        var t1: Time.Timer;
        if logLevel == LogLevel.DEBUG {
            t1 = new Time.Timer();
            t1.clear();
            t1.start();
        }

        if writeNil {
            H5LTmake_dataset_WAR(fileId, '/%s/values'.format(group).c_str(), 1,
                        c_ptrTo([valuesList.size:uint(64)]), getHDF5Type(uint(8)), nil);
            H5LTmake_dataset_WAR(fileId, '/%s/segments'.format(group).c_str(), 1,
                        c_ptrTo([segmentsList.size:uint(64)]),getHDF5Type(int), nil);
        } else {
            H5LTmake_dataset_WAR(fileId, '/%s/values'.format(group).c_str(), 1,
                        c_ptrTo([valuesList.size:uint(64)]), getHDF5Type(uint(8)),
                        c_ptrTo(valuesList.toArray()));
            if ( !segmentsList.isEmpty() ) {
                H5LTmake_dataset_WAR(fileId, '/%s/segments'.format(group).c_str(), 1,
                        c_ptrTo([segmentsList.size:uint(64)]),getHDF5Type(int),
                        c_ptrTo(segmentsList.toArray()));
            }
        }


        if logLevel == LogLevel.DEBUG {           
            t1.stop();  
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                  "Time for writing Strings to hdf5 file on locale %i: %.17r".format(
                       idx,t1.elapsed()));        
        }
    }
    
    /*
     * Returns a boolean indicating whether this is the last locale
     */
    private proc isLastLocale(idx: int) : bool {
        return idx == numLocales-1;
    }


}
