module GenSymIO {
    use HDF5;
    use IO;
    use CPtr;
    use Path;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use FileSystem;
    use Sort;
    use CommAggregation;
    use NumPyDType;
    use List;
    use Map;
    use PrivateDist;
    use Reflection;
    use Errors;
    use Logging;
    use Message;
    use ServerConfig;
    
    const gsLogger = new Logger();
  
    if v {
        gsLogger.level = LogLevel.DEBUG;
    } else {
        gsLogger.level = LogLevel.INFO;
    } 

    config const GenSymIO_DEBUG = false;
    config const SEGARRAY_OFFSET_NAME = "segments";
    config const SEGARRAY_VALUE_NAME = "values";
    config const NULL_STRINGS_VALUE = 0:uint(8);
    config const TRUNCATE: int = 0;
    config const APPEND: int = 1;

    /*
     * Creates a pdarray server-side and returns the SymTab name used to
     * retrieve the pdarray from the SymTab.
     */
    proc arrayMsg(cmd: string, payload: bytes, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string;
        var (dtypeBytes, sizeBytes, data) = payload.splitMsgToTuple(b" ", 3);
        var dtype = str2dtype(try! dtypeBytes.decode());
        var size = try! sizeBytes:int;
        var tmpf:file;
        overMemLimit(2*8*size);

        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "dtype: %t size: %i".format(dtype,size));

        // Write the data payload composing the pdarray to a memory buffer
        try {
            tmpf = openmem();
            var tmpw = tmpf.writer(kind=iobig);
            tmpw.write(data);
            try! tmpw.close();
        } catch {
            var errorMsg = "Could not write to memory buffer";
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);            
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // Get the next name from the SymTab cache
        var rname = st.nextName();

        /*
         * Read the data payload from the memory buffer, encapsulate
         * within a SymEntry, and write to the SymTab cache  
         */
        try {
            var tmpr = tmpf.reader(kind=iobig, start=0);
            if dtype == DType.Int64 {
                var entryInt = new shared SymEntry(size, int);
                tmpr.read(entryInt.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryInt);
            } else if dtype == DType.Float64 {
                var entryReal = new shared SymEntry(size, real);
                tmpr.read(entryReal.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryReal);
            } else if dtype == DType.Bool {
                var entryBool = new shared SymEntry(size, bool);
                tmpr.read(entryBool.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryBool);
            } else if dtype == DType.UInt8 {
                var entryUInt = new shared SymEntry(size, uint(8));
                tmpr.read(entryUInt.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryUInt);
            } else {
                tmpr.close();
                tmpf.close();

                var errorMsg = "Unhandled data type %s".format(dtypeBytes);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);            
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            tmpr.close();
            tmpf.close();
        } catch {
            var errorMsg = "Could not read from memory buffer into SymEntry";
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);            
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        /*
         * Return message indicating the SymTab name corresponding to the
         * newly-created pdarray
         */
        repMsg = "created " + st.attrib(rname);
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);         
        return new MsgTuple(repMsg, MsgType.NORMAL);
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
        var tmpf: file;
        try {
            tmpf = openmem();
            var tmpw = tmpf.writer(kind=iobig);
            if entry.dtype == DType.Int64 {
                tmpw.write(toSymEntry(entry, int).a);
            } else if entry.dtype == DType.Float64 {
                tmpw.write(toSymEntry(entry, real).a);
            } else if entry.dtype == DType.Bool {
                tmpw.write(toSymEntry(entry, bool).a);
            } else if entry.dtype == DType.UInt8 {
                tmpw.write(toSymEntry(entry, uint(8)).a);
            } else {
                var errorMsg = "Error: Unhandled dtype %s".format(entry.dtype);                
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);            
                return try! b"Error: Unhandled dtype %s".format(entry.dtype);
            }
            tmpw.close();
        } catch {
            try! tmpf.close();
            return b"Error: Unable to write SymEntry to memory buffer";
        }

        try {
            var tmpr = tmpf.reader(kind=iobig, start=0);
            tmpr.readbytes(arrayBytes);
            tmpr.close();
            tmpf.close();
        } catch {
            return b"Error: Unable to copy array from memory buffer to string";
        }
        //var repMsg = try! "Array: %i".format(arraystr.length) + arraystr;
        /*
         Engin: fwiw, if you want to achieve the above, you can:

         return b"Array: %i %|t".format(arrayBytes.length, arrayBytes);

         But I think the main problem is how to separate the length from the data
         */
       return arrayBytes;
    }

    /*
     * Converts the JSON array to a string pdarray
     */
    proc jsonToPdArray(json: string, size: int) throws {
        var f = opentmp();
        var w = f.writer();
        w.write(json);
        w.close();
        var r = f.reader(start=0);
        var array: [0..#size] string;
        r.readf("%jt", array);
        r.close();
        f.close();
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
     * Spawns a separate Chapel process that executes and returns the 
     * result of the h5ls command
     */
    proc lshdfMsg(cmd: string, payload: string,
                                st: borrowed SymTab): MsgTuple throws {
        // reqMsg: "lshdf [<json_filename>]"
        use Spawn;
        const tmpfile = "/tmp/arkouda.lshdf.output";
        var repMsg: string;
        var (jsonfile) = payload.splitMsgToTuple(1);

        var filename: string;
        try {
            filename = jsonToPdArray(jsonfile, 1)[0];
        } catch {
            var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(
                                     1, jsonfile);                                     
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);                                    
        }

        // Attempt to interpret filename as a glob expression and ls the first result
        var tmp = glob(filename);

        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                          "glob expanded filename: %s to size: %i files".format(filename, tmp.size));

        if tmp.size <= 0 {
            var errorMsg = "No files matching %s".format(filename);
            
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        filename = tmp[tmp.domain.first];
        var exitCode: int;
        var errMsg: string;

        try {
            if exists(tmpfile) {
                remove(tmpfile);
            }
            var cmd = try! "h5ls \"%s\" > \"%s\"".format(filename, tmpfile);
            var sub = spawnshell(cmd);

            sub.wait();

            exitCode = sub.exit_status;
            
            var f = open(tmpfile, iomode.r);
            var r = f.reader(start=0);
            r.readstring(repMsg);
            r.close();
            f.close();
            remove(tmpfile);
        } catch e : Error {
            var errorMsg = "failed to spawn process and read output %t".format(e);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if exitCode != 0 {
            var errorMsg = "error opening %s, check file permissions".format(filename);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        } else {
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
    }

    /* Read dataset from HDF5 files into arkouda symbol table. */
    proc readhdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string;
        // reqMsg = "readhdf <dsetName> <nfiles> [<json_filenames>]"
        var (dsetName, strictFlag, nfilesStr, jsonfiles) = payload.splitMsgToTuple(4);
        var strictTypes: bool = true;
        if (strictFlag.toLower() == "false") {
          strictTypes = false;
        }

        var nfiles = try! nfilesStr:int;
        var filelist: [0..#nfiles] string;

        try {
            filelist = jsonToPdArray(jsonfiles, nfiles);
        } catch {
            var errorMsg = "Error: could not decode json filenames via tempfile (%i files: %s)".format(
                                                                 nfiles, jsonfiles);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);                                                           
        }

        var filedom = filelist.domain;
        var filenames: [filedom] string;

        if filelist.size == 1 {
            var tmp = glob(filelist[0]);

            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                               "glob expanded %s to %i files".format(filelist[0], tmp.size));
            if tmp.size == 0 {
                var errorMsg = "Error: no files matching %s".format(filelist[0]);
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
        for (i, fname) in zip(filedom, filenames) {
            try {
                (segArrayFlags[i], dclasses[i], bytesizes[i], signFlags[i]) = get_dtype(fname, dsetName);
            } catch e: FileNotFoundError {
                var errorMsg = "File %s not found".format(fname);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            } catch e: PermissionError {
                var errorMsg = "Permission error opening %s: %s".format(fname,e.message());
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            } catch e: DatasetNotFoundError {
                var errorMsg = "Dataset %s not found in file %s".format(dsetName,fname);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            } catch e: NotHDF5FileError {
                var errorMsg = "The file %s is not an HDF5 file: %s".format(fname,e.message());
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            } catch e: HDF5FileFormatError {
                var errorMsg = "HDF5 format error %s for file %s".format(e.message(),fname);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);           
            } catch e: SegArrayError {
                var errorMsg = "SegmentedArray error: %s".format(e.message());
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            } catch e: Error {
                var errorMsg = "Other error %s".format(e.message());
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        const isSegArray = segArrayFlags[filedom.first];
        const dataclass = dclasses[filedom.first];
        const bytesize = bytesizes[filedom.first];
        const isSigned = signFlags[filedom.first];
        for (name, sa, dc, bs, sf) in zip(filenames, segArrayFlags, dclasses, bytesizes, signFlags) {
            if ((sa != isSegArray) || (dc != dataclass)) {
                var errorMsg = "inconsistent dtype in dataset %s of file %s".format(dsetName, name);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);              
            } else if (strictTypes && ((bs != bytesize) || (sf != isSigned))) {
                var errorMsg = "inconsistent precision or sign in dataset %s of file %s\nWith strictTypes, mixing of precision and signedness not allowed (set strictTypes=False to suppress)".format(dsetName, name);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);                            
            }
        }
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "Verified all dtypes across files");

        var subdoms: [filedom] domain(1);
        var segSubdoms: [filedom] domain(1);
        var len: int;
        var nSeg: int;
        try {
            if isSegArray {
                (segSubdoms, nSeg) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                (subdoms, len) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);
            } else {
                (subdoms, len) = get_subdoms(filenames, dsetName);
            }
        } catch e: HDF5RankError {
            var errorMsg = notImplementedError("readhdf", try! "Rank %i arrays".format(e.rank));
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        } catch e: Error {
            var errorMsg = "Other error: %s".format(e.message());
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR); 
        }
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "Got subdomains and total length");

        select (isSegArray, dataclass) {
            when (true, C_HDF5.H5T_INTEGER) {
                if (bytesize != 1) || isSigned {
                    var errorMsg = "Detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".
                                            format(isSegArray, dataclass, bytesize, isSigned);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);                                             
                }

                var entrySeg = new shared SymEntry(nSeg, int);
                read_files_into_distributed_array(entrySeg.a, segSubdoms, filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                fixupSegBoundaries(entrySeg.a, segSubdoms, subdoms);
                var entryVal = new shared SymEntry(len, uint(8));
                read_files_into_distributed_array(entryVal.a, subdoms, filenames, 
                                                         dsetName + "/" + SEGARRAY_VALUE_NAME);

                var segName = st.nextName();
                st.addEntry(segName, entrySeg);
                var valName = st.nextName();
                st.addEntry(valName, entryVal);
                
                var repMsg = "created " + st.attrib(segName) + " +created " + st.attrib(valName);
                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when (false, C_HDF5.H5T_INTEGER) {
                var entryInt = new shared SymEntry(len, int);
                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "Initialized int entry");                
                read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
                var rname = st.nextName();
                st.addEntry(rname, entryInt);

                var repMsg = "created " + st.attrib(rname);
                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when (false, C_HDF5.H5T_FLOAT) {
                var entryReal = new shared SymEntry(len, real);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                             "Initialized float entry");
                read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
                var rname = st.nextName();
                st.addEntry(rname, entryReal);

                var repMsg = "created " + st.attrib(rname);
                gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            otherwise {
                var errorMsg = "Detected unhandled datatype: segmented? " +
                               "%t, class %i, size %i, signed? %t".format(isSegArray, 
                               dataclass, bytesize, isSigned);
                gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    /* 
     * Reads all datasets from 1..n HDF5 files into an Arkouda symbol table. 
     */
    proc readAllHdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        // reqMsg = "readAllHdf <ndsets> <nfiles> [<json_dsetname>] | [<json_filenames>]"
        var repMsg: string;
        // May need a more robust delimiter then " | "
        var (strictFlag, ndsetsStr, nfilesStr, arraysStr) = payload.splitMsgToTuple(4);
        var strictTypes: bool = true;
        if (strictFlag.toLower() == "false") {
          strictTypes = false;
        }
        var (jsondsets, jsonfiles) = arraysStr.splitMsgToTuple(" | ",2);
        var ndsets = try! ndsetsStr:int;
        var nfiles = try! nfilesStr:int;
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

        var dsetdom = dsetlist.domain;
        var filedom = filelist.domain;
        var dsetnames: [dsetdom] string;
        var filenames: [filedom] string;
        dsetnames = dsetlist;

        if filelist.size == 1 {
            var tmp = glob(filelist[0]);
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "glob expanded %s to %i files".format(filelist[0], tmp.size));
            if tmp.size == 0 {
                var errorMsg = "No files matching %s".format(filelist[0]);
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
        var rnames: string;
        for dsetName in dsetnames do {
            for (i, fname) in zip(filedom, filenames) {
                try {
                    (segArrayFlags[i], dclasses[i], bytesizes[i], signFlags[i]) = get_dtype(fname, dsetName);
                } catch e: FileNotFoundError {
                    var errorMsg = "File %s not found".format(fname);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                } catch e: PermissionError {
                    var errorMsg = "Permission error %s opening %s".format(e.message(),fname);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                } catch e: DatasetNotFoundError {
                    var errorMsg = "Dataset %s not found in file %s".format(dsetName,fname);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                } catch e: NotHDF5FileError {
                    var errorMsg = "The file %s is not an HDF5 file: %s".format(fname,e.message());
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                } catch e: SegArrayError {
                    var errorMsg = "SegmentedArray error: %s".format(e.message());
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                } catch e : Error {
                    var errorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
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
            var len: int;
            var nSeg: int;
            try {
                if isSegArray {
                    (segSubdoms, nSeg) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                    (subdoms, len) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);
                } else {
                    (subdoms, len) = get_subdoms(filenames, dsetName);
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
                    var entrySeg = new shared SymEntry(nSeg, int);
                    read_files_into_distributed_array(entrySeg.a, segSubdoms, filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                    fixupSegBoundaries(entrySeg.a, segSubdoms, subdoms);
                    var entryVal = new shared SymEntry(len, uint(8));
                    read_files_into_distributed_array(entryVal.a, subdoms, filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);
                    var segName = st.nextName();
                    st.addEntry(segName, entrySeg);
                    var valName = st.nextName();
                    st.addEntry(valName, entryVal);
                    rnames = rnames + "created " + st.attrib(segName) + " +created " + st.attrib(valName) + " , ";
                }
                when (false, C_HDF5.H5T_INTEGER) {
                    var entryInt = new shared SymEntry(len, int);
                    gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "Initialized int entry for dataset %s".format(dsetName));

                    read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
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
                    rnames = rnames + "created " + st.attrib(rname) + " , ";
                }
                when (false, C_HDF5.H5T_FLOAT) {
                    var entryReal = new shared SymEntry(len, real);
                    gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                      "Initialized float entry");
                    read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
                    var rname = st.nextName();
                    st.addEntry(rname, entryReal);
                    rnames = rnames + "created " + st.attrib(rname) + " , ";
                }
                otherwise {
                    var errorMsg = "detected unhandled datatype: segmented? %t, class %i, size %i, " +
                                   "signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
                    gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
        }

        repMsg = rnames.strip(" , ", leading = false, trailing = true);
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    proc fixupSegBoundaries(a: [?D] int, segSubdoms: [?fD] domain(1), valSubdoms: [fD] domain(1)) {
        var boundaries: [fD] int; // First index of each region that needs to be raised
        var diffs: [fD] int;// Amount each region must be raised over previous region
        forall (i, sd, vd, b) in zip(fD, segSubdoms, valSubdoms, boundaries) {
            b = sd.low; // Boundary is index of first segment in file
            // Height increase of next region is number of bytes in current region
            if (i < fD.high) {
                diffs[i+1] = vd.size;
            }
        }
        // Insert height increases at region boundaries
        var sparseDiffs: [D] int;
        forall (b, d) in zip(boundaries, diffs) with (var agg = newDstAggregator(int)) {
            agg.copy(sparseDiffs[b], d);
        }
        // Make plateaus from peaks
        var corrections = + scan sparseDiffs;
        // Raise the segment offsets by the plateaus
        a += corrections;
    }

    /* 
     * Retrieves the datatype of the dataset read from HDF5 
     */
    proc get_dtype(filename: string, dsetName: string) throws {
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
            if isStringsDataset(file_id, dsetName) {
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
            defer {
                C_HDF5.H5Fclose(file_id);
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
                           check file permissions".format(filename);
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
        var boolDataset: bool;

        try {
            boolDataset = isBooleanDataset(fileId, dsetName);
            /*
             * Put close function call in defer block to ensure it's invoked, 
             * which prevents accumulation of open file descriptors
             */
            defer {
                C_HDF5.H5Fclose(fileId);
            }
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
        for (i, filename) in zip(FD, filenames) {
            try {
                var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                           C_HDF5.H5P_DEFAULT);
                var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now
                var dName = try! getReadDsetName(file_id, dsetName);

                // Read array length into dims[0]
                C_HDF5.HDF5_WAR.H5LTget_dataset_info_WAR(file_id, dName.c_str(), 
                                           c_ptrTo(dims), nil, nil);
                defer {
                    /*
                     * Put close function call in defer block to ensure it's invoked, 
                     * which prevents accumulation of open file descriptors
                     */
                    C_HDF5.H5Fclose(file_id);
                }
                lengths[i] = dims[0]: int;
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
        return (subdoms, (+ reduce lengths));
    }

    /* This function gets called when A is a BlockDist or DefaultRectangular array. */
    proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), 
                                                 filenames: [FD] string, dsetName: string)
        where (MyDmap == Dmap.blockDist || MyDmap == Dmap.defaultRectangular) {
            try! gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "entry.a.targetLocales() = %t".format(A.targetLocales()));
            try! gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "Filedomains: %t".format(filedomains));

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

                            try! gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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
        var (arrayName, dsetName, modeStr, jsonfile, dataType)
            = payload.splitMsgToTuple(5);

        var mode = try! modeStr: int;
        var filename: string;
        var entry = st.lookup(arrayName);

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
                    var e = toSymEntry(entry, uint(8));
                    warnFlag = write1DDistStrings(filename, mode, dsetName, e.a, DType.UInt8);
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
                                                                array_type: DType) throws {
        var prefix: string;
        var extension: string;  
        var warnFlag: bool;      
        
        (prefix,extension) = getFileMetadata(filename);
 
        // Generate the filenames based upon the number of targetLocales.
        var filenames = generateFilenames(prefix, extension, A);
        
        // Generate a list of matching filenames to test against. 
        var matchingFilenames = getMatchingFilenames(prefix, extension);
        
        // Create files with groups needed to persist values and segments pdarrays
        var group = getGroup(dsetName);
        warnFlag = processFilenames(filenames, matchingFilenames, mode, A, group);
        
        /*
         * The leadingSliceIndices object, which is a globally-scoped PrivateSpace 
         * array, contains the leading slice index for each locale, which is used 
         * to remove the uint(8) characters moved to the previous locale; this
         * situation occurs when a string spans two locales.
         *
         * The trailingSliceIndices PrivateSpace is used in the special case 
         * where the majority of a large string spanning two locales is the sole
         * string on a locale; in this case, the trailing slice index is used
         * to move the smaller string chunk to the locale containing the large
         * string chunk that is the sole string chunk on a locale.
         *
         * The isSingleString PrivateSpace indicates whether each locale contains
         * chars corresponding to one string/string segment, which occurs if 
         * (1) the char array contains no null uint(8) characters or (2) there is
         * only one null uint(8) char at the end of the string/string segment
         *
         * The endsWithCompleteString PrivateSpace indicates whether the values
         * array for each locale ends with complete string, meaning that the last
         * character in the local slice is a null uint(8) char.
         */
        var leadingSliceIndices: [PrivateSpace] int;    
        var trailingSliceIndices: [PrivateSpace] int;
        var isSingleString: [PrivateSpace] bool;
        var endsWithCompleteString: [PrivateSpace] bool;

        /*
         * Loop through all locales and set (1) leadingSliceIndices, which are
         * used to remove leading uint(8) characters from the local slice that
         * complete a string started in the previous locale (2) trailingSliceIndices,
         * which are used to removing trailing uint(8) characters used to start
         * strings that are completed in the next locale locale (3) isSingleString,
         * which indicates if a locale contains a Strings values array that
         * corresponds to only one complete string or string segment and (4)
         * endsWithCompleteString, which indicates if the local slice array has 
         * a complete string at the end of the array.
         */
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) 
                      with (ref leadingSliceIndices, ref trailingSliceIndices, 
                            ref isSingleString, ref endsWithCompleteString) do on loc {
            generateValuesMetadata(idx,leadingSliceIndices, trailingSliceIndices, 
                                        isSingleString, endsWithCompleteString, A);
        }
                                                       
        /*
         * Iterate through each locale and (1) open the hdf5 file corresponding to the
         * locale (2) prepare values and segments lists to be written (3) write each
         * list as a Chapel array to the open hdf5 file and (4) close the hdf5 file
         */
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with 
                        (ref leadingSliceIndices, ref trailingSliceIndices) do on loc {
            const myFilename = filenames[idx];
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                   "%s exists? %t".format(myFilename, exists(myFilename)));

            var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), 
                                       C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
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

            /*
             * Check for the possibility that 1..n strings in the values array span 
             * two neighboring locales; by seeing if the final character in the local 
             * slice is the null uint(8) character. If it is not, then the last string 
             * is only a partial string and the remainder of the string is in the 
             * next locale
             */
            if A.localSlice(locDom).back() != NULL_STRINGS_VALUE {
                /*
                 * Since the last value of the local slice is other than the uint(8) null
                 * character, this means the last string in the current locale (idx) spans 
                 * the current AND next locale. Consequently, need to do the following:
                 * 1. Add all current locale values to a list
                 * 2. Obtain remaining uint(8) values from the next locale
                 */
                var charList : list(uint(8));
                var segmentsList : list(int);

                /* 
                 * Generate the values and segments lists from the locale domain; these
                 * lists will be used to generate post-shuffle values and segments arrays
                 * to be written to hdf5
                 */
                (charList, segmentsList) = sliceToValuesAndSegments(A.localSlice(locDom));

                /*
                 * If this locale contains a single string/string segment (and therefore no 
                 * leading slice or trailing slice), and is not the first locale, retrieve 
                 * the trailing chars from the previous locale, if applicable, to set the
                 * correct starting chars for the lone string/string segment on this locale.
                 * 
                 * Note: if this is the first locale, there are no trailing chars from a
                 * previous locale, so this code block is not executed in this case.
                 */
                if isSingleString[idx] && idx > 0 {
                    var trailingIndex = trailingSliceIndices[idx-1];

                    if trailingIndex > -1 {
                        /*
                         * There are 1..n chars to be shuffled from the previous locale
                         * (idx-1) to complete the beginning of the one string assigned 
                         * to the current locale (idx)
                         */
                        var trailingValuesList : list(uint(8));
                        on Locales[idx-1] {
                            const locDom = A.localSubdomain();

                            for (value, i) in zip(A.localSlice(locDom),
                                                        0..A.localSlice(locDom).size-1) {
                                if i >= trailingIndex {
                                    trailingValuesList.append(value:uint(8));
                                }
                            }
                        }
                        charList.insert(0, trailingValuesList);
                    }
                }

                /*
                 * Now that the start of the first string of the current locale (idx) is correct,
                 * shuffle chars to place a complete string at the end the current locale. 
                 * There are two possible scenarios to account for. First, the next locale 
                 * has a leadingSliceIndex > -1. If so, the chars up to the leadingSliceIndex 
                 * will be shuffled from the next locale (idx+1) to complete the last string 
                 * in the current locale (idx). In the second scenario, the next locale is 
                 * the last locale in the in the Arkouda cluster If so, all of the chars 
                 * from the next locale (idx+1) are shuffled to the current locale (idx).
                 */
                if leadingSliceIndices[idx+1] > -1 || isLastLocale(idx+1) {
                    on Locales[idx+1] {
                        const locDom = A.localSubdomain();
                        var sliceList: list(uint(8), parSafe=false);

                        /*
                         * Iterate through the local slice values for the next locale and add
                         * each to the sliceList until the null uint(8) character is reached;
                         * this subset of the next locale (idx+1) chars corresponds to the
                         * chars that complete the last string of the current locale (idx)
                         */
                        for value in A.localSlice(locDom) {
                            if value != NULL_STRINGS_VALUE {
                                sliceList.append(value:uint(8));
                            } else {
                                break;
                            }
                        }                       
                        charList.extend(sliceList);   
                    }
                }

                /* 
                 * To prepare for writing revised values array to hdf5, do the following, 
                 * if applicable:
                 * 1. Remove the leading slice characters shuffled to previous locale
                 * 2. Remove the trailing slice characters shuffled to the next locale
                 * 3. If (2) does not apply, add null uint(8) char to end of the valuesList
                 */
                var leadingSliceIndex = leadingSliceIndices[idx]:int;
                var trailingSliceIndex = trailingSliceIndices[idx]:int;

                var valuesList: list(uint(8), parSafe=false);

                /*
                 * Verify if the current locale (idx) contains chars shuffled to the previous 
                 * locale (idx-1) by checking the leadingSliceIndex, the number of strings in 
                 * the current locale, and whether the preceding locale ends with a complete
                 * string. If (1) the leadingSliceIndex > -1, (2) this locale contains 2..n 
                 * strings, and (3) the previous locale does not end with a complete string
                 * this means that the charList contains chars that were shuffled to complete
                 * the last string from the previous locale (idx-1) . If so, generate
                 * a new valuesList that has those values sliced out. Otherwise, set the
                 * valuesList reference to the charList
                 */
                 if leadingSliceIndex > -1 && !isSingleString[idx] 
                                                       && !endsWithCompleteString[idx-1] {
                     /*
                      * Since the leading slice was used to complete the last string in
                      * the previous locale (idx-1), slice those chars from the charList
                      */
                     (valuesList, segmentsList) = 
                                         adjustForLeadingSlice(leadingSliceIndex, charList);
                 } else {
                     valuesList = charList;
                 }

                 /*
                  * Verify if the current locale contains chars shuffled to the next locale 
                  * (idx+1) because the next locale only has one string/string segment and
                  * the current locale's trailingSliceIndex > -1. If so, remove the
                  * chars starting with the trailingSliceIndex, which will place the null 
                  * uint(8) char is at the end of the valuesList. Otherwise, manually 
                  * add the null uint(8) char to the end of the valuesList.
                  */
                 if trailingSliceIndex > -1 && isSingleString[idx+1] {
                     var sliceIndex = segmentsList.last();
                     (valuesList, segmentsList) = 
                                          adjustForTrailingSlice(sliceIndex, valuesList);                
                 } else {
                     valuesList.append(NULL_STRINGS_VALUE);            
                 }
                                  
                 // Write the finalized valuesList and segmentsList to the hdf5 group
                 writeStringsToHdf(myFileID, group, valuesList, segmentsList);
             } else {
                 /*
                  * The current local slice (idx) ends with the uint(8) null character,  
                  * which is the value required to ensure correct read logic; with this
                  * confirmed, check to see if the current locale (idx) slice contains
                  * 1..n chars that compose a string from the previous (idx-1) locale.
                  */
                 var leadingSliceIndex = leadingSliceIndices[idx]:int;

                 if leadingSliceIndex == -1 {
                     /*
                      * Since the leadingSliceIndex is -1, the current local slice (idx) does 
                      * not contain chars from the previous locale (idx-1).
                      */
                     var valuesList : list(uint(8));
                     var segmentsList : list(int);

                     (valuesList, segmentsList) = sliceToValuesAndSegments(A.localSlice(locDom));

                     /*
                      * If (1) this locale (idx) contains one string/string segment and (2) ends 
                      * with the null uint(8) char, check to see if the trailingSliceIndex from 
                      * the previous locale (idx-1) is > -1. If so, the chars following the 
                      * trailingSliceIndex from the previous locale (idx-1) complete the one 
                      * string/string segment within the current locale (idx). 
                      */
                     if isSingleString[idx] && idx > 0 {
                         var trailingIndex = trailingSliceIndices[idx-1];

                         if trailingIndex > -1 {
                             var trailingValuesList : list(uint(8));
                             on Locales[idx-1] {
                                 const locDom = A.localSubdomain();  
                                 for (value, i) in zip(A.localSlice(locDom), 
                                                               0..A.localSlice(locDom).size-1) {
                                     if i >= trailingIndex {
                                         trailingValuesList.append(value:uint(8));
                                     }
                                 } 
                             }
                             //prepend the current locale valuesList with the trailingValuesList
                             valuesList.insert(0, trailingValuesList);
                         }
                     }
                      
                     /*
                      * Account for the special case where the following is true about the
                      * current locale (idx):
                      *
                      * 1. This is the last locale
                      * 2. There is one partial string started in the previous locale
                      * 3. The previous locale has no trailing slice to complete the partial
                      *    string in the current locale
                      *
                      * In this very special case, (1) move the current locale (idx) chars to 
                      * the previous locale (idx-1) and (2) clear out the current locale
                      * segments list because the current locale values list is now empty.
                      */                     
                     if isLastLocale(idx) {
                         if !endsWithCompleteString[idx-1] && isSingleString[idx] 
                                                        && trailingSliceIndices[idx-1] == -1 {
                             valuesList.clear();
                             segmentsList.clear();
                         }
                      }

                      // Write the finalized valuesList and segmentsList to the hdf5 group
                      writeStringsToHdf(myFileID, group, valuesList, segmentsList);
                  } else {
                      /*
                       * The local slice (idx) does possibly contain chars from previous locale 
                       * (idx-1). Accordingly, generate a corresponding valuesList that can be 
                       * sliced and check to see if the previous locale ends with a complete
                       * string. If not, (1) adjust the valuesList by slicing the chars out that
                       * correspond to chars shuffled to the previous locale (idx-1) and 
                       * (2) generate a new, corresponding Strings segments list. 
                       */  
                      var charList : list(uint(8));
                      var segmentsList : list(int);
                      var valuesList : list(uint(8));
                     
                      (charList, segmentsList) = sliceToValuesAndSegments(A.localSlice(locDom)); 
          
                      /*
                       * Check to see if previous locale (idx-1) ends with a complete string.
                       * If not, then the leading slice of this string was used to complete
                       * the last string in the previous locale, so slice those chars.
                       */
                      if !endsWithCompleteString(idx-1) {
                          (valuesList, segmentsList) = 
                                             adjustForLeadingSlice(leadingSliceIndex, charList);
                      } else {
                          valuesList = charList;
                      }

                      // Write the finalized valuesList and segmentsList to the hdf5 group
                      writeStringsToHdf(myFileID, group, valuesList, segmentsList);
                    }
                }
            
            // Close the file now that the values and segments pdarrays have been written
            C_HDF5.H5Fclose(myFileID);
        }
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
            H5LTmake_dataset_WAR(myFileID, myDsetName.c_str(), 1, c_ptrTo(dims),
                                      dType, c_ptrTo(A.localSlice(locDom)));

            // Close the file now that the 1..n pdarrays have been written
            C_HDF5.H5Fclose(myFileID);
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
              
              prepareGroup(file_id, group);

              if file_id < 0 { // Negative file_id means error
                  throw getErrorWithContext(
                                    msg="The file %s does not exist".format(filenames[loc.id]),
                                    lineNumber=getLineNumber(), 
                                    routineName=getRoutineName(), 
                                    moduleName=getModuleName(), 
                                    errorClass='FileNotFoundError');
              }

              /*
               * Close the file now that it has been created and, if applicable, the 
               * Strings group derived from the dsetName has been created.
               */
              C_HDF5.H5Fclose(file_id);
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
     * If APPEND mode, checks to see if the matchingFilenams matches the filenames
     * array and, if not, raises a MismatchedAppendError. If in TRUNCATE mode, creates
     * the files matching the filenames. If 1..n of the filenames exist, returns 
     * warning to the user that 1..n files were overwritten.
     */
    proc processFilenames(filenames: [] string, matchingFilenames: [] string, mode: int, A) throws {
      // if appending, make sure number of files hasn't changed and all are present
      var warnFlag: bool;
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

              if file_id < 0 { // Negative file_id means error
                  throw getErrorWithContext(
                                     msg="The file %s does not exist".format(filenames[loc.id]),
                                     lineNumber=getLineNumber(), 
                                     routineName=getRoutineName(), 
                                     moduleName=getModuleName(), 
                                     errorClass='FileNotFoundError');
              }

              /*
               * Close the file now that it has been created and, if applicable, the 
               * Strings group derived from the dsetName has been created.
               */
              C_HDF5.H5Fclose(file_id);
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
     * Generates an array of filenames to be matched in APPEND mode and to be
     * checked in TRUNCATE mode that will warn the user that 1..n files are
     * being overwritten.
     */
    proc getMatchingFilenames(prefix : string, extension : string) throws {
        return glob(try! "%s_LOCALE*%s".format(prefix, extension));    
    }

    /*
     * Generates values array metadata required to partition the corresponding strings
     * across 1..n locales via shuffle operations. The metadata includes leading and
     * trailing slice indices as well as flags indicating whether the values array
     * contains one string and if the values array ends with a complete string.
     */
    private proc generateValuesMetadata(idx : int, leadingSliceIndices, 
                       trailingSliceIndices, isSingleString, endsWithCompleteString, A) {
        /*
         * Generate the leadlingSliceIndex, which is used to (1) indicate the chars to be
         * pulled down to the previous locale to complete the last string there and (2)
         * filter out the chars from the current locale that were used to complete the 
         * string in the previous locale, along with the null uint(8) char. 
         *
         * Next, generate the trailingSliceIndex, which is used to (1) indicate the chars
         * to be pulled up to the next locale to complete the first string in that 
         * locale and (2) filter out the chars from the current locale used to complete
         * the first string in the next locale. 
         
         * Finally, set the endsWithCompleteString value, which indicates if the values
         * array has the null uint(8) as the final character, which means the final 
         * string in the locale is complete (does not span to the next locale).
         */
        on Locales[idx] {
            const locDom = A.localSubdomain();
            var leadingSliceSet = false;

            //Initialize both indices to -1 to indicate neither exists for locale
            leadingSliceIndices[idx] = -1;
            trailingSliceIndices[idx] = -1;

            for (value, i) in zip(A.localSlice(locDom), 0..A.localSlice(locDom).size-1) {
            
                /*
                 * Check all chars leading up to the last char in the values array. If a
                 * char is the null uint(8) char and is not the last char, this is a segment
                 * index that could be a leadingSliceIndex or a trailingSliceIndex.
                 */
                if i < A.localSlice(locDom).size-1 {
                    if value == NULL_STRINGS_VALUE {
                        /*
                         * The first null char of the values array is the leadingSliceIndex,
                         * which wil be used to pull chars from current locale to complete
                         * the string started in the previous locale, if applicable.
                         */
                        if !leadingSliceSet {
                            leadingSliceIndices[idx] = i + 1;
                            leadingSliceSet = true; 
                        } else {
                            /*
                             * If the leadingSliceIndex has already been set, the next null
                             * char is a candidate to be the trailingSliceIndex.
                             */
                             trailingSliceIndices[idx] = i + 1;
                        }
                    }
                } else {
                    /*
                     * Since this is the last character within the array, check to see if it 
                     * is a null char. If it is, that means that the last string in this values 
                     * array does not span to the next locale. Consequently, (1) no chars from
                     * the next locale will be used to complete the last string in this locale
                     * and (2) no chars from this locale will be sliced to complete the first
                     * string in the next locale.
                     */
                     if value == NULL_STRINGS_VALUE {
                        endsWithCompleteString[idx] = true;
                     } else {
                        endsWithCompleteString[idx] = false;
                     }
                }
            }    
        
            if leadingSliceIndices[idx] > -1 || trailingSliceIndices[idx] > -1 {    
                /*
                 * If either of the indices are > -1, this means there's 2..n null characters
                 * in the string corresponding to the values array, which means the values
                 * array contains 2..n string segments/complete strings.
                 */   
                isSingleString[idx] = false;
            } else {
                /*
                 * Since there is neither a leadingSliceIndex nor a trailingSliceIndex for 
                 * this locale, it only contains a single complete string or string segment.
                 */
                isSingleString[idx] = true;
            }

            /* 
             * For the special case of this being the first locale, set the leadingSliceIndex 
             * to -1 since there is no previous locale that has an incomplete string at the
             * end that will require chars sliced from locale 0 to complete. If there is one
             * null uint(8) char that is not at the end of the values array, this is the 
             * trailingSliceIndex for the first locale.
             */
            if idx == 0 {
                if leadingSliceIndices[idx] > -1 {
                    trailingSliceIndices[idx] = leadingSliceIndices[idx];
                }
                leadingSliceIndices[idx] = -1;
            }
            
            /*
             * For the special case of this being the last locale, set the trailingSliceIndex 
             * to -1 since there is no next locale to shuffle trailing slice to.
             */
            if isLastLocale(idx) {
                trailingSliceIndices[idx] = -1;
            }
        }
    }

    /*
     * Processes a local Strings slice into (1) a uint(8) values list for use in methods 
     * that finalize the values array elements following any shuffle operations and (2)
     * a segments list representing starting indices for each string in the values list.
     */
    private proc sliceToValuesAndSegments(rawChars) {
        var charList: list(uint(8), parSafe=false);
        var indices: list(int, parSafe=false);

        //initialize segments with index to first char in values
        indices.append(0);
        
        for (value, i) in zip(rawChars, 0..rawChars.size-1) do {
            charList.append(value:uint(8));
            /*
             * If the char is the null uint(8) char, check to see if it is the 
             * last char. If not, added to the indices. If it is the last char,  
             * don't add, because it is the correct ending char for a Strings 
             * values array to be written to a locale.
             */ 
            if value == NULL_STRINGS_VALUE && i < rawChars.size-1 {
                indices.append(i+1);
            }
        }

        return (charList, indices);
    }

    /*
     * Adjusts for the shuffling of a leading char sequence to the 
     * previous locale by (1) slicing leading chars that compose
     * a string started in the previous locale and returning (1) 
     * a new values list that composes all of the strings that start
     * in the current locale and (2) a new segments list that 
     * corresponds to the new values list
     */
    private proc adjustForLeadingSlice(sliceIndex : int,
                                   charList : list(uint(8))) {
        var valuesList: list(uint(8), parSafe=false);
        var indices: list(int);
        var i: int = 0;
        indices.append(0);
        
        var segmentsBound = charList.size - sliceIndex - 1;

        for value in charList(sliceIndex..charList.size-1)  {
            valuesList.append(value:uint(8));
            
            /*
             * If the value is a null char and is not the last char
             * in the list, then it is a segment use to delimit
             * strings within the corresponding values array.
             */
            if value == NULL_STRINGS_VALUE && i < segmentsBound {
                indices.append(i+1);
            }
            i+=1;
        }

        return (valuesList,indices);
    }

    /* 
     * Adjusts for the shuffling of a trailing char sequence to the next 
     * locale by (1) slicing trailing chars that correspond to 1..n
     * chars composing a string that completes in the next locale,
     * returning (1) a new list that composes all strings that end in the 
     * current locale and (2) returns a new segments list corresponding
     * to the new values list for the current locale
     */
    private proc adjustForTrailingSlice(sliceIndex : int,
                                   charList : list(uint(8))) {
        var valuesList: list(uint(8), parSafe=false);
        var indices: list(int);
        var i: int = 0;
        indices.append(0);

        for value in charList(0..sliceIndex-1)  {
            valuesList.append(value:uint(8));
            if value == NULL_STRINGS_VALUE && i < sliceIndex-1 {
                indices.append(i+1);
            }
            i+=1;
        }

        return (valuesList,indices);
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
        var groupId = C_HDF5.H5Gcreate2(fileId, "/%s".format(group).c_str(),
              C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        C_HDF5.H5Gclose(groupId);
    }
    
    /*
     * Writes the values and segments lists to hdf5 within a group.
     */
    private proc writeStringsToHdf(fileId: int, group: string, 
                              valuesList: list(uint(8)), segmentsList: list(int)) throws {
        H5LTmake_dataset_WAR(fileId, '/%s/values'.format(group).c_str(), 1,
                     c_ptrTo([valuesList.size:uint(64)]), getHDF5Type(uint(8)),
                            c_ptrTo(valuesList.toArray()));

        H5LTmake_dataset_WAR(fileId, '/%s/segments'.format(group).c_str(), 1,
                     c_ptrTo([segmentsList.size:uint(64)]),getHDF5Type(int),
                           c_ptrTo(segmentsList.toArray()));
    }
    
    /*
     * Returns a boolean indicating whether this is the last locale
     */
    private proc isLastLocale(idx: int) : bool {
        return idx == numLocales-1;
    }
}
