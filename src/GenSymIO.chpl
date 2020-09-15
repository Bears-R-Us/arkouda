module GenSymIO {
    use HDF5;
    use IO;
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
    proc arrayMsg(cmd: string, payload: bytes, st: borrowed SymTab): string {
        var repMsg: string;
        var (dtypeBytes, sizeBytes, data) = payload.splitMsgToTuple(3);
        var dtype = str2dtype(try! dtypeBytes.decode());
        var size = try! sizeBytes:int;
        var tmpf:file;

        // Write the data payload composing the pdarray to a memory buffer
        try {
            tmpf = openmem();
            var tmpw = tmpf.writer(kind=iobig);
            tmpw.write(data);
            try! tmpw.close();
        } catch {
            return "Error: Could not write to memory buffer";
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
                return try! "Error: Unhandled data type %s".format(dtypeBytes);
            }
            tmpr.close();
            tmpf.close();
        } catch {
            return "Error: Could not read from memory buffer into SymEntry";
        }
        /*
         * Return message indicating the SymTab name corresponding to the
         * newly-created pdarray
         */
        return try! "created " + st.attrib(rname);
    }

    /*
     * Outputs the pdarray as a Numpy ndarray in the form of a 
     * Chapel Bytes object
     */
    proc tondarrayMsg(cmd: string, payload: bytes, st: 
                                          borrowed SymTab): bytes throws {
        var arrayBytes: bytes;
        var entryStr = payload.decode();
        var entry = st.lookup(entryStr);
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

    class DatasetNotFoundError: Error {proc init() {}}
    class NotHDF5FileError: Error {proc init() {}}
    class MismatchedAppendError: Error {proc init() {}}
    class WriteModeError: Error { proc init() {} }
    class SegArrayError: Error {proc init() {}}

    /*
     * Converts the JSON array to a pdarray
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
     * Spawns a separate Chapel process that executes and returns the 
     * result of the h5ls command
     */
    proc lshdfMsg(cmd: string, payload: bytes,
                                st: borrowed SymTab): string throws {
        // reqMsg: "lshdf [<json_filename>]"
        use Spawn;
        const tmpfile = "/tmp/arkouda.lshdf.output";
        var repMsg: string;
        var (jsonfile) = payload.decode().splitMsgToTuple(1);

        var filename: string;
        try {
            filename = jsonToPdArray(jsonfile, 1)[0];
        } catch {
            return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(1, jsonfile);
        }

        // Attempt to interpret filename as a glob expression and ls the first result
        var tmp = glob(filename);
        if GenSymIO_DEBUG {
            writeln(try! "glob expanded %s to %i files".format(filename, tmp.size));
        }
        if tmp.size <= 0 {
            return try! "Error: no files matching %s".format(filename);
        }
        filename = tmp[tmp.domain.first];
        var exitCode: int;
        try {
            if exists(tmpfile) {
                remove(tmpfile);
            }
            var cmd = try! "h5ls \"%s\" > \"%s\"".format(filename, tmpfile);
            var sub = spawnshell(cmd);
            // sub.stdout.readstring(repMsg);
            sub.wait();
            exitCode = sub.exit_status;
            var f = open(tmpfile, iomode.r);
            var r = f.reader(start=0);
            r.readstring(repMsg);
            r.close();
            f.close();
            remove(tmpfile);
        } catch {
            return "Error: failed to spawn process and read output";
        }

        if exitCode != 0 {
            return try! "Error: %s".format(repMsg);
        } else {
            return repMsg;
        }
    }

    /* Read dataset from HDF5 files into arkouda symbol table. */
    proc readhdfMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        var repMsg: string;
        // reqMsg = "readhdf <dsetName> <nfiles> [<json_filenames>]"
        var (dsetName, nfilesStr, jsonfiles) = payload.decode().splitMsgToTuple(3);
        var nfiles = try! nfilesStr:int;
        var filelist: [0..#nfiles] string;
        try {
            filelist = jsonToPdArray(jsonfiles, nfiles);
        } catch {
            return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
        }
        var filedom = filelist.domain;
        var filenames: [filedom] string;
        if filelist.size == 1 {
            var tmp = glob(filelist[0]);
            if GenSymIO_DEBUG {
                writeln(try! "glob expanded %s to %i files".format(filelist[0], tmp.size));
            }
            if tmp.size == 0 {
                return try! "Error: no files matching %s".format(filelist[0]);
            }
            // Glob returns filenames in weird order. Sort for consistency
            // sort(tmp);
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
                return try! "Error: file not found: %s".format(fname);
            } catch e: PermissionError {
                return try! "Error: permission error on %s".format(fname);
            } catch e: DatasetNotFoundError {
                return try! "Error: dataset %s not found in file %s".format(dsetName, fname);
            } catch e: NotHDF5FileError {
                return try! "Error: cannot open as HDF5 file %s".format(fname);
            } catch e: SegArrayError {
                return try! "Error: expected segmented array but could not find sub-datasets '%s' and '%s'".
                                                                   format(SEGARRAY_OFFSET_NAME, SEGARRAY_VALUE_NAME);
            } catch {
                // Need a catch-all for non-throwing function
                return try! "Error: unknown cause";
            }
        }
        const isSegArray = segArrayFlags[filedom.first];
        const dataclass = dclasses[filedom.first];
        const bytesize = bytesizes[filedom.first];
        const isSigned = signFlags[filedom.first];
        for (name, sa, dc, bs, sf) in zip(filenames, segArrayFlags, dclasses, bytesizes, signFlags) {
            if (sa != isSegArray) || (dc != dataclass) || (bs != bytesize) || (sf != isSigned) {
                return try! "Error: inconsistent dtype in dataset %s of file %s".format(dsetName, name);
            }
        }
        if GenSymIO_DEBUG {
            writeln("Verified all dtypes across files");
        }
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
            return notImplementedError("readhdf", try! "Rank %i arrays".format(e.rank));
        } catch {
            return try! "Error: unknown cause";
        }
        if GenSymIO_DEBUG {
            writeln("Got subdomains and total length");
        }

        select (isSegArray, dataclass) {
            when (true, C_HDF5.H5T_INTEGER) {
                if (bytesize != 1) || isSigned {
                    return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".
                                            format(isSegArray, dataclass, bytesize, isSigned);
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
                return try! "created " + st.attrib(segName) + " +created " + st.attrib(valName);
            }
            when (false, C_HDF5.H5T_INTEGER) {
                var entryInt = new shared SymEntry(len, int);
                if GenSymIO_DEBUG {
                    writeln("Initialized int entry"); try! stdout.flush();
                }
                read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
                var rname = st.nextName();
                st.addEntry(rname, entryInt);
                return try! "created " + st.attrib(rname);
            }
            when (false, C_HDF5.H5T_FLOAT) {
                var entryReal = new shared SymEntry(len, real);
                if GenSymIO_DEBUG {
                    writeln("Initialized float entry"); try! stdout.flush();
                }
                read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
                var rname = st.nextName();
                st.addEntry(rname, entryReal);
                return try! "created " + st.attrib(rname);
            }
            otherwise {
                return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
            }
        }
    }

    /* 
     * Reads all datasets from 1..n HDF5 files into an Arkouda symbol table. 
     */
    proc readAllHdfMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        // reqMsg = "readAllHdf <ndsets> <nfiles> [<json_dsetname>] | [<json_filenames>]"
        var repMsg: string;
        // May need a more robust delimiter then " | "
        var (ndsetsStr, nfilesStr, arraysStr) = payload.decode().splitMsgToTuple(3);
        var (jsondsets, jsonfiles) = arraysStr.splitMsgToTuple(" | ",2);
        var ndsets = try! ndsetsStr:int;
        var nfiles = try! nfilesStr:int;
        var dsetlist: [0..#ndsets] string;
        var filelist: [0..#nfiles] string;
        try {
            dsetlist = jsonToPdArray(jsondsets, ndsets);
        } catch {
            return try! "Error: could not decode json dataset names via tempfile (%i files: %s)".format(ndsets, jsondsets);
        }
        try {
            filelist = jsonToPdArray(jsonfiles, nfiles);
        } catch {
            return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
        }
        var dsetdom = dsetlist.domain;
        var filedom = filelist.domain;
        var dsetnames: [dsetdom] string;
        var filenames: [filedom] string;
        dsetnames = dsetlist;

        if filelist.size == 1 {
            var tmp = glob(filelist[0]);
            if GenSymIO_DEBUG {
                writeln(try! "glob expanded %s to %i files".format(filelist[0], tmp.size));
            }
            if tmp.size == 0 {
                return try! "Error: no files matching %s".format(filelist[0]);
            }
            // Glob returns filenames in weird order. Sort for consistency
            // sort(tmp);
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
                    return try! "Error: file not found: %s".format(fname);
                } catch e: PermissionError {
                    return try! "Error: permission error on %s".format(fname);
                } catch e: DatasetNotFoundError {
                    return try! "Error: dataset %s not found in file %s".format(dsetName, fname);
                } catch e: NotHDF5FileError {
                    return try! "Error: cannot open as HDF5 file %s".format(fname);
                } catch e: SegArrayError {
                    return try! "Error: expected segmented array but could not find sub-datasets '%s' and '%s'".
                                          format(SEGARRAY_OFFSET_NAME, SEGARRAY_VALUE_NAME);
                } catch {
                    // Need a catch-all for non-throwing function
                    return try! "Error: unknown cause";
                }
            }
            const isSegArray = segArrayFlags[filedom.first];
            const dataclass = dclasses[filedom.first];
            const bytesize = bytesizes[filedom.first];
            const isSigned = signFlags[filedom.first];
            for (name, sa, dc, bs, sf) in zip(filenames, segArrayFlags, dclasses, bytesizes, signFlags) {
                if (sa != isSegArray) || (dc != dataclass) || (bs != bytesize) || (sf != isSigned) {
                    return try! "Error: inconsistent dtype in dataset %s of file %s".format(dsetName, name);
                }
            }
            if GenSymIO_DEBUG {
                writeln("Verified all dtypes across files for dataset ", dsetName);
            }
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
                return notImplementedError("readhdf", try! "Rank %i arrays".format(e.rank));
            } catch {
                return try! "Error: unknown cause";
            }
            if GenSymIO_DEBUG {
                writeln("Got subdomains and total length for dataset ", dsetName);
            }
            select (isSegArray, dataclass) {
                when (true, C_HDF5.H5T_INTEGER) {
                    if (bytesize != 1) || isSigned {
                        return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
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
                    if GenSymIO_DEBUG {
                        writeln("Initialized int entry for dataset ", dsetName); try! stdout.flush();
                    }
                    read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
                    var rname = st.nextName();
                    st.addEntry(rname, entryInt);
                    rnames = rnames + "created " + st.attrib(rname) + " , ";
                }
                when (false, C_HDF5.H5T_FLOAT) {
                    var entryReal = new shared SymEntry(len, real);
                    if GenSymIO_DEBUG {
                        writeln("Initialized float entry"); try! stdout.flush();
                    }
                    read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
                    var rname = st.nextName();
                    st.addEntry(rname, entryReal);
                    rnames = rnames + "created " + st.attrib(rname) + " , ";
                }
                otherwise {
                    return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
                }
            }
        }
        return try! rnames.strip(" , ", leading = false, trailing = true);
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

    /* Get the class of the HDF5 datatype for the dataset. */
    proc get_dtype(filename: string, dsetName: string) throws {
        const READABLE = (S_IRUSR | S_IRGRP | S_IROTH);
        if !exists(filename) {
            throw new owned FileNotFoundError();
        }
        if !(getMode(filename) & READABLE) {
            throw new owned PermissionError();
        }
        var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
        if file_id < 0 { // HF5open returns negative value on failure
            throw new owned NotHDF5FileError();
        }
        if !C_HDF5.H5Lexists(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT) {
            throw new owned DatasetNotFoundError();
        }
        var dataclass: C_HDF5.H5T_class_t;
        var bytesize: int;
        var isSigned: bool;
        var isSegArray: bool;

        try {
            (dataclass, bytesize, isSigned) = get_dataset_info(file_id, dsetName);
            isSegArray = false;
        } catch e:DatasetNotFoundError {
            var group_id = C_HDF5.H5Gopen2(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT);
            if (group_id < 0) {
                try! writeln("The dataset is neither at the root of the HDF5 file nor within a group");
                throw new owned SegArrayError();
            }
            C_HDF5.H5Gclose(group_id);
            var offsetDset = dsetName + "/" + SEGARRAY_OFFSET_NAME;
            var valueDset = dsetName + "/" + SEGARRAY_VALUE_NAME;
            var (offsetClass, offsetByteSize, offsetSign) = try get_dataset_info(file_id, offsetDset);
            if (offsetClass != C_HDF5.H5T_INTEGER) {
                throw new owned SegArrayError();
            }
            try (dataclass, bytesize, isSigned) = get_dataset_info(file_id, valueDset);
            isSegArray = true;
        } catch e {
            throw e;
        }
        C_HDF5.H5Fclose(file_id);
        return (isSegArray, dataclass, bytesize, isSigned);
    }

    proc get_dataset_info(file_id, dsetName) throws {
        var dset = C_HDF5.H5Dopen(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT);
        if (dset < 0) {
            throw new owned DatasetNotFoundError();
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
     *  Get the subdomains of the distributed array represented by each file, 
     *  as well as the total length of the array. 
     */
    proc get_subdoms(filenames: [?FD] string, dsetName: string) throws {
        use SysCTypes;

        var lengths: [FD] int;
        for (i, filename) in zip(FD, filenames) {
            var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
            var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now
//      var dsetRank: c_int;
//      // Verify 1D array
//      C_HDF5.H5LTget_dataset_ndims(file_id, dsetName.c_str(), dsetRank);
//      if dsetRank != 1 {
//        // TODO: change this to a throw
//        // halt("Expected 1D array, got rank " + dsetRank);
//        throw new owned HDF5RankError(dsetRank, filename, dsetName);
//      }
            // Read array length into dims[0]
            C_HDF5.HDF5_WAR.H5LTget_dataset_info_WAR(file_id, dsetName.c_str(), c_ptrTo(dims), nil, nil);
            C_HDF5.H5Fclose(file_id);
            lengths[i] = dims[0]: int;
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
            if GenSymIO_DEBUG {
                writeln("entry.a.targetLocales() = ", A.targetLocales()); try! stdout.flush();
                writeln("Filedomains: ", filedomains); try! stdout.flush();
            }
            coforall loc in A.targetLocales() do on loc {
                // Create local copies of args
                var locFiles = filenames;
                var locFiledoms = filedomains;
                var locDset = dsetName;
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
                                dataset = C_HDF5.H5Dopen(file_id, locDset.c_str(), C_HDF5.H5P_DEFAULT);
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
                            if GenSymIO_DEBUG {
                                writeln("Locale ", loc, ", intersection ", intersection, ", dataset slice ", 
                                        (intersection.low - filedom.low, intersection.high - filedom.low));
                            }

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
                readHDF5Dataset(file_id, dsetName, AA);
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

    proc tohdfMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        var (arrayName, dsetName, modeStr, jsonfile, dataType)
            = payload.decode().splitMsgToTuple(5);

        var mode = try! modeStr: int;
        var filename: string;
        var entry = st.lookup(arrayName);

        try {
            filename = jsonToPdArray(jsonfile, 1)[0];
        } catch {
            return try! "Error: could not decode json filenames via tempfile " +
                                                      "(%i files: %s)".format(1, jsonfile);
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
                    return unrecognizedTypeError("tohdf", dtype2str(entry.dtype));
                }
            }
        } catch e: FileNotFoundError {
              return try! "Error: unable to open file for writing: %s".format(filename);
        } catch e: MismatchedAppendError {
              return "Error: appending to existing files must be done with the same number" +
                      "of locales. Try saving with a different directory or filename prefix?";
        } catch e: WriteModeError {
              return "Error: cannot append the non-existent file %s. Please save the file in standard truncate mode".format(filename);
        } catch e: Error {
              return "Error: problem writing to file %s".format(e);
        }
        if warnFlag {
            return "Warning: possibly overwriting existing files matching filename pattern";
        } else {
            return "wrote array to file";
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
        
        //Generate a list of matching filenames to test against. 
        var matchingFilenames = getMatchingFilenames(prefix, extension);
        
        var group: string;
 
        if isStringsDataset(dsetName) {
            group = getGroup(dsetName);
            warnFlag = processFilenames(filenames, matchingFilenames, mode, A, group);
        } else {
            warnFlag = false;
        }
        
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
         */
        var leadingSliceIndices: [PrivateSpace] int;    
        var trailingSliceIndices: [PrivateSpace] int;
        var isSingleString: [PrivateSpace] bool;
        var endsWithCompleteString: [PrivateSpace] bool;

        /*
         * If this is a Strings dataset, loop through all locales and set 
         * (1) leadingSliceIndices, which are used to remove leading uint(8) characters 
         * from the local slice that complete a string started in the previous locale and
         * (2) trailingSliceIndices, which are used to start strings that are completed in
         * the new locale.remove that belongs to the previous locale.
         */
        if isStringsDataset(dsetName) {
            coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) 
                          with (ref leadingSliceIndices, ref trailingSliceIndices, 
                                ref isSingleString, ref endsWithCompleteString) do on loc {
                generateSliceIndices(idx,leadingSliceIndices, trailingSliceIndices, 
                                            isSingleString, endsWithCompleteString, A);
            }
        }
                                                       
        /*
         * Iterate through each locale and (1) open the hdf5 file corresponding to the
         * locale (2) prepare segments or values pdarray to be written (3) write pdarray to open
         * hdf5 file and (4) close the hdf5 file
         */
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with 
                        (ref leadingSliceIndices, ref trailingSliceIndices) do on loc {
            const myFilename = filenames[idx];
            if GenSymIO_DEBUG {
                writeln(try! "%s exists? %t".format(myFilename, exists(myFilename)));
            }
            var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), 
                                       C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
            const locDom = A.localSubdomain();
            var dims: [0..#1] C_HDF5.hsize_t;
            dims[0] = locDom.size: C_HDF5.hsize_t;
            var myDsetName = "/" + dsetName;

            use C_HDF5.HDF5_WAR;

            /*
             * Since this is Strings values array, confirm if it's in append mode. If
             * so, the Strings dataset is going to be appended to an hdf5 file as a 
             * set of values and segments arrays within a group named after the 
             * dsetName parameter.
             */
            if mode == APPEND {
                prepareStringsGroup(myFileID, group);
            }

            /*
             * Check for the possibility that 1..n strings span two neighboring locales;
             * by seeing if the final character in the local slice is the null uint(8)
             * character. If it is not, then the last string is only a partial string.
             */
            if A.localSlice(locDom).back() != NULL_STRINGS_VALUE {
              /*
               * Since the last value of the local slice is other than the uint(8) null
               * character, this means the last string in the current, local slice spans 
               * the current AND next locale. Consequently, need to do the following:
               * 1. Add all current locale slice values to a list
               * 2. Obtain remaining uint(8) values from the next locale
               */
              var charList : list(uint(8));
              var segmentsList : list(int);

              (charList, segmentsList) = sliceToValuesAndSegments(A.localSlice(locDom));

              if isSingleString[idx] && idx > 0 {
                  var trailingIndex = trailingSliceIndices[idx-1];

                  if trailingIndex > -1 {
                      var trailingValuesList : list(uint(8));
                      on Locales[idx-1] {
                          const locDom = A.localSubdomain();

                          if trailingIndex == A.localSlice(locDom).size-1 {
                              trailingValuesList.append(A.localSlice(locDom)[trailingIndex]:uint(8));
                          } else {
                              for (value, i) in zip(A.localSlice(locDom),
                                                               0..A.localSlice(locDom).size-1) {
                                  if i > trailingIndex {
                                      trailingValuesList.append(value:uint(8));
                                  }
                              }
                          }
                      }

                      charList.insert(0, trailingValuesList);
                  }
                  if segmentsList.isEmpty() {
                      segmentsList.append(0);
                  }
              }

              /*
               * On the next locale do the following:
               * 
               * 1. Retrieve the non-null uint(8) chars from the start of the local 
               *    slice until the next null uint(8) character is encountered
               * 2. If the slice list is a subset of the values array on the next 
               *     locale, append the charList for this locale.
               */
              on Locales[idx+1] {
                  const locDom = A.localSubdomain();
                  var sliceList: list(uint(8), parSafe=true);

                  /*
                   * Iterate through the local slice values for the next locale and add
                   * each to the valuesList, until the null uint(8) character is reached.
                   * This subset of chars corresponds to the chars that complete the 
                   * last string of the previous locale (idx) if the
                   */
                   for (value, i) in zip(A.localSlice(locDom),
                                                      0..A.localSlice(locDom).size-1) {
                       if value != NULL_STRINGS_VALUE {
                           sliceList.append(value:uint(8));
                       } else {
                           break;
                       }
                   }

                   /*
                    * Check for the situation where only one string or string segment maps to this  
                    * locale, which is indicated by the sliceList size matching the local slice array
                    * size; in such a case there is only one segment.  If so, then keep the uint(0)
                    * chars here and shuffle chars from the previous locale to start the one and only
                    * string of the current locale. If not, then take the leading slice from this
                    * next locale and add to the current locale values list.
                    */
                   if sliceList.size != A.localSlice(locDom).size {
                       charList.extend(sliceList);
                   }                
               }

               /* 
                * To prepare for writing revised values array to hdf5, do the following:
                * 1. Add null uint(8) char to the end of the array so reads work correctly
                * 2. Adjust the dims[0] value, which is the revised length of the valuesList
                */
               var leadingSliceIndex = leadingSliceIndices[idx]:int;
               var trailingSliceIndex = trailingSliceIndices[idx]:int;

               var valuesList: list(uint(8), parSafe=true);

               /*
                * Now check to see if the current locale contains chars from the previous 
                * locale by checking the leadingSliceIndex. If the leadingSliceIndex > -1, and  
                * this locale has 2..n strings, this means that the charList contains chars  
                * that compose the last string from the previous locale. If so, generate a new 
                * valuesList that has those values sliced out.
                */
                if leadingSliceIndex > -1 && !isSingleString[idx] {
                    (valuesList, segmentsList) = adjustForLeadingSlice(leadingSliceIndex, charList);
                } else {
                    valuesList = charList;
                }

                /*
                 * Now check to see if the current locale contains chars that need to be shuffled to
                 * the next locale because the next locale has one string only. If so, then remove
                 * those characters, at which point the null uint(8) char is at the end of the values
                 * list. Otherwise, add the null uint(8) char to the end of the values list
                 */
                if trailingSliceIndex > -1 && isSingleString[idx+1] {
                    var sliceIndex = segmentsList.last();
                    (valuesList, segmentsList) = adjustForTrailingSlice(sliceIndex, valuesList);                
                } else {
                    valuesList.append(NULL_STRINGS_VALUE);            
                }

                if segmentsList.isEmpty() {
                    writeln("HAVING TO MANUALLY APPEND SEGMENTS FOR LOCALE %t".format(idx));
                    segmentsList.append(0);
                }
                
                // Update the dimensions per the possibly re-sized valuesList
                dims[0] = valuesList.size:uint(64);              
                
                /*
                 * Write the valuesList containing the uint(8) characters missing from the
                 * current locale slice along with retrieved from the next locale to hdf5
                 */
                H5LTmake_dataset_WAR(myFileID, '/%s/values'.format(group).c_str(), 1,
                        c_ptrTo(dims), getHDF5Type(A.eltType), c_ptrTo(valuesList.toArray()));

                H5LTmake_dataset_WAR(myFileID, '/%s/segments'.format(group).c_str(), 1,
                                           c_ptrTo([segmentsList.size:uint(64)]),getHDF5Type(int),
                                           c_ptrTo(segmentsList.toArray()));
              } else {
                  /*
                   * The local slice ends with the uint(8) null character, which is the 
                   * required value to ensure correct read logic, so next check to see if 
                   * this local slice contains 1..n chars that compose a string from the 
                   * previous locale.
                   */
                  var leadingSliceIndex = leadingSliceIndices[idx]:int;

                  if leadingSliceIndex == -1 {
                      /*
                       * The local slice ends with the uint(8) null character, which means it's
                       * last string does not span two locales. Since the local slice also 
                       * not does not contain chars from previous locale, simply convert the
                       * local slice to a values and segments list and write the resulting
                       * arrays out to hdf5.
                       */
                      var valuesList : list(uint(8));
                      var segmentsList : list(int);

                      (valuesList, segmentsList) = sliceToValuesAndSegments(A.localSlice(locDom));

                      /*
                       * If (1) this locale ends with the null uint(8) char and (2) contains one string, 
                       * check to see if the trailingSliceIndex from the previous locale is > -1. If it
                       * is, this means the chars following the trailingSliceIndex complete the one
                       * and only string within this locale. 
                       */
                      if isSingleString[idx] && idx > 0 {
                          var trailingIndex = trailingSliceIndices[idx-1];
                          if trailingIndex > -1 {
                              var trailingValuesList : list(uint(8));
                              on Locales[idx-1] {
                                  const locDom = A.localSubdomain();
                                  if trailingIndex == A.localSlice(locDom).size-1 {
                                      trailingValuesList.append(A.localSlice(locDom)[trailingIndex]:uint(8));
                                  } else {
                                      for (value, i) in zip(A.localSlice(locDom), 
                                                                       0..A.localSlice(locDom).size-1) {
                                          if i > trailingIndex {
                                              trailingValuesList.append(value:uint(8));
                                          }
                                      } 
                                  }
                              }

                              valuesList.insert(0, trailingValuesList);
                          }
                      }
                      
                      /*
                       * Covers the special case where the following is true about the current locale:
                       *
                       * 1. This is the last locale
                       * 2. There is one partial string started in the previous locale
                       * 3. The previous locale has no trailing slice to complete the partial string
                       *    in the current locale
                       *
                       * In this very special case, (1) move the current locale chars to the previous 
                       * locale's values list and (2) clear out the current locale segments list
                       * because this locale's values list is now empty
                       */                     
                      if idx == numLocales-1 {
                          if !endsWithCompleteString[idx-1] && isSingleString[idx] 
                                                                && trailingSliceIndices[idx-1] == -1 {
                              writeln("CLEARING OUT FOR LOCALE %t".format(idx));
                              valuesList.clear();
                              segmentsList.clear();
                          }
                      }

                      if !valuesList.isEmpty() && segmentsList.isEmpty() {
                          writeln("MANUALLY APPENDING SEGMENTS for LOCALE %t".format(idx));
                          segmentsList.append(0);
                      }
                    
                      // Update the dimensions per the re-sized Strings values list
                      dims[0] = valuesList.size:uint(64);

                      H5LTmake_dataset_WAR(myFileID, '/%s/values'.format(group).c_str(), 1,
                              c_ptrTo(dims), getHDF5Type(A.eltType), c_ptrTo(valuesList.toArray()));

                      H5LTmake_dataset_WAR(myFileID, '/%s/segments'.format(group).c_str(), 1,
                                           c_ptrTo([segmentsList.size:uint(64)]),getHDF5Type(int),
                                           c_ptrTo(segmentsList.toArray()));
                  } else {
                      /*
                       * The local slice does contain chars from previous locale, so (1)
                       * generate a corresponding Strings value list that can be sliced,
                       * and (2) adjust the Strings values list by slicing the chars out
                       * that correspond to chars from previous locale, and (3) adjust 
                       * the dims value per the size of the updated Strings value list. 
                       */
                       
                      var charList : list(uint(8));
                      var segmentsList : list(int);
                      var valuesList : list(uint(8));
                     
                      (charList, segmentsList) = sliceToValuesAndSegments(A.localSlice(locDom));           
                      (valuesList, segmentsList) = adjustForLeadingSlice(leadingSliceIndex, charList);
                       
                      if isSingleString[idx] && idx > 0 {
                         var trailingIndex = trailingSliceIndices[idx-1];

                         if trailingIndex > -1 {
                             var trailingValuesList : list(uint(8));
                             on Locales[idx-1] {
                                 const locDom = A.localSubdomain();
                                 for (value, i) in zip(A.localSlice(locDom), 0..A.localSlice(locDom).size-1) {
                                     if i > trailingIndex {
                                         trailingValuesList.append(value:uint(8));
                                     }
                                 }
                             }
                             valuesList.insert(0, trailingValuesList);
                         }
                      }

                      if segmentsList.isEmpty() {
                          segmentsList.append(0);
                      }

                      // Update the dimensions per the re-sized Strings values list
                      dims[0] = valuesList.size:uint(64);

                      H5LTmake_dataset_WAR(myFileID, '/%s/values'.format(group).c_str(), 1,
                                        c_ptrTo(dims), getHDF5Type(A.eltType),
                                        c_ptrTo(valuesList.toArray()));

                      H5LTmake_dataset_WAR(myFileID, '/%s/segments'.format(group).c_str(), 1,
                                           c_ptrTo([segmentsList.size:uint(64)]),getHDF5Type(int),
                                           c_ptrTo(segmentsList.toArray()));
                    }
                }
            
            // Close the file now that the 1..n pdarrays have been written
            C_HDF5.H5Fclose(myFileID);
        }
        return warnFlag;
    }

    /*
     * Writes the float, int, or bool pdarray out to hdf5
     */
    private proc write1DDistArray(filename: string, mode: int, dsetName: string, A, 
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
            if GenSymIO_DEBUG {
                writeln(try! "%s exists? %t".format(myFilename, exists(myFilename)));
            }
            var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), 
                                       C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
            const locDom = A.localSubdomain();
            var dims: [0..#1] C_HDF5.hsize_t;
            dims[0] = locDom.size: C_HDF5.hsize_t;
            var myDsetName = "/" + dsetName;

            use C_HDF5.HDF5_WAR;

            /*
             * Write the local slice out to the top-level group of the hdf5 file.
             */
            H5LTmake_dataset_WAR(myFileID, myDsetName.c_str(), 1, c_ptrTo(dims),
                                      getHDF5Type(A.eltType), c_ptrTo(A.localSlice(locDom)));

            // Close the file now that the 1..n pdarrays have been written
            C_HDF5.H5Fclose(myFileID);
        }
        return warnFlag;
    }

    /*
     * Returns a tuple composed of a file prefix and extension to be used to generate
     * locale-specific filenames to be written to.
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
    proc generateFilenames(prefix : string, extension : string, A) : [] string { 
        // Generate the filenames based upon the number of targetLocales.
        var filenames: [0..#A.targetLocales().size] string;
        for i in 0..#A.targetLocales().size {
            filenames[i] = try! "%s_LOCALE%s%s".format(prefix, i:string, extension);
        }
        return filenames;
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
              throw new owned WriteModeError();
          }

          /*
           * There is a mismatch between the number of files to be appended to and the 
           * number of files actually on the file system. This typically happens when 
           * a file append is attempted where the number of locales between the file 
           * creates and updates changes.
           */
          if !allexist || (matchingFilenames.size != filenames.size) {
              throw new owned MismatchedAppendError();
          }

      } else if mode == TRUNCATE { // if truncating, create new file per locale
          if matchingFilenames.size > 0 {
              warnFlag = true;
          } else {
              warnFlag = false;
          }

          for loc in 0..#A.targetLocales().size {
              /*
               * When done with a coforall over locales, only locale 0's file gets created
               * correctly, whereas hhe other locales' files have corrupted headers.
               */
              //filenames[loc] = try! "%s_LOCALE%s%s".format(prefix, loc:string, extension);
              var file_id: C_HDF5.hid_t;

              if GenSymIO_DEBUG {
                  writeln("Creating or truncating file");
              }

              file_id = C_HDF5.H5Fcreate(filenames[loc].c_str(), C_HDF5.H5F_ACC_TRUNC,
                                                        C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
              
              prepareStringsGroup(file_id, group);

              if file_id < 0 { // Negative file_id means error
                  throw new owned FileNotFoundError();
              }

              /*
               * Close the file now that it has been created and, if applicable, the 
               * Strings group derived from the dsetName has been created.
               */
              C_HDF5.H5Fclose(file_id);
           }
        } else {
            throw new IllegalArgumentError("The mode %t is invalid".format(mode));
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
          for f in filenames {
            allexist &= try! exists(f);
          }

          if !allexist || (matchingFilenames.size != filenames.size) {
              throw new owned MismatchedAppendError();
          }
      } else if mode == TRUNCATE { // if truncating, create new file per locale
          if matchingFilenames.size > 0 {
              warnFlag = true;
          } else {
              warnFlag = false;
          }

          for loc in 0..#A.targetLocales().size {
              /*
               * When done with a coforall over locales, only locale 0's file gets created
               * correctly, whereas hhe other locales' files have corrupted headers.
               */
              //filenames[loc] = try! "%s_LOCALE%s%s".format(prefix, loc:string, extension);
              var file_id: C_HDF5.hid_t;

              if GenSymIO_DEBUG {
                  writeln("Creating or truncating file");
              }

              file_id = C_HDF5.H5Fcreate(filenames[loc].c_str(), C_HDF5.H5F_ACC_TRUNC,
                                                      C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);

              if file_id < 0 { // Negative file_id means error
                  throw new owned FileNotFoundError();
              }

              /*
               * Close the file now that it has been created and, if applicable, the 
               * Strings group derived from the dsetName has been created.
               */
              C_HDF5.H5Fclose(file_id);
           }
        } else {
            throw new IllegalArgumentError("The mode %t is invalid".format(mode));
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
     * Generates the slice index for the locale Strings values array and adds it to the 
     * indices parameter. Note: the slice index will be used to remove characters from 
     * the current locale that correspond to the last string of the previous locale.
     */
    private proc generateSliceIndices(idx : int, leadingSliceIndices, trailingSliceIndices, 
                               isSingleString, endsWithCompleteString, A) {
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
            var leadingSliceIndex = -1;
            var trailingSliceIndex = -1;

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
                            leadingSliceIndex = i + 1;
                            leadingSliceSet = true; 
                        } else {
                            /*
                             * If the leadingSliceIndex has already been set, the next null
                             * char is a candidate to be the trailingSliceIndex.
                             */
                             trailingSliceIndex = i;
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
                     writeln("THE LOCALE %t ENDS WITH null %t".format(idx, endsWithCompleteString[idx]);
                }
            }    

            leadingSliceIndices[idx] = leadingSliceIndex;
            trailingSliceIndices[idx] = trailingSliceIndex;
        
            if leadingSliceIndices[idx] > -1 || trailingSliceIndices[idx] > -1 {    
                /*
                 * If either of the indices are > -1, this means there's 2..n null characters
                 * in the string corresponding to the values array, which means the values
                 * array contains 2..n string segments and therefore not a single string.
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
                if leadingSliceIndices[idx] > 0 {
                    trailingSliceIndices[idx] = leadingSliceIndices[idx];
                }
                leadingSliceIndices[idx] = -1;
            }
            
            /*
             * For the special case of this being the last locale, set the trailingSliceIndex to
             * -1 since there is no next locale to slice chars from the current locale to complete.
             */
            if idx == numLocales-1 {
                trailingSliceIndices[idx] = -1;
            }
        }
    }

    /*
     * Processes a local Strings slice into (1) a uint(8) values list for use in methods 
     * that finalize the values array elements following any shuffle operations and (2)
     * a segments list represending starting indices for each string in the values list.
     */
    private proc sliceToValuesAndSegments(rawChars) {
        var charList: list(uint(8), parSafe=true);
        var indices: list(int, parSafe=true);

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
        var valuesList: list(uint(8), parSafe=true);
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
     * to the new values list
     */
    private proc adjustForTrailingSlice(sliceIndex : int,
                                   charList : list(uint(8))) {
        var valuesList: list(uint(8), parSafe=true);
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
            throw new IllegalArgumentError('The Strings dataset must be in form {dset}/values');
        } else {
            return values[0];
        }
    }

    /*
     * Creates an HDF5 Group named via the group parameter to store a String
     * object's segments and values pdarrays.
     * 
     * Note: The file corresponding to the fileId must be open prior to 
     * attempting the group create.
     */
    private proc prepareStringsGroup(fileId: int, group: string) throws {
        var groupId = C_HDF5.H5Gcreate2(fileId, "/%s".format(group).c_str(),
              C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        C_HDF5.H5Gclose(groupId);
    }

    /*
     * Returns a boolean indicating whether the data set is a Strings 
     * dataset corresponding to a Strings save operation.
     */
    private proc isStringsDataset(dataset: string) : bool {
        return dataset.find(needle="values") > -1;
    }
}