module GenHdf5 {
    
    use CommAggregation;
    use CPtr;
    use FileSystem;
    use HDF5;
    use IO;
    use List;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use Reflection;
    use Sort;
    use SymArrayDmap;
    
    import ak;
    import ServerConfig;
    import ServerErrors;
    import ServerErrorStrings;
    import Logging;

    private config const logLevel = ServerConfig.logLevel;
    const ghLogger = new Logging.Logger(logLevel);

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
            ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
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
            ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                        "checking if isBooleanDataset %t with file %s".format(e.message()));
        }
        return boolDataset;
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
            ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                    "checking if isStringsDataset %t".format(e.message())); 
        }

        return groupExists > -1;
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
     * Retrieves the datatype of the dataset read from HDF5 
     */
    proc get_dtype_from_hdf5(filename: string, dsetName: string) throws {
        const READABLE = (S_IRUSR | S_IRGRP | S_IROTH);

        if !exists(filename) {
            throw ServerErrors.getErrorWithContext(
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
            throw ServerErrors.getErrorWithContext(
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
            throw ServerErrors.getErrorWithContext(
                           msg="in accessing %s HDF5 file content".format(filename),
                           lineNumber=getLineNumber(), 
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(), 
                           errorClass="HDF5FileFormatError");            
        }

        var dName = getReadDsetName(file_id, dsetName);

        if !C_HDF5.H5Lexists(file_id, dName.c_str(), C_HDF5.H5P_DEFAULT) {
            C_HDF5.H5Fclose(file_id);
            throw ServerErrors.getErrorWithContext(
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
                var offsetDset = dsetName + "/" + ak.io.SEGARRAY_OFFSET_NAME;
                var (offsetClass, offsetByteSize, offsetSign) = 
                                           try get_dataset_info(file_id, offsetDset);
                if (offsetClass != C_HDF5.H5T_INTEGER) {
                    throw ServerErrors.getErrorWithContext(
                       msg="dataset %s has incorrect one or more sub-datasets" +
                       " %s %s".format(dsetName, ak.io.SEGARRAY_OFFSET_NAME, ak.io.SEGARRAY_VALUE_NAME), 
                       lineNumber=getLineNumber(),
                       routineName=getRoutineName(),
                       moduleName=getModuleName(),
                       errorClass='SegArrayError');                    
                }
                var valueDset = dsetName + "/" + ak.io.SEGARRAY_VALUE_NAME;
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
            throw ServerErrors.getErrorWithContext( 
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
        throw ServerErrors.getErrorWithContext(
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
            throw ServerErrors.getErrorWithContext( 
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

    /*
     * This function gets called when A is a BlockDist or DefaultRectangular array.
     */
    proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), 
                                                 filenames: [FD] string, dsetName: string)
        where (MyDmap == Dmap.blockDist || MyDmap == Dmap.defaultRectangular) {
            try! ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                     "entry.a.targetLocales() = %t".format(A.targetLocales()));
            try! ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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

                            try! ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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

    /*
     * This function is called when A is a CyclicDist array.
     */
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
                defer { // Close the file on exit
                    C_HDF5.H5Fclose(file_id);
                }

                var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now
                var dName = try! getReadDsetName(file_id, dsetName);

                // Read array length into dims[0]
                C_HDF5.HDF5_WAR.H5LTget_dataset_info_WAR(file_id, dName.c_str(), 
                                           c_ptrTo(dims), nil, nil);
                lengths[i] = dims[0]: int;
            } catch e: Error {
                throw ServerErrors.getErrorWithContext(
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
     * Reads all datasets from 1..n HDF5 files into an Arkouda symbol table. 
     */
    proc readAllHdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        // reqMsg = "readAllHdf <ndsets> <nfiles> [<json_dsetname>] | [<json_filenames>]"
        var repMsg: string;
        // May need a more robust delimiter then " | "
        var (strictFlag, ndsetsStr, nfilesStr, allowErrorsFlag, arraysStr) = payload.splitMsgToTuple(5);
        var strictTypes: bool = true;
        if (strictFlag.toLower().strip() == "false") {
          strictTypes = false;
        }

        var allowErrors: bool = "true" == allowErrorsFlag.toLower(); // default is false
        if allowErrors {
            ghLogger.warn(getModuleName(), getRoutineName(), getLineNumber(), "Allowing file read errors");            
        }

        // Test arg casting so we can send error message instead of failing
        if (!checkCast(ndsetsStr, int)) {
            var errMsg = "Number of datasets:`%s` could not be cast to an integer".format(ndsetsStr);
            ghLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errMsg);
            return new MsgTuple(errMsg, MsgType.ERROR);
        }
        if (!checkCast(nfilesStr, int)) {
            var errMsg = "Number of files:`%s` could not be cast to an integer".format(nfilesStr);
            ghLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errMsg);
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
            ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        try {
            filelist = jsonToPdArray(jsonfiles, nfiles);
        } catch {
            var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
            ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var dsetdom = dsetlist.domain;
        var filedom = filelist.domain;
        var dsetnames: [dsetdom] string;
        var filenames: [filedom] string;
        dsetnames = dsetlist;

        if filelist.size == 1 {
            if filelist[0].strip().size == 0 {
                var errorMsg = "filelist was empty.";
                ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            var tmp = glob(filelist[0]);
            ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "glob expanded %s to %i files".format(filelist[0], tmp.size));
            if tmp.size == 0 {
                var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
                ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
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
        for dsetName in dsetnames do {
            for (i, fname) in zip(filedom, filenames) {
                var hadError = false;
                try {
                    (segArrayFlags[i], dclasses[i], bytesizes[i], signFlags[i]) = get_dtype_from_hdf5(fname, dsetName);
                } catch e: FileNotFoundError {
                    fileErrorMsg = "File %s not found".format(fname);
                    ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: PermissionError {
                    fileErrorMsg = "Permission error %s opening %s".format(e.message(),fname);
                    ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: ServerErrors.DatasetNotFoundError {
                    fileErrorMsg = "Dataset %s not found in file %s".format(dsetName,fname);
                    ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: ServerErrors.NotHDF5FileError {
                    fileErrorMsg = "The file %s is not an HDF5 file: %s".format(fname,e.message());
                    ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: ServerErrors.SegArrayError {
                    fileErrorMsg = "SegmentedArray error: %s".format(e.message());
                    ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e : Error {
                    fileErrorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                    ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
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
                  ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                  return new MsgTuple(errorMsg, MsgType.ERROR);
              } else if (strictTypes && ((bs != bytesize) || (sf != isSigned))) {
                  var errorMsg = "Inconsistent precision or sign in dataset %s of file %s\nWith strictTypes, mixing of precision and signedness not allowed (set strictTypes=False to suppress)".format(dsetName, name);
                  ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                  return new MsgTuple(errorMsg, MsgType.ERROR);
              }
            }

            ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Verified all dtypes across files for dataset %s".format(dsetName));
            var subdoms: [filedom] domain(1);
            var segSubdoms: [filedom] domain(1);
            var len: int;
            var nSeg: int;
            try {
                if isSegArray {
                    (segSubdoms, nSeg) = get_subdoms(filenames, dsetName + "/" + ak.io.SEGARRAY_OFFSET_NAME);
                    (subdoms, len) = get_subdoms(filenames, dsetName + "/" + ak.io.SEGARRAY_VALUE_NAME);
                } else {
                    (subdoms, len) = get_subdoms(filenames, dsetName);
                }
            } catch e: ServerErrors.HDF5RankError {
                var errorMsg = ServerErrorStrings.notImplementedError("readhdf", "Rank %i arrays".format(e.rank));
                ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            } catch e: Error {
                var errorMsg = "Other error in accessing dataset %s: %s".format(dsetName,e.message());
                ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }

            ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Got subdomains and total length for dataset %s".format(dsetName));

            select (isSegArray, dataclass) {
                when (true, C_HDF5.H5T_INTEGER) {
                    if (bytesize != 1) || isSigned {
                        var errorMsg = "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(
                                                isSegArray, dataclass, bytesize, isSigned);
                        ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                    var entrySeg = new shared SymEntry(nSeg, int);
                    read_files_into_distributed_array(entrySeg.a, segSubdoms, filenames, dsetName + "/" + ak.io.SEGARRAY_OFFSET_NAME);
                    fixupSegBoundaries(entrySeg.a, segSubdoms, subdoms);
                    var entryVal = new shared SymEntry(len, uint(8));
                    read_files_into_distributed_array(entryVal.a, subdoms, filenames, dsetName + "/" + ak.io.SEGARRAY_VALUE_NAME);
                    var segName = st.nextName();
                    st.addEntry(segName, entrySeg);
                    var valName = st.nextName();
                    st.addEntry(valName, entryVal);
                    rnames.append((dsetName, "seg_string", "%s+%s".format(segName, valName)));
                }
                when (false, C_HDF5.H5T_INTEGER) {
                    var entryInt = new shared SymEntry(len, int);
                    ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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
                    rnames.append((dsetName, "pdarray", rname));
                }
                when (false, C_HDF5.H5T_FLOAT) {
                    var entryReal = new shared SymEntry(len, real);
                    ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                      "Initialized float entry");
                    read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
                    var rname = st.nextName();
                    st.addEntry(rname, entryReal);
                    rnames.append((dsetName, "pdarray", rname));
                }
                otherwise {
                    var errorMsg = "detected unhandled datatype: segmented? %t, class %i, size %i, " +
                                   "signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
                    ghLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
        }

        if allowErrors && fileErrorCount > 0 {
            ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "allowErrors:true, fileErrorCount:%t".format(fileErrorCount));
        }
        repMsg = _buildReadAllHdfMsgJson(rnames, allowErrors, fileErrorCount, fileErrors, st);
        ghLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
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
}