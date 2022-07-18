module HDF5MultiDim {
    use IO;
    use CTypes;
    use Reflection;
    use HDF5;
    use FileIO;
    use FileSystem;
    use AryUtil;
    use NumPyDType;

    use Logging;
    use ServerConfig;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use CommAggregation;
    use ServerErrors;
    use ServerErrorStrings;

    config const TRUNCATE: int = 0;
    config const APPEND: int = 1;

    private config const logLevel = ServerConfig.logLevel;
    const h5tLogger = new Logger(logLevel);

    require "c_helpers/help_h5ls.h", "c_helpers/help_h5ls.c";

    private extern proc c_incrementCounter(data:c_void_ptr);
    private extern proc c_append_HDF5_fieldname(data:c_void_ptr, name:c_string);

    config const FLAT: int = 0;
    config const MULTI_DIM: int = 1;

    /*
    * Wrapper to call c function used to count the number of attributes
    */
    proc _count_attrs(loc_id:C_HDF5.hid_t, name:c_void_ptr, info:c_void_ptr, data:c_void_ptr){
        c_incrementCounter(data);
        return 0;
    }

    /*
    * Wrapper to call out to c function to add attribute name to list
    */
    proc _list_attributes(loc_id:C_HDF5.hid_t, name:c_void_ptr, info:c_void_ptr, data:c_void_ptr){
        var obj_name = name:c_string;

        c_append_HDF5_fieldname(data, obj_name);

        return 0;
    }

    /*
    * Take a multidimensional array and flatten is to a 1 X (prod of dimensions)
    */
    proc _flatten(a: [?D] int, shape: [?sD] int){
        var dim_prod: [sD] int = (* scan shape) / shape;
        var flat_size: int = (* reduce shape);

        var flat: [0..#flat_size] int;
        for coords in D{
            var idx: int;
            for i in 0..#coords.size {
                idx += coords[i] * dim_prod[(dim_prod.size-(i+1))];
            }
            flat[idx] = a[coords];
        }

        return flat;
    }

    /*
    Returns a bool indicating if the file provided exists or not 
    */
    proc doesFileExists(filename: string): bool throws {
        var prefix: string;
        var extension: string;
        (prefix,extension) = getFileMetadata(filename);

        var matchingFiles = glob("%s*%s".format(prefix, extension));

        return matchingFiles.size > 0;
    }

    /*
    Determines if the file exists and creates if not.
    In the event of truncation, remove the file if it already exists
    */
    proc prepFile(filename: string, mode: int) throws {
        var fileExists: bool = doesFileExists(filename);

        if ((!fileExists && mode == APPEND) || mode == TRUNCATE) {
            if (mode == TRUNCATE && fileExists){
                remove(filename);
            }
            // create the file
            var file_id: C_HDF5.hid_t = C_HDF5.H5Fcreate(filename.c_str(), C_HDF5.H5F_ACC_TRUNC, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            defer { // Close file upon exiting scope
                C_HDF5.H5Fclose(file_id);
            }

            if file_id < 0 { // Negative file_id means error
                throw getErrorWithContext(msg="Failure to create/overwrite file, %s".format(filename),
                                            lineNumber=getLineNumber(), 
                                            routineName=getRoutineName(), 
                                            moduleName=getModuleName(), 
                                            errorClass='FileNotFoundError');
            }

            // TODO - add the meta data to the file. This is not done yet because current model does not make sense as it is overwritten (or not updated on append)
        }
    }

    /*
    Reads an HDF5 dataset and builds the components as pdarrays
    The resulting data is always in flattened form (as expected by ArrayView)
    Adds a pdarray containing the flattened data and a pdarray containing the shape of the object to the symbol table
    */
    proc read_hdf_multi_msg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        // Currently always load flat as row major
        var (filename, dset_name) = payload.splitMsgToTuple(2);

        var file_id: C_HDF5.hid_t;
        var dset_id: C_HDF5.hid_t;

        file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);

        if file_id < 0 { // HF5open returns negative value on failure
            C_HDF5.H5Fclose(file_id);
            var errorMsg = "Failure accessing file %s.".format(filename);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        dset_id = C_HDF5.H5Dopen(file_id, dset_name.c_str(), C_HDF5.H5P_DEFAULT);
        if dset_id < 0 { // HF5open returns negative value on failure
            C_HDF5.H5Fclose(file_id);
            var errorMsg = "Failure accessing dataset %s.".format(dset_name);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);     
        }
        

        // check if rank is attr and then get.
        var rank: int;
        if C_HDF5.H5Aexists_by_name(dset_id, ".".c_str(), "Rank", C_HDF5.H5P_DEFAULT) > 0 {
            var rank_id: C_HDF5.hid_t = C_HDF5.H5Aopen_by_name(dset_id, ".".c_str(), "Rank", C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            var attr_type: C_HDF5.hid_t = C_HDF5.H5Aget_type(rank_id);
            C_HDF5.H5Aread(rank_id, getHDF5Type(int), c_ptrTo(rank));
        }
        else{
            // Return error that file does not have required attrs
            var errorMsg = "Rank Attribute was not located in %s. This attribute is required to process multi-dimensional data.".format(filename);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // check if shape attr is present and read it
        var shape: [0..#rank] int;
        if C_HDF5.H5Aexists_by_name(dset_id, ".".c_str(), "Shape", C_HDF5.H5P_DEFAULT) > 0 {
            var shape_id: C_HDF5.hid_t = C_HDF5.H5Aopen_by_name(dset_id, ".".c_str(), "Shape", C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            var attr_type: C_HDF5.hid_t = C_HDF5.H5Aget_type(shape_id);
            C_HDF5.H5Aread(shape_id, getHDF5Type(shape.eltType), c_ptrTo(shape));
        }
        else {
            // Return error that file does not have required attrs
            var errorMsg = "Shape Attribute was not located in %s. This attribute is required to process multi-dimensional data.".format(filename);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var dset_domain: domain(3) = { // domain cannot init with runtime var. Must be set at compile
            0..#shape[0],
            0..#shape[1],
            0..#shape[2]
        };

        var data: [dset_domain] int;

        C_HDF5.H5Dread(dset_id, getHDF5Type(data.eltType), C_HDF5.H5S_ALL, C_HDF5.H5S_ALL, C_HDF5.H5P_DEFAULT, c_ptrTo(data));

        var flat = _flatten(data, shape);

        var fname = st.nextName();
        st.addEntry(fname, new shared SymEntry(flat));

        var sname = st.nextName();
        st.addEntry(sname, new shared SymEntry(shape));

        // Close the open hdf5 objects
        C_HDF5.H5Dclose(dset_id);
        C_HDF5.H5Fclose(file_id);

        var repMsg: string = "created " + st.attrib(sname) + "+created " + st.attrib(fname);

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc writeMultiDimDset(file_id: C_HDF5.hid_t, dset_name: string, A, shape: SymEntry, method: int, arrType: DType) throws{
        // Convert the Chapel dtype to HDF5
        var dtype_id: C_HDF5.hid_t;
        select arrType {
            when DType.Int64 {
                dtype_id = getHDF5Type(int);
            }
            when DType.UInt64 {
                dtype_id = getHDF5Type(uint);
            }
            when DType.Float64 {
                dtype_id = getHDF5Type(real);
            }
            otherwise {
                throw getErrorWithContext(
                           msg="Invalid Data Type: %s".format(dtype2str(arrType)),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
            }
        }

        // write the dataset based in the format provided
        var rank: c_int;
        select (method) {
            when (FLAT) { // store data as a flattened single array
                rank = 1:c_int;
                var dims: [0..#1] C_HDF5.hsize_t;
                dims[0] = (* reduce shape.a):C_HDF5.hsize_t;
                C_HDF5.H5LTmake_dataset(file_id, dset_name.c_str(), rank, dims, dtype_id, A.ptr);
            }
            when (MULTI_DIM) { // store the data as the multi-dimensional object
                rank = shape.size:c_int;
                // reshape the data based on the shape passed.
                var dims: [0..#shape.size] C_HDF5.hsize_t;
                forall i in 0..#shape.size{
                    dims[i] = shape.a[i]:C_HDF5.hsize_t;
                }
                C_HDF5.H5LTmake_dataset(file_id, dset_name.c_str(), rank, dims, dtype_id, A.ptr);
            }
            otherwise {
                throw getErrorWithContext(
                           msg="Unknown storage method. Expecting 0 (flat) or 1 (multi). Found %s".format(method),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
            }
        }
        
    }

    /*
    * Takes a multidimensional array obj and writes it into and HDF5 dataset.
    * Provides the ability to store the data flat or multidimensional.
    */
    proc write_hdf_multi_msg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        var (flat_name, shape_name, order_str, filename, dset_name, mode_str, method_str) = payload.splitMsgToTuple(7);
        
        var method: int;
        try {
            method = method_str:int;
        } catch {
            var errorMsg = "Could not convert method to numeric";
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var entry = st.lookup(flat_name);
        var entryDtype = DType.UNDEF;
        if (entry.isAssignableTo(SymbolEntryType.TypedArraySymEntry)) {
            entryDtype = (entry: borrowed GenSymEntry).dtype;
        } else if (entry.isAssignableTo(SymbolEntryType.SegStringSymEntry)) {
            entryDtype = (entry: borrowed SegStringSymEntry).dtype;
        } else {
            var errorMsg = "writehdf_multi Unsupported SymbolEntryType:%t".format(entry.entryType);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var shape_sym: borrowed GenSymEntry = getGenericTypedArrayEntry(shape_name, st);
        var shape = toSymEntry(shape_sym, int);

        var mode: int;
        try {
            mode = mode_str: int;
        } catch {
            var errorMsg = "Could not convert mode to numeric";
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // create the file if it does not exist or if we are truncating
        prepFile(filename, mode);
        var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);

        if file_id < 0 { // HF5open returns negative value on failure
            C_HDF5.H5Fclose(file_id);
            var errorMsg = "Failure accessing file %s.".format(filename);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);     
        }

        // validate that the dataset does not already exist
        var dset_exists: int = C_HDF5.H5Lexists(file_id, dset_name.c_str(), C_HDF5.H5P_DEFAULT);
        if dset_exists > 0 {
            var errorMsg = "A dataset named %s already exists in %s. Overwriting is not currently supported. Please choose another name.".format(dset_name, filename);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        else if dset_exists < -1 {
            var errorMsg = "Failure validating the status of dataset named %s.".format(dset_name);
            h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        select entryDtype {
            when DType.Int64 {
                var flat = toSymEntry(toGenSymEntry(entry), int);
                var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                writeMultiDimDset(file_id, dset_name, localFlat, shape, method, DType.Int64);
            }
            when DType.UInt64 {
                var flat = toSymEntry(toGenSymEntry(entry), uint);
                var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                writeMultiDimDset(file_id, dset_name, localFlat, shape, method, DType.UInt64);
            }
            when DType.Float64 {
                var flat = toSymEntry(toGenSymEntry(entry), real);
                var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                writeMultiDimDset(file_id, dset_name, localFlat, shape, method, DType.Float64);
            }
            otherwise {
                var errorMsg = unrecognizedTypeError("writehdf_multi", dtype2str(entryDtype));
                h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }

        //open the created dset so we can add attributes.
        var dset_id: C_HDF5.hid_t = C_HDF5.H5Dopen(file_id, dset_name.c_str(), C_HDF5.H5P_DEFAULT);

        // Create the attribute space
        var attrSpaceId: C_HDF5.hid_t = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        var attr_id: C_HDF5.hid_t;

        // Create the objectType. This will be important when merging with other read/write functionality.
        var objType: string = "ArrayView";
        var attrStringType = C_HDF5.H5Tcopy(C_HDF5.H5T_C_S1): C_HDF5.hid_t;
        C_HDF5.H5Tset_size(attrStringType, objType.size:uint(64) + 1); // ensure space for NULL terminator
        C_HDF5.H5Tset_strpad(attrStringType, C_HDF5.H5T_STR_NULLTERM);
        C_HDF5.H5Tset_cset(attrStringType, C_HDF5.H5T_CSET_UTF8);
        attr_id = C_HDF5.H5Acreate2(dset_id, "ObjType".c_str(), attrStringType, attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        var type_chars = c_calloc(c_char, objType.size+1);
        for (c, i) in zip(objType.codepoints(), 0..<objType.size) {
            type_chars[i] = c:c_char;
        }
        type_chars[objType.size] = 0:c_char; // ensure NULL termination
        C_HDF5.H5Awrite(attr_id, attrStringType, type_chars);
        C_HDF5.H5Aclose(attr_id);

        // Store the rank of the dataset. Required to read so that shape can be built
        attr_id = C_HDF5.H5Acreate2(dset_id, "Rank".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(shape.size));
        C_HDF5.H5Aclose(attr_id);

        // Indicates if the data is stored as Flat or Mutli-dimensional.
        attr_id = C_HDF5.H5Acreate2(dset_id, "Format".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(method));
        C_HDF5.H5Aclose(attr_id);

        C_HDF5.H5Sclose(attrSpaceId);
        attrSpaceId= C_HDF5.H5Screate(C_HDF5.H5S_SIMPLE);
        var adim: [0..#1] C_HDF5.hsize_t = shape.size:C_HDF5.hsize_t;
        C_HDF5.H5Sset_extent_simple(attrSpaceId, 1, c_ptrTo(adim), c_ptrTo(adim));

        attr_id = C_HDF5.H5Acreate2(dset_id, "Shape".c_str(), getHDF5Type(shape.a.eltType), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        var localShape = new lowLevelLocalizingSlice(shape.a, 0..#shape.size);
        C_HDF5.H5Awrite(attr_id, getHDF5Type(shape.a.eltType), localShape.ptr);
        C_HDF5.H5Aclose(attr_id);


        // Close the open hdf5 objects
        C_HDF5.H5Sclose(attrSpaceId);
        C_HDF5.H5Dclose(dset_id);
        C_HDF5.H5Fclose(file_id);

        var repMsg: string = "Dataset written successfully!";

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }


    proc registerMe() {
        use CommandMap;
        registerFunction("readhdf_multi", read_hdf_multi_msg, getModuleName());
        registerFunction("writehdf_multi", write_hdf_multi_msg, getModuleName());
    }
}