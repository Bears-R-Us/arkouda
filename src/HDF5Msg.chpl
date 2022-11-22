module HDF5Msg {
    use CTypes;
    use FileSystem;
    use HDF5;
    use IO;
    use List;
    use Map;
    use PrivateDist;
    use Reflection;
    use Set;
    use Time only;
    use AryUtil;

    use CommAggregation;
    use FileIO;
    use FileSystem;
    use GenSymIO;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use NumPyDType;
    use ServerConfig;
    use ServerErrors;
    use ServerErrorStrings;
    use SegmentedString;
    use Sort;

    private config const logLevel = ServerConfig.logLevel;
    const h5Logger = new Logger(logLevel);

    const ARKOUDA_HDF5_FILE_METADATA_GROUP = "/_arkouda_metadata";
    const ARKOUDA_HDF5_ARKOUDA_VERSION_KEY = "arkouda_version"; // see ServerConfig.arkoudaVersion
    type ARKOUDA_HDF5_ARKOUDA_VERSION_TYPE = c_string;
    const ARKOUDA_HDF5_FILE_VERSION_KEY = "file_version";
    const ARKOUDA_HDF5_FILE_VERSION_VAL = 2.0:real(32);
    type ARKOUDA_HDF5_FILE_VERSION_TYPE = real(32);
    config const SEGSTRING_OFFSET_NAME = "segments";
    config const SEGSTRING_VALUE_NAME = "values";

    enum ObjType {
      ARRAYVIEW=0,
      PDARRAY=1,
      STRINGS=2
    };

    config const TRUNCATE: int = 0;
    config const APPEND: int = 1;

    config const SINGLE_FILE: int = 0;
    config const MULTI_FILE: int = 1;

    require "c_helpers/help_h5ls.h", "c_helpers/help_h5ls.c";
    private extern proc c_get_HDF5_obj_type(loc_id:C_HDF5.hid_t, name:c_string, obj_type:c_ptr(C_HDF5.H5O_type_t)):C_HDF5.herr_t;
    private extern proc c_strlen(s:c_ptr(c_char)):c_size_t;
    private extern proc c_incrementCounter(data:c_void_ptr);
    private extern proc c_append_HDF5_fieldname(data:c_void_ptr, name:c_string);

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
        Returns the C_HDF5.hid_t corresponding to the provided Chapel type
    */
    proc getDataType(type t) : C_HDF5.hid_t {
        if t == bool {
            return C_HDF5.H5T_NATIVE_HBOOL;
        }
        else {
            return getHDF5Type(t);
        }
    }

    /*
        Validates that the provided write mode is APPEND or TRUNCATE
        mode: int

        If mode is not 1 (Append) or 0 (Truncate) error
    */
    proc validateWriteMode(mode: int) throws {
        if (mode != APPEND && mode != TRUNCATE) {
            throw getErrorWithContext(
                           msg="Unknown write mode %i found.".format(mode),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }
    }

    /*
        Prepare the file for writing data to single file
    */
    proc prepFiles(filename: string, mode: int): string throws {
        // validate the write mode
        validateWriteMode(mode);

        var prefix: string;
        var extension: string;
        (prefix,extension) = getFileMetadata(filename);

        var filenames: [0..#1] string = ["%s%s".format(prefix, extension)];
        var matchingFilenames = glob("%s*%s".format(prefix, extension));

        const f = filenames[0];
        
        var fileExists: bool = matchingFilenames.size > 0;
        if (mode == TRUNCATE || (mode == APPEND && !fileExists)) {
            if (mode == TRUNCATE && fileExists){
                remove(f);
            }

            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "Creating or truncating file");

            // create the file
            var file_id: C_HDF5.hid_t = C_HDF5.H5Fcreate(f.c_str(), C_HDF5.H5F_ACC_TRUNC, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            defer { // Close file upon exiting scope
                C_HDF5.H5Fclose(file_id);
            }

            if file_id < 0 { // Negative file_id means error
                throw getErrorWithContext(msg="The file %s cannot be created".format(f),
                                            lineNumber=getLineNumber(), 
                                            routineName=getRoutineName(), 
                                            moduleName=getModuleName(), 
                                            errorClass='FileNotFoundError');
            }
        }
        return f;
    }

    /*
        Prepare the files required to write files distributed across locales
        A is the entry to be written.
    */
    proc prepFiles(filename: string, mode: int, A): [] string throws {
        // validate the write mode
        validateWriteMode(mode);

        var prefix: string;
        var extension: string;
        (prefix,extension) = getFileMetadata(filename);

        var targetSize = A.targetLocales().size;
        var filenames: [0..#targetSize] string;
        forall i in 0..#targetSize {
            filenames[i] = generateFilename(prefix, extension, i);
        }
        fioLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "generateFilenames targetLocales.size %i, filenames.size %i".format(targetSize, filenames.size));

        var matchingFilenames = glob("%s_LOCALE*%s".format(prefix, extension));
        var filesExist: bool = matchingFilenames.size > 0;

        if (mode == TRUNCATE || (mode == APPEND && !filesExist)) {
            coforall loc in A.targetLocales() do on loc {
                var file_id: C_HDF5.hid_t;
                var fn = filenames[loc.id].localize();
                var existList = glob(fn);
                if mode == TRUNCATE && existList.size == 1 {
                    remove(fn);
                }
                h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "Creating or truncating file");

                file_id = C_HDF5.H5Fcreate(fn.c_str(), C_HDF5.H5F_ACC_TRUNC,
                                                            C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
                defer { // Close file upon exiting scope
                    C_HDF5.H5Fclose(file_id);
                }

                if file_id < 0 { // Negative file_id means error
                  throw getErrorWithContext(
                                    msg="The file %s cannot be created".format(fn),
                                    lineNumber=getLineNumber(), 
                                    routineName=getRoutineName(), 
                                    moduleName=getModuleName(), 
                                    errorClass='FileNotFoundError');
              }
            }
        }
        else if mode == APPEND {
            if filenames.size != matchingFilenames.size {
                throw getErrorWithContext(
                           msg="Cannot append when the number of existing filenames does not match the expected.",
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
            }
        }
        return filenames;
    }

    /*
        Validate that the dataset name provided does not already exist
    */
    proc validateDataset(file_id: C_HDF5.hid_t, filename: string, dset_name: string) throws {
        // validate that the dataset does not already exist
        var dset_exists: int = C_HDF5.H5Lexists(file_id, dset_name.localize().c_str(), C_HDF5.H5P_DEFAULT);
        if dset_exists > 0 {
            throw getErrorWithContext(
                           msg="A dataset named %s already exists in %s. Overwriting is not currently supported. Please choose another name.".format(dset_name, filename),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }
        else if dset_exists < 0 {
            throw getErrorWithContext(
                           msg="Failure validating the status of dataset named %s.".format(dset_name),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }
    }

    /*
        Validate that the group does not already exist.
        If it does not exist, it is created.
    */
    proc validateGroup(file_id: C_HDF5.hid_t, filename: string, group: string) throws {
        var group_exists: int = C_HDF5.H5Lexists(file_id, group.localize().c_str(), C_HDF5.H5P_DEFAULT);
        if group_exists > 0 {
            throw getErrorWithContext(
                           msg="A group named %s already exists in %s. Overwriting is not currently supported. Please choose another name.".format(group, filename),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }
        else if group_exists < 0 {
            throw getErrorWithContext(
                           msg="Failure validating the status of group named %s.".format(group),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }
        else {
            // create the group
            var groupId: C_HDF5.hid_t = C_HDF5.H5Gcreate2(file_id, "/%s".format(group).c_str(),
                                    C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            C_HDF5.H5Gclose(groupId);
        }
    }

    /*
        Write Arkouda metadata attributes to the provided object. 
        
        file_id: C_HDF5.hid_t
            ID of the file the attributes are to be written to. This should be the id of a group or dataset

        objName: string
            Name of the group or dataset the attributes are being written to.

        objType: string
            The type of the object stored in the parent. ArrayView, pdarray, or strings

        dtype: C_HDF5.hid_t
            id of the C_HDF5 datatype of the data contained in the object. Used to check for boolean datasets
    */
    proc writeArkoudaMetaData(file_id: C_HDF5.hid_t, objName: string, objType: string, dtype: C_HDF5.hid_t) throws {
        var obj_id: C_HDF5.hid_t = C_HDF5.H5Oopen(file_id, objName.localize().c_str(), C_HDF5.H5P_DEFAULT);

        // Create the attribute space
        var attrSpaceId: C_HDF5.hid_t = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        var attr_id: C_HDF5.hid_t;

        var attrStringType = C_HDF5.H5Tcopy(C_HDF5.H5T_C_S1): C_HDF5.hid_t;

         // Create the objectType. This will be important when merging with other read/write functionality.
        attr_id = C_HDF5.H5Acreate2(obj_id, "ObjType".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        var t: ObjType = objType.toUpper(): ObjType;
        var t_int: int = t: int;
        C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(t_int));
        C_HDF5.H5Aclose(attr_id);

        // write attribute for boolean
        if dtype == C_HDF5.H5T_NATIVE_HBOOL {
            attr_id = C_HDF5.H5Acreate2(obj_id, "isBool".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            var isBool: int = 1;
            C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(isBool));
            C_HDF5.H5Aclose(attr_id);
        }

        var attrFileVersionType = getHDF5Type(ARKOUDA_HDF5_FILE_VERSION_TYPE);
        var attrId = C_HDF5.H5Acreate2(obj_id,
                          ARKOUDA_HDF5_FILE_VERSION_KEY.c_str(),
                          attrFileVersionType,
                          attrSpaceId,
                          C_HDF5.H5P_DEFAULT,
                          C_HDF5.H5P_DEFAULT);
        
        // H5Awrite requires a pointer and we have a const, so we need a variable ref we can turn into a pointer
        var fileVersion = ARKOUDA_HDF5_FILE_VERSION_VAL;
        C_HDF5.H5Awrite(attrId, attrFileVersionType, c_ptrTo(fileVersion));
        C_HDF5.H5Aclose(attrId);

        // var attrStringType = C_HDF5.H5Tcopy(C_HDF5.H5T_C_S1): C_HDF5.hid_t;
        C_HDF5.H5Tset_size(attrStringType, arkoudaVersion.size:uint(64) + 1); // ensure space for NULL terminator
        C_HDF5.H5Tset_strpad(attrStringType, C_HDF5.H5T_STR_NULLTERM);
        
        attrId = C_HDF5.H5Acreate2(obj_id,
                            ARKOUDA_HDF5_ARKOUDA_VERSION_KEY.c_str(),
                            attrStringType,
                            attrSpaceId,
                            C_HDF5.H5P_DEFAULT,
                            C_HDF5.H5P_DEFAULT);

        // For the value, we need to build a ptr to a char[]; c_string doesn't work because it is a const char*        
        var akVersion = c_calloc(c_char, arkoudaVersion.size+1);
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
        C_HDF5.H5Oclose(obj_id);
    }

    /*
        Writes Attributes specific to a multidimensional array.
            - objType = ArrayView
            - Rank: int - rank of the dataset
            - Shape: [] int - stores the shape of object.
        Calls to writeArkoudaMetaData to write the arkouda metadata
    */
    proc writeArrayViewAttrs(file_id: C_HDF5.hid_t, dset_name: string, objType: string, shape: SymEntry, dtype:C_HDF5.hid_t) throws {
        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Writing ArrayView Attrs");
        //open the created dset so we can add attributes.
        var dset_id: C_HDF5.hid_t = C_HDF5.H5Dopen(file_id, dset_name.localize().c_str(), C_HDF5.H5P_DEFAULT);

        // Create the attribute space
        var attrSpaceId: C_HDF5.hid_t = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        var attr_id: C_HDF5.hid_t;

        // Store the rank of the dataset. Required to read so that shape can be built
        attr_id = C_HDF5.H5Acreate2(dset_id, "Rank".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        var s = shape.size; // needed to localize in the event that shape is not local.
        C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(s));
        C_HDF5.H5Aclose(attr_id);

        C_HDF5.H5Sclose(attrSpaceId);
        attrSpaceId= C_HDF5.H5Screate(C_HDF5.H5S_SIMPLE);
        var adim: [0..#1] C_HDF5.hsize_t = shape.size:C_HDF5.hsize_t;
        C_HDF5.H5Sset_extent_simple(attrSpaceId, 1, c_ptrTo(adim), c_ptrTo(adim));

        attr_id = C_HDF5.H5Acreate2(dset_id, "Shape".c_str(), getHDF5Type(shape.a.eltType), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        var localShape = new lowLevelLocalizingSlice(shape.a, 0..#shape.size);
        C_HDF5.H5Awrite(attr_id, getHDF5Type(shape.a.eltType), localShape.ptr);
        C_HDF5.H5Aclose(attr_id);

        // close the space and the dataset
        C_HDF5.H5Sclose(attrSpaceId);
        C_HDF5.H5Dclose(dset_id);

        // add arkouda meta data attributes
        writeArkoudaMetaData(file_id, dset_name, objType, dtype);
    }

    /*
        writes 1D array to dataset in single file
    */
    proc writeLocalDset(file_id: C_HDF5.hid_t, dset_name: string, A, dimension: int, type t) throws{
        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Writing dataset, %s".format(dset_name));
        // Convert the Chapel dtype to HDF5
        var dtype_id: C_HDF5.hid_t = getDataType(t);

        // always store multidimensional arrays as flattened array
        var dims: [0..#1] C_HDF5.hsize_t;
        dims[0] = dimension:C_HDF5.hsize_t;
        C_HDF5.H5LTmake_dataset(file_id, dset_name.c_str(), 1:c_int, dims, dtype_id, A.ptr);
    }

    /*
        write 1d array to dataset in files distributed over locales
    */
    proc writeDistDset(filenames: [] string, dset_name: string, objType: string, A, st: borrowed SymTab, shape_name: string = "") throws {
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) do on loc {
            const localeFilename = filenames[idx];
            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "%s exists? %t".format(localeFilename, exists(localeFilename)));

            var file_id = C_HDF5.H5Fopen(localeFilename.c_str(), C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
            defer { // Close the file on scope exit
                C_HDF5.H5Fclose(file_id);
            }

            // validate that the dataset does not already exist
            validateDataset(file_id, localeFilename, dset_name);

            const locDom = A.localSubdomain();
            var dims: [0..#1] C_HDF5.hsize_t;
            dims[0] = locDom.size: C_HDF5.hsize_t;

            use C_HDF5.HDF5_WAR;

            var dType: C_HDF5.hid_t = getDataType(A);

            /*
            * Depending upon the datatype, write the local slice out to the top-level
            * or nested, named group within the hdf5 file corresponding to the locale.
            */
            if locDom.size <= 0 {
                H5LTmake_dataset_WAR(file_id, dset_name.localize().c_str(), 1, c_ptrTo(dims), dType, nil);
            } else {
                H5LTmake_dataset_WAR(file_id, dset_name.localize().c_str(), 1, c_ptrTo(dims), dType, c_ptrTo(A.localSlice(locDom)));
            }

            // write the appropriate attributes
            if shape_name != "" {
                // write attributes for multi-dim array
                var shape_sym: borrowed GenSymEntry = getGenericTypedArrayEntry(shape_name, st);
                var shape = toSymEntry(shape_sym, int);
                writeArrayViewAttrs(file_id, dset_name, objType, shape, dType);
            }
            else {
                // write attributes for arkouda meta info otherwise
                writeArkoudaMetaData(file_id, dset_name, objType, dType);
            }
        }
    }

    /*
        Process and write an Arkouda ArrayView to HDF5.
    */
    proc arrayView_tohdfMsg(msgArgs: MessageArgs, st: borrowed SymTab) throws {
        // access integer representation of APPEND/TRUNCATE
        var mode: int = msgArgs.get("write_mode").getIntValue();

        var filename: string = msgArgs.getValueOf("filename");
        var entry = st.lookup(msgArgs.getValueOf("values"));
        var file_format = msgArgs.get("file_format").getIntValue();

        const entryDtype = msgArgs.get("values").getDType();

        var dset_name = msgArgs.getValueOf("dset");
        const objType = msgArgs.getValueOf("objType");

        select file_format {
            when SINGLE_FILE {
                var f = prepFiles(filename, mode);
                var file_id = C_HDF5.H5Fopen(f.c_str(), C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
                if file_id < 0 { // HF5open returns negative value on failure
                    C_HDF5.H5Fclose(file_id);
                    var errorMsg = "Failure accessing file %s.".format(f);
                    throw getErrorWithContext(
                           msg=errorMsg,
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="FileNotFoundError");
                }

                // validate that the dataset does not already exist
                validateDataset(file_id, f, dset_name);

                var shape_sym: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("shape"), st);
                var shape = toSymEntry(shape_sym, int);
                var dims: int = * reduce shape.a;

                var dtype: C_HDF5.hid_t;
                
                select entryDtype {
                    when DType.Int64 {
                        var flat = toSymEntry(toGenSymEntry(entry), int);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, dims, int);
                        dtype = getHDF5Type(int);
                    }
                    when DType.UInt64 {
                        var flat = toSymEntry(toGenSymEntry(entry), uint);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, dims, uint);
                        dtype = getHDF5Type(uint);
                    }
                    when DType.Float64 {
                        var flat = toSymEntry(toGenSymEntry(entry), real);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, dims, real);
                        dtype = getHDF5Type(real);
                    }
                    when DType.Bool {
                        var flat = toSymEntry(toGenSymEntry(entry), bool);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, dims, bool);
                        dtype = C_HDF5.H5T_NATIVE_HBOOL;
                    }
                    otherwise {
                        var errorMsg = unrecognizedTypeError("arrayView_tohdfMsg", dtype2str(entryDtype));
                        throw getErrorWithContext(
                           msg=errorMsg,
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="TypeError");
                    }
                }
                // write attributes for multi-dim array
                writeArrayViewAttrs(file_id, dset_name, objType, shape, dtype);
                C_HDF5.H5Fclose(file_id);
            }
            when MULTI_FILE {
                select entryDtype {
                    when DType.Int64 {
                        var e = toSymEntry(toGenSymEntry(entry), int);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st, msgArgs.getValueOf("shape"));
                    }
                    when DType.UInt64 {
                        var e = toSymEntry(toGenSymEntry(entry), uint);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st, msgArgs.getValueOf("shape"));
                    }
                    when DType.Float64 {
                        var e = toSymEntry(toGenSymEntry(entry), real);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st, msgArgs.getValueOf("shape"));
                    }
                    when DType.Bool {
                        var e = toSymEntry(toGenSymEntry(entry), bool);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st, msgArgs.getValueOf("shape"));
                    }
                    otherwise {
                        var errorMsg = unrecognizedTypeError("multiDimArray_tohdfMsg", dtype2str(entryDtype));
                        throw getErrorWithContext(
                           msg=errorMsg,
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="TypeError");
                    }
                }
            }
            otherwise {
                throw getErrorWithContext(
                           msg="Unknown file format. Expecting 0 (single file) or 1 (file per locale). Found %i".format(file_format),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
            }
        }
    }

    /*
        Process and write an Arkouda pdarray to HDF5.
    */
    proc pdarray_tohdfMsg(msgArgs: MessageArgs, st: borrowed SymTab) throws {
        var mode: int = msgArgs.get("write_mode").getIntValue();

        var filename: string = msgArgs.getValueOf("filename");
        var entry = st.lookup(msgArgs.getValueOf("values"));
        var file_format = msgArgs.get("file_format").getIntValue();

        const entryDtype = msgArgs.get("values").getDType();

        var dset_name = msgArgs.getValueOf("dset");
        const objType = msgArgs.getValueOf("objType");
        var dtype: C_HDF5.hid_t;

        select file_format {
            when SINGLE_FILE {
                var f = prepFiles(filename, mode);
                var file_id = C_HDF5.H5Fopen(f.c_str(), C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
                if file_id < 0 { // HF5open returns negative value on failure
                    C_HDF5.H5Fclose(file_id);
                    var errorMsg = "Failure accessing file %s.".format(f);
                    throw getErrorWithContext(
                           msg=errorMsg,
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="FileNotFoundError");
                }

                // validate that the dataset does not already exist
                validateDataset(file_id, f, dset_name);

                select entryDtype {
                    when DType.Int64 {
                        var flat = toSymEntry(toGenSymEntry(entry), int);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, flat.size, int);
                        dtype = getHDF5Type(int);
                    }
                    when DType.UInt64 {
                        var flat = toSymEntry(toGenSymEntry(entry), uint);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, flat.size, uint);
                        dtype = getHDF5Type(uint);
                    }
                    when DType.Float64 {
                        var flat = toSymEntry(toGenSymEntry(entry), real);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, flat.size, real);
                        dtype = getHDF5Type(real);
                    }
                    when DType.Bool {
                        var flat = toSymEntry(toGenSymEntry(entry), bool);
                        var localFlat = new lowLevelLocalizingSlice(flat.a, 0..#flat.size);
                        writeLocalDset(file_id, dset_name, localFlat, flat.size, bool);
                        dtype = C_HDF5.H5T_NATIVE_HBOOL;
                    }
                    otherwise {
                        var errorMsg = unrecognizedTypeError("pdarray_tohdfmsg", dtype2str(entryDtype));
                        throw getErrorWithContext(
                           msg=errorMsg,
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="TypeError");
                    }
                }
                // write attributes for arkouda meta info
                writeArkoudaMetaData(file_id, dset_name, objType, dtype);
                C_HDF5.H5Fclose(file_id);
            }
            when MULTI_FILE {
                select entryDtype {
                    when DType.Int64 {
                        var e = toSymEntry(toGenSymEntry(entry), int);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st);
                    }
                    when DType.UInt64 {
                        var e = toSymEntry(toGenSymEntry(entry), uint);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st);
                    }
                    when DType.Float64 {
                        var e = toSymEntry(toGenSymEntry(entry), real);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st);
                    }
                    when DType.Bool {
                        var e = toSymEntry(toGenSymEntry(entry), bool);
                        var filenames = prepFiles(filename, mode, e.a);
                        writeDistDset(filenames, dset_name, objType, e.a, st);
                    }
                    otherwise {
                        var errorMsg = unrecognizedTypeError("pdarray_tohdfmsg", dtype2str(entryDtype));
                        throw getErrorWithContext(
                           msg=errorMsg,
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="TypeError");
                    }
                }
            }
            otherwise {
                throw getErrorWithContext(
                           msg="Unknown file format. Expecting 0 (single file) or 1 (file per locale). Found %i".format(file_format),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
            }
        }
    }

    /**
     * Writes empty "Strings" components to the designated parent group in the HDF5 file
     * :arg fileId: HDF5 file id
     * :type fileId: int
     *
     * :arg group: parent dataset / group name for values and segments
     * :type group: string
     *
     * :arg writeOffsets: boolean switch for whether or not to write offsets/segements to file
     * :type writeOffsets: bool
     */
    private proc writeNilStringsGroupToHdf(fileId: int, group: string, writeOffsets: bool) throws {
        var dset_id: C_HDF5.hid_t;
        C_HDF5.H5LTmake_dataset_WAR(fileId, "/%s/values".format(group).c_str(), 1,
                c_ptrTo([0:uint(64)]), getHDF5Type(uint(8)), nil);

        dset_id = C_HDF5.H5Dopen(fileId, "/%s/values".format(group).c_str(), C_HDF5.H5P_DEFAULT);

        // Create the attribute space
        var attrSpaceId: C_HDF5.hid_t = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        var attr_id: C_HDF5.hid_t;

        // Create the objectType. This will be important when merging with other read/write functionality.
        attr_id = C_HDF5.H5Acreate2(dset_id, "ObjType".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        var t: ObjType = ObjType.PDARRAY;
        var t_int: int = t: int;
        C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(t_int));
        C_HDF5.H5Aclose(attr_id);
        C_HDF5.H5Sclose(attrSpaceId);
        C_HDF5.H5Dclose(dset_id);

        if (writeOffsets) {
            C_HDF5.H5LTmake_dataset_WAR(fileId, "/%s/segments".format(group).c_str(), 1,
                c_ptrTo([0:uint(64)]), getHDF5Type(int), nil);

            dset_id = C_HDF5.H5Dopen(fileId, "/%s/segments".format(group).c_str(), C_HDF5.H5P_DEFAULT);

            attrSpaceId = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
            attr_id = C_HDF5.H5Acreate2(dset_id, "ObjType".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            var t: ObjType = ObjType.PDARRAY;
            var t_int: int = t: int;
            C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(t_int));
            C_HDF5.H5Aclose(attr_id);
            C_HDF5.H5Sclose(attrSpaceId);
            C_HDF5.H5Dclose(dset_id);
        }
    }

    /**
     * Writes the given Stings component array to HDF5 within a group.
     * :arg fileId: HDF5 file id
     * :type fileId: int
     *
     * :arg group: parent dataset / group name to write designated component
     * :type group: string
     *
     * :arg component: name of the component to write, should be either values or segments
     * :type component: string
     *
     * :arg items: the array containing the data to be written for te specified Strings array component
     * :type items: [] ?etype
     */
    private proc writeStringsComponentToHdf(fileId: int, group: string, component: string, items: [] ?etype) throws {
        C_HDF5.H5LTmake_dataset_WAR(fileId, '/%s/%s'.format(group, component).c_str(), 1,
                c_ptrTo([items.size:uint(64)]), getHDF5Type(etype), c_ptrTo(items));

        var dset_id: C_HDF5.hid_t = C_HDF5.H5Dopen(fileId, '/%s/%s'.format(group, component).c_str(), C_HDF5.H5P_DEFAULT);

        // Create the attribute space
        var attrSpaceId: C_HDF5.hid_t = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        var attr_id: C_HDF5.hid_t;

        // Create the objectType. This will be important when merging with other read/write functionality.
        attr_id = C_HDF5.H5Acreate2(dset_id, "ObjType".c_str(), getHDF5Type(int), attrSpaceId, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        var t: ObjType = ObjType.PDARRAY;
        var t_int: int = t: int;
        C_HDF5.H5Awrite(attr_id, getHDF5Type(int), c_ptrTo(t_int));
        C_HDF5.H5Sclose(attrSpaceId);
        C_HDF5.H5Aclose(attr_id);
        C_HDF5.H5Dclose(dset_id);
    }

    /*
        Process and write an Arkouda Strings (SegmentedString) object to HDF5.
    */
    proc strings_tohdfMsg(msgArgs: MessageArgs, st: borrowed SymTab) throws {
        use C_HDF5.HDF5_WAR;
        var mode: int = msgArgs.get("write_mode").getIntValue();

        var filename: string = msgArgs.getValueOf("filename");
        var file_format = msgArgs.get("file_format").getIntValue();
        var group = msgArgs.getValueOf("dset");
        var writeOffsets = msgArgs.get("save_offsets").getBoolValue();

        var entry:SegStringSymEntry = toSegStringSymEntry(st.lookup(msgArgs.getValueOf("values")));
        var segString = new SegString("", entry);

        const objType = msgArgs.getValueOf("objType");

        select file_format {
            when SINGLE_FILE {
                var f = prepFiles(filename, mode);
                var file_id = C_HDF5.H5Fopen(f.c_str(), C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
                if file_id < 0 { // HF5open returns negative value on failure
                    C_HDF5.H5Fclose(file_id);
                    var errorMsg = "Failure accessing file %s.".format(f);
                    throw getErrorWithContext(
                           msg=errorMsg,
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="FileNotFoundError");
                }

                // create the group
                validateGroup(file_id, f, group);

                //localize values and write dataset
                var localVals = new lowLevelLocalizingSlice(segString.values.a, 0..#segString.values.size);
                var val_dims: [0..#1] C_HDF5.hsize_t = [segString.values.size:C_HDF5.hsize_t];
                C_HDF5.H5LTmake_dataset(file_id, "/%s/values".format(group).c_str(), 1:c_int, val_dims, getHDF5Type(uint(8)), localVals.ptr);
                
                if (writeOffsets) {
                    //localize offsets and write dataset
                    var localOffsets = new lowLevelLocalizingSlice(segString.offsets.a, 0..#segString.size);
                    var off_dims: [0..#1] C_HDF5.hsize_t = [segString.offsets.size:C_HDF5.hsize_t];
                    C_HDF5.H5LTmake_dataset(file_id, "/%s/segments".format(group).c_str(), 1:c_int, off_dims, getHDF5Type(int), localOffsets.ptr);
                }
                writeArkoudaMetaData(file_id, group, objType, getHDF5Type(uint(8)));
                C_HDF5.H5Fclose(file_id);
            }
            when MULTI_FILE {
                var filenames = prepFiles(filename, mode, segString.offsets.a);

                ref ss = segString;
                var A = ss.offsets.a;
                const lastOffset = A[A.domain.high];
                const lastValIdx = ss.values.a.domain.high;

                // For each locale gather the string bytes corresponding to the offsets in its local domain
                coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with (ref ss) do on loc {
                    /*
                    * Generate metadata such as file name, file id, and dataset name
                    * for each file to be written
                    */
                    const f = filenames[idx];
                    var file_id = C_HDF5.H5Fopen(f.c_str(), C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
                    defer { // Close the file on exit
                        C_HDF5.H5Fclose(file_id);
                    }
                    const locDom = A.localSubdomain();
                    var dims: [0..#1] C_HDF5.hsize_t;
                    dims[0] = locDom.size: C_HDF5.hsize_t;

                    // create the group
                    validateGroup(file_id, f, group);

                    if (locDom.isEmpty() || locDom.size <= 0) { // shouldn't need the second clause, but in case negative number is returned
                        // Case where num_elements < num_locales, we need to write a nil into this locale's file
                        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "write1DDistStringsAggregators: locale.id %i has empty locDom.size %i, will get empty dataset."
                            .format(loc.id, locDom.size));
                        writeNilStringsGroupToHdf(file_id, group, writeOffsets);
                        // write attributes for arkouda meta info
                        writeArkoudaMetaData(file_id, group, objType, getHDF5Type(uint(8)));
                    } else {
                        var localOffsets = A[locDom];
                        var startValIdx = localOffsets[locDom.low];
                        /*
                        * The locale's last offset is the START idx of its last string, but we need to know where the END of it is located.
                        * thus...
                        * If this slice is the tail of the offsets, we set our endValIdx to the last index in the bytes/values array.
                        * Else get the next offset value and back up one to get the ending position of the last string we are responsible for
                        */
                        var endValIdx = if (lastOffset == localOffsets[locDom.high]) then lastValIdx else A[locDom.high + 1] - 1;
                        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "locale %i, writing strings offsets[%i..%i] corresponding to values[%i..%i], lastValIdx %i"
                            .format(loc.id, locDom.low, locDom.high, startValIdx, endValIdx, lastValIdx));
                        
                        var valIdxRange = startValIdx..endValIdx;
                        var localVals: [valIdxRange] uint(8);
                        ref olda = ss.values.a;
                        forall (localVal, valIdx) in zip(localVals, valIdxRange) with (var agg = newSrcAggregator(uint(8))) {
                            // Copy the remote value at index position valIdx to our local array
                            agg.copy(localVal, olda[valIdx]); // in SrcAgg, the Right Hand Side is REMOTE
                        }

                        // localVals is now a local copy of the gathered string bytes, write that component to HDF5
                        writeStringsComponentToHdf(file_id, group, "values", localVals);
                        
                        if (writeOffsets) { // if specified write the offsets component to HDF5
                            // Re-zero offsets so local file is zero based see also fixupSegBoundaries performed during read
                            localOffsets = localOffsets - startValIdx;
                            writeStringsComponentToHdf(file_id, group, "segments", localOffsets);
                        }
                        // write attributes for arkouda meta info
                        writeArkoudaMetaData(file_id, group, objType, getHDF5Type(uint(8)));
                    }
                }
            }
            otherwise {
                throw getErrorWithContext(
                           msg="Unknown file format. Expecting 0 (single file) or 1 (file per locale). Found %i".format(file_format),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
            }
        }
    }

    /*
        Parse and exectue tohdf message.
        Determines the type of the object to be written and calls the corresponding write functionality.
    */
    proc tohdfMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var objType: ObjType = msgArgs.getValueOf("objType").toUpper(): ObjType; // pdarray, Strings, ArrayView

        select objType {
            when ObjType.ARRAYVIEW {
                // call handler for arrayview write msg
                arrayView_tohdfMsg(msgArgs, st);
            }
            when ObjType.PDARRAY {
                // call handler for pdarray write
                pdarray_tohdfMsg(msgArgs, st);
            }
            when ObjType.STRINGS {
                // call handler for strings write
                strings_tohdfMsg(msgArgs, st);
            }
            otherwise {
                var errorMsg = "Unable to write object type %s to HDF5 file.".format(objType);
                h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        var repMsg: string = "Dataset written successfully!";
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
     * Returns boolean indicating whether the file is a valid HDF5 file.
     * Note: if the file cannot be opened due to permissions, throws
     * a PermissionError
     */
    proc isHdf5File(filename : string) : bool throws {
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
        C_HDF5.H5Literate(fid, C_HDF5.H5_INDEX_NAME, C_HDF5.H5_ITER_NATIVE, idx_p, c_ptrTo(_get_item_count), c_ptrTo(nfields));
        
        // Allocate space for array of strings
        var c_field_names = c_calloc(c_char, 255 * nfields);
        idx_p = 0:C_HDF5.hsize_t; // reset our iteration counter
        C_HDF5.H5Literate(fid, C_HDF5.H5_INDEX_NAME, C_HDF5.H5_ITER_NATIVE, idx_p, c_ptrTo(_simulate_h5ls), c_field_names);
        var pos = c_strlen(c_field_names):int;
        var items = createStringWithNewBuffer(c_field_names, pos, pos+1);
        c_free(c_field_names);
        return items;
    }

    proc lshdfMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string;

        // Retrieve filename from payload
        var filename: string = msgArgs.getValueOf("filename");
        if filename.isEmpty() {
            var errorMsg = "Filename was Empty";
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        // If the filename represents a glob pattern, retrieve the locale 0 filename
        if isGlobPattern(filename) {
            // Attempt to interpret filename as a glob expression and ls the first result
            var tmp = glob(filename);
            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "glob-expanded filename: %s to size: %i files".format(filename, tmp.size));

            if tmp.size <= 0 {
                var errorMsg = "Cannot retrieve filename from glob expression %s, check file name or format".format(filename);
                h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            
            // Set filename to globbed filename corresponding to locale 0
            filename = tmp[tmp.domain.first];
        }
        
        // Check to see if the file exists. If not, return an error message
        if !exists(filename) {
            var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filename);
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg,MsgType.ERROR);
        } 

        if !isHdf5File(filename) {
            var errorMsg = "File %s is not an HDF5 file".format(filename);
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg,MsgType.ERROR);
        }
        
        try {

            var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
            defer { C_HDF5.H5Fclose(file_id); } // ensure file is closed
            repMsg = simulate_h5ls(file_id);
            var items = new list(repMsg.split(",")); // convert to json

            repMsg = "%jt".format(items);
        } catch e : Error {
            var errorMsg = "Failed to process HDF5 file %t".format(e.message());
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
     *  Get the subdomains of the distributed array represented by each file, 
     *  as well as the total length of the array. 
     */
    proc get_subdoms(filenames: [?FD] string, dsetName: string, validFiles: [] bool) throws {
        use CTypes;

        var lengths: [FD] int;
        var skips = new set(string); // Case where there is no data in the file for this dsetName
        for (i, filename, isValid) in zip(FD, filenames, validFiles) {
            try {
                // if file had and error, it should be skipped.
                if !isValid {
                    skips.add(filename);
                    h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Adding invalid file to skips, %s".format(filename));
                    continue;
                }
                var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                           C_HDF5.H5P_DEFAULT);
                defer { // Close the file on exit
                    C_HDF5.H5Fclose(file_id);
                }

                var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now

                // Read array length into dims[0]
                C_HDF5.HDF5_WAR.H5LTget_dataset_info_WAR(file_id, dsetName.c_str(), 
                                           c_ptrTo(dims), nil, nil);
                lengths[i] = dims[0]: int;
                if lengths[i] == 0 {
                    skips.add(filename);
                    h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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

    /* 
        Write data from HDF5 dataset into a distributed array.
        This function gets called when A is a BlockDist or DefaultRectangular array. 
    */
    proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), 
                                                 filenames: [FD] string, dsetName: string, skips: set(string)) throws 
        where (MyDmap == Dmap.blockDist || MyDmap == Dmap.defaultRectangular)
    {
        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "entry.a.targetLocales() = %t".format(A.targetLocales()));
        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "Filedomains: %t".format(filedomains));
        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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
                    h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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
                                try! dataset = C_HDF5.H5Dopen(file_id, dsetName.localize().c_str(), C_HDF5.H5P_DEFAULT);
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

                            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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

    /*
        Determine if the dataset contains boolean values
    */
    proc isBoolDataset(filename: string, dset: string): bool throws {
        var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                           C_HDF5.H5P_DEFAULT);
        defer { // Close the file on exit
            C_HDF5.H5Fclose(file_id);
        }
        var boolDataset: bool;
        try {
            var dset_id: C_HDF5.hid_t = C_HDF5.H5Dopen(file_id, dset.c_str(), C_HDF5.H5P_DEFAULT);
            var isBool: int;
            if C_HDF5.H5Aexists_by_name(dset_id, ".".c_str(), "isBool", C_HDF5.H5P_DEFAULT) > 0 {
                var isBool_id: C_HDF5.hid_t = C_HDF5.H5Aopen_by_name(dset_id, ".".c_str(), "isBool", C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
                C_HDF5.H5Aread(isBool_id, getHDF5Type(int), c_ptrTo(isBool));
                boolDataset = if isBool == 1 then true else false;
            }
            else{
                boolDataset = false;
            }
            C_HDF5.H5Dclose(dset_id);
        } catch e: Error {
            /*
             * If there's an actual error, print it here. :TODO: revisit this
             * catch block after confirming the best way to handle HDF5 error
             */
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),
                        "checking if isBoolDataset %t with file %s".format(e.message()));
        }
        return boolDataset;
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
        if(1 == a.size) { // short circuit case where we only have one string/segment
            return;
        }
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
                h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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
        Read an ArrayView object from the files provided into a distributed array
    */
    proc arrayView_readhdfMsg(filenames: [?fD] string, dset: string, dataclass, bytesize: int, isSigned: bool, validFiles: [] bool, st: borrowed SymTab): (string, string, string) throws {
        var subdoms: [fD] domain(1);
        var skips = new set(string);
        var len: int;
        (subdoms, len, skips) = get_subdoms(filenames, dset, validFiles);

        var file_id = C_HDF5.H5Fopen(filenames[0].c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                           C_HDF5.H5P_DEFAULT);
        var dset_id: C_HDF5.hid_t = C_HDF5.H5Dopen(file_id, dset.c_str(), C_HDF5.H5P_DEFAULT);

        // check if rank is attr and then get.
        var rank: int;
        if C_HDF5.H5Aexists_by_name(dset_id, ".".c_str(), "Rank", C_HDF5.H5P_DEFAULT) > 0 {
            var rank_id: C_HDF5.hid_t = C_HDF5.H5Aopen_by_name(dset_id, ".".c_str(), "Rank", C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            var attr_type: C_HDF5.hid_t = C_HDF5.H5Aget_type(rank_id);
            C_HDF5.H5Aread(rank_id, getHDF5Type(int), c_ptrTo(rank));
        }
        else{
            // Return error that file does not have required attrs
            var errorMsg = "Rank Attribute was not located in %s. This attribute is required to process multi-dimensional data.".format(filenames[0]);
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw getErrorWithContext(
                             msg=errorMsg,
                             lineNumber=getLineNumber(), 
                             routineName=getRoutineName(), 
                             moduleName=getModuleName(), 
                             errorClass='AttributeNotFoundError');
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
            var errorMsg = "Shape Attribute was not located in %s. This attribute is required to process multi-dimensional data.".format(filenames[0]);
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw getErrorWithContext(
                             msg=errorMsg,
                             lineNumber=getLineNumber(), 
                             routineName=getRoutineName(), 
                             moduleName=getModuleName(), 
                             errorClass='AttributeNotFoundError');
        }

        C_HDF5.H5Dclose(dset_id);
        C_HDF5.H5Fclose(file_id);
        
        var sname = st.nextName();
        st.addEntry(sname, new shared SymEntry(shape));
        select dataclass {
            when C_HDF5.H5T_INTEGER {
                // identify the index of the first valid file
                var (v, idx) = maxloc reduce zip(validFiles, validFiles.domain);
                if (!isSigned && 8 == bytesize) {
                    var entryUInt = new shared SymEntry(len, uint);
                    h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Initialized uint entry for dataset %s".format(dset));
                    read_files_into_distributed_array(entryUInt.a, subdoms, filenames, dset, skips);
                    var rname = st.nextName();
                    if isBoolDataset(filenames[idx], dset) {
                        var entryBool = new shared SymEntry(len, bool);
                        entryBool.a = entryUInt.a:bool;
                        st.addEntry(rname, entryBool);
                    } else {
                        // Not a boolean dataset, so add original SymEntry to SymTable
                        st.addEntry(rname, entryUInt);
                    }
                    st.addEntry(rname, entryUInt);
                    return (dset, "ArrayView", "%s+%s".format(rname, sname));
                }
                else {
                    var entryInt = new shared SymEntry(len, int);
                    h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Initialized int entry for dataset %s".format(dset));
                    read_files_into_distributed_array(entryInt.a, subdoms, filenames, dset, skips);
                    var rname = st.nextName();
                    if isBoolDataset(filenames[idx], dset) {
                        var entryBool = new shared SymEntry(len, bool);
                        entryBool.a = entryInt.a:bool;
                        st.addEntry(rname, entryBool);
                    } else {
                        // Not a boolean dataset, so add original SymEntry to SymTable
                        st.addEntry(rname, entryInt);
                    }
                    return (dset, "ArrayView", "%s+%s".format(rname, sname));
                }
            }
            when C_HDF5.H5T_FLOAT {
                var entryReal = new shared SymEntry(len, real);
                h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                    "Initialized float entry");
                read_files_into_distributed_array(entryReal.a, subdoms, filenames, dset, skips);
                var rname = st.nextName();
                st.addEntry(rname, entryReal);
                return (dset, "ArrayView", "%s+%s".format(rname, sname));
            }
            otherwise {
                var errorMsg = "detected unhandled datatype: objType? ArrayView, class %i, size %i, " +
                                "signed? %t".format(dataclass, bytesize, isSigned);
                h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                throw getErrorWithContext(
                            msg=errorMsg,
                            lineNumber=getLineNumber(), 
                            routineName=getRoutineName(), 
                            moduleName=getModuleName(), 
                            errorClass='UnhandledDatatypeError');
            }
        }
    }

    /*
        Read an pdarray object from the files provided into a distributed array
    */
    proc pdarray_readhdfMsg(filenames: [?fD] string, dset: string, dataclass, bytesize: int, isSigned: bool, validFiles: [] bool, st: borrowed SymTab): (string, string, string) throws {
        var subdoms: [fD] domain(1);
        var skips = new set(string);
        var len: int;
        (subdoms, len, skips) = get_subdoms(filenames, dset, validFiles);
        select dataclass {
            when C_HDF5.H5T_INTEGER {
                // identify the index of the first valid file
                var (v, idx) = maxloc reduce zip(validFiles, validFiles.domain);
                if (!isSigned && 8 == bytesize) {
                    var entryUInt = new shared SymEntry(len, uint);
                    h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Initialized uint entry for dataset %s".format(dset));
                    read_files_into_distributed_array(entryUInt.a, subdoms, filenames, dset, skips);
                    var rname = st.nextName();
                    if isBoolDataset(filenames[idx], dset) {
                        var entryBool = new shared SymEntry(len, bool);
                        entryBool.a = entryUInt.a:bool;
                        st.addEntry(rname, entryBool);
                    } else {
                        // Not a boolean dataset, so add original SymEntry to SymTable
                        st.addEntry(rname, entryUInt);
                    }
                    return (dset, "pdarray", rname);
                }
                else {
                    var entryInt = new shared SymEntry(len, int);
                    h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Initialized int entry for dataset %s".format(dset));
                    read_files_into_distributed_array(entryInt.a, subdoms, filenames, dset, skips);
                    var rname = st.nextName();
                    if isBoolDataset(filenames[idx], dset) {
                        var entryBool = new shared SymEntry(len, bool);
                        entryBool.a = entryInt.a:bool;
                        st.addEntry(rname, entryBool);
                    } else {
                        // Not a boolean dataset, so add original SymEntry to SymTable
                        st.addEntry(rname, entryInt);
                    }
                    return (dset, "pdarray", rname);
                }
            }
            when C_HDF5.H5T_FLOAT {
                var entryReal = new shared SymEntry(len, real);
                h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                                    "Initialized float entry");
                read_files_into_distributed_array(entryReal.a, subdoms, filenames, dset, skips);
                var rname = st.nextName();
                st.addEntry(rname, entryReal);
                return (dset, "pdarray", rname);
            }
            otherwise {
                var errorMsg = "detected unhandled datatype: objType? pdarray, class %i, size %i, " +
                                "signed? %t".format(dataclass, bytesize, isSigned);
                h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                throw getErrorWithContext(
                            msg=errorMsg,
                            lineNumber=getLineNumber(), 
                            routineName=getRoutineName(), 
                            moduleName=getModuleName(), 
                            errorClass='UnhandledDatatypeError');
            }
        }
    }

    /*
        Read an strings object from the files provided into a distributed array
    */
    proc strings_readhdfMsg(filenames: [?fD] string, dset: string, dataclass, bytesize: int, isSigned: bool, calcStringOffsets: bool, validFiles: [] bool, st: borrowed SymTab): (string, string, string) throws {
        var subdoms: [fD] domain(1);
        var segSubdoms: [fD] domain(1);
        var skips = new set(string);
        var len: int;
        var nSeg: int;
        if (!calcStringOffsets) {
            (segSubdoms, nSeg, skips) = get_subdoms(filenames, dset + "/" + SEGSTRING_OFFSET_NAME, validFiles);
        }
        (subdoms, len, skips) = get_subdoms(filenames, dset + "/" + SEGSTRING_VALUE_NAME, validFiles);

        if (bytesize != 1) || isSigned {
            var errorMsg = "Error: detected unhandled datatype: objType? SegString, class %i, size %i, signed? %t".format(
                                    dataclass, bytesize, isSigned);
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw getErrorWithContext(
                            msg=errorMsg,
                            lineNumber=getLineNumber(), 
                            routineName=getRoutineName(), 
                            moduleName=getModuleName(), 
                            errorClass='UnhandledDatatypeError');
        }

        // Load the strings bytes/values first
        var entryVal = new shared SymEntry(len, uint(8));
        read_files_into_distributed_array(entryVal.a, subdoms, filenames, dset + "/" + SEGSTRING_VALUE_NAME, skips);

        proc _buildEntryCalcOffsets(): shared SymEntry throws {
            var offsetsArray = segmentedCalcOffsets(entryVal.a, entryVal.a.domain);
            return new shared SymEntry(offsetsArray);
        }

        proc _buildEntryLoadOffsets() throws {
            var offsetsEntry = new shared SymEntry(nSeg, int);
            read_files_into_distributed_array(offsetsEntry.a, segSubdoms, filenames, dset + "/" + SEGSTRING_OFFSET_NAME, skips);
            fixupSegBoundaries(offsetsEntry.a, segSubdoms, subdoms);
            return offsetsEntry;
        }

        var entrySeg = if (calcStringOffsets || nSeg < 1 || !skips.isEmpty()) then _buildEntryCalcOffsets() else _buildEntryLoadOffsets();

        var stringsEntry = assembleSegStringFromParts(entrySeg, entryVal, st);
        return (dset, "seg_string", "%s+%t".format(stringsEntry.name, stringsEntry.nBytes));
    }

    /*
        Reads the ObjType attribute from a given object. 
        Returns the string representation
    */
    proc getObjType(file_id: C_HDF5.hid_t, dset: string): ObjType throws {
        var obj_id: C_HDF5.hid_t;

        obj_id = C_HDF5.H5Oopen(file_id, dset.c_str(), C_HDF5.H5P_DEFAULT);
        if obj_id < 0 {
            throw getErrorWithContext(
                           msg="Dataset, %s, not found.".format(dset),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }

        var objType_int: int = -1;
        if C_HDF5.H5Aexists_by_name(obj_id, ".".c_str(), "ObjType", C_HDF5.H5P_DEFAULT) > 0 {
            var objType_id: C_HDF5.hid_t = C_HDF5.H5Aopen_by_name(obj_id, ".".c_str(), "ObjType", C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            C_HDF5.H5Aread(objType_id, getHDF5Type(int), c_ptrTo(objType_int));
            C_HDF5.H5Aclose(objType_id);
        }
        else{
            // work around to handle old formats that do not store meta data.
            // It is assumed that any objects in this case are storing strings or pdarray
            if C_HDF5.H5Lexists(obj_id, "/values".c_str(), C_HDF5.H5P_DEFAULT) > 0{
                // this means that the obj is a group and contains a strings obj
                objType_int = ObjType.STRINGS: int;
            }
            else {
                objType_int = ObjType.PDARRAY: int;
            }
        }
        // Close the open hdf5 objects
        C_HDF5.H5Oclose(obj_id);
        return objType_int:ObjType;
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

    /*
        Access information about the dataset in the given file.
        Used to detect errors when reading
    */
    proc get_info(filename: string, dsetName: string, calcStringOffsets: bool) throws {
        // Verify that the file exists
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

        if C_HDF5.H5Lexists(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT) <= 0 {
            C_HDF5.H5Fclose(file_id);
            throw getErrorWithContext(
                 msg="The dataset %s does not exist in the file %s".format(dsetName, 
                                                filename),
                 lineNumber=getLineNumber(), 
                 routineName=getRoutineName(), 
                 moduleName=getModuleName(), 
                 errorClass='DatasetNotFoundError');
        }

        var objType: ObjType;
        var dataclass: C_HDF5.H5T_class_t;
        var bytesize: int;
        var isSigned: bool;
        try {
            objType = getObjType(file_id, dsetName);
            if objType == ObjType.STRINGS {
                if ( !calcStringOffsets ) {
                    var offsetDset = dsetName + "/" + SEGSTRING_OFFSET_NAME;
                    var (offsetClass, offsetByteSize, offsetSign) = 
                                            try get_dataset_info(file_id, offsetDset);
                    if (offsetClass != C_HDF5.H5T_INTEGER) {
                        throw getErrorWithContext(
                            msg="dataset %s has incorrect one or more sub-datasets" +
                            " %s %s".format(dsetName,SEGSTRING_OFFSET_NAME,SEGSTRING_VALUE_NAME), 
                            lineNumber=getLineNumber(),
                            routineName=getRoutineName(),
                            moduleName=getModuleName(),
                            errorClass='SegStringError');                    
                    }
                }
                var valueDset = dsetName + "/" + SEGSTRING_VALUE_NAME;
                try (dataclass, bytesize, isSigned) = 
                                           try get_dataset_info(file_id, valueDset);           
            } else {
                (dataclass, bytesize, isSigned) = get_dataset_info(file_id, dsetName);
            }
        } catch e : Error {
            //:TODO: recommend revisiting this catch block 
            throw getErrorWithContext( 
                msg="in get_info %s".format(e.message()), 
                lineNumber=getLineNumber(),
                routineName=getRoutineName(),
                moduleName=getModuleName(),
                errorClass='Error');
        }
        C_HDF5.H5Fclose(file_id);
        return (objType, dataclass, bytesize, isSigned);
    }

    /*
        Read HDF5 files into an Arkouda Object
    */
    proc readAllHdfMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var strictTypes: bool = msgArgs.get("strict_types").getBoolValue();

        var allowErrors: bool = msgArgs.get("allow_errors").getBoolValue(); // default is false
        if allowErrors {
            h5Logger.warn(getModuleName(), getRoutineName(), getLineNumber(), "Allowing file read errors");
        }

        var calcStringOffsets: bool = msgArgs.get("calc_string_offsets").getBoolValue(); // default is false
        if calcStringOffsets {
            h5Logger.warn(getModuleName(), getRoutineName(), getLineNumber(),
                "Calculating string array offsets instead of reading from HDF5");
        }

        var ndsets = msgArgs.get("dset_size").getIntValue();
        var dsetlist: [0..#ndsets] string;
        try {
            dsetlist = msgArgs.get("dsets").getList(ndsets);
        } catch {
            // limit length of dataset names to 2000 chars
            var n: int = 1000;
            var jsondsets = msgArgs.getValueOf("dsets");
            var dsets: string = if jsondsets.size > 2*n then jsondsets[0..#n]+'...'+jsondsets[jsondsets.size-n..#n] else jsondsets;
            var errorMsg = "Could not decode json dataset names via tempfile (%i files: %s)".format(
                                                ndsets, dsets);
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var nfiles = msgArgs.get("filename_size").getIntValue();
        var filelist: [0..#nfiles] string;
        try {
            filelist = msgArgs.get("filenames").getList(nfiles);
        } catch {
            // limit length of file names to 2000 chars
            var n: int = 1000;
            var jsonfiles = msgArgs.getValueOf("filenames");
            var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
            var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, files);
            h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        var filedom = filelist.domain;
        var filenames: [filedom] string;

        if filelist.size == 1 {
            if filelist[0].strip().size == 0 {
                var errorMsg = "filelist was empty.";
                h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            var tmp = glob(filelist[0]);
            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "glob expanded %s to %i files".format(filelist[0], tmp.size));
            if tmp.size == 0 {
                var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
                h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
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
        
        var objTypeList: [filedom] ObjType;
        var dclasses: [filedom] C_HDF5.hid_t;
        var bytesizes: [filedom] int;
        var signFlags: [filedom] bool;
        var validFiles: [filedom] bool = true;
        var rtnData: list((string, string, string));
        var fileErrors: list(string);
        var fileErrorCount:int = 0;
        var fileErrorMsg:string = "";
        const AK_META_GROUP = ARKOUDA_HDF5_FILE_METADATA_GROUP(1..ARKOUDA_HDF5_FILE_METADATA_GROUP.size-1); // strip leading slash
        for dsetName in dsetlist do {
            if dsetName == AK_META_GROUP { // Legacy code to ignore meta group. Meta data no longer in group
                continue;
            }
            for (i, fname) in zip(filedom, filenames) {
                var hadError = false;
                try {
                    (objTypeList[i], dclasses[i], bytesizes[i], signFlags[i]) = get_info(fname, dsetName, calcStringOffsets);
                } catch e: FileNotFoundError {
                    fileErrorMsg = "File %s not found".format(fname);
                    h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: PermissionError {
                    fileErrorMsg = "Permission error %s opening %s".format(e.message(),fname);
                    h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: DatasetNotFoundError {
                    fileErrorMsg = "Dataset %s not found in file %s".format(dsetName,fname);
                    h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: NotHDF5FileError {
                    fileErrorMsg = "The file %s is not an HDF5 file: %s".format(fname,e.message());
                    h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e: SegStringError {
                    fileErrorMsg = "SegmentedString error: %s".format(e.message());
                    h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                } catch e : Error {
                    fileErrorMsg = "Other error in accessing file %s: %s".format(fname,e.message());
                    h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),fileErrorMsg);
                    hadError = true;
                    if !allowErrors { return new MsgTuple(fileErrorMsg, MsgType.ERROR); }
                }

                if hadError {
                    // Keep running total, but we'll only report back the first 10
                    if fileErrorCount < 10 {
                        fileErrors.append(fileErrorMsg.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip());
                    }
                    fileErrorCount += 1;
                    validFiles[i] = false;
                }
            }

            
            // identify the index of the first valid file
            var (v, idx) = maxloc reduce zip(validFiles, validFiles.domain);
            const objType = objTypeList[idx];
            const dataclass = dclasses[idx];
            const bytesize = bytesizes[idx];
            const isSigned = signFlags[idx];
            for (isValid, name, ot, dc, bs, sf) in zip(validFiles, filenames, objTypeList, dclasses, bytesizes, signFlags) {
                if isValid {
                    if (ot != objType) {
                        var errorMsg = "Inconsistent objecttype in dataset %s of file %s. Expected: %s, Found: %s".format(dsetName, name, objType:string, ot:string);
                        h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                    else if (dc != dataclass) {
                        var errorMsg = "Inconsistent dtype in dataset %s of file %s".format(dsetName, name);
                        h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    } else if (strictTypes && ((bs != bytesize) || (sf != isSigned))) {
                        var errorMsg = "Inconsistent precision or sign in dataset %s of file %s\nWith strictTypes, mixing of precision and signedness not allowed (set strictTypes=False to suppress)".format(dsetName, name);
                        h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Verified all dtypes across files for dataset %s".format(dsetName));

            select objType {
                when ObjType.ARRAYVIEW {
                    rtnData.append(arrayView_readhdfMsg(filenames, dsetName, dataclass, bytesize, isSigned, validFiles, st));
                }
                when ObjType.PDARRAY {
                    rtnData.append(pdarray_readhdfMsg(filenames, dsetName, dataclass, bytesize, isSigned, validFiles, st));
                }
                when ObjType.STRINGS {
                    rtnData.append(strings_readhdfMsg(filenames, dsetName, dataclass, bytesize, isSigned, calcStringOffsets, validFiles, st));
                }
                otherwise {
                    var errorMsg = "Unkwown object type found";
                    h5Logger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
        }
        if allowErrors && fileErrorCount > 0 {
            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "allowErrors:true, fileErrorCount:%t".format(fileErrorCount));
        }
        var repMsg: string = _buildReadAllMsgJson(rtnData, allowErrors, fileErrorCount, fileErrors, st);
        h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg,MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("lshdf", lshdfMsg, getModuleName());
    registerFunction("readAllHdf", readAllHdfMsg, getModuleName());
    registerFunction("tohdf", tohdfMsg, getModuleName());
}
