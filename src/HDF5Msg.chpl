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

    const ARKOUDA_HDF5_FILE_VERSION_VAL = 1.0:real(32);

    config const TRUNCATE: int = 0;
    config const APPEND: int = 1;

    config const SINGLE_FILE: int = 0;
    config const MULTI_FILE: int = 1;

    require "c_helpers/help_h5ls.h", "c_helpers/help_h5ls.c";
    private extern proc c_get_HDF5_obj_type(loc_id:C_HDF5.hid_t, name:c_string, obj_type:c_ptr(C_HDF5.H5O_type_t)):C_HDF5.herr_t;
    private extern proc c_strlen(s:c_ptr(c_char)):c_size_t;
    private extern proc c_incrementCounter(data:c_void_ptr);
    private extern proc c_append_HDF5_fieldname(data:c_void_ptr, name:c_string);

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

    proc prepFiles_Single(filenames: [] string, matchingFilenames: [] string, mode: int) throws {
        // validate the write mode
        validateWriteMode(mode);

        const filename = filenames[0;]

        var fileExists: bool = filename == matchingFilenames[0];
        if (mode == TRUNCATE || (mode == APPEND && !filesExist) {
            if (mode == TRUNCATE && fileExists){
                remove(filename);
            }

            h5Logger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "Creating or truncating file");

            // create the file
            var file_id: C_HDF5.hid_t = C_HDF5.H5Fcreate(filename.c_str(), C_HDF5.H5F_ACC_TRUNC, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
            defer { // Close file upon exiting scope
                C_HDF5.H5Fclose(file_id);
            }

            if file_id < 0 { // Negative file_id means error
                throw getErrorWithContext(msg="The file %s cannot be created".format(filename),
                                            lineNumber=getLineNumber(), 
                                            routineName=getRoutineName(), 
                                            moduleName=getModuleName(), 
                                            errorClass='FileNotFoundError');
            }
        }
    }

    proc prepFiles_Distributed(filenames: [] string, matchingFilenames: [] string, mode: int, A) throws {
        // validate the write mode
        validateWriteMode(mode);

        var filesExist: bool = matchingFilenames.size > 0;

        if (mode == TRUNCATE || (mode == APPEND && !filesExist) {
            coforall loc in A.targetLocales() do on loc {
                var file_id: C_HDF5.hid_t;
                var fn = filenames[loc.id].localize();
                if mode == TRUNCATE and glob(fn).size == 1{
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
            if filesnames.size != matchingFilenames.size {
                throw getErrorWithContext(
                           msg="Cannot append when the number of existing filenames does not match the expected.",
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
            }
        }
    }

    proc prepFiles(filename, mode: int, file_format: int, A): [] string throws {
        var prefix: string;
        var extension: string;
        (prefix,extension) = getFileMetadata(filename);

        select file_format{
            when SINGLE_FILE {
                var filenames: [0..#1] string = ["%s%s".format(prefix, extension)];
                var matchingFilenames = glob("%s*%s".format(prefix, extension));
                prepFiles_Single(filenames, matchingFilenames, mode);
                return filenames;
            }
            when MULTI_FILE {
                var filenames: [0..#targetLocalesSize] string;
                forall i in 0..#targetLocalesSize {
                    filenames[i] = generateFilename(prefix, extension, i);
                }
                fioLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "generateFilenames targetLocales.size %i, filenames.size %i".format(targetLocalesSize, filenames.size));

                var matchingFilenames = glob("%s_LOCALE*%s".format(prefix, extension));
                prepFiles_Distributed(filenames, matchingFilenames, mode, A);
                return filenames;
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

    proc validateDataset(file_id: C_HDF5.hid_t, filename: string, dset_name: string) throws {
        // validate that the dataset does not already exist
        var dset_exists: int = C_HDF5.H5Lexists(file_id, dset_name.c_str(), C_HDF5.H5P_DEFAULT);
        if dset_exists > 0 {
            throw getErrorWithContext(
                           msg="A dataset named %s already exists in %s. Overwriting is not currently supported. Please choose another name.".format(dset_name, filename),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }
        else if dset_exists < -1 {
            throw getErrorWithContext(
                           msg="Failure validating the status of dataset named %s.".format(dset_name),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="IllegalArgumentError");
        }
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

        // always store multidimensional arrays as flattened array
        var dims: [0..#1] C_HDF5.hsize_t = (* reduce shape.a):C_HDF5.hsize_t;
        C_HDF5.H5LTmake_dataset(file_id, dset_name.c_str(), 1:c_int, dims, dtype_id, A.ptr);
    }

    proc writeArkoudaVersionMeta(dset_id: C_HDF5.hid_t) {
        // Create the attribute space
        var attrSpaceId: C_HDF5.hid_t = C_HDF5.H5Screate(C_HDF5.H5S_SCALAR);
        var attr_id: C_HDF5.hid_t;

        var attrFileVersionType = getHDF5Type(ARKOUDA_HDF5_FILE_VERSION_TYPE);
        var attrId = C_HDF5.H5Acreate2(dset_id,
                          ARKOUDA_HDF5_FILE_VERSION_KEY.c_str(),
                          attrFileVersionType,
                          attrSpaceId,
                          C_HDF5.H5P_DEFAULT,
                          C_HDF5.H5P_DEFAULT);
        
        // H5Awrite requires a pointer and we have a const, so we need a variable ref we can turn into a pointer
        var fileVersion = ARKOUDA_HDF5_FILE_VERSION_VAL;
        C_HDF5.H5Awrite(attrId, attrFileVersionType, c_ptrTo(fileVersion));
        C_HDF5.H5Aclose(attrId);

        var attrStringType = C_HDF5.H5Tcopy(C_HDF5.H5T_C_S1): C_HDF5.hid_t;
        C_HDF5.H5Tset_size(attrStringType, arkoudaVersion.size:uint(64) + 1); // ensure space for NULL terminator
        C_HDF5.H5Tset_strpad(attrStringType, C_HDF5.H5T_STR_NULLTERM);
        
        attrId = C_HDF5.H5Acreate2(dset_id,
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
    }

    proc writeMultiDimArrayAttrs(file_id: C_HDF5.hid_t, dset_name: string, shape: SymEntry) {
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
        writeArkoudaVersionMeta(dset_id);
    }

    proc multiDimArray_tohdfMsg(msgArgs: MessageArgs, st: borrowed SymTab) throws {
        // access integer representation of APPEND/TRUNCATE
        var mode: int = msgArgs.get("write_mode").getIntValue();

        var filename: string = msgArgs.getValueOf("filename");
        var entry = st.lookup(msgArgs.getValueOf("values"));
        var file_format = msgArgs.get("file_format").getIntValue();

        var filenames = prepFiles(filename, mode, file_format);
        var dset_name = msgArgs.getValueOf("dset");

        var shape_sym: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("shape"), st);
        var shape = toSymEntry(shape_sym, int);

        select file_format {
            when SINGLE_FILE {
                var file_id = C_HDF5.H5Fopen(filenames[0], C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
                if file_id < 0 { // HF5open returns negative value on failure
                    C_HDF5.H5Fclose(file_id);
                    var errorMsg = "Failure accessing file %s.".format(filename);
                    h5tLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);     
                }

                // validate that the dataset does not already exist
                validateDataset(file_id, filenames[0], dset_name);
                
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
                //TODO - write the attributes to the dataset
            }
            when MULTI_FILE {
                coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) do on loc {
                    const localeFilename = filenames[idx];
                    h5tLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
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
                        H5LTmake_dataset_WAR(file_id, dset_name.c_str(), 1, c_ptrTo(dims), dType, nil);
                    } else {
                        H5LTmake_dataset_WAR(file_id, dset_name.c_str(), 1, c_ptrTo(dims), dType, c_ptrTo(A.localSlice(locDom)));
                    }
                    // TODO - write the attributes to the dataset
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

    proc tohdfMsg(cmd: string, payload: string, argSize: int, st: borrowed SymTab): MsgTuple throws {
        var msgArgs = parseMessageArgs(payload, argSize);

        var objType: string = msgArgs.getValueOf("objType"); // pdarray, Strings, ArrayView

        select objType.upper() {
            when "ARRAYVIEW" {
                // call handler for arrayview write msg
                multiDimArray_tohdfMsg(msgArgs, st);
            }
            when "PDARRAY" {
                // TODO - call handler for pdarray write msg
            }
            when "STRINGS" {
                // TODO - call handler for SegString write msg
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

    use CommandMap;
    // TODO - lshdf and readhdf messages
    registerFunction("tohdf", tohdfMsg, getModuleName());
}