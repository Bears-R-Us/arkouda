module GenSymIO {
    use HDF5;
    use Time only;
    use IO;
    use CPtr;
    use Path;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use FileSystem;
    use Sort;
    use NumPyDType;
    use List;
    use Map;
    use PrivateDist;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use ServerConfig;
    use Search;
    use IndexingMsg;

    import ak;
    
    private config const logLevel = ServerConfig.logLevel;
    const gsLogger = new Logger(logLevel);

    config const GenSymIO_DEBUG = false;
    config const NULL_STRINGS_VALUE = 0:uint(8);
    config const TRUNCATE: int = 0;
    config const APPEND: int = 1;

    /*
     * Creates a pdarray server-side and returns the SymTab name used to
     * retrieve the pdarray from the SymTab.
     */
    proc arrayMsg(cmd: string, payload: bytes, st: borrowed SymTab): MsgTuple throws {
        // Set up our return items
        var msgType = MsgType.NORMAL;
        var msg:string = "";
        var rname:string = "";

        var (dtypeBytes, sizeBytes, data) = payload.splitMsgToTuple(b" ", 3);
        var dtype = DType.UNDEF;
        var size:int;
        try {
            dtype = str2dtype(dtypeBytes.decode());
            size = sizeBytes:int;
        } catch {
            var errorMsg = "Error parsing/decoding either dtypeBytes or size";
            gsLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        overMemLimit(2*8*size);

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

        if (MsgType.ERROR != msgType) {  // success condition
            // Set up return message indicating SymTab name corresponding to new pdarray
            msg = "created " + st.attrib(rname);
            gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),msg);
        }
        return new MsgTuple(msg, msgType);
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
     * Indicates whether the filename represents a glob expression as opposed to
     * an specific filename
     */
    proc isGlobPattern(filename: string): bool throws {
        return filename.endsWith("*");
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
        
        var exitCode: int;

        try {
            if exists(tmpfile) {
                remove(tmpfile);
            }

            var cmd = try! "h5ls \"%s\" > \"%s\"".format(filename, tmpfile);
            var sub = spawnshell(cmd);

            sub.wait();

            exitCode = sub.exit_status;
            
            var f = open(tmpfile, iomode.r);
            defer {  // This will ensure we try to close f when we exit the proc scope.
                ak.io.ensureClose(f);
                try { remove(tmpfile); } catch {}
            }
            var r = f.reader(start=0);
            r.readstring(repMsg);
            r.close();
        } catch e : Error {
            var errorMsg = "failed to spawn process and execute ls: %t".format(e.message());
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if exitCode != 0 {
            var errorMsg = "could not execute ls on %s, check file permissions or format".format(filename);
            gsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        } else {
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
    }

    /*
     * Delegate call to our specific hdf5 module
     */
    proc readAllHdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        return ak.io.GenHdf5.readAllHdfMsg(cmd, payload, st);
    }

    proc tohdfMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {               
        var (arrayName, dsetName, modeStr, jsonfile, 
                                      dataType, segsName) = payload.splitMsgToTuple(6);

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
                    /*
                     * Look up the values and segments arrays, both of which are needed to write
                     * uint8 arrays such as Strings out to external systems.
                     */
                    var e = toSymEntry(entry, uint(8));
                    var segsEntry = st.lookup(segsName);                   
                    var s_e = toSymEntry(segsEntry, int);
                    warnFlag = write1DDistStrings(filename, mode, dsetName, e.a, DType.UInt8,s_e.a);
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
                                                                array_type: DType, SA) throws {
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
             generateStringsMetadata(idx,shuffleLeftIndices, shuffleRightIndices, 
                          isSingleString, endsWithCompleteString, charArraySize, A, SA);
        }

        t1.stop();  
        var elapsed = t1.elapsed();
        gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "Time for generating all values metadata: %.17r".format(elapsed));   
                                       
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
                     'locale %i does not end with null char, need left or right shuffle'.format(
                                              idx));

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
                     charList = new list(adjustForRightShuffle(
                                                  shuffleRightIndex,charList));
                     gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        'adjusted locale %i for right shuffle to locale %i'.format(
                                        idx,idx+1));
                 } else {
                     charList.append(NULL_STRINGS_VALUE);
                     gsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        'no adjustment for right shuffle from locale %i to locale %i'.format(
                                        idx,idx+1));        
                 }
                 
                 // Generate the segments list now that the char list is finalized
                 var segmentsList = generateFinalSegmentsList(charList,idx);
             
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
                     var segmentsList = generateFinalSegmentsList(charList,idx);
 
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
                      var segmentsList = generateFinalSegmentsList(charList,idx);

                      // Write the finalized valuesList and segmentsList to the hdf5 group
                      writeStringsToHdf(myFileID, idx, group, charList, segmentsList);
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
            H5LTmake_dataset_WAR(myFileID, myDsetName.c_str(), 1, c_ptrTo(dims),
                                      dType, c_ptrTo(A.localSlice(locDom)));
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
        var groupId = C_HDF5.H5Gcreate2(fileId, "/%s".format(group).c_str(),
              C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        C_HDF5.H5Gclose(groupId);
    }
    
    /*
     * Writes the values and segments lists to hdf5 within a group.
     */
    private proc writeStringsToHdf(fileId: int, idx: int, group: string, 
                              valuesList: list(uint(8)), segmentsList: list(int)) throws {
        // initialize timer
        var t1: Time.Timer;
        if logLevel == LogLevel.DEBUG {
            t1 = new Time.Timer();
            t1.clear();
            t1.start();
        }

        H5LTmake_dataset_WAR(fileId, '/%s/values'.format(group).c_str(), 1,
                     c_ptrTo([valuesList.size:uint(64)]), getHDF5Type(uint(8)),
                            c_ptrTo(valuesList.toArray()));

        H5LTmake_dataset_WAR(fileId, '/%s/segments'.format(group).c_str(), 1,
                     c_ptrTo([segmentsList.size:uint(64)]),getHDF5Type(int),
                           c_ptrTo(segmentsList.toArray()));

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
