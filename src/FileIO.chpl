

module FileIO {
    use IO;
    use GenSymIO;
    use FileSystem;
    use Map;
    use Path;
    use Reflection;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrors;
    use Sort;

    use ServerConfig, Logging, CommandMap;
    private config const logLevel = ServerConfig.logLevel;
    const fioLogger = new Logger(logLevel);
    
    enum FileType {HDF5, ARROW, PARQUET, UNKNOWN};

    proc appendFile(filePath : string, line : string) throws {
        var writer : channel;
        if exists(filePath) {
            var aFile = try! open(filePath, iomode.rw);
            writer = try! aFile.writer(start=aFile.size);

        } else {
              var aFile = try! open(filePath, iomode.cwr);
              writer = try! aFile.writer();
        }

        writer.writeln(line);
        writer.flush();
        writer.close();
    }

    proc getLineFromFile(filePath : string, lineIndex : int=-1) : string throws {
        var aFile = try! open(filePath, iomode.rw);
        var lines = try! aFile.lines();
        var line : string;
        var returnLine : string;
        var i = 1;

        for line in lines do {
            returnLine = line;
            if i == lineIndex {
                break;
            } else {
                i+=1;
            }
        }

        return returnLine.strip();
    }

    proc delimitedFileToMap(filePath : string, delimiter : string=',') : map {
        var fileMap : map(keyType=string, valType=string, parSafe=false) = 
                         new map(keyType=string,valType=string,parSafe=false);
        var aFile = try! open(filePath, iomode.rw);
        var lines = try! aFile.lines();
        var line : string;
        for line in lines do {
            const values = line.split(delimiter);
            fileMap.add(values[0], values[1]);
        }
        return fileMap;
    }

    proc initDirectory(filePath : string) throws {
        if !isDir(filePath) {
           try! mkdir(name=filePath, parents=true);
        } 
    }

    /*
     * Ensure the file is closed, disregard errors
     */
    proc ensureClose(tmpf:file): bool {
        var success = true;
        try {
            tmpf.close();
        } catch {
            success = false;
        }
        return success;
    }

    /*
     * Indicates whether the filename represents a glob expression as opposed to
     * an specific filename
     */
    proc isGlobPattern(filename: string): bool throws {
        return filename.endsWith("*");
    }

    /*
     * Generates a list of filenames to be written to based upon a file prefix,
     * extension, and number of locales.
     */
    proc generateFilenames(prefix : string, extension : string, targetLocalesSize:int) : [] string throws { 
        // Generate the filenames based upon the number of targetLocales.
        var filenames: [0..#targetLocalesSize] string;
        for i in 0..#targetLocalesSize {
            filenames[i] = generateFilename(prefix, extension, i);
        }
        fioLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "generateFilenames targetLocales.size %i, filenames.size %i".format(targetLocalesSize, filenames.size));

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
     * Generates an array of filenames to be matched in APPEND mode and to be
     * checked in TRUNCATE mode that will warn the user that 1..n files are
     * being overwritten.
     */
    proc getMatchingFilenames(prefix : string, extension : string) throws {
        return glob("%s_LOCALE*%s".format(prefix, extension));    
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

    // File Magic headers for supported formats
    const MAGIC_PARQUET:bytes = b"\x50\x41\x52\x31"; // 4 bytes "PAR1"
    const MAGIC_HDF5:bytes = b"\x89\x48\x44\x46\x0d\x0a\x1a\x0a"; // 8 bytes "\211HDF\r\n\032\n"
    const MAGIC_ARROW:bytes = b"\x41\x52\x52\x4F\x57\x31\x00\x00"; // 6 bytes "ARROW1" padded to 8 with nulls

    /* Determine FileType based on public File Magic for supported types
      :arg header: file header
      :type header: bytes

      :returns: FileType from enum.FileType
    */
    proc getFileTypeByMagic(header:bytes): FileType {
        var t = FileType.UNKNOWN;
        var length = header.size;
        if (length >= 4 && MAGIC_PARQUET == header.this(0..3)) {
            t = FileType.PARQUET;
        } else if (length >= 8 && MAGIC_HDF5 == header.this(0..7)) {
            t = FileType.HDF5;
        } else if (length >= 8 && MAGIC_ARROW == header.this(0..7)) {
            t = FileType.ARROW;
        }
        return t;
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

    proc getFirstEightBytesFromFile(path:string):bytes throws {
        var f:file = open(path, iomode.r);
        var reader = f.reader(kind=ionative);
        var header:bytes;
        if (reader.binary()) {
          reader.readbytes(header, 8);
        } else {
          throw getErrorWithContext(
                     msg="File reader was not in binary mode",
                     getLineNumber(),
                     getRoutineName(),
                     getModuleName(),
                     errorClass="IOError");
        }
        try {
          f.close();
        } catch e {
          throw getErrorWithContext(
                     msg=e:string,
                     getLineNumber(),
                     getRoutineName(),
                     getModuleName(),
                     errorClass="IOError");
        }
        return header;
    }

    proc getFileType(filename: string) throws {
      return getFileTypeByMagic(getFirstEightBytesFromFile(filename));
    }

    proc getFileTypeMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
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
        return new MsgTuple(errorMsg, MsgType.ERROR);                                    
      }

      // If the filename represents a glob pattern, retrieve the locale 0 filename
      if isGlobPattern(filename) {
        // Attempt to interpret filename as a glob expression and ls the first result
        var tmp = glob(filename);

        if tmp.size <= 0 {
          var errorMsg = "Cannot retrieve filename from glob expression %s, check file name or format".format(filename);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
            
        // Set filename to globbed filename corresponding to locale 0
        filename = tmp[tmp.domain.first];
      }

      if !exists(filename) {
        var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filename);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      } 

      select getFileType(filename) {
        when FileType.HDF5 {
          return new MsgTuple("HDF5", MsgType.NORMAL);
        }
        when FileType.PARQUET {
          return new MsgTuple("Parquet", MsgType.NORMAL);
        } otherwise {
          var errorMsg = "Unsupported file type; Parquet and HDF5 are only supported formats";
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }

    proc lsAnyMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
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
        return new MsgTuple(errorMsg, MsgType.ERROR);                                    
      }

      // If the filename represents a glob pattern, retrieve the locale 0 filename
      if isGlobPattern(filename) {
        // Attempt to interpret filename as a glob expression and ls the first result
        var tmp = glob(filename);

        if tmp.size <= 0 {
          var errorMsg = "Cannot retrieve filename from glob expression %s, check file name or format".format(filename);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
            
        // Set filename to globbed filename corresponding to locale 0
        filename = tmp[tmp.domain.first];
      }

      if !exists(filename) {
        var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filename);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      } 

      select getFileType(filename) {
        when FileType.HDF5 {
          return executeCommand("lshdf", payload, st);
        }
        when FileType.PARQUET {
          return executeCommand("lspq", payload, st);
        } otherwise {
          var errorMsg = "Unsupported file type; Parquet and HDF5 are only supported formats";
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }

    proc readAnyMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var (strictFlag, ndsetsStr, nfilesStr, allowErrorsFlag, calcStringOffsetsFlag, arraysStr) = payload.splitMsgToTuple(6);
      var (jsondsets, jsonfiles) = arraysStr.splitMsgToTuple(" | ",2);

      if (!checkCast(nfilesStr, int)) {
        var errMsg = "Number of files:`%s` could not be cast to an integer".format(nfilesStr);
        return new MsgTuple(errMsg, MsgType.ERROR);
      }
      var nfiles = nfilesStr:int; // Error checked above
      var filelist: [0..#nfiles] string;
      
      try {
        filelist = jsonToPdArray(jsonfiles, nfiles);
      } catch {
        var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
      
      var filedom = filelist.domain;
      var filenames: [filedom] string;

      if filelist.size == 1 {
        if filelist[0].strip().size == 0 {
          var errorMsg = "filelist was empty.";
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        var tmp = glob(filelist[0]);
        if tmp.size == 0 {
          var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        // Glob returns filenames in weird order. Sort for consistency
        sort(tmp);
        filedom = tmp.domain;
        filenames = tmp;
      } else {
        filenames = filelist;
      }

      if !exists(filenames[0]) {
        var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filenames[0]);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      } 

      select getFileType(filenames[0]) {
        when FileType.HDF5 {
          return executeCommand("readAllHdf", payload, st);
        }
        when FileType.PARQUET {
          return executeCommand("readAllParquet", payload, st);
        } otherwise {
          var errorMsg = "Unsupported file type; Parquet and HDF5 are only supported formats";
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }
}
