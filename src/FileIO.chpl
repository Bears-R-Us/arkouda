

module FileIO {
    use GenSymIO;
    use FileSystem;
    use Path;
    use Reflection;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrors;
    use Sort;
    use Map;
    use IOUtils;

    use ServerConfig, Logging, CommandMap;
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const fioLogger = new Logger(logLevel, logChannel);
    
    enum FileType {HDF5, ARROW, PARQUET, CSV, UNKNOWN};

    proc appendFile(filePath : string, line : string) throws {
        var writer;
        if exists(filePath) {
            var aFile = open(filePath, ioMode.rw);
            writer = aFile.writer(region=aFile.size.., locking=false);
        } else {
            var aFile = open(filePath, ioMode.cwr);
            writer = aFile.writer(locking=false);
        }

        writer.writeln(line);
        writer.flush();
        writer.close();
    }

    proc writeToFile(filePath : string, line : string) throws {
        var writer = openWriter(filePath);

        writer.writeln(line);
        writer.flush();
        writer.close();
    }
    
    proc writeLinesToFile(filePath : string, lines : string) throws {
        var writer = openWriter(filePath);

        for line in lines {
            writer.writeln(line);
        }
        writer.flush();
        writer.close();
    }

    proc getLineFromFile(filePath : string, lineIndex : int=-1) : string throws {
        var lines = openReader(filePath).lines();
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

    proc getLineFromFile(path: string, match: string) throws {
        var reader = openReader(path);
        var returnLine: string;

        for line in reader.lines() {
            if line.find(match) >= 0 {
                returnLine = line;
                break;
            }
        }

        return returnLine;
    }

    proc delimitedFileToMap(filePath : string, delimiter : string=',') : map {
        var fileMap : map(keyType=string, valType=string) = 
                         new map(keyType=string,valType=string);
        var aFile = try! open(filePath, ioMode.rw);
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
     * Delete files matching a prefix and following the pattern <prefix>_LOCALE*.
     */
    proc deleteMatchingFilenamesMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var prefix = msgArgs["prefix"].toScalar(string);
      var extension: string;
      (prefix, extension) = getFileMetadata(prefix);
      deleteMatchingFilenames(prefix, extension);

      var repMsg = "Files deleted successfully!";
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc deleteMatchingFilenames(prefix : string, extension : string) throws {
      const filenames = getMatchingFilenames(prefix, extension);
      forall filename in filenames{
        deleteFile(filename);
      }
    }

    proc deleteFile(filename: string) throws {
      try {
        remove(filename);
        fioLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
          "File %s has been deleted successfully.".format(filename));
      } catch e {
        fioLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
          "Error deleting file: %s".format(e.message()));
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

    // File Magic headers for supported formats
    const MAGIC_PARQUET:bytes = b"\x50\x41\x52\x31"; // 4 bytes "PAR1"
    const MAGIC_HDF5:bytes = b"\x89\x48\x44\x46\x0d\x0a\x1a\x0a"; // 8 bytes "\211HDF\r\n\032\n"
    const MAGIC_ARROW:bytes = b"\x41\x52\x52\x4F\x57\x31\x00\x00"; // 6 bytes "ARROW1" padded to 8 with nulls
    const MAGIC_CSV:bytes = b"\x2a\x2a\x48\x45\x41\x44\x45\x52"; // 8 bytes "**HEADER"

    /* Determine FileType based on public File Magic for supported types
      :arg header: file header
      :type header: bytes

      :returns: FileType from enum.FileType
    */
    proc getFileTypeByMagic(header:bytes): FileType {
        var t = FileType.UNKNOWN;
        var length = header.size;
        if (length >= 4 && MAGIC_PARQUET == header[0..3]) {
            t = FileType.PARQUET;
        } else if (length >= 8 && MAGIC_HDF5 == header[0..7]) {
            t = FileType.HDF5;
        } else if (length >= 8 && MAGIC_ARROW == header[0..7]) {
            t = FileType.ARROW;
        } else if (length >= 8 && MAGIC_CSV == header[0..7]) {
          t = FileType.CSV;
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
        if d1.strides==strideKind.one && d2.strides==strideKind.one {
            return {low..high};
        } else {
            var stride = max(d1.stride, d2.stride);
            return {low..high by stride};
        }
    }

    proc getFirstEightBytesFromFile(path:string):bytes throws {
        var f:file = open(path, ioMode.r);
        var reader = f.reader(locking=false, deserializer=new binarySerializer(endian=endianness.native));
        var header:bytes;
        if reader.deserializerType == binarySerializer {
          reader.readBytes(header, 8);
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
                     msg="%?".format(e),
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

    proc getFileTypeMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var filename = msgArgs.getValueOf("filename");

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
        }
        when FileType.CSV {
          return new MsgTuple("CSV", MsgType.NORMAL);
        } otherwise {
          var errorMsg = "Unsupported file type; Parquet, HDF5 and CSV are only supported formats";
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }

    proc lsAnyMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      // Retrieve filename from payload
      var filename: string = msgArgs.getValueOf("filename");
      if filename.isEmpty() {
        var errorMsg = "Empty Filename Provided.";
        fioLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
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
      fioLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "FILENAME: %s".format(filename));
      if !exists(filename) {
        var errorMsg = "File %s does not exist in a location accessible to Arkouda".format(filename);
        return new MsgTuple(errorMsg,MsgType.ERROR);
      } 

      select getFileType(filename) {
        when FileType.HDF5 {
          return executeCommand("lshdf", msgArgs, st);
        }
        when FileType.PARQUET {
          return executeCommand("lspq", msgArgs, st);
        } when FileType.CSV {
          return executeCommand("lscsv", msgArgs, st);
        } otherwise {
          var errorMsg = "Unsupported file type; Parquet and HDF5 are only supported formats";
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }

    proc globExpansionMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var nfiles = msgArgs.get("file_count").getIntValue();
      var filelist: [0..#nfiles] string;
      
      // attempt to read file name list
      try {
          filelist = msgArgs.get("filenames").getList(nfiles);
      } catch {
          // limit length of file names to 2000 chars
          var n: int = 1000;
          var jsonfiles = msgArgs.getValueOf("filenames");
          var files: string = if jsonfiles.size > 2*n then jsonfiles[0..#n]+'...'+jsonfiles[jsonfiles.size-n..#n] else jsonfiles;
          var errorMsg = "Could not decode json filenames via tempfile (%i files: %s)".format(nfiles, files);
          fioLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }

      var filedom = filelist.domain;
      var filenames: [filedom] string;

      if filelist.size == 1 {
          if filelist[0].strip().size == 0 {
              var errorMsg = "filelist was empty.";
              fioLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
          }
          var tmp = glob(filelist[0]);
          fioLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "glob expanded %s to %i files".format(filelist[0], tmp.size));
          if tmp.size == 0 {
              var errorMsg = "The wildcarded filename %s either corresponds to files inaccessible to Arkouda or files of an invalid format".format(filelist[0]);
              fioLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
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
      return new MsgTuple(formatJson(filenames), MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("deleteMatchingFilenames", deleteMatchingFilenamesMsg, getModuleName());
}
