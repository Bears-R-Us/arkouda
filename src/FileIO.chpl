

module FileIO {
    use IO;
    use FileSystem;
    use Map;
    use Path;

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

}
