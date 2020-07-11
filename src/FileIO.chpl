

module FileIO {
    use IO;
    use FileSystem;
    use Map;
    use Path;

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
}
