module ArkoudaFileCompat {
  import IO.{openmem, iomode, fileReader as chapelFileReader, iokind, itemReaderInternal, file};
  import IO.open as fileOpen;
  enum ioMode {
    r = 1,
    cw = 2,
    rw = 3,
    cwr = 4,
  }

  proc openMemFile() throws {
    return openmem();
  }

  proc open(path: string, mode: ioMode) throws {
    var oldMode: iomode;
    select mode {
      when ioMode.r {
        oldMode = iomode.r;
      }
      when ioMode.cw {
        oldMode = iomode.cw;
      }
      when ioMode.rw {
        oldMode = iomode.rw;
      }
      when ioMode.cwr {
        oldMode = iomode.cwr;
      }
    }
    return fileOpen(path, oldMode);
  }

  record fileReader {
    type chapelFileReader;

    proc lines() {
      var ret = new itemReaderInternal(string, iokind.dynamic, true, chapelFileReader);
      return ret;
    }
  }
  proc chapelFileReader.bytesRead(ref a, b) throws {
    this.readbytes(a,b);
  }
}
