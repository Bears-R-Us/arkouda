module ArkoudaFileCompat {
  use IO;
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
    return open(path, oldMode);
  }

  record fileReader {
    type t;

    proc lines() {
      var ret = new itemReaderInternal(string, iokind.dynamic, true, t);
      return ret;
    }
  }
  proc channel.bytesRead(ref a, b) throws {
    this.readbytes(a,b);
  }
}
