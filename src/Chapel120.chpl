//
// This module is designed to provide some backwards compatibility
// with Chapel 1.20 for arkouda.
//
module Chapel120 {
  //
  // TODO: Would be cool / smart if Chapel code could query compiler
  // version number directly to avoid needing this param and the
  // last resort overload below...  See Chapel issue #5491.
  //
  config param version120 = false;
  
  use ZMQ;
  use IO;

  proc Socket.recv(type T: bytes) throws where version120 {
    // there is no validation in 1.20, so use string for ZMQ but return bytes
    return this.recv(string):bytes;
  }

  proc Socket.send(msg: bytes) throws where version120 {
    // there is no validation in 1.20, so use string for ZMQ but return bytes
    this.send(convertToString(msg));
  }

  proc bytes.format(args ...?k): bytes throws where version120 {
    var s = convertToString(this);
    return s.format((...args)):bytes;
  }

  proc channel.readbytes(ref str_out:bytes, len:int(64) = -1):bool throws 
                                                               where version120 {
    var s: string;
    const ret = this.readstring(s, len);
    str_out = s:bytes;
    return ret;
  }

  private inline proc convertToString(b: bytes): string {
    return createStringWithBorrowedBuffer(b.buff, b.len, b._size);
  }

}
