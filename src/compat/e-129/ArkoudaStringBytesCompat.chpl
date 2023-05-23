module ArkoudaStringBytesCompat {

  inline proc type
  string.createBorrowingBuffer(x: string): string {
    return createStringWithBorrowedBuffer(x);
  }

  inline proc type
  string.createBorrowingBuffer(x: c_string,
                               length=x.size)
                               : string throws {
    return createStringWithBorrowedBuffer(x, length);
  }

  inline proc type
  string.createBorrowingBuffer(x: c_ptr(?t),
                               length: int,
                               size: int
                               ): string throws {
    return createStringWithBorrowedBuffer(x, length, size);
  }

  inline proc type
  string.createAdoptingBuffer(x: c_string, length=x.size): string throws {
    return createStringWithOwnedBuffer(x, length);
  }

  inline proc type
  string.createAdoptingBuffer(x: c_ptr(?t),
                              length: int,
                              size: int
                              ): string throws {
    return createStringWithOwnedBuffer(x, length, size);
  }

  inline proc type
  string.createCopyingBuffer(x: c_string,
                             length=x.size,
                             policy=decodePolicy.strict
                             ): string throws {
    return createStringWithNewBuffer(x, length, policy);
  }

  inline proc type
  string.createCopyingBuffer(x: c_ptr(?t),
                             length: int,
                             size=length+1,
                             policy=decodePolicy.strict
                             ): string throws {
    return createStringWithNewBuffer(x, length, size, policy);
  }




  inline proc type
  bytes.createBorrowingBuffer(x: bytes): bytes {
    return createBytesWithBorrowedBuffer(x);
  }

  inline proc type
  bytes.createBorrowingBuffer(x: c_string,
                              length=x.size
                              ): bytes {
    return createBytesWithBorrowedBuffer(x, length);
  }

  inline proc type
  bytes.createBorrowingBuffer(x: c_ptr(?t),
                              length: int,
                              size: int
                              ): bytes {
    return createBytesWithBorrowedBuffer(x, length, size);
  }

  inline proc type
  bytes.createAdoptingBuffer(x: c_string, length=x.size): bytes {
    return createBytesWithOwnedBuffer(x, length);
  }

  inline proc type
  bytes.createAdoptingBuffer(x: c_ptr(?t), length: int, size: int): bytes {
    return createBytesWithOwnedBuffer(x, length, size);
  }

  inline proc type
  bytes.createCopyingBuffer(x: c_string, length=x.size): bytes {
    return createBytesWithNewBuffer(x, length);
  }

  inline proc type
  bytes.createCopyingBuffer(x: c_ptr(?t),
                            length: int,
                            size=length+1
                            ): bytes {
    return createBytesWithNewBuffer(x, length, size);
  }
  
}
