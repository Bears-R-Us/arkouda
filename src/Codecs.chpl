module Codecs {
  use idna;
  use iconv;

  use AryUtil;
  use CTypes;

  proc encodeStr(obj: c_ptr(uint(8)), inBufSize: int, outBufSize: int, toEncoding: string = "UTF-8", fromEncoding: string = "UTF-8"): [] uint(8) throws {
    if toEncoding == "IDNA" {
      if fromEncoding != "UTF-8" {
        throw new Error("Only encoding from UTF-8 is supported for IDNA encoding");
      } else if outBufSize == 1 {
        var ret: [0..0] uint(8) = 0x00;
        return ret;
      }
      var cRes: c_ptrConst(c_char);
      var rc = idn2_to_ascii_lz(obj:c_ptrConst(c_char), cRes, 0);
      if (rc != IDNA_SUCCESS) {
        throw new Error("Encode failed");
      }
      var tmp = cRes: c_ptr(uint(8));
      var ret: [0..#outBufSize] uint(8);
      for i in ret.domain do ret[i] = (tmp+i).deref();
      idn2_free(cRes: c_ptr(void));
      return ret;
    } else if fromEncoding == "IDNA" {
      if toEncoding != "UTF-8" {
        throw new Error("Only encoding to UTF-8 is supported for IDNA encoding");
      } else if outBufSize == 1 {
        var ret: [0..0] uint(8) = 0x00;
        return ret;
      }
      var cRes: c_ptrConst(c_char);
      var rc = idn2_to_unicode_8z8z(obj:c_ptrConst(c_char), cRes, 0);
      if (rc != IDNA_SUCCESS) {
        throw new Error("Decode failed");
      }
      var tmp = cRes: c_ptr(uint(8));
      var ret: [0..#outBufSize] uint(8);
      for i in ret.domain do ret[i] = (tmp+i).deref();
      idn2_free(cRes: c_ptr(void));
      return ret;
    } else {
      var cd = libiconv_open(toEncoding.c_str(), fromEncoding.c_str());
      if cd == (-1):libiconv_t then
        throw new Error("Unsupported encoding: " + toEncoding + " " + fromEncoding);
      var inBuf = obj:c_ptrConst(c_char);
      // Null terminator already accounted for
      var inSize = (inBufSize): c_size_t;

      var chplRes: [0..#(outBufSize+1)] uint(8);
      var outSize = (outBufSize+1): c_size_t;
      
      var outBuf = c_ptrTo(chplRes):c_ptrConst(c_char);

      if libiconv(cd, inBuf, inSize, outBuf, outSize) != 0 then
        throw new Error("Encoding to " + toEncoding + " failed");
      libiconv_close(cd);

      return chplRes;
    }
  }

    proc getBufLength(obj: c_ptr(uint(8)), inBufSize: int, toEncoding: string = "IDNA", fromEncoding: string = "UTF-8"): int throws {
      if toEncoding == "IDNA" {
        var cRes: c_ptrConst(c_char);
        var rc = idn2_to_ascii_lz(obj:c_ptrConst(c_char), cRes, 0);
        if (rc != IDNA_SUCCESS) {
          // Error condition, we just want this to be empty string
          idn2_free(cRes: c_ptr(void));
          return 1;
        }
        var tmp = bytes.createCopyingBuffer(cRes);
        idn2_free(cRes: c_ptr(void));
        return tmp.size+1;
      } else if fromEncoding == "IDNA" {
        // Check valid round trip characters
        var validChars = idn2_lookup_u8(obj, nil, 0);
        if validChars != 0 {
          return 1;
        }
        var cRes: c_ptrConst(c_char);
        var rc = idn2_to_unicode_8z8z(obj:c_ptrConst(c_char), cRes, 0);
        if (rc != IDNA_SUCCESS) {
          // Error condition, we just want this to be empty string
          idn2_free(cRes: c_ptr(void));
          return 1;
        }
        var tmp = bytes.createCopyingBuffer(cRes);
        idn2_free(cRes: c_ptr(void));
        return tmp.size+1;
      } else {
        var cd = libiconv_open(toEncoding.c_str(), fromEncoding.c_str());
        if cd == (-1):libiconv_t then
          throw new Error("Unsupported encoding: " + toEncoding + " " + fromEncoding);
        var inBuf = obj:c_ptrConst(c_char);
        // Add 1 for null terminator
        var inSize = (inBufSize): c_size_t;

        // This buffer must be 4x the size of the input
        // buffer, as that is the maximum number of bytes
        // that can be allocated for the output (which is
        // UTF-16, the others are smaller)
        var chplRes: [0..#(inBufSize*4)] uint(8) = 0x00;
        var origSize = chplRes.size;
        var outSize = chplRes.size: c_size_t;
      
        var outBuf = c_ptrTo(chplRes):c_ptrConst(c_char);

        if libiconv(cd, inBuf, inSize, outBuf, outSize) != 0 then
          throw new Error("Getting buf length for " + toEncoding + " failed");
        libiconv_close(cd);
      
        // For some encodings we have to handle the additional null
        // terminators that are added onto the end
        return (origSize:int-outSize:int-1);
      }
    }
}
