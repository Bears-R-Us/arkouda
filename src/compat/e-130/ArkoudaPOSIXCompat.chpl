module ArkoudaPOSIXCompat {
  use CTypes;

  inline proc memcpy(dest:c_void_ptr, const src:c_void_ptr, n: c_size_t) {
    c_memcpy(dest, src, n);
  }
}
