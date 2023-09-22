module ArkoudaCTypesCompat {
  public use CTypes;

  private proc offset_ARRAY_ELEMENTS {
    extern const CHPL_RT_MD_ARRAY_ELEMENTS:chpl_mem_descInt_t;
    pragma "fn synchronization free"
    extern proc chpl_memhook_md_num(): chpl_mem_descInt_t;
    return CHPL_RT_MD_ARRAY_ELEMENTS - chpl_memhook_md_num();
  }
  
  inline proc allocate(type eltType, size: c_size_t, clear: bool = false,
                       alignment: c_size_t = 0) : c_ptr(eltType) {
    const alloc_size = size * c_sizeof(eltType);
    const aligned : bool = (alignment != 0);
    var ptr : c_void_ptr = nil;

    // pick runtime allocation function based on requested zeroing + alignment
    if (!aligned) {
      if (clear) {
        // normal calloc
        ptr = chpl_here_calloc(alloc_size, 1, offset_ARRAY_ELEMENTS);
      } else {
        // normal malloc
        ptr = chpl_here_alloc(alloc_size, offset_ARRAY_ELEMENTS);
      }
    } else {
      // check alignment, size restriction
      // Alignment of 0 is our sentinel value for no specified alignment,
      // so no need to check for it.
      if boundsChecking {
        use Math;
        var one:c_size_t = 1;
        // Round the alignment up to the nearest power of 2
        var p = log2(alignment); // power of 2 rounded down
        // compute alignment rounded up
        if (one << p) < alignment then
          p += 1;
        assert(alignment <= (one << p));
        if alignment != (one << p) then
          halt("allocate called with non-power-of-2 alignment ", alignment);
        if alignment < c_sizeof(c_void_ptr) then
          halt("allocate called with alignment smaller than pointer size");
      }

      // normal aligned alloc, whether we clear after or not
      ptr = chpl_here_aligned_alloc(alignment, alloc_size,
                                    offset_ARRAY_ELEMENTS);

      if (clear) {
        // there is no aligned calloc; have to aligned_alloc + memset to 0
        c_memset(ptr, 0, alloc_size);
      }
    }

    return ptr : c_ptr(eltType);
  }

  inline proc deallocate(data: c_void_ptr) {
    chpl_here_free(data);
  }

  type c_string_ptr = c_string;
}
