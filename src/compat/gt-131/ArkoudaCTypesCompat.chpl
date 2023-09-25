module ArkoudaCTypesCompat {
  // Use an only clause including everything except the deprecated c_void_ptr,
  // so we can define it as an alias for its replacement c_ptr(void).
  // Using an except clause would not work here as the reference to the
  // deprecated name c_void_ptr would warn.
  public use CTypes only c_float, c_double, cFileTypeHasPointer, c_FILE, c_ptr,
         c_ptrConst, c_array, c_ptrTo, c_ptrToConst, cPtrToLogicalValue,
         c_addrOf, c_addrOfConst, c_sizeof, c_offsetof, allocate, deallocate;
  // Need to manually use ChapelSysCTypes here, as it is `public use`d in CTypes
  // but doesn't get picked up by the above use with only clause.
  public use ChapelSysCTypes;

  // Redefine c_void_ptr to a usable form, without deprecation warning.
  type c_ptr_void = c_ptr(void);

  type c_string_ptr = c_ptrConst(c_char);
}
