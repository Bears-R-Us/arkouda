module ArkoudaCTypesCompat {
  public use CTypes;
  type c_string_ptr = c_string;
  type c_ptr_void = c_void_ptr;
}
