module CommPrimitives {
  use CTypes;
  use Communication;
  public import Communication.get as GET;
  public import Communication.put as PUT;

  inline proc getAddr(const ref p): c_ptr(p.type) {
    // TODO can this use c_ptrTo?
    return __primitive("_wide_get_addr", p): c_ptr(p.type);
  }
}
