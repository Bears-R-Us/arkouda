module ServerRegistration {
  proc doRegister() {
    import KExtremeMsg;
    KExtremeMsg.registerMe();
    import ArraySetopsMsg;
    ArraySetopsMsg.registerMe();
  }
}
