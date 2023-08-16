module ArkoudaMapCompat {
  public use Map;

  proc map.addOrReplace(in k: keyType, in v: valType) {
    this.addOrSet(k, v);
  }
}
