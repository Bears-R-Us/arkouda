module ArkoudaBlockCompat {
  use BlockDist;
  proc type Block.createDomain(D) {
    return newBlockDom(D);
  }
}