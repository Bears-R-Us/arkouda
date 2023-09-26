module ArkoudaBlockCompat {
  use BlockDist;

  type blockDist = Block;

  proc BlockDom.distribution {
    return this.dist;
  }
}
