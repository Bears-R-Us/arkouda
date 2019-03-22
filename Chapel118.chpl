//
// This module is designed to provide some backwards compatibility
// with Chapel 1.18 for arkouda.
//
module Chapel118 {
  //
  // TODO: Would be cool / smart if Chapel code could query compiler
  // version number directly to avoid needing this param and the
  // last resort overload below...  See Chapel issue #5491.
  //
  config param version118 = false;

  // If `version118` is set, forward .contains() on a domain to
  // .member()
  //
  proc _domain.contains(i) where version118 {
    return this.member(i);
  }

  //
  // This is a trick to provide a hint to someone using 1.18 (or
  // earlier) to throw the -sversion118 flag without breaking 1.19
  //
  pragma "last resort"
  proc _domain.contains(i) {
    compilerError("Couldn't find <domain>.contains(:"+i.type:string+")\n"+
                  "Maybe try recompiling with -sversion118?");
  }
}
