module ArkoudaRegexCompat {
  public import Regex.regex;

  // Previous releases of Chapel do not support throwing
  // initializers and to work around that, we had to run
  // `compile` for each call to match, rather than only
  // at regex creation, so here we are updating 1.30 to
  // keep the old behavior around so that performance doesn't
  // take a hit for previous releases
  proc compile(const pattern) throws {
    return new regex(pattern);
  }
}
