module ArkoudaRegexCompat {
  import Regex.regex as chapelRegex;
  import Regex.regexMatch;
  import Regex.compile;

  record regex {
    type eltType;
    const pattern: eltType;

    proc init(pat: ?t) {
      eltType = t;
      pattern = pat;
    }
    
    proc match(name) throws {
      const cp = compile(pattern);
      return cp.match(name);
    }
    proc search(text):regexMatch throws {
      const cp = compile(pattern);
      return cp.search(text);
    }
    iter matches(text, param captures=0) throws {
      const cp = compile(pattern);
      for match in cp.matches(text, captures) {
        yield match;
      }
    }
  }
}