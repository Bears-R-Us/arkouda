module Baz {
  use Foo;  // Baz depends on foo()
  
  proc baz() {
    writeln("In Baz.baz(), calling foo():");
    foo();
  }

  proc main() {
    baz();
  }
}
