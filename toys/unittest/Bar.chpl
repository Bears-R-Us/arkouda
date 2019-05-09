module Bar {
  proc bar(x: int) {
    writeln("In Bar.bar(", x, ")");
  }

  proc main() {
    bar(2);
    bar(-2);
  }
}
