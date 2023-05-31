module ArkoudaListCompat {
  use List;

  proc ref list.pushBack(in x: this.eltType): int {
    return this.append(x);
  }

  proc ref list.pushBack(other: list(this.eltType)) lifetime this < other {
    this.append(other);
  }

  proc ref list.pushBack(other: [] this.eltType) lifetime this < other {
    this.append(other);
  }

  proc ref list.pushBack(other: range(this.eltType, ?)) lifetime this < other {
    this.append(other);
  }

  proc ref list.popBack(): this.eltType {
    return this.pop();
  }

  proc ref list.replace(i: int, x: this.eltType): bool {
    return this.set(i, x);
  }
}
