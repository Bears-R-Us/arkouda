module ArkoudaListCompat {
  use List;

  proc ref list.pushBack(x: this.eltType): int {
    return this.append(x);
  }

  proc ref list.pushBack(other: list(this.eltType)) {
    this.append(other);
  }

  proc ref list.pushBack(other: [] this.elType) {
    this.append(other);
  }

  proc ref list.pushBack(other: range(this.eltType, ?)) {
    this.append(other);
  }

  proc ref list.popBack(): this.elType {
    return this.pop();
  }

  proc ref list.replace(i: int, x: this.eltType): bool {
    return this.set(i, x);
  }
}
