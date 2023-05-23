module ArkoudaListCompat {
  use List;

  proc list.pushBack(x: this.eltType): int {
    return this.append(x);
  }

  proc list.pushBack(other: list(this.eltType)) {
    this.append(other);
  }

  proc list.pushBack(other: [] this.elType) {
    this.append(other);
  }

  proc list.pushBack(other: range(this.eltType, ?)) {
    this.append(other);
  }

  proc list.popBack(): this.elType {
    return this.pop();
  }

  proc list.replace(i: int, x: this.eltType): bool {
    return this.set(i, x);
  }
}
