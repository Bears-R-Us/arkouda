module UInt128 {
  use IO;
  use Sort;

  // 128-bit unsigned integer as two 64-bit limbs
  record UInt128 {
    var hi: uint(64);
    var lo: uint(64);

    proc init() {
      this.hi = 0;
      this.lo = 0;
    }

    proc init(x: uint(64)) {
      this.hi = 0;
      this.lo = x;
    }

    proc init(hi: uint(64), lo: uint(64)) {
      this.hi = hi;
      this.lo = lo;
    }

    //
    // Custom printing: write as hex without using BigInteger.
    // Example output: 0x1f2a... (no leading zeros beyond what’s needed).
    //
    proc writeThis(f) throws {
      // Special-case zero so we don't print an empty string.
      if this.hi == 0 && this.lo == 0 {
        f.write("0");
        return;
      }

      // Hex digits as strings so we don't need 'char' type.
      const hexDigits: [0..15] string =
        ("0", "1", "2", "3", "4", "5", "6", "7",
        "8", "9", "a", "b", "c", "d", "e", "f");

      var s: string = "0x";
      var started = false;

      // We have 128 bits -> 32 hex nibbles (0 = most significant).
      for k in 0..31 {
        var nibble: uint(64);

        if k < 16 {
          // High 64 bits
          const shiftAmt = (15:uint(64) - k:uint(64)) * 4:uint(64);
          nibble = (this.hi >> shiftAmt) & 0xf:uint(64);
        } else {
          // Low 64 bits
          const shiftAmt = (31:uint(64) - k:uint(64)) * 4:uint(64);
          nibble = (this.lo >> shiftAmt) & 0xf:uint(64);
        }

        if nibble == 0 && !started {
          // Skip leading zeros
          continue;
        }

        started = true;
        s += hexDigits[nibble:int];
      }

      if !started {
        // All bits were zero (defensive, though we early-returned above)
        s += "0";
      }

      f.write(s);
    }

  }

  // Alias if you like primitive-ish spelling
  type uint128 = UInt128;

  // ===== Bit shifting helpers =====

  // Internal: left shift by an unsigned 64-bit amount
  inline proc _shiftLeft(a: UInt128, amt: uint(64)): UInt128 {
    if amt == 0:uint(64) {
      return a;
    } else if amt < 64:uint(64) {
      const hi = (a.hi << amt) | (a.lo >> (64:uint(64) - amt));
      const lo = a.lo << amt;
      return new UInt128(hi, lo);
    } else if amt == 64:uint(64) {
      return new UInt128(a.lo, 0:uint(64));
    } else if amt < 128:uint(64) {
      const s = amt - 64:uint(64);
      const hi = a.lo << s;
      return new UInt128(hi, 0:uint(64));
    } else {
      // Shift by >= 128 bits -> zero
      return new UInt128(0:uint(64), 0:uint(64));
    }
  }

  // Internal: logical right shift by an unsigned 64-bit amount
  inline proc _shiftRight(a: UInt128, amt: uint(64)): UInt128 {
    if amt == 0:uint(64) {
      return a;
    } else if amt < 64:uint(64) {
      const hi = a.hi >> amt;
      const lo = (a.lo >> amt) | (a.hi << (64:uint(64) - amt));
      return new UInt128(hi, lo);
    } else if amt == 64:uint(64) {
      return new UInt128(0:uint(64), a.hi);
    } else if amt < 128:uint(64) {
      const s = amt - 64:uint(64);
      const lo = a.hi >> s;
      return new UInt128(0:uint(64), lo);
    } else {
      // Shift by >= 128 bits -> zero
      return new UInt128(0:uint(64), 0:uint(64));
    }
  }

  // ----- Public operators: UInt128 << shift_count -----

  // Shift by unsigned count (uint / uint(64))
  operator <<(a: UInt128, b: uint): UInt128 {
    return _shiftLeft(a, b: uint(64));
  }

  // Shift by signed count (int / int(64)); negative is a bug
  operator <<(a: UInt128, b: int): UInt128 {
    if b < 0 then
      halt("uint128 << negative shift count");
    return _shiftLeft(a, b: uint(64));
  }

  // ----- Public operators: UInt128 >> shift_count -----

  // Logical right shift by unsigned count
  operator >>(a: UInt128, b: uint): UInt128 {
    return _shiftRight(a, b: uint(64));
  }

  // Logical right shift by signed count; negative is a bug
  operator >>(a: UInt128, b: int): UInt128 {
    if b < 0 then
      halt("uint128 >> negative shift count");
    return _shiftRight(a, b: uint(64));
  }

  // ===== Comparisons =====

  operator ==(a: UInt128, b: UInt128): bool {
    return a.hi == b.hi && a.lo == b.lo;
  }

  operator !=(a: UInt128, b: UInt128): bool {
    return !(a == b);
  }

  operator <(a: UInt128, b: UInt128): bool {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
  }

  operator >(a: UInt128, b: UInt128): bool {
    return b < a;
  }

  operator <=(a: UInt128, b: UInt128): bool {
    return !(b < a);
  }

  operator >=(a: UInt128, b: UInt128): bool {
    return !(a < b);
  }

  // ===== Simple casts for interoperability (optional but handy) =====

  proc _cast(type t: UInt128, x: uint(64)) {
    return new UInt128(x);
  }

  proc _cast(type t: uint(64), x: UInt128) {
    return x.lo;
  }
}
