module UInt128 {
  use IO;
  use Sort;
  use BigInteger;
  use Math;

  // 128-bit unsigned integer as two 64-bit limbs
  record UInt128 {
    var hi: uint(64);
    var lo: uint(64);

    proc init() {
      this.hi = 0;
      this.lo = 0;
    }

    proc init=(x: UInt128) {
      this.hi = x.hi;
      this.lo = x.lo;
    }

    // from unsigned
    proc init=(x: uint(64)) {
      this.hi = 0:uint(64);
      this.lo = x;
    }

    // Two’s-complement wrap (sign-extend from 64 -> 128)
    proc init=(x: int(64)) {
      // sign extension: if x is negative, high limb becomes all 1s, else 0
      this.hi = if x < 0 then ~0:uint(64) else 0:uint(64);
      this.lo = x:uint(64); // preserves the raw low bits
    }

    proc init(x: uint(64)) {
      this.hi = 0;
      this.lo = x;
    }

    proc init(hi: uint(64), lo: uint(64)) {
      this.hi = hi;
      this.lo = lo;
    }

    // Allow `x: UInt128` casts from int(64)
    operator :(x: int(64), type t: UInt128): UInt128 {
      return new UInt128(x);
    }

    // Allow `x: UInt128` casts from uint(64)
    operator :(x: uint(64), type t: UInt128): UInt128 {
      return new UInt128(x);
    }

    operator :(x: bigint, type t: UInt128): UInt128 {
      const two64  = (1:bigint) << 64;
      const two128 = (1:bigint) << 128;
      const mask64  = two64  - 1;
      const mask128 = two128 - 1;

      // Wrap modulo 2^128 (works for negative too)
      const y = x & mask128;

      const lo = (y & mask64):uint(64);
      const hi = (y >> 64):uint(64);

      return new UInt128(hi, lo);
    }

    operator :(x: real(64), type t: UInt128): UInt128 {
      if !isFinite(x) then
        halt("cannot cast non-finite real to UInt128: ", x);

      const tx = trunc(x);
      if tx != x then
        halt("cannot cast non-integer real to UInt128: ", x);

      var b: bigint;
      b.set(tx);          // <-- key: convert real->bigint via set()
      return b:UInt128;   // uses your bigint->UInt128 operator
    }

    // UInt128 -> int(64) (wrap: take low limb, interpret as signed)
    operator :(x: UInt128, type t: int(64)): int(64) {
      return x.lo:int(64);
    }

    // UInt128 -> uint(w): wrap (take low bits)
    operator :(x: UInt128, type t: uint(?w)): uint(w) {
      return x.lo:uint(w);
    }

    operator :(x: UInt128, type t: real(64)): real(64) {
      return ldExp(x.hi:real(64), 64) + x.lo:real(64);
    }

    operator :(x: UInt128, type t: bigint): bigint {
      var b: bigint;
      // b = hi<<64 + lo
      b = (x.hi:bigint) << 64;
      b += x.lo:bigint;
      return b;
    }

    // UInt128 -> bool: true iff nonzero
    operator :(x: UInt128, type t: bool): bool {
      return (x.hi != 0:uint(64)) || (x.lo != 0:uint(64));
    }

    operator :(x: UInt128, type t: string): string {
      // Uses writeThis under the hood
      return "%?".format(x);
    }

    operator :(s: string, type t: UInt128): UInt128 {
      const b = parseBigintStrict(s);
      return b:UInt128; // wrap or checked depending on your bigint->UInt128 operator
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

  proc numBits(type t) param where t == UInt128 do return 128;

  // helper: parse string to bigint (decimal or 0x hex), strict
  private proc parseBigintStrict(s: string): bigint {
    var t = s.strip();
    if t.size == 0 then halt("empty string");

    var b: bigint;

    // bigint parsing: use set() if that's what works in your environment
    // Support hex with 0x/0X
    if t.startsWith("0x") || t.startsWith("0X") {
      // parse hex manually into bigint
      // (bigint doesn't always accept 0x prefix depending on implementation)
      var acc: bigint;
      acc = 0;

      for ch in t[2..] {
        var v: int;
        if ch >= "0" && ch <= "9" then v = (ch.byte(0) - "0".byte(0)):int;
        else if ch >= "a" && ch <= "f" then v = 10 + (ch.byte(0) - "a".byte(0)):int;
        else if ch >= "A" && ch <= "F" then v = 10 + (ch.byte(0) - "A".byte(0)):int;
        else halt("invalid hex digit in: " + s);

        acc *= 16;
        acc += v;
      }
      b = acc;
    } else {
      // decimal: rely on bigint's string parsing
      // If `t:bigint` doesn't compile, use `b.set(t)` if available.
      b = t:bigint;
    }

    return b;
  }
}
