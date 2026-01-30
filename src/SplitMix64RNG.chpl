// SplitMix64: fast 64-bit mixer suitable for stateless RNG / hashing.
// Works by design with overflow modulo 2^64, so we use uint(64) everywhere.

module SplitMix64RNG {
  // Constants from the canonical SplitMix64 reference.
  private param GAMMA: uint(64) = 0x9E3779B97F4A7C15:uint(64);
  private param C1:    uint(64) = 0xBF58476D1CE4E5B9:uint(64);
  private param C2:    uint(64) = 0x94D049BB133111EB:uint(64);

  // A single SplitMix64 mix step.
  // Given an input x, returns a well-scrambled 64-bit value.
  inline proc splitmix64(xIn: uint(64)): uint(64) {
    var x = xIn + GAMMA;
    x = (x ^ (x >> 30)) * C1;
    x = (x ^ (x >> 27)) * C2;
    x = x ^ (x >> 31);
    return x;
  }

  // Derive a per-stream key from seed and stream id.
  // This makes streams "independent enough" for analytics/sampling/shuffle keys.
  inline proc streamKey(seed: uint(64), stream: uint(64)): uint(64) {
    // Any deterministic combination is fine; we mix it to avoid structure.
    return splitmix64(seed ^ (stream * 0xD2B74407B1CE6E93:uint(64)));
  }

  // Stateless uint64 RNG at a global index i:
  // r[i] = splitmix64(i ^ key(seed, stream))
  inline proc randU64At(i: uint(64), seed: uint(64), stream: uint(64) = 0:uint(64)): uint(64) {
    const k = streamKey(seed, stream);
    return splitmix64(i ^ k);
  }

  // Convert uint64 to a uniform float64 in [0,1) using top 53 bits (exact grid).
  inline proc u64ToUniform01(r: uint(64)): real(64) {
    const top53 = r >> 11;                 // keep 53 MSBs
    return top53:real(64) * 0x1.0p-53;     // 2^-53
  }

  // Fill an array with uint64 randomness based on *global* indices.
  // startIdx is the global index of A.domain.low (for Arkouda you'd pass the locale's global offset).
  @arkouda.registerCommand
  proc fillRandU64(ref A: [?d] ?t, seed: uint, stream: uint = 0: uint, startIdx: uint = 0:uint)
  where (t == uint(64)) && (d.rank == 1) {
    const k = streamKey(seed, stream);
    forall idx in d with (ref A){      
      // idx is the domain index type; cast to uint(64) for a stable global counter
      const i = startIdx + idx:uint(64);
      A[idx] = splitmix64(i ^ k);
    }
  }

  // Fill an array with uniform float64 in [0,1).
  @arkouda.registerCommand
  proc fillUniform01(ref A: [?d] ?t, seed: uint, stream: uint = 0:uint, startIdx: uint = 0:uint) throws
  where (t == real) && (d.rank == 1){
    const k = streamKey(seed, stream);
    forall idx in d with (ref A){      
      const i = startIdx + idx:uint(64);
      A[idx] = u64ToUniform01(splitmix64(i ^ k));
    }
  }
}
