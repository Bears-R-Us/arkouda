module MurmurHash {

  private inline proc getblock32(key: [?D] uint(8), word: int): uint(32) {
    var res: uint(32);
    for param j in 0..3 {
      res |= (key[D.low + 4*word + j]: uint(32)) << (3 - j);
    }
    return res;
  }

  private inline proc ROTL32(x: uint(32), r: int): uint(32) {
    return (x << r) | (x >> (32 - r));
  }

  private inline proc fmix32(in h: uint(32)): uint(32) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  proc MurmurHash3_128(blocks: [?D] uint(8), seed: uint(32) = 0): 2*uint(64) {
    const nblocks: int = D.size / 16;
    const len = D.size;
    var h1 = seed, h2 = seed, h3 = seed, h4 = seed;
    const c1: uint(32) = 0x239b961b;
    const c2: uint(32) = 0xab0e9789;
    const c3: uint(32) = 0x38b34ae5; 
    const c4: uint(32) = 0xa1e38b93;

    for i in 0..#nblocks {
      var k1 = getblock32(blocks,i*4+0);
      var k2 = getblock32(blocks,i*4+1);
      var k3 = getblock32(blocks,i*4+2);
      var k4 = getblock32(blocks,i*4+3);

      k1 *= c1; k1  = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
      
      h1 = ROTL32(h1,19); h1 += h2; h1 = h1*5+0x561ccd1b;
      
      k2 *= c2; k2  = ROTL32(k2,16); k2 *= c3; h2 ^= k2;
      
      h2 = ROTL32(h2,17); h2 += h3; h2 = h2*5+0x0bcaa747;
      
      k3 *= c3; k3  = ROTL32(k3,17); k3 *= c4; h3 ^= k3;
      
      h3 = ROTL32(h3,15); h3 += h4; h3 = h3*5+0x96cd1c35;
      
      k4 *= c4; k4  = ROTL32(k4,18); k4 *= c1; h4 ^= k4;
      
      h4 = ROTL32(h4,13); h4 += h1; h4 = h4*5+0x32ac3b17;
    }

    var k1: uint(32) = 0;
    var k2: uint(32) = 0;
    var k3: uint(32) = 0;
    var k4: uint(32) = 0;

    var taildom = {D.low+nblocks*16..D.high};
    const t = len & 15;
    var tail: [0..#t] uint(8) = blocks[taildom];
    //ref tail = blocks[D.low+nblocks*16..D.high];
    
    if (t == 15) {k4 ^= tail[14] << 16;}
    if (t >= 14) {k4 ^= tail[13] << 8;}
    if (t >= 13) {k4 ^= tail[12]; k4 = ROTL32(k4, 18); k4 *= c1; h4 ^= k4;}
    
    if (t >= 12) {k3 ^= tail[11] << 24;}
    if (t >= 11) {k3 ^= tail[10] << 16;}
    if (t >= 10) {k3 ^= tail[ 9] << 8;}
    if (t >= 9)  {k3 ^= tail[ 8]; k3 *= c3; k3  = ROTL32(k3,17); k3 *= c4; h3 ^= k3;}

    if (t >= 8)  {k2 ^= tail[ 7] << 24;}
    if (t >= 7)  {k2 ^= tail[ 6] << 16;}
    if (t >= 6)  {k2 ^= tail[ 5] << 8;}
    if (t >= 5)  {k2 ^= tail[ 4]; k2 *= c2; k2  = ROTL32(k2,16); k2 *= c3; h2 ^= k2;}

    if (t >= 4)  {k1 ^= tail[ 3] << 24;}
    if (t >= 3)  {k1 ^= tail[ 2] << 16;}
    if (t >= 2)  {k1 ^= tail[ 1] << 8;}
    if (t >= 1)  {k1 ^= tail[ 0] << 0; k1 *= c1; k1  = ROTL32(k1,15); k1 *= c2; h1 ^= k1;}

    h1 ^= len:uint(32); h2 ^= len:uint(32); h3 ^= len:uint(32); h4 ^= len:uint(32);

    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;

    h1 = fmix32(h1);
    h2 = fmix32(h2);
    h3 = fmix32(h3);
    h4 = fmix32(h4);

    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;

    var res: 2*uint(64);
    res[1] |= (h1: uint(64)) << 32;
    res[1] |= (h2: uint(64));
    res[2] |= (h3: uint(64)) << 32;
    res[2] |= (h4: uint(64));
    return res;
  }
}