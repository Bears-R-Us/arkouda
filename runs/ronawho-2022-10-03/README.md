Arkouda argsort runs from recent scalability improvements

Sort scalability graphs for https://github.com/Bears-R-Us/arkouda/issues/1404.
Includes argsort results on 240 nodes of an SGI InfiniBand system with recent
optimizations. Some of the optimizations were to upstream Chapel so I had to
use older Chapel versions to test that and since modern Arkouda isn't
compatible I had to use older Arkouda as well. However, there aren't really any
performance changes to either Chapel or Arkouda outside the ones we're trying
to test that impact sort performance. Below are the configs tested:
 - Chapel 1.25 and Arkouda from that era (baseline)
 - Chapel 1.27 and Arkouda from that era (chapel scan opts)
 - Chapel 1.28 and Arkouda from that era (ak bucket exchange improvements)
 - Chapel 1.28 and latest Arkouda (ak aggregator offsetting)

Hardware Configuration:
---
SGI 8600:
 - EDR-100 InfiniBand network
 - 40-core Cascade Lake (dual-socket Intel Xeon Gold 6242R)
 - 192 GB RAM
