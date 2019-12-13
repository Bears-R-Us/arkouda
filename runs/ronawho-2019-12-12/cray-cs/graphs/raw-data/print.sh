#!/usr/bin/env bash

node_counts=(1 2 4 8 16 32 64 128 256 512)
files=(arkouda.stream.py arkouda.argsort.py arkouda.gather.py arkouda.scatter.py)
#files=(arkouda.stream.py arkouda.argsort.py arkouda.gather.py arkouda.scatter.py arkouda.scan.py arkouda.reduce.py)
for f in "${files[@]}"; do 
  for i in "${node_counts[@]}"; do
    out_file="$f.$i.out"
    if  [ -f "$out_file" ]; then 
      echo "$out_file"
      cat $out_file 2>/dev/null | grep "Average time"
      cat $out_file 2>/dev/null | grep "Average rate"
    echo ""
    fi
  done
  echo ""
  echo ""
done
