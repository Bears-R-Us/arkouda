echo "In run.sh script!"

date

chpl --version
module list 

BENCHMARKS="stream gather scatter argsort"
node_counts=(1 2 4 8 16 32)
for i in {1..1}; do
  for nodes in "${node_counts[@]}"; do 
    dir=$PWD/ak-perf-$nodes
    ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS
  done
done

date

echo "DONE"
