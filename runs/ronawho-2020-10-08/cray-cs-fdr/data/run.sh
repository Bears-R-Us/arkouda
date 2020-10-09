echo "In run.sh script!"

date

chpl --version
module list

BENCHMARKS="stream argsort gather scatter reduce scan"
node_counts=(1 2 4 8 16 32)

for benchmark in ${BENCHMARKS}; do
  for i in {1..1}; do
    for nodes in "${node_counts[@]}"; do
      dir=$PWD/ak-perf-$nodes
      rm -rf $dir
      ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $benchmark --size=$((2**30)) --trials=1
    done
  done
  ./print.py

  mv "$benchmark.dat" "lg-$benchmark.dat"

  for i in {1..1}; do
    for nodes in "${node_counts[@]}"; do
      dir=$PWD/ak-perf-$nodes
      rm -rf $dir
      ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $benchmark
    done
  done
  ./print.py
done


date

cd $workdir

echo "DONE"
