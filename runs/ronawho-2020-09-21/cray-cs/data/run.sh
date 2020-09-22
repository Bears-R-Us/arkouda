echo "In run.sh script!"

date

chpl --version
module list

BENCHMARKS="scatter argsort"
node_counts=(1 2 4 8 16 32)

for i in {1..1}; do
  for nodes in "${node_counts[@]}"; do
    dir=$PWD/ak-perf-$nodes
    rm -rf $dir
    ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS --size=$((2**30)) --trials=1
  done
done
./print.py

for benchmark in ${BENCHMARKS}; do
  mv "$benchmark.dat" "lg-$benchmark.dat"
done

for i in {1..1}; do
  for nodes in "${node_counts[@]}"; do
    dir=$PWD/ak-perf-$nodes
    rm -rf $dir
    ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS
  done
done
./print.py


date

cd $workdir

echo "DONE"
