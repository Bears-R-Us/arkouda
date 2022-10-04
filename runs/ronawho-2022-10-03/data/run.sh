echo "In run.sh script!"

date

chpl --version
module list

BENCHMARKS="argsort"
node_counts=(1 2 4 8 16 32 64 128 240)

for nodes in "${node_counts[@]}"; do
  dir=$PWD/ak-perf-$nodes
  rm -rf $dir
  ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS --size=$((2**30)) --trials=1
done
./print.py
mkdir -p 2-to-30
for benchmark in ${BENCHMARKS}; do
  mv "$benchmark.dat" "2-to-30/"
done

###
#
#for nodes in "${node_counts[@]}"; do
#  dir=$PWD/ak-perf-$nodes
#  rm -rf $dir
#  ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS
#done
#./print.py
#mkdir -p 10-to-8
#for benchmark in ${BENCHMARKS}; do
#  mv "$benchmark.dat" "10-to-8/" #done
#
###

for nodes in "${node_counts[@]}"; do
  dir=$PWD/ak-perf-$nodes
  rm -rf $dir
  ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS --size=$((2**26))
done
./print.py
mkdir -p 2-to-26
for benchmark in ${BENCHMARKS}; do
  mv "$benchmark.dat" "2-to-26/"
done

###

for nodes in "${node_counts[@]}"; do
  dir=$PWD/ak-perf-$nodes
  rm -rf $dir
  ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS --size=$((2**16))
done
./print.py
mkdir -p 2-to-16
for benchmark in ${BENCHMARKS}; do
  mv "$benchmark.dat" "2-to-16/"
done

###

for nodes in "${node_counts[@]}"; do
  dir=$PWD/ak-perf-$nodes
  rm -rf $dir
  ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS --size=$((2**10))
done
./print.py
mkdir -p 2-to-10
for benchmark in ${BENCHMARKS}; do
  mv "$benchmark.dat" "2-to-10/"
done

###

date

cd $workdir

echo "DONE"
