echo "In run.sh script!"

date

chpl --version
module list 

BENCHMARKS="stream gather scatter argsort"
node_counts=(1 2 4 8 16 32 64 128 256 512)

BENCHMARKS="argsort"
node_counts=(512)

#node_counts=(16)
for i in {1..1}; do
  for nodes in "${node_counts[@]}"; do
    dir=$PWD/ak-perf-$nodes
    ./benchmarks/run_benchmarks.py -nl $nodes --dat-dir $dir --gen-graphs $BENCHMARKS
  done
done

#for i in {1..1}; do
#  for nodes in "${node_counts[@]}"; do 
#    dir=$PWD/ak-lg-perf-$nodes
#    run_perf_lg $nodes benchmarks/stream.py   --size=2147483648
#    run_perf_lg $nodes benchmarks/argsort.py  --size=2147483648
#    run_perf_lg $nodes benchmarks/gather.py   --index-size=2147483648 --value-size=2147483648
#    run_perf_lg $nodes benchmarks/scatter.py  --index-size=2147483648 --value-size=2147483648
#    wait
#  done
#done

date

cd $workdir

echo "DONE"
