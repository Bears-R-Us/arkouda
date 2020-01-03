echo "In run.sh script!"

HOST=""
PORT=""
function start_server {
  nodes=$1
  echo "Running ./arkouda_server -nl $nodes"
  server_info=$PWD/ak-server-info.txt
  rm -f $server_info

  ./arkouda_server -nl $nodes --logging=false --v=false &>$server_info &

  while  ! grep -q "server listening on " $server_info ; do sleep 1; done
  str=$(grep "server listening on " $server_info)
  str=${str#"server listening on "}
  HOST=$(echo $str | awk -F: '{print $1}')
  PORT=$(echo $str | awk -F: '{print $2}')
  echo "Started running on $HOST $PORT"
  echo ""
}

function stop_server {
  echo ""
  echo "Shutting down server"
  ./tests/shutdown.py $HOST $PORT &>/dev/null
}

function run_perf {
  base=$(basename $2)
  out_file="arkouda.$base.$1.out"
  echo "Running $2 with $1 nodes!"
  ./$2 $HOST $PORT >> $out_file
  echo "" >> $out_file
}

function run_perf_lg {
  start_server $nodes
  module unload craype-hugepages16M
  base=$(basename $2)
  out_file="arkouda.$base.lg.$1.out"
  echo "Running $2 large problem size with $1 nodes!"
  ./$2 $HOST $PORT --trials=1 $3 $4 >> $out_file
  echo "" >> $out_file
  module load craype-hugepages16M
  stop_server
}

date

chpl --version
module list 

node_counts=(1 2 4 8 16 32 64 128 256 512)
for i in {1..1}; do
  for nodes in "${node_counts[@]}"; do 
    start_server $nodes

    module unload craype-hugepages16M
    run_perf $nodes benchmarks/stream.py 
    run_perf $nodes benchmarks/argsort.py 
    run_perf $nodes benchmarks/gather.py 
    run_perf $nodes benchmarks/scatter.py
    run_perf $nodes benchmarks/scan.py
    run_perf $nodes benchmarks/reduce.py
    module load craype-hugepages16M

    stop_server
    wait
  done
done

for i in {1..1}; do
  for nodes in "${node_counts[@]}"; do 
    run_perf_lg $nodes benchmarks/stream.py   --size=2147483648
    run_perf_lg $nodes benchmarks/argsort.py  --size=2147483648
    run_perf_lg $nodes benchmarks/gather.py   --index-size=2147483648 --value-size=2147483648
    run_perf_lg $nodes benchmarks/scatter.py  --index-size=2147483648 --value-size=2147483648
    wait
  done
done

date

cd $workdir

echo "DONE"
