group_size=(10 100 1000)
n_ranks=(1 2 4 8)

cd build
make -j

run() {
    gr=$1
    rank=$2
    NMC_NUM_THREADS=1 srun -n $rank -c 1 ./miniapp/brunel/brunel_miniapp.exe -n 10000 -m 2500 -e 30 -p 0.1 -w 1.2 -d 1 -g 0.5 -r 5 -t 100 -s 1 -G $gr
}

IFS=

for gr in "${group_size[@]}"; do 
    for rank in "${n_ranks[@]}"; do
        output=$(run $gr $rank)
        spikes=$(echo $output | awk '/there/ {print $3}')
        echo "group_size = "$gr" n_ranks = "$rank" spikes = "$spikes
    done
done

cd ../../../


