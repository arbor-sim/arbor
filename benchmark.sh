run() {
    seed=$1
    srun ./build/example/miniapp/brunel/brunel_miniapp.exe -n 100 -m 25 -e 30 -p 0.1 -w 1.2 -d 1 -g 0.5 -l 5 -t 100 -s 1 -G 10 -S $seed -f
}

IFS=

seed=0
i=1

total_spikes=0
num_of_runs=100

while [ $i -le $num_of_runs ]
do
    seed=$((seed+10))
    i=$((i+1))
    echo "Seed = "$seed

    output=$(run $seed)
    mv ./spikes_0.gdf ./build/spikes_$seed

    spikes=$(echo $output | awk '/there were/ {print $3}')
    echo "There were "$spikes" spikes"

    total_spikes=$((total_spikes+spikes))
done

echo "Average number of spikes = "$(echo "$total_spikes/$num_of_runs" | bc -l)

