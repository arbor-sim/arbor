# Parameters of the simulation.
n_exc=100           # exc population size
n_inh=$((n_exc/4))  # inh popoulation size
n_ext=30           # poisson population size
prop=0.1            # prop of connections from each population
weight=1.2            # exc connections weight
rel_inh_strength=0.5 # relative strength of inhibitory connections
delay=1             # delay of all connections
rate=5              # rate of Poisson neruons
time=100            # simulation time
dt=1                # timestep (ignored)
group_size=100       # size of cell groups

# Multicore parameters.
#n_ranks=(1 2 4 9 18)
n_ranks=(1)
n_cores=(2)

# Runs the simulation with given parameters on n_rank ranks and n_core cores.
run() {
    n_rank=$1
    n_core=$2
    group_size=$((2*n_exc))

    # Use multithreading for 36 cores and otherwise no.
    if [ $n_core -eq 36 ]
    then
        srun -n $n_rank -c $n_core ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
    else
        srun -n $n_rank -c $n_core --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
    fi
}

run_temp() {
    group_size=150000
    #ssrun -n 1 -c 5 ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
    ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
}

# Preserve the newline characters by setting this empty (field splitting).
IFS=

cd $SCRATCH/nestmc-proto/build
make -j
cd ../miniapp/brunel

vary_n_exc=(1000 10000 100000)

for n_exc in ${vary_n_exc[@]}
do
    #echo "Setting n_exc = "$n_exc"..."

    file="scale_"$n_exc".txt"

    rm scale_*.txt

    n_inh=$((n_exc/4))

    for n_rank in ${n_ranks[@]}
    do
        #echo "  setting n_core = "$n_core"..."
        # Take the output of the simulation.
        output=$(run_temp)
        #output=$(run $n_rank $n_core)
        #output=$(run_locally)
        #echo "  "$output
        # Find the duration of the simulation from stdout.
        setup=$(echo $output | awk '/setup/ {print $2}')
        model_init=$(echo $output | awk '/model-init/ {print $2}')
        model_simulate=$(echo $output | awk '/model-simulate/ {print $2}')
        echo $n_core" "$setup" "$model_init" "$model_simulate
        # Output n_core and the duration to a file.
        echo $n_core" "$setup" "$model_init" "$model_simulate >> $file
    done
    prop=$(echo "$prop/10.0" | bc -l)
done






