# Parameters of the simulation.
n_exc=100           # exc population size
n_inh=$((n_exc/4))  # inh popoulation size
n_ext=100           # poisson population size
prop=0.1            # prop of connections from each population
weight=1.2            # exc connections weight
rel_inh_strength=0.5 # relative strength of inhibitory connections
delay=1             # delay of all connections
rate=5              # rate of Poisson neruons
time=1000            # simulation time
dt=1                # timestep (ignored)
group_size=50       # size of cell groups
optimised=1         # if 1, optimisation turned on

# Multicore parameters.
n_ranks=1
n_cores=(1 2 4 9 18)

# Runs the simulation with given parameters on n_rank ranks and n_core cores.
run() {
    n_rank=$1
    n_core=$2
    opt=$3

    # Use multithreading for 36 cores and otherwise no.
    if [ $n_core -eq 36 ]
    then
        if [ $opt -eq 1 ]
        then
            NMC_NUM_THREADS=$n_core srun -n $n_rank -c $n_core ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size -o
        else
            NMC_NUM_THREADS=$n_core srun -n $n_rank -c $n_core ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
        fi
    else
        if [ $opt -eq 1 ]
        then
            NMC_NUM_THREADS=$n_core srun -n $n_rank -c $n_core --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size -o
        else
            NMC_NUM_THREADS=$n_core srun -n $n_rank -c $n_core --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
        fi
    fi
}

run_locally() {
    ./build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size -f
}

# Preserve the newline characters by setting this empty (field splitting).
IFS=

cd $SCRATCH/nestmc-proto/build
make -j
cd ../miniapp/brunel

vary_n_exc=(100 1000 10000)

for n_exc in ${vary_n_exc[@]}
do
    #echo "Setting n_exc = "$n_exc"..."

    file="scaling_cores_"$n_exc".txt"

    rm scaling_cores_*

    n_inh=$((n_exc/4))
    n_ext=$((n_exc))

    for n_core in ${n_cores[@]}
    do
        #echo "  setting n_core = "$n_core"..."
        # Take the output of the simulation.
        output=$(run $n_ranks $n_core $optimised)
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






