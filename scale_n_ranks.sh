# PARAMETERS OF THE SIMULATION
n_exc=100         # exc population size
n_inh=$((n_exc/4))       # inh popoulation size
prop=0.05           # prop of connections from each population
weight=1.2          # exc connections weight
rel_inh_strength=1  # relative strength of inhibitory connections
delay=0.1           # delay of all connections
rate=1              # rate of Poisson neruons
time=10           # simulation time
dt=1                # timestep (ignored)
group_size=10       # size of cell groups

# MULTICORE PARAMETERS
n_ranks=(1 2 4 9 18 36)
n_cores=1
nmc_num_threads=2

# Runs the simulation with given parameters on n_rank ranks and n_core cores.
run() {
    n_rank=$1
    n_core=$2
    srun -n $n_rank -c $n_core ./build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
}

run_locally() {
    NMC_NUM_THREADS=2 ./build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size
}

# Preserve the newline characters by setting this empty (field splitting).
IFS=

rm scaling_ranks.txt

for n_rank in ${n_ranks[@]}
do
    # Take the output of the simulation.
    output=$(run n_rank n_cores)
    #output=$(run_locally)

    # Find the duration of the simulation from stdout.
    duration=$(echo $output | awk '/model-simulate/ {print $2}')

    echo $n_rank" "$duration

    # Output n_rank and the duration to a file.
    echo $n_rank" "$duration >> "scaling_ranks.txt"
done
