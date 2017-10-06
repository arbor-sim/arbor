# Parameters of the simulation.
n_ext=30           # poisson population size
prop=0.95            # prop of connections from each population
weight=1.2            # exc connections weight
rel_inh_strength=0.5 # relative strength of inhibitory connections
delay=1             # delay of all connections
rate=5              # rate of Poisson neruons
time=100            # simulation time
dt=1                # timestep (ignored)
group_size=300000       # size of cell groups

run() {
    nvprof --profile-from-start off --unified-memory-profiling off ../../build/miniapp/brunel/brunel_miniapp.exe -n $n_exc -m $n_inh -e $n_ext -p $prop -w $weight -d $delay -g $rel_inh_strength -r $rate -t $time -s $dt -G $group_size |& grep advance_kernel |awk '{print $4}'
}

# Preserve the newline characters by setting this empty (field splitting).
IFS=

cd /users/kabicm/nestmc-proto/build
make -j
cd ../miniapp/brunel

vary_n=(128 512 1024 2048 4096 8192 16384 32768 65536 131072 262144)

n_rep=5

rm scale_*.txt

for n in ${vary_n[@]}; do
    n_exc=$((3*n/4))
    n_inh=$((n/4))
    rep=0

    while [ $rep -lt $n_rep ]; do
        file="scale_"$n"_"$rep".txt"
        # Take the output of the simulation.
        avg_kernel_time=$(run)
        # echo "Output is "$output
        #avg_kernel_time=$(output |& grep advance_kernel |awk '{print $4}')

        # Find the duration of the simulation from stdout.
        echo $n" "$rep" "$avg_kernel_time

        # Output to a file.
        echo $n" "$rep" "$avg_kernel_time >> $file

        rep=$((rep+1))
    done
    prop=$(echo "$prop/2.0" | bc -l)
done







