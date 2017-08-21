#######################################################
# Small problem (#neurons = 2 * 100 + 25)
#######################################################

# Vary the number of cores, while keeping the other parameters fixed.
# The connection probabiity is p = 0.1

NMC_NUM_THREADS=1 srun -n 1 -c 1 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 100 -m 25 -e 100 -p 0.1 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=2 srun -n 1 -c 2 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 100 -m 25 -e 100 -p 0.1 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=4 srun -n 1 -c 4 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 100 -m 25 -e 100 -p 0.1 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=9 srun -n 1 -c 9 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 100 -m 25 -e 100 -p 0.1 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=18 srun -n 1 -c 18 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 100 -m 25 -e 100 -p 0.1 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

# For 36 cores, we turn on the hyperthreading.
NMC_NUM_THREADS=36 srun -n 1 -c 36 ../../build/miniapp/brunel/brunel_miniapp.exe -n 100 -m 25 -e 100 -p 0.1 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

#######################################################
# Medium problem (#neurons = 2 * 1000 + 250)
#######################################################

# Increase all 3 populations 10 times. Decrease the connection probability down to p = 0.01
# to keep the in degree to each neuron the same as before.

NMC_NUM_THREADS=1 srun -n 1 -c 1 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 1000 -m 250 -e 1000 -p 0.01 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=2 srun -n 1 -c 2 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 1000 -m 250 -e 1000 -p 0.01 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=4 srun -n 1 -c 4 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 1000 -m 250 -e 1000 -p 0.01 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=9 srun -n 1 -c 9 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 1000 -m 250 -e 1000 -p 0.01 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=18 srun -n 1 -c 18 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 1000 -m 250 -e 1000 -p 0.01 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

# For 36 cores, we turn on the hyperthreading.
NMC_NUM_THREADS=36 srun -n 1 -c 36 ../../build/miniapp/brunel/brunel_miniapp.exe -n 1000 -m 250 -e 1000 -p 0.01 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

#######################################################
# Large problem (#neurons = 2 * 10000 + 2500)
#######################################################

# Again, increase all 3 populations 10 times. Decrease the connection probability down to p = 0.001
# to keep the in degree to each neuron the same as before.

NMC_NUM_THREADS=1 srun -n 1 -c 1 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 10000 -m 2500 -e 10000 -p 0.001 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=2 srun -n 1 -c 2 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 10000 -m 2500 -e 10000 -p 0.001 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=4 srun -n 1 -c 4 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 10000 -m 2500 -e 10000 -p 0.001 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=9 srun -n 1 -c 9 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 10000 -m 2500 -e 10000 -p 0.001 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

NMC_NUM_THREADS=18 srun -n 1 -c 18 --hint=nomultithread ../../build/miniapp/brunel/brunel_miniapp.exe -n 10000 -m 2500 -e 10000 -p 0.001 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o

# For 36 cores, we turn on the hyperthreading.
NMC_NUM_THREADS=36 srun -n 1 -c 36 ../../build/miniapp/brunel/brunel_miniapp.exe -n 10000 -m 2500 -e 10000 -p 0.001 -w 1.2 -d 1 -g 0.5 -r 5 -t 1000 -s 1 -G 50 -o


