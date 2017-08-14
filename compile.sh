cd build
make -j
cd miniapp/brunel
NMC_NUM_THREADS=1 ./brunel_miniapp.exe -n 400 -m 100 -e 400 -p 0.1 -w 1.2 -d 0.1 -g 0.5 -r 5 -t 1000 -s 1 -G 100 -f -o
cd ../../
