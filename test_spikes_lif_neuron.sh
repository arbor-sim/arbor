cd build

make -j test.exe

cd tests

./test.exe || true

cd ../../tests/unit/

python3 plot_spikes.py
