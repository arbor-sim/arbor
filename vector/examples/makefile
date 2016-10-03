FLAGS=-std=c++11 -O3

ifeq ($(PE_ENV),GNU)
	FLAGS+=-fopenmp
	CC=g++
endif
ifeq ($(PE_ENV),INTEL)
	FLAGS+=-openmp
	CC=icpc
endif

FLAGS+=-march=core-avx2
#FLAGS+=-mavx

all : stream.omp stream.mpi

stream.omp : stream_omp.cpp
	$(CC) ${FLAGS} -I ../include stream_omp.cpp -o stream.omp

stream.mpi : stream_omp.cpp
	CC ${FLAGS} -I ../include stream_mpi.cpp -o stream.mpi

clean :
	rm -rf stream.omp stream.mpi
