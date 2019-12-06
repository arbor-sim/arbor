#pragma once

#include <vector>
#include <algorithm>

#include <mpi.h>

#include <arbor/spike.hpp>

std::vector<arb::spike> gather_spikes(const std::vector<arb::spike>& values, MPI_Comm comm) {
	// Arbor spikes follow the NEURON format: <uint gid, uint local_id, float time>
    // TODO: Refactor so that local_id does not have to be send. 
    int size;
    MPI_Comm_size(comm, &size);

    std::vector<int> counts(size);
    int n_local = values.size()*sizeof(arb::spike);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displ(size+1);
    for (int i=0; i<size; ++i) {
        displ[i+1] = displ[i] + counts[i];
    }

    std::vector<arb::spike> buffer(displ.back()/sizeof(arb::spike));
    MPI_Allgatherv(
            const_cast<arb::spike*>(values.data()), n_local, MPI_CHAR,  // send buffer
            buffer.data(), counts.data(), displ.data(), MPI_CHAR,       // receive buffer
            comm);

    return buffer;
}

int mpi_rank(MPI_Comm c) {
    int result;
    MPI_Comm_rank(c, &result);
    return result;
}

int mpi_size(MPI_Comm c) {
    int result;
    MPI_Comm_size(c, &result);
    return result;
}

template <typename T>
void print_vec_comm(const std::string& src, const std::vector<T>& v) {
    std::cerr << src << ": [";
    for (auto x: v) std::cerr << x << " ";
    std::cerr << "]" << std::endl;
}

template <typename T>
void print_vec_comm(const std::string& src, const std::vector<T>& v, MPI_Comm comm) {
    int rank = mpi_rank(comm);
    int size = mpi_size(comm);
    for (int i=0; i<size; ++i) {
        if (i==rank) {
            print_vec_comm(src, v);
        }
        MPI_Barrier(comm);
    }
}

int broadcast(int local, MPI_Comm comm, int root) {
    int result = local;
    MPI_Bcast(&result, 1, MPI_INT, root, comm);
    return result;
}

unsigned broadcast(unsigned local, MPI_Comm comm, int root) {
    int result = local;
    MPI_Bcast(&result, 1, MPI_UNSIGNED, root, comm);
    return result;
}


float broadcast(float local, MPI_Comm comm, int root) {
    float result = local;
    MPI_Bcast(&result, 1, MPI_FLOAT, root, comm);
    return result;
}

struct comm_info {
    int global_size; //
    int global_rank; //
    int local_rank; //
    bool is_arbor; //
    bool is_nest;  //
    int arbor_size; //
    int nest_size; //
    int arbor_root; //
    int nest_root; //
    MPI_Comm comm; //
};

inline
std::ostream& operator<<(std::ostream& o, comm_info i) {
    return o << "global ( rank "<< i.global_rank << ", size " << i.global_size<< ")\n"
             << "local rank " << i.local_rank << "\n"
             << (i.is_arbor? "arbor": "nest") << "\n"
             << (i.is_nest? "nest": "arbor") << "\n"
             << "arbor (root " << i.arbor_root << ", size " << i.arbor_size << ")\n"
             << "nest (root " << i.nest_root << ", size " << i.nest_size << ")\n";
}

comm_info get_comm_info(bool is_arbor) {
    static_assert((sizeof(arb::spike) % alignof(arb::spike)) == 0);

    comm_info info;
    info.is_arbor = is_arbor;
    info.is_nest = !is_arbor;

    info.global_rank = mpi_rank(MPI_COMM_WORLD);
    info.global_size = mpi_size(MPI_COMM_WORLD);

    // split MPI_COMM_WORLD: all arbor go into split 1
    int color = is_arbor? 1: 0;
    MPI_Comm_split(MPI_COMM_WORLD, color, info.global_rank, &info.comm);

    int local_size  = mpi_size(info.comm);
    info.local_rank = mpi_rank(info.comm);

    info.arbor_size = is_arbor? local_size: info.global_size - local_size;
    info.nest_size = info.global_size - info.arbor_size;

    std::vector<int> local_ranks(local_size);
    MPI_Allgather(&info.global_rank, 1, MPI_INT, local_ranks.data(), 1, MPI_INT, info.comm);
    std::sort(local_ranks.begin(), local_ranks.end());

    auto first_missing = [](const std::vector<int>& x) {
        auto it = std::adjacent_find(x.begin(), x.end(), [](int l, int r){return (r-l)!=1;});
        return it==x.end()? x.back()+1: (*it)+1;
    };

    if (info.is_arbor) {
        info.arbor_root = local_ranks.front();
        info.nest_root = info.arbor_root == 0 ? first_missing(local_ranks) : 0;
    }
    else {
        info.nest_root = local_ranks.front();
        info.arbor_root = info.nest_root == 0 ? first_missing(local_ranks) : 0;
    }

    return info;
}

template<typename F>
void on_local_rank_zero(comm_info& info, F func) {
    if (info.local_rank != 0) return;
    func();
}
