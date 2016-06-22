#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>

#include <mpi.h>
#include "utils.hpp" 
#include "utils.hpp" 
namespace mpi {

    template <typename T>
    struct mpi_traits {
        constexpr static size_t count() {
            return sizeof(T);
        }
        constexpr static MPI_Datatype mpi_type() {
            return MPI_CHAR;
        }
        constexpr static bool is_mpi_native_type() {
            return false;
        }
    };

    #define MAKE_TRAITS(T,M)     \
    template <>                 \
    struct mpi_traits<T> {  \
        constexpr static size_t count()            { return 1; } \
        constexpr static MPI_Datatype mpi_type()   { return M; } \
        constexpr static bool is_mpi_native_type() { return true; } \
    };

    MAKE_TRAITS(double, MPI_DOUBLE)
    MAKE_TRAITS(float,  MPI_FLOAT)
    MAKE_TRAITS(int,    MPI_INT)
    MAKE_TRAITS(long int, MPI_LONG)
    MAKE_TRAITS(char,   MPI_CHAR)
    MAKE_TRAITS(size_t, MPI_UNSIGNED_LONG)
    static_assert(sizeof(size_t)==sizeof(unsigned long),
                  "size_t and unsigned long are not equivalent");

    bool init(int *argc, char ***argv);
    bool finalize();
    bool is_root();
    int rank();
    int size();
    void barrier();

    template <typename T>
    std::vector<T> gather(T value, int root) {
        using traits = mpi_traits<T>;
        auto buffer_size = (rank()==root) ? size() : 0;
        std::vector<T> buffer(buffer_size);

        MPI_Gather( &value,        traits::count(), traits::mpi_type(), // send buffer
                    buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
                    root, MPI_COMM_WORLD);

        return buffer;
    }

    template <typename T>
    std::vector<T> gather_all(T value) {
        using traits = mpi_traits<T>;
        std::vector<T> buffer(size());

        MPI_Allgather( &value,        traits::count(), traits::mpi_type(), // send buffer
                       buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
                       MPI_COMM_WORLD);

        return buffer;
    }

    template <typename T>
    std::vector<T> gather_all(const std::vector<T> &values) {
        using traits = mpi_traits<T>;
        auto counts = gather_all(int(values.size()));
        for(auto& c : counts) {
            c *= traits::count();
        }
        auto displs = algorithms::make_map(counts);

        std::vector<T> buffer(displs.back()/traits::count());

        MPI_Allgatherv(
            // send buffer
            values.data(), counts[rank()], traits::mpi_type(),
            // receive buffer
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(),
            MPI_COMM_WORLD
        );

        return buffer;
    }

    template <typename T>
    T reduce(T value, MPI_Op op, int root) {
        using traits = mpi_traits<T>;
        static_assert(traits::is_mpi_native_type(),
                      "can only perform reductions on MPI native types");

        T result;

        MPI_Reduce(&value, &result, 1, traits::mpi_type(), op, root, MPI_COMM_WORLD);

        return result;
    }

    template <typename T>
    T reduce(T value, MPI_Op op) {
        using traits = mpi_traits<T>;
        static_assert(traits::is_mpi_native_type(),
                      "can only perform reductions on MPI native types");

        T result;

        MPI_Allreduce(&value, &result, 1, traits::mpi_type(), op, MPI_COMM_WORLD);

        return result;
    }

    template <typename T>
    std::pair<T,T> minmax(T value) {
        return {reduce<T>(value, MPI_MIN), reduce<T>(value, MPI_MAX)};
    }

    template <typename T>
    std::pair<T,T> minmax(T value, int root) {
        return {reduce<T>(value, MPI_MIN, root), reduce<T>(value, MPI_MAX, root)};
    }

    bool ballot(bool vote);

    template <typename T>
    T broadcast(T value, int root) {
        using traits = mpi_traits<T>;

        MPI_Bcast(&value, traits::count(), traits::mpi_type(), root, MPI_COMM_WORLD);

        return value;
    }

    template <typename T>
    T broadcast(int root) {
        using traits = mpi_traits<T>;
        T value;

        MPI_Bcast(&value, traits::count(), traits::mpi_type(), root, MPI_COMM_WORLD);

        return value;
    }

} // namespace mpi
