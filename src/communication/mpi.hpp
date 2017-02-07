#pragma once

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

#include <cassert>

#include <mpi.h>

#include <algorithms.hpp>
#include <communication/gathered_vector.hpp>
#include <util/debug.hpp>
#include <profiling/profiler.hpp>


namespace nest {
namespace mc {
namespace mpi {

    // prototypes
    void init(int *argc, char ***argv);
    void finalize();
    bool is_root();
    int rank();
    int size();
    void barrier();
    bool ballot(bool vote);

    // type traits for automatically setting MPI_Datatype information
    // for C++ types
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
        /* constexpr */ static MPI_Datatype mpi_type()   { return M; } \
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

    // Gather individual values of type T from each rank into a std::vector on
    // the root rank.
    // T must be trivially copyable
    template<typename T>
    std::vector<T> gather(T value, int root) {
        static_assert(
            true,//std::is_trivially_copyable<T>::value,
            "gather can only be performed on trivally copyable types");

        using traits = mpi_traits<T>;
        auto buffer_size = (rank()==root) ? size() : 0;
        std::vector<T> buffer(buffer_size);

        PE("MPI", "Gather");
        MPI_Gather( &value,        traits::count(), traits::mpi_type(), // send buffer
                    buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
                    root, MPI_COMM_WORLD);
        PL(2);

        return buffer;
    }

    // Gather individual values of type T from each rank into a std::vector on
    // the every rank.
    // T must be trivially copyable
    template <typename T>
    std::vector<T> gather_all(T value) {
        static_assert(
            true,//std::is_trivially_copyable<T>::value,
            "gather_all can only be performed on trivally copyable types");

        using traits = mpi_traits<T>;
        std::vector<T> buffer(size());

        PE("MPI", "Allgather");
        MPI_Allgather( &value,        traits::count(), traits::mpi_type(), // send buffer
                       buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
                       MPI_COMM_WORLD);
        PL(2);

        return buffer;
    }

    template <typename T>
    std::vector<T> gather_all(const std::vector<T>& values) {
        static_assert(
            true,//std::is_trivially_copyable<T>::value,
            "gather_all can only be performed on trivally copyable types");

        using traits = mpi_traits<T>;
        auto counts = gather_all(int(values.size()));
        for (auto& c : counts) {
            c *= traits::count();
        }
        auto displs = algorithms::make_index(counts);

        std::vector<T> buffer(displs.back()/traits::count());

        PE("MPI", "Allgatherv");
        MPI_Allgatherv(
            // send buffer
            values.data(), counts[rank()], traits::mpi_type(),
            // receive buffer
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(),
            MPI_COMM_WORLD
        );
        PL(2);

        return buffer;
    }

    /// Gather all of a distributed vector
    /// Retains the meta data (i.e. vector partition)
    template <typename T>
    gathered_vector<T> gather_all_with_partition(const std::vector<T>& values) {
        using gathered_type = gathered_vector<T>;
        using count_type = typename gathered_vector<T>::count_type;
        using traits = mpi_traits<T>;

        // We have to use int for the count and displs vectors instead
        // of count_type because these are used as arguments to MPI_Allgatherv
        // which expects int arguments.
        auto counts = gather_all(int(values.size()));
        for (auto& c : counts) {
            c *= traits::count();
        }
        auto displs = algorithms::make_index(counts);

        std::vector<T> buffer(displs.back()/traits::count());

        PE("MPI", "Allgatherv-partition");
        MPI_Allgatherv(
            // send buffer
            values.data(), counts[rank()], traits::mpi_type(),
            // receive buffer
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(),
            MPI_COMM_WORLD
        );
        PL(2);

        for (auto& d : displs) {
            d /= traits::count();
        }

        return gathered_type(
            std::move(buffer),
            std::vector<count_type>(displs.begin(), displs.end())
        );
    }

    template <typename T>
    T reduce(T value, MPI_Op op, int root) {
        using traits = mpi_traits<T>;
        static_assert(
            traits::is_mpi_native_type(),
            "can only perform reductions on MPI native types");

        T result;

        PE("MPI", "Reduce");
        MPI_Reduce(&value, &result, 1, traits::mpi_type(), op, root, MPI_COMM_WORLD);
        PL(2);

        return result;
    }

    template <typename T>
    T reduce(T value, MPI_Op op) {
        using traits = mpi_traits<T>;
        static_assert(
            traits::is_mpi_native_type(),
            "can only perform reductions on MPI native types");

        T result;

        PE("MPI", "Allreduce");
        MPI_Allreduce(&value, &result, 1, traits::mpi_type(), op, MPI_COMM_WORLD);
        PL(2);

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

    template <typename T>
    T broadcast(T value, int root) {
        static_assert(
            true,//std::is_trivially_copyable<T>::value,
            "broadcast can only be performed on trivally copyable types");

        using traits = mpi_traits<T>;

        PE("MPI", "Bcast");
        MPI_Bcast(&value, traits::count(), traits::mpi_type(), root, MPI_COMM_WORLD);
        PL(2);

        return value;
    }

    template <typename T>
    T broadcast(int root) {
        static_assert(
            true,//std::is_trivially_copyable<T>::value,
            "broadcast can only be performed on trivally copyable types");

        using traits = mpi_traits<T>;
        T value;

        PE("MPI", "Bcast-void");
        MPI_Bcast(&value, traits::count(), traits::mpi_type(), root, MPI_COMM_WORLD);
        PL(2);

        return value;
    }

} // namespace mpi
} // namespace mc
} // namespace nest
