#pragma once

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <arbor/export.hpp>
#include <arbor/assert.hpp>
#include <arbor/communication/mpi_error.hpp>

#include "communication/gathered_vector.hpp"
#include "profile/profiler_macro.hpp"
#include "util/rangeutil.hpp"
#include "util/partition.hpp"

namespace arb {
namespace mpi {

// prototypes
ARB_ARBOR_API int rank(MPI_Comm);
ARB_ARBOR_API int size(MPI_Comm);
ARB_ARBOR_API void barrier(MPI_Comm);

#define MPI_OR_THROW(fn, ...)\
while (int r_ = fn(__VA_ARGS__)) throw mpi_error(r_, #fn)

// Type traits for automatically setting MPI_Datatype information for C++ types.
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

MAKE_TRAITS(float,              MPI_FLOAT)
MAKE_TRAITS(double,             MPI_DOUBLE)
MAKE_TRAITS(char,               MPI_CHAR)
MAKE_TRAITS(int,                MPI_INT)
MAKE_TRAITS(unsigned,           MPI_UNSIGNED)
MAKE_TRAITS(long,               MPI_LONG)
MAKE_TRAITS(unsigned long,      MPI_UNSIGNED_LONG)
MAKE_TRAITS(long long,          MPI_LONG_LONG)
MAKE_TRAITS(unsigned long long, MPI_UNSIGNED_LONG_LONG)

static_assert(std::is_same<std::size_t, unsigned>::value ||
              std::is_same<std::size_t, unsigned long>::value ||
              std::is_same<std::size_t, unsigned long long>::value,
              "size_t is not the same as any MPI unsigned type");

// Gather individual values of type T from each rank into a std::vector on
// the root rank.
// T must be trivially copyable.
template<typename T>
std::vector<T> gather(T value, int root, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    auto buffer_size = (rank(comm)==root) ? size(comm) : 0;
    std::vector<T> buffer(buffer_size);

    MPI_OR_THROW(MPI_Gather,
                &value,        traits::count(), traits::mpi_type(), // send buffer
                buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
                root, comm);

    return buffer;
}

// Gather individual values of type T from each rank into a std::vector on
// the every rank.
// T must be trivially copyable
template <typename T>
std::vector<T> gather_all(T value, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    std::vector<T> buffer(size(comm));

    MPI_OR_THROW(MPI_Allgather,
            &value,        traits::count(), traits::mpi_type(), // send buffer
            buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
            comm);

    return buffer;
}

// Specialize gather for std::string.
inline std::vector<std::string> gather(std::string str, int root, MPI_Comm comm) {
    using traits = mpi_traits<char>;

    std::vector<int> counts, displs;
    counts = gather_all(int(str.size()), comm);
    util::make_partition(displs, counts);

    std::vector<char> buffer(displs.back());

    // const_cast required for MPI implementations that don't use const* in
    // their interfaces.
    std::string::value_type* ptr = const_cast<std::string::value_type*>(str.data());
    MPI_OR_THROW(MPI_Gatherv,
            ptr, counts[rank(comm)], traits::mpi_type(),                       // send
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(),   // receive
            root, comm);

    // Unpack the raw string data into a vector of strings.
    std::vector<std::string> result;
    auto nranks = size(comm);
    result.reserve(nranks);
    for (auto i=0; i<nranks; ++i) {
        result.push_back(std::string(buffer.data()+displs[i], counts[i]));
    }
    return result;
}

template <typename T>
std::vector<T> gather_all(const std::vector<T>& values, MPI_Comm comm) {

    using traits = mpi_traits<T>;
    std::vector<int> counts, displs;
    counts = gather_all(int(values.size()), comm);
    for (auto& c : counts) {
        c *= traits::count();
    }
    util::make_partition(displs, counts);

    std::vector<T> buffer(displs.back()/traits::count());
    MPI_OR_THROW(MPI_Allgatherv,
            // const_cast required for MPI implementations that don't use const* in their interfaces
            const_cast<T*>(values.data()), counts[rank(comm)], traits::mpi_type(),  // send buffer
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(), // receive buffer
            comm);

    return buffer;
}

inline std::vector<std::string> gather_all(const std::vector<std::string>& values, MPI_Comm comm) {
    using traits = mpi_traits<char>;
    std::vector<int> counts_individual, counts_total, displs_individual, displs_total;

    // vector of individual string sizes
    std::vector<int> individual_sizes(values.size());
    std::transform(values.begin(), values.end(), individual_sizes.begin(), [](const std::string& val){return int(val.size());});

    counts_individual = gather_all(individual_sizes, comm);
    counts_total      = gather_all(util::sum(individual_sizes, 0), comm);

    util::make_partition(displs_total, counts_total);
    std::vector<char> buffer(displs_total.back());

    // Concatenate string data
    std::string values_concat;
    for (const auto& v: values) {
        values_concat += v;
    }

    // Cast to ptr
    // const_cast required for MPI implementations that don't use const* in
    // their interfaces.
    std::string::value_type* ptr = const_cast<std::string::value_type*>(values_concat.data());
    MPI_OR_THROW(MPI_Allgatherv,
                 ptr, counts_total[rank(comm)], traits::mpi_type(),  // send buffer
                 buffer.data(), counts_total.data(), displs_total.data(), traits::mpi_type(), // receive buffer
                 comm);

    // Construct the vector of strings
    std::vector<std::string> string_buffer;
    string_buffer.reserve(counts_individual.size());

    auto displs_individual_part = util::make_partition(displs_individual, counts_individual);
    for (const auto& str_range: displs_individual_part) {
        string_buffer.emplace_back(buffer.begin()+str_range.first, buffer.begin()+str_range.second);
    }

    return string_buffer;
}

template <typename T>
std::vector<std::vector<T>> gather_all(const std::vector<std::vector<T>>& values, MPI_Comm comm) {
    std::vector<unsigned long> counts_internal, displs_internal;

    // Vector of individual vector sizes
    std::vector<unsigned long> internal_sizes(values.size());
    std::transform(values.begin(), values.end(), internal_sizes.begin(), [](const auto& val){return int(val.size());});

    counts_internal = gather_all(internal_sizes, comm);
    auto displs_internal_part = util::make_partition(displs_internal, counts_internal);

    // Concatenate all internal vector data
    std::vector<T> values_concat;
    for (const auto& v: values) {
        values_concat.insert(values_concat.end(), v.begin(), v.end());
    }

    // Gather all concatenated vector data
    auto global_vec_concat = gather_all(values_concat, comm);

    // Construct the vector of vectors
    std::vector<std::vector<T>> global_vec;
    global_vec.reserve(displs_internal_part.size());

    for (const auto& internal_vec_range: displs_internal_part) {
        global_vec.emplace_back(global_vec_concat.begin()+internal_vec_range.first,
                                global_vec_concat.begin()+internal_vec_range.second);
    }

    return global_vec;
}

/// Gather all of a distributed vector
/// Retains the meta data (i.e. vector partition)
template <typename T>
gathered_vector<T> gather_all_with_partition(const std::vector<T>& values, MPI_Comm comm) {
    using gathered_type = gathered_vector<T>;
    using count_type = typename gathered_vector<T>::count_type;
    using traits = mpi_traits<T>;

    // We have to use int for the count and displs vectors instead
    // of count_type because these are used as arguments to MPI_Allgatherv
    // which expects int arguments.
    std::vector<int> counts, displs;
    counts = gather_all(int(values.size()), comm);
    for (auto& c : counts) {
        c *= traits::count();
    }
    util::make_partition(displs, counts);

    std::vector<T> buffer(displs.back()/traits::count());

    MPI_OR_THROW(MPI_Allgatherv,
            // const_cast required for MPI implementations that don't use const* in their interfaces
            const_cast<T*>(values.data()), counts[rank(comm)], traits::mpi_type(), // send buffer
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(), // receive buffer
            comm);

    for (auto& d : displs) {
        d /= traits::count();
    }

    return gathered_type(
        std::move(buffer),
        std::vector<count_type>(displs.begin(), displs.end())
    );
}

template <typename T>
T reduce(T value, MPI_Op op, int root, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    static_assert(traits::is_mpi_native_type(),
                  "can only perform reductions on MPI native types");

    T result;

    MPI_OR_THROW(MPI_Reduce,
        &value, &result, 1, traits::mpi_type(), op, root, comm);

    return result;
}

template <typename T>
T reduce(T value, MPI_Op op, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    static_assert(traits::is_mpi_native_type(),
                  "can only perform reductions on MPI native types");

    T result;

    MPI_Allreduce(&value, &result, 1, traits::mpi_type(), op, comm);

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
T broadcast(T value, int root, MPI_Comm comm) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "broadcast can only be performed on trivally copyable types");

    using traits = mpi_traits<T>;

    MPI_OR_THROW(MPI_Bcast,
        &value, traits::count(), traits::mpi_type(), root, comm);

    return value;
}

template <typename T>
T broadcast(int root, MPI_Comm comm) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "broadcast can only be performed on trivally copyable types");

    using traits = mpi_traits<T>;
    T value;

    MPI_OR_THROW(MPI_Bcast,
        &value, traits::count(), traits::mpi_type(), root, comm);

    return value;
}

} // namespace mpi
} // namespace arb
