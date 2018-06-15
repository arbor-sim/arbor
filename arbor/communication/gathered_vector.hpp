#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

#include <util/rangeutil.hpp>

namespace arb {

template <typename T>
class gathered_vector {
public:
    using value_type = T;
    using count_type = unsigned;

    gathered_vector(std::vector<value_type>&& v, std::vector<count_type>&& p) :
        values_(std::move(v)),
        partition_(std::move(p))
    {
        arb_assert(util::is_sorted(partition_));
        arb_assert(partition_.back() == values_.size());
    }

    /// the partition of distribution
    const std::vector<count_type>& partition() const {
        return partition_;
    }

    /// the number of entries in the gathered vector in partition i
    count_type count(std::size_t i) const {
        return partition_[i+1] - partition_[i];
    }

    /// the values in the gathered vector
    const std::vector<value_type>& values() const {
        return values_;
    }

    /// the size of the gathered vector
    std::size_t size() const {
        return values_.size();
    }

private:
    std::vector<value_type> values_;
    std::vector<count_type> partition_;
};

} // namespace arb
