#include <algorithm>
#include <string>
#include <vector>

#include <arbor/distributed_context.hpp>
#include <arbor/spike.hpp>

namespace arb {

struct dry_run_context_impl {

    explicit dry_run_context_impl(int num_ranks): num_ranks_(num_ranks) {};

    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        using count_type = typename gathered_vector<arb::spike>::count_type;
        std::vector<arb::spike> shift_local_spikes(local_spikes);
        std::vector<arb::spike> gathered_spikes;
        gathered_spikes.reserve(local_spikes.size()*num_ranks_);

        for (unsigned i = 0; i < num_ranks_; i++) {
            gathered_spikes.insert(gathered_spikes.end(), shift_local_spikes.begin(), shift_local_spikes.end());
            std::for_each(shift_local_spikes.begin(), shift_local_spikes.end(),
                          [&](arb::spike& s) {s.source += num_cells_per_tile_;});
            /*for (unsigned j = 0; j < shift_local_spikes.size(); j++) {
                shift_local_spikes[j].source +=  num_cells_per_tile_;
            }*/
        }
        std::vector<count_type> partition;
        partition.push_back(0u);
        int count = 0;
        for(int i = 0; i < num_ranks_; i++) {
            partition.push_back(static_cast<count_type>(count+=local_spikes.size()));
        }

        return gathered_vector<arb::spike>(
                std::move(std::vector<arb::spike>(gathered_spikes)),
                std::move(std::vector<count_type>(partition.begin(), partition.end()))
        );
    }

    int id() const { return 0; }

    int size() const { return num_ranks_; }

    template <typename T>
    T min(T value) const { return value; }

    template <typename T>
    T max(T value) const { return value; }

    template <typename T>
    T sum(T value) const { return value * num_ranks_; }

    template <typename T>
    std::vector<T> gather(T value, int) const {
        std::vector<T> gathered_v;
        for (unsigned i = 0; i < num_ranks_; i++) {
            gathered_v.push_back(value);
        }
        return gathered_v;
    }

    void set_num_cells(unsigned total_cells_) {
        num_cells_per_tile_ = total_cells_/num_ranks_;
    }

    void barrier() const {}

    std::string name() const { return "dry_run"; }

    unsigned num_ranks_;
    unsigned num_cells_per_tile_;
};

std::shared_ptr<distributed_context> dry_run_context(unsigned num_ranks) {
    return std::make_shared<distributed_context>(dry_run_context_impl(num_ranks));
}

} // namespace arb
