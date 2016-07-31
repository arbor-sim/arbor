#include <cstdlib>
#include <vector>

#include <catypes.hpp>
#include <cell_group.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <fvm_cell.hpp>
#include <recipe.hpp>

#include "trace_sampler.hpp"

namespace nest {
namespace mc {

struct model {
    using cell_group_type = cell_group<fvm::fvm_cell<double, cell_local_size_type>>;
    using time_type = cell_group_type::time_type;
    using communicator_type = communication::communicator<communication::global_policy>;

    model(const recipe &rec, cell_gid_type cell_from, cell_gid_type cell_to, time_type sample_dt);

    void reset();

    time_type run(time_type tfinal, time_type dt);

    void write_traces() const;

    void add_artificial_spike(cell_member_type source) {
        add_artificial_spike(source, t_);
    }

    void add_artificial_spike(cell_member_type source, time_type tspike);

    std::size_t num_spikes() const { return communicator_.num_spikes(); }

private:
    time_type t_ = 0.;
    std::vector<cell_group_type> cell_groups_;
    std::vector<std::vector<sample_to_trace>> samplers_;
    communicator_type communicator_;
};

} // namespace mc
} // namespace nest
