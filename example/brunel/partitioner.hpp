#include <communication/global_policy.hpp>
#include <domain_decomposition.hpp>
#include <hardware/node_info.hpp>
#include <recipe.hpp>

namespace arb {
    domain_decomposition decompose(const recipe& rec, const unsigned group_size) {
        struct partition_gid_domain {
            partition_gid_domain(std::vector<cell_gid_type> divs):
                gid_divisions(std::move(divs))
            {}

            int operator()(cell_gid_type gid) const {
                auto gid_part = util::partition_view(gid_divisions);
                return gid_part.index(gid);
            }

            const std::vector<cell_gid_type> gid_divisions;
        };

        cell_size_type num_global_cells = rec.num_cells();
        unsigned num_domains = communication::global_policy::size();
        int domain_id = communication::global_policy::id();

        auto dom_size = [&](unsigned dom) -> cell_gid_type {
            const cell_gid_type B = num_global_cells/num_domains;
            const cell_gid_type R = num_global_cells - num_domains*B;
            return B + (dom<R);
        };

        // Global load balance
        std::vector<cell_gid_type> gid_divisions;
        auto gid_part = make_partition(
            gid_divisions, util::transform_view(util::make_span(0, num_domains), dom_size));

        auto range = gid_part[domain_id];
        cell_size_type num_local_cells = range.second - range.first;

        unsigned num_groups = num_local_cells / group_size + (num_local_cells%group_size== 0 ? 0 : 1);
        std::vector<group_description> groups;

        // Local load balance
        // i.e. all the groups that the current rank (domain) owns
        for (unsigned i = 0; i < num_groups; ++i) {
            unsigned start = i * group_size;
            unsigned end = std::min(start + group_size, num_local_cells);
            std::vector<cell_gid_type> group_elements;

            for (unsigned j = start; j < end; ++j) {
                group_elements.push_back(j);
            }

            groups.push_back({cell_kind::lif_neuron, std::move(group_elements), backend_kind::multicore});
        }

        domain_decomposition d;
        d.num_domains = num_domains;
        d.domain_id = domain_id;
        d.num_local_cells = num_local_cells;
        d.num_global_cells = num_global_cells;
        d.groups = std::move(groups);
        d.gid_domain = partition_gid_domain(std::move(gid_divisions));

        return d;
    }
}
