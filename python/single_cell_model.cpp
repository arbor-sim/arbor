#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

/*
#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/any.hpp>
#include <arbor/util/unique_any.hpp>

#include "conversion.hpp"
#include "morph_parse.hpp"
#include "schedule.hpp"
#include "strprintf.hpp"
*/

#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>

#include "error.hpp"

namespace pyarb {

struct probe_site {
    arb::mlocation site;
    double frequency;     // [Hz]
};

// Used internally by the single cell model.
struct single_cell_recipe: arb::recipe {
    const arb::cable_cell& cell_;

    // todo: make these references
    const std::vector<probe_site>& probes_;
    const std::vector<arb::event_generator>& generators_;

    single_cell_recipe(
            const arb::cable_cell& c,
            const std::vector<probe_site>& probes,
            const std::vector<arb::event_generator>& gens):
        cell_(c), probes_(probes), generators_(gens)
    {}

    virtual arb::cell_size_type num_cells() const override {
        return 1;
    }

    virtual arb::util::unique_any get_cell_description(arb::cell_gid_type gid) {
        return cell_;
    }

    virtual arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
        return arb::cell_kind::cable;
    }

    virtual arb::cell_size_type num_sources(arb::cell_gid_type) const override {
        return cell_.detectors().size();
    }

    virtual arb::cell_size_type num_targets(arb::cell_gid_type) const override {
        return cell_.synapses().size();
    }

    virtual arb::cell_size_type num_probes(arb::cell_gid_type)  const override {
        return probes_.size();
    }

    virtual std::vector<arb::event_generator> event_generators(arb::cell_gid_type) const override {
        return generators_;
    }

    virtual arb::cell_size_type num_gap_junction_sites(arb::cell_gid_type gid)  const override {
        return 0; // no gap junctions on a single cell model
    }

    virtual std::vector<arb::cell_connection> connections_on(arb::cell_gid_type) const override {
        return {}; // no connections on a single cell model
    }

    virtual std::vector<arb::gap_junction_connection> gap_junctions_on(arb::cell_gid_type) const override {
        return {}; // no gap junctions on a single cell model
    }

    virtual arb::probe_info get_probe(arb::cell_member_type probe_id) const override {
        // TODO: return something meaningful
        throw arb::bad_probe_id(probe_id);
    }

    virtual arb::util::any get_global_properties(arb::cell_kind) const override {
        // TODO: return something meaningful
        return arb::util::any{};
    }
};

class single_cell_model {
    arb::cable_cell cell_;
    std::vector<probe_site> probes_;
    std::vector<arb::event_generator> generators_;
    arb::simulation sim_;

public:
    single_cell_model(arb::cable_cell c):
        cell_(std::move(c)) {}

    // m.probe('voltage', arbor.location(2,0.5))
    void probe(const std::string& what, const arb::mlocation& where, double frequency) {
        if (what != "voltage") {
            throw pyarb_error(
                util::pprintf("{} does not name a valid variable to trace (currently only 'voltage' is supported)", what));
        }
        if (frequency<=0) {
            throw pyarb_error(
                util::pprintf("sampling frequency is not greater than zero", what));
        }
        if (where.branch>=cell_.num_branches()) {
            throw pyarb_error(
                util::pprintf("invalid location", what));
        }
        probes_.push_back({where, frequency});
    }

    void run(double tfinal) {
        sim_ = arb::simulation();
    }
};

} // namespace pyarb

