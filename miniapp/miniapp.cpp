#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <memory>

#include <json/src/json.hpp>

#include <common_types.hpp>
#include <cell.hpp>
#include <cell_group.hpp>
#include <fvm_cell.hpp>
#include <mechanism_catalogue.hpp>
#include <model.hpp>
#include <threading/threading.hpp>
#include <profiling/profiler.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <util/ioutil.hpp>
#include <util/optional.hpp>

#include "io.hpp"
#include "miniapp_recipes.hpp"
#include "trace_sampler.hpp"

using namespace nest::mc;

using global_policy = communication::global_policy;

using lowered_cell = fvm::fvm_cell<double, cell_local_size_type>;
using model_type = model<lowered_cell>;
using sample_trace_type = sample_trace<model_type::time_type, model_type::value_type>;

void banner();
std::unique_ptr<recipe> make_recipe(const io::cl_options&);
std::unique_ptr<sample_trace_type> make_trace(cell_member_type probe_id, probe_spec probe);
std::pair<cell_gid_type, cell_gid_type> distribute_cells(cell_size_type ncells);

void write_trace_json(const sample_trace_type& trace, const std::string& prefix = "trace_");

int main(int argc, char** argv) {
    nest::mc::communication::global_policy_guard global_guard(argc, argv);

    try {
        std::cout << util::mask_stream(global_policy::id()==0);
        banner();

        // read parameters
        io::cl_options options = io::read_options(argc, argv);
        std::cout << options << "\n";
        std::cout << "\n";
        std::cout << ":: simulation to " << options.tfinal << " ms in "
                  << std::ceil(options.tfinal / options.dt) << " steps of "
                  << options.dt << " ms" << std::endl;

        auto recipe = make_recipe(options);
        auto cell_range = distribute_cells(recipe->num_cells());

        // build model from recipe
        model_type m(*recipe, cell_range.first, cell_range.second);

        // inject some artificial spikes, 1 per 20 neurons.
        cell_gid_type spike_cell = 20*((cell_range.first+19)/20);
        for (; spike_cell<cell_range.second; spike_cell+=20) {
            m.add_artificial_spike({spike_cell,0u});
        }

        // attach samplers to all probes
        std::vector<std::unique_ptr<sample_trace_type>> traces;
        const model_type::time_type sample_dt = 0.1;
        for (auto probe: m.probes()) {
            traces.push_back(make_trace(probe.id, probe.probe));
            m.attach_sampler(probe.id, make_trace_sampler(traces.back().get(),sample_dt));
        }

        // run model
        m.run(options.tfinal, options.dt);
        util::profiler_output(0.001);

        std::cout << "there were " << m.num_spikes() << " spikes\n";

        // save traces
        for (const auto& trace: traces) {
            write_trace_json(*trace.get());
        }
    }
    catch (io::usage_error& e) {
        // only print usage/startup errors on master
        std::cerr << util::mask_stream(global_policy::id()==0);
        std::cerr << e.what() << "\n";
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    return 0;
}

std::pair<cell_gid_type, cell_gid_type> distribute_cells(cell_size_type num_cells) {
    // crude load balancing:
    auto num_domains = communication::global_policy::size();
    auto domain_id = communication::global_policy::id();

    cell_gid_type cell_from = (cell_gid_type)(num_cells*(domain_id/(double)num_domains));
    cell_gid_type cell_to = (cell_gid_type)(num_cells*((domain_id+1)/(double)num_domains));

    return {cell_from, cell_to};
}

void banner() {
    std::cout << "====================\n";
    std::cout << "  starting miniapp\n";
    std::cout << "  - " << threading::description() << " threading support\n";
    std::cout << "  - communication policy: " << global_policy::name() << "\n";
    std::cout << "====================\n";
}

std::unique_ptr<recipe> make_recipe(const io::cl_options& options) {
    basic_recipe_param p;

    p.num_compartments = options.compartments_per_segment;
    p.num_synapses = options.all_to_all? options.cells-1: options.synapses_per_cell;
    p.synapse_type = options.syn_type;

    if (options.all_to_all) {
        return make_basic_kgraph_recipe(options.cells, p);
    }
    else {
        return make_basic_rgraph_recipe(options.cells, p);
    }
}

std::unique_ptr<sample_trace_type> make_trace(cell_member_type probe_id, probe_spec probe) {
    std::string name = "";
    std::string units = "";
    
    switch (probe.kind) {
    case probeKind::membrane_voltage:
        name = "v";
        units = "mV";
        break;
    case probeKind::membrane_current:
        name = "i";
        units = "mA/cm²";
        break; 
    default: ;
    }
    name += probe.location.segment? "dend" : "soma";

    return util::make_unique<sample_trace_type>(probe_id, name, units);
}

void write_trace_json(const sample_trace_type& trace, const std::string& prefix) {
    auto path = prefix + std::to_string(trace.probe_id.gid) +
                "." + std::to_string(trace.probe_id.index) + "_" + trace.name + ".json";

    nlohmann::json jrep;
    jrep["name"] = trace.name;
    jrep["units"] = trace.units;
    jrep["cell"] = trace.probe_id.gid;
    jrep["probe"] = trace.probe_id.index;

    auto& jt = jrep["data"]["time"];
    auto& jy = jrep["data"][trace.name];

    for (const auto& sample: trace.samples) {
        jt.push_back(sample.time);
        jy.push_back(sample.value);
    }
    std::ofstream file(path);
    file << std::setw(1) << jrep << std::endl;
}


