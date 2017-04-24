#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <json/json.hpp>

#include <backends/fvm.hpp>
#include <common_types.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <io/exporter_spike_file.hpp>
#include <model.hpp>
#include <profiling/profiler.hpp>
#include <profiling/meter_manager.hpp>
#include <threading/threading.hpp>
#include <util/debug.hpp>
#include <util/ioutil.hpp>
#include <util/nop.hpp>
#include <util/optional.hpp>
#include <util/span.hpp>

#include "io.hpp"
#include "miniapp_recipes.hpp"
#include "trace_sampler.hpp"

using namespace nest::mc;

using global_policy = communication::global_policy;
#ifdef NMC_HAVE_CUDA
using lowered_cell = fvm::fvm_multicell<gpu::backend>;
#else
using lowered_cell = fvm::fvm_multicell<multicore::backend>;
#endif
using model_type = model<lowered_cell>;
using sample_trace_type = sample_trace<model_type::time_type, model_type::value_type>;
using file_export_type = io::exporter_spike_file<global_policy>;
void banner();
std::unique_ptr<recipe> make_recipe(const io::cl_options&, const probe_distribution&);
std::unique_ptr<sample_trace_type> make_trace(cell_member_type probe_id, probe_spec probe);
std::pair<cell_gid_type, cell_gid_type> distribute_cells(cell_size_type ncells);
std::vector<cell_gid_type> partition_domains(cell_size_type ncells);
using communicator_type = communication::communicator<communication::global_policy>;


void write_trace_json(const sample_trace_type& trace, const std::string& prefix = "trace_");
void report_compartment_stats(const recipe&);

int main(int argc, char** argv) {
    nest::mc::communication::global_policy_guard global_guard(argc, argv);

    try {
        nest::mc::util::meter_manager meters;
        meters.checkpoint("start");

        std::cout << util::mask_stream(global_policy::id()==0);
        // read parameters
        io::cl_options options = io::read_options(argc, argv, global_policy::id()==0);

        // If compiled in dry run mode we have to set up the dry run
        // communicator to simulate the number of ranks that may have been set
        // as a command line parameter (if not, it is 1 rank by default)
        if (global_policy::kind() == communication::global_policy_kind::dryrun) {
            // Dry run mode requires that each rank has the same number of cells.
            // Here we increase the total number of cells if required to ensure
            // that this condition is satisfied.
            auto cells_per_rank = options.cells/options.dry_run_ranks;
            if (options.cells % options.dry_run_ranks) {
                ++cells_per_rank;
                options.cells = cells_per_rank*options.dry_run_ranks;
            }

            global_policy::set_sizes(options.dry_run_ranks, cells_per_rank);
        }

        banner();

        meters.checkpoint("global setup");

        // determine what to attach probes to
        probe_distribution pdist;
        pdist.proportion = options.probe_ratio;
        pdist.all_segments = !options.probe_soma_only;

        auto recipe = make_recipe(options, pdist);
        auto cell_range = distribute_cells(recipe->num_cells());
        auto domain_partition = partition_domains(recipe->num_cells());

        std::vector<cell_gid_type> group_divisions;
        for (auto i = cell_range.first; i<cell_range.second; i+=options.group_size) {
            group_divisions.push_back(i);
        }
        group_divisions.push_back(cell_range.second);

        EXPECTS(group_divisions.front() == cell_range.first);
        EXPECTS(group_divisions.back() == cell_range.second);

        auto register_exporter = [] (const io::cl_options& options) {
            return
                util::make_unique<file_export_type>(
                    options.file_name, options.output_path,
                    options.file_extension, options.over_write);
        };

        model_type m(*recipe, util::partition_view(group_divisions), util::partition_view(domain_partition));
        if (options.report_compartments) {
            report_compartment_stats(*recipe);
        }

        // Specify event binning/coalescing.
        auto binning_policy =
            options.bin_dt==0? binning_kind::none:
            options.bin_regular? binning_kind::regular:
            binning_kind::following;

        m.set_binning_policy(binning_policy, options.bin_dt);

        // Inject some artificial spikes, 1 per 20 neurons.
        cell_gid_type first_spike_cell = 20*((cell_range.first+19)/20);
        for (auto c=first_spike_cell; c<cell_range.second; c+=20) {
            m.add_artificial_spike({c, 0});
        }

        // Attach samplers to all probes
        std::vector<std::unique_ptr<sample_trace_type>> traces;
        const model_type::time_type sample_dt = 0.1;
        for (auto probe: m.probes()) {
            if (options.trace_max_gid && probe.id.gid>*options.trace_max_gid) {
                continue;
            }

            traces.push_back(make_trace(probe.id, probe.probe));
            m.attach_sampler(probe.id, make_trace_sampler(traces.back().get(), sample_dt));
        }

        // Initialize the spike exporting interface
        std::unique_ptr<file_export_type> file_exporter;
        if (options.spike_file_output) {
            if (options.single_file_per_rank) {
                file_exporter = register_exporter(options);
                m.set_local_spike_callback(
                    [&](const std::vector<spike>& spikes) {
                        file_exporter->output(spikes);
                    });
            }
            else if(communication::global_policy::id()==0) {
                file_exporter = register_exporter(options);
                m.set_global_spike_callback(
                    [&](const std::vector<spike>& spikes) {
                       file_exporter->output(spikes);
                    });
            }
        }

        meters.checkpoint("model initialization");

        // run model
        m.run(options.tfinal, options.dt);

        meters.checkpoint("time stepping");

        // output profile and diagnostic feedback
        auto const num_steps = options.tfinal / options.dt;
        util::profiler_output(0.001, m.num_cells()*num_steps, options.profile_only_zero);
        std::cout << "there were " << m.num_spikes() << " spikes\n";

        // save traces
        for (const auto& trace: traces) {
            write_trace_json(*trace.get(), options.trace_prefix);
        }

        meters.checkpoint("output");

        util::save_to_file(meters, "meters.json");
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
    // Crude load balancing:
    // divide [0, num_cells) into num_domains non-overlapping, contiguous blocks
    // of size as close to equal as possible.

    auto num_domains = communication::global_policy::size();
    auto domain_id = communication::global_policy::id();

    cell_gid_type cell_from = (cell_gid_type)(num_cells*(domain_id/(double)num_domains));
    cell_gid_type cell_to = (cell_gid_type)(num_cells*((domain_id+1)/(double)num_domains));

    return {cell_from, cell_to};
}

std::vector<cell_gid_type> partition_domains(cell_size_type num_cells) {
    using nest::mc::util::make_span;
    using nest::mc::util::partition_functional;
    
    const auto num_domains = communication::global_policy::size();
    const auto cells_per_domain = (cell_gid_type) (num_cells/(double)num_domains);

    std::vector<cell_gid_type> partition;
    make_partition(partition_functional,
                   partition,
                   make_span(0, num_domains),
                   [=] (domain_gid_type) {return cells_per_domain;});
    
    return partition;
}

void banner() {
    std::cout << "====================\n";
    std::cout << "  starting miniapp\n";
    std::cout << "  - " << threading::description() << " threading support\n";
    std::cout << "  - communication policy: " << std::to_string(global_policy::kind()) << " (" << global_policy::size() << ")\n";
#ifdef NMC_HAVE_CUDA
    std::cout << "  - gpu support: on\n";
#else
    std::cout << "  - gpu support: off\n";
#endif
    std::cout << "====================\n";
}

std::unique_ptr<recipe> make_recipe(const io::cl_options& options, const probe_distribution& pdist) {
    basic_recipe_param p;

    if (options.morphologies) {
        std::cout << "loading morphologies...\n";
        p.morphologies.clear();
        load_swc_morphology_glob(p.morphologies, options.morphologies.get());
        std::cout << "loading morphologies: " << p.morphologies.size() << " loaded.\n";
    }
    p.morphology_round_robin = options.morph_rr;

    p.num_compartments = options.compartments_per_segment;
    p.num_synapses = options.all_to_all? options.cells-1: options.synapses_per_cell;
    p.synapse_type = options.syn_type;

    if (options.all_to_all) {
        return make_basic_kgraph_recipe(options.cells, p, pdist);
    }
    else if (options.ring) {
        return make_basic_ring_recipe(options.cells, p, pdist);
    }
    else {
        return make_basic_rgraph_recipe(options.cells, p, pdist);
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

void report_compartment_stats(const recipe& rec) {
std::size_t ncell = rec.num_cells();
    std::size_t ncomp_total = 0;
    std::size_t ncomp_min = std::numeric_limits<std::size_t>::max();
    std::size_t ncomp_max = 0;

    for (std::size_t i = 0; i<ncell; ++i) {
        std::size_t ncomp = rec.get_cell(i).num_compartments();
        ncomp_total += ncomp;
        ncomp_min = std::min(ncomp_min, ncomp);
        ncomp_max = std::max(ncomp_max, ncomp);
    }

    std::cout << "compartments/cell: min=" << ncomp_min <<"; max=" << ncomp_max << "; mean=" << (double)ncomp_total/ncell << "\n";
}
