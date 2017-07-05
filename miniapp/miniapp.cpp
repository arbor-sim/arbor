#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <json/json.hpp>

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
#include <util/config.hpp>
#include <util/debug.hpp>
#include <util/ioutil.hpp>
#include <util/nop.hpp>

#include "io.hpp"
#include "miniapp_recipes.hpp"
#include "trace_sampler.hpp"

using namespace nest::mc;

using global_policy = communication::global_policy;
using sample_trace_type = sample_trace<time_type, double>;
using file_export_type = io::exporter_spike_file<global_policy>;
void banner();
std::unique_ptr<recipe> make_recipe(const io::cl_options&, const probe_distribution&);
std::unique_ptr<sample_trace_type> make_trace(probe_record probe);
using communicator_type = communication::communicator<communication::global_policy>;

void write_trace_json(const sample_trace_type& trace, const std::string& prefix = "trace_");
void write_trace_csv(const sample_trace_type& trace, const std::string& prefix = "trace_");
void report_compartment_stats(const recipe&);

int main(int argc, char** argv) {
    nest::mc::communication::global_policy_guard global_guard(argc, argv);

    try {
        nest::mc::util::meter_manager meters;
        meters.start();

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

        meters.checkpoint("setup");

        // determine what to attach probes to
        probe_distribution pdist;
        pdist.proportion = options.probe_ratio;
        pdist.all_segments = !options.probe_soma_only;

        auto recipe = make_recipe(options, pdist);

        auto register_exporter = [] (const io::cl_options& options) {
            return
                util::make_unique<file_export_type>(
                    options.file_name, options.output_path,
                    options.file_extension, options.over_write);
        };

        group_rules rules;
        rules.policy = config::has_cuda?
            backend_policy::prefer_gpu: backend_policy::use_multicore;
        rules.target_group_size = options.group_size;
        auto decomp = domain_decomposition(*recipe, rules);

        model m(*recipe, decomp);

        if (options.report_compartments) {
            report_compartment_stats(*recipe);
        }

        // Specify event binning/coalescing.
        auto binning_policy =
            options.bin_dt==0? binning_kind::none:
            options.bin_regular? binning_kind::regular:
            binning_kind::following;

        m.set_binning_policy(binning_policy, options.bin_dt);

        // Attach samplers to all probes
        std::vector<std::unique_ptr<sample_trace_type>> traces;
        const time_type sample_dt = options.sample_dt;
        for (auto probe: m.probes()) {
            if (options.trace_max_gid && probe.id.gid>*options.trace_max_gid) {
                continue;
            }

            traces.push_back(make_trace(probe));
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

        meters.checkpoint("model-init");

        // run model
        m.run(options.tfinal, options.dt);

        meters.checkpoint("model-simulate");

        // output profile and diagnostic feedback
        util::profiler_output(0.001, options.profile_only_zero);
        std::cout << "there were " << m.num_spikes() << " spikes\n";

        // save traces
        auto write_trace = options.trace_format=="json"? write_trace_json: write_trace_csv;
        for (const auto& trace: traces) {
            write_trace(*trace.get(), options.trace_prefix);
        }

        auto report = util::make_meter_report(meters);
        std::cout << report;
        if (global_policy::id()==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << util::to_json(report) << "\n";
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

void banner() {
    std::cout << "====================\n";
    std::cout << "  starting miniapp\n";
    std::cout << "  - " << threading::description() << " threading support\n";
    std::cout << "  - communication policy: " << std::to_string(global_policy::kind()) << " (" << global_policy::size() << ")\n";
    std::cout << "  - gpu support: " << (config::has_cuda? "on": "off") << "\n";
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

    // TODO: Put all recipe parameters in the recipes file
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

std::unique_ptr<sample_trace_type> make_trace(probe_record probe) {
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

    return util::make_unique<sample_trace_type>(probe.id, name, units);
}

void write_trace_csv(const sample_trace_type& trace, const std::string& prefix) {
    auto path = prefix + std::to_string(trace.probe_id.gid) +
                "." + std::to_string(trace.probe_id.index) + "_" + trace.name + ".csv";

    std::ofstream file(path);
    file << "# cell: " << trace.probe_id.gid << "\n";
    file << "# probe: " << trace.probe_id.index << "\n";
    file << "time_ms, " << trace.name << "_" << trace.units << "\n";

    for (const auto& sample: trace.samples) {
        file << util::strprintf("% 20.15f, % 20.15f\n", sample.time, sample.value);
    }
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
        std::size_t ncomp = 0;
        auto c = rec.get_cell_description(i);
        if (auto ptr = util::any_cast<cell>(&c)) {
            ncomp = ptr->num_compartments();
        }
        ncomp_total += ncomp;
        ncomp_min = std::min(ncomp_min, ncomp);
        ncomp_max = std::max(ncomp_max, ncomp);
    }

    std::cout << "compartments/cell: min=" << ncomp_min <<"; max=" << ncomp_max << "; mean=" << (double)ncomp_total/ncell << "\n";
}
