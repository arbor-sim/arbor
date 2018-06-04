#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <json/json.hpp>

#include <common_types.hpp>
#include <communication/distributed_context.hpp>
#include <cell.hpp>
#include <hardware/gpu.hpp>
#include <hardware/node_info.hpp>
#include <io/exporter_spike_file.hpp>
#include <load_balance.hpp>
#include <simulation.hpp>
#include <profiling/profiler.hpp>
#include <profiling/meter_manager.hpp>
#include <sampling.hpp>
#include <schedule.hpp>
#include <threading/threading.hpp>
#include <util/any.hpp>
#include <util/config.hpp>
#include <util/debug.hpp>
#include <util/ioutil.hpp>
#include <util/nop.hpp>

#include "io.hpp"
#include "miniapp_recipes.hpp"
#include "trace.hpp"

using namespace arb;

using util::any_cast;
using util::make_span;

void banner(hw::node_info, const distributed_context*);
std::unique_ptr<recipe> make_recipe(const io::cl_options&, const probe_distribution&);
sample_trace make_trace(const probe_info& probe);

void report_compartment_stats(const recipe&);

int main(int argc, char** argv) {
    // default serial context
    distributed_context context;

    try {
        #ifdef ARB_HAVE_MPI
        mpi::scoped_guard guard(&argc, &argv);
        context = mpi_context(MPI_COMM_WORLD);
        #endif

        util::meter_manager meters(&context);
        meters.start();

        std::cout << util::mask_stream(context.id()==0);
        // read parameters
        io::cl_options options = io::read_options(argc, argv, context.id()==0);

        // TODO: add dry run mode

        // Use a node description that uses the number of threads used by the
        // threading back end, and 1 gpu if available.
        hw::node_info nd;
        nd.num_cpu_cores = threading::num_threads();
        nd.num_gpus = hw::num_gpus()>0? 1: 0;
        banner(nd, &context);

        meters.checkpoint("setup");

        // determine what to attach probes to
        probe_distribution pdist;
        pdist.proportion = options.probe_ratio;
        pdist.all_segments = !options.probe_soma_only;

        auto recipe = make_recipe(options, pdist);
        if (options.report_compartments) {
            report_compartment_stats(*recipe);
        }

        auto register_exporter = [] (const io::cl_options& options) {
            return
                util::make_unique<io::exporter_spike_file>(
                    options.file_name, options.output_path,
                    options.file_extension, options.over_write);
        };

        auto decomp = partition_load_balance(*recipe, nd, &context);
        simulation sim(*recipe, decomp, &context);

        // Set up samplers for probes on local cable cells, as requested
        // by command line options.
        std::vector<sample_trace> sample_traces;
        for (const auto& g: decomp.groups) {
            if (g.kind==cell_kind::cable1d_neuron) {
                for (auto gid: g.gids) {
                    if (options.trace_max_gid && gid>*options.trace_max_gid) {
                        continue;
                    }

                    for (cell_lid_type j: make_span(0, recipe->num_probes(gid))) {
                        sample_traces.push_back(make_trace(recipe->get_probe({gid, j})));
                    }
                }
            }
        }

        auto ssched = regular_schedule(options.sample_dt);
        for (auto& trace: sample_traces) {
            sim.add_sampler(one_probe(trace.probe_id), ssched, make_simple_sampler(trace.samples));
        }

        // Specify event binning/coalescing.
        auto binning_policy =
            options.bin_dt==0? binning_kind::none:
            options.bin_regular? binning_kind::regular:
            binning_kind::following;

        sim.set_binning_policy(binning_policy, options.bin_dt);

        // Initialize the spike exporting interface
        std::unique_ptr<io::exporter_spike_file> file_exporter;
        if (options.spike_file_output) {
            if (options.single_file_per_rank) {
                file_exporter = register_exporter(options);
                sim.set_local_spike_callback(
                    [&](const std::vector<spike>& spikes) {
                        file_exporter->output(spikes);
                    });
            }
            else if(context.id()==0) {
                file_exporter = register_exporter(options);
                sim.set_global_spike_callback(
                    [&](const std::vector<spike>& spikes) {
                       file_exporter->output(spikes);
                    });
            }
        }

        meters.checkpoint("model-init");

        // run model
        sim.run(options.tfinal, options.dt);

        meters.checkpoint("model-simulate");

        // output profile and diagnostic feedback
        auto profile = util::profiler_summary();
        std::cout << profile << "\n";
        std::cout << "\nthere were " << sim.num_spikes() << " spikes\n";

        // save traces
        auto write_trace = options.trace_format=="json"? write_trace_json: write_trace_csv;
        for (const auto& trace: sample_traces) {
            write_trace(trace, options.trace_prefix);
        }

        auto report = util::make_meter_report(meters);
        std::cout << report;
        if (context.id()==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << util::to_json(report) << "\n";
        }
    }
    catch (io::usage_error& e) {
        // only print usage/startup errors on master
        std::cerr << util::mask_stream(context.id()==0);
        std::cerr << e.what() << "\n";
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    return 0;
}

void banner(hw::node_info nd, const distributed_context* ctx) {
    std::cout << "==========================================\n";
    std::cout << "  Arbor miniapp\n";
    std::cout << "  - distributed : " << ctx->size()
              << " (" << ctx->name() << ")\n";
    std::cout << "  - threads     : " << nd.num_cpu_cores
              << " (" << threading::description() << ")\n";
    std::cout << "  - gpus        : " << nd.num_gpus << "\n";
    std::cout << "==========================================\n";
}

std::unique_ptr<recipe> make_recipe(const io::cl_options& options, const probe_distribution& pdist) {
    basic_recipe_param p;

    if (options.morphologies) {
        std::cout << "loading morphologies...\n";
        p.morphologies.clear();
        load_swc_morphology_glob(p.morphologies, options.morphologies.value());
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

sample_trace make_trace(const probe_info& probe) {
    std::string name = "";
    std::string units = "";

    auto addr = any_cast<cell_probe_address>(probe.address);
    switch (addr.kind) {
    case cell_probe_address::membrane_voltage:
        name = "v";
        units = "mV";
        break;
    case cell_probe_address::membrane_current:
        name = "i";
        units = "mA/cm²";
        break;
    default: ;
    }
    name += addr.location.segment? "dend" : "soma";

    return sample_trace{probe.id, name, units};
}

void report_compartment_stats(const recipe& rec) {
    std::size_t ncell = rec.num_cells();
    std::size_t ncomp_total = 0;
    std::size_t ncomp_min = std::numeric_limits<std::size_t>::max();
    std::size_t ncomp_max = 0;

    for (std::size_t i = 0; i<ncell; ++i) {
        std::size_t ncomp = 0;
        auto c = rec.get_cell_description(i);
        if (auto ptr = any_cast<cell>(&c)) {
            ncomp = ptr->num_compartments();
        }
        ncomp_total += ncomp;
        ncomp_min = std::min(ncomp_min, ncomp);
        ncomp_max = std::max(ncomp_max, ncomp);
    }

    std::cout << "compartments/cell: min=" << ncomp_min <<"; max=" << ncomp_max << "; mean=" << (double)ncomp_total/ncell << "\n";
}
