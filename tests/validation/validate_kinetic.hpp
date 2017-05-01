#include <json/json.hpp>

#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <simple_sampler.hpp>
#include <util/rangeutil.hpp>

#include "../test_common_cells.hpp"
#include "convergence_test.hpp"
#include "trace_analysis.hpp"
#include "validation_data.hpp"

void run_kinetic_dt(
    nest::mc::backend_policy backend,
    nest::mc::cell& c,
    float t_end,
    nlohmann::json meta,
    const std::string& ref_file)
{
    using namespace nest::mc;

    float sample_dt = .025f;
    sampler_info samplers[] = {
        {"soma.mid", {0u, 0u}, simple_sampler(sample_dt)}
    };

    meta["sim"] = "nestmc";
    meta["backend_policy"] = to_string(backend);
    convergence_test_runner<float> runner("dt", samplers, meta);
    runner.load_reference_data(ref_file);

    model model(singleton_recipe{c}, {1u, backend});

    auto exclude = stimulus_ends(c);

    // use dt = 0.05, 0.02, 0.01, 0.005, 0.002,  ...
    double max_oo_dt = std::round(1.0/g_trace_io.min_dt());
    for (double base = 100; ; base *= 10) {
        for (double multiple: {5., 2., 1.}) {
            double oo_dt = base/multiple;
            if (oo_dt>max_oo_dt) goto end;

            model.reset();
            float dt = float(1./oo_dt);
            runner.run(model, dt, t_end, dt, exclude);
        }
    }

end:
    runner.report();
    runner.assert_all_convergence();
}

void validate_kinetic_kin1(nest::mc::backend_policy backend) {
    using namespace nest::mc;

    // 20 µm diameter soma with single mechanism, current probe
    cell c;
    auto soma = c.add_soma(10);
    c.add_probe({{0, 0.5}, probeKind::membrane_current});
    soma->add_mechanism(std::string("test_kin1"));

    nlohmann::json meta = {
        {"model", "test_kin1"},
        {"name", "membrane current"},
        {"units", "nA"}
    };

    run_kinetic_dt(backend, c, 100.f, meta, "numeric_kin1.json");
}

void validate_kinetic_kinlva(nest::mc::backend_policy backend) {
    using namespace nest::mc;

    // 20 µm diameter soma with single mechanism, current probe
    cell c;
    auto soma = c.add_soma(10);
    c.add_probe({{0, 0.5}, probeKind::membrane_voltage});
    c.add_stimulus({0,0.5}, {20., 130., -0.025});
    soma->add_mechanism(std::string("test_kinlva"));

    nlohmann::json meta = {
        {"model", "test_kinlva"},
        {"name", "membrane voltage"},
        {"units", "mV"}
    };

    run_kinetic_dt(backend, c, 300.f, meta, "numeric_kinlva.json");
}

