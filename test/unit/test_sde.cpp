
#include "../gtest.h"

#include <arborio/label_parse.hpp>

#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>
#include <arbor/schedule.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/util/any_ptr.hpp>

#include <arborenv/default_env.hpp>

#include "unit_test_catalogue.hpp"
#include "../simple_recipes.hpp"

using namespace arb;
using namespace arborio::literals;

// depends on the r123 generator
constexpr unsigned cache_size = 4;

// This recipe generates simple 1D cells (consisting of 2 segments)
class sde_recipe: public simple_recipe_base {
public:
    sde_recipe(unsigned ncell, unsigned ncvs, label_dict labels, decor dec)
    : simple_recipe_base()
    , ncell_(ncell) {
        // add unit test catalogue
        cell_gprop_.catalogue.import(make_unit_test_catalogue(), "");

        // set cvs explicitly
        double const cv_size = 1.0;
        dec.set_default(cv_policy_max_extent(cv_size));

        // generate cells
        unsigned const n1 = ncvs/2;
        for (unsigned int i=0; i<ncell_; ++i) {
            segment_tree tree;
            tree.append(mnpos, {i*20., 0, 0.0, 4.0}, {i*20., 0, n1*cv_size, 4.0}, 1);
            tree.append(0, {i*20., 0, ncvs*cv_size, 4.0}, 2);
            cells_.push_back(cable_cell(morphology(tree), labels, dec));
        }
    }

    cell_size_type num_cells() const override {
        return ncell_;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        return cells_[gid];
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }

    std::any get_global_properties(cell_kind) const override {
        return cell_gprop_;
    }
    
    sde_recipe& add_probe(probe_tag tag, std::any address) {
        for (unsigned i=0; i<cells_.size(); ++i) {
            simple_recipe_base::add_probe(i, tag, address);
        }
        return *this;
    }

private:
    unsigned ncell_;
    std::vector<cable_cell> cells_;
};

// generate a label dictionary with locations for synapses
label_dict make_label_dict(unsigned nsynapse) {
    auto t_1 = "(tag 1)"_reg;
    auto t_2 = "(tag 2)"_reg;
    auto locs = *arborio::parse_locset_expression("(uniform (all) 0 " + std::to_string(nsynapse-1) + " 0)");

    label_dict labels;
    labels.set("left", t_1);
    labels.set("right", t_2);
    labels.set("locs", locs);

    return labels;
}

// compute mean and variance
std::pair<double, double> statistics(std::vector<double> const & samples);

// t test
void t_test_(double t, double mu, double mean, double var, std::size_t n) {
    double const lower = mean - t*std::sqrt(var/n);
    double const upper = mean + t*std::sqrt(var/n);
    EXPECT_TRUE(lower <= mu);
    EXPECT_TRUE(mu <= upper);
}

void t_test(double mu, double mean, double var, std::size_t n) {
    double const t = 3.291; // 99.9% confidence for ∞ samples
    t_test_(t, mu, mean, var, n);
}

void t_test_1000(double mu, double mean, double var, std::size_t n) {
    double const t = 3.300; // 99.9% confidence for 1000 samples
    t_test_(t, mu, mean, var, n);
}

// Test quality of generated random numbers.
// In order to cover all situations, we have multiple cells running on multiple threads.
// Furthermore, multiple (and duplicated) stochastic point mechanisms and multiple (and duplicated)
// stochastic density mechanisms are added for each cell. The simulation is run for several time
// steps, and we sample the relevant random values each time.
// The quality of the random numbers is assesed by checking
// - uniqueness
// - mean
// - variance
TEST(sde, normality) {
    // simulation parameters
    unsigned ncells = 4;
    unsigned nsynapses = 100;
    unsigned ncvs = 100;
    double const dt = 0.5;
    unsigned nsteps = 50;

    // make labels (and locations for synapses)
    auto labels = make_label_dict(nsynapses);

    // Decorations with a bunch of stochastic processes
    // Duplicate mechanisms added on purpose in order test generation of unique random values
    decor dec;
    dec.paint("(all)"_reg , density("hh"));
    dec.paint("(all)"_reg , density("mean_reverting_stochastic_density_process"));
    dec.paint("(all)"_reg , density("mean_reverting_stochastic_density_process/sigma=0.2"));
    dec.paint("(all)"_reg , density("mean_reverting_stochastic_density_process2"));
    dec.paint("(all)"_reg , density("mean_reverting_stochastic_density_process2/sigma=0.2"));
    dec.place(*labels.locset("locs"), synapse("mean_reverting_stochastic_process"), "synapses");
    dec.place(*labels.locset("locs"), synapse("mean_reverting_stochastic_process"), "synapses");
    dec.place(*labels.locset("locs"), synapse("mean_reverting_stochastic_process2"), "synapses");
    dec.place(*labels.locset("locs"), synapse("mean_reverting_stochastic_process2"), "synapses");

    // a basic sampler: stores result in a vector
    auto sampler_ = [ncells, nsteps] (std::vector<double>& results, unsigned count,
        probe_metadata pm, std::size_t n, sample_record const * samples) {

        auto* cables_ptr = arb::util::any_cast<const arb::mcable_list*>(pm.meta);
        auto* point_info_ptr = arb::util::any_cast<const std::vector<arb::cable_probe_point_info>*>(pm.meta);
        assert((cables_ptr != nullptr) || (point_info_ptr != nullptr));

        unsigned n_entities = cables_ptr ? cables_ptr->size() : point_info_ptr->size();
        unsigned tag = pm.tag;
        assert(n_entities == count);

        unsigned offset = pm.id.gid*(nsteps-1)*n_entities + n_entities*tag;
        unsigned stride = n_entities*cache_size;
        for (std::size_t i = 0; i<n; ++i) {
            auto* value_range = arb::util::any_cast<const arb::cable_sample_range*>(samples[i].data);
            assert(value_range);
            const auto& [lo, hi] = *value_range;
            assert(n_entities==hi-lo);
            for (unsigned j = 0; j<n_entities; ++j) {
                results[offset + stride*i + j] = lo[j];
            }
        }
    };

    // concrete sampler for random variables used for white noise W
    std::vector<double> results_W(ncells*(nsteps-1)*(2*nsynapses));
    auto synapse_sampler_W = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_W, 2*nsynapses, pm, n, samples);
    };

    // concrete sampler for random variables used for white noise Q
    std::vector<double> results_Q(ncells*(nsteps-1)*(2*nsynapses));
    auto synapse_sampler_Q = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_Q, 2*nsynapses, pm, n, samples);
    };

    // concrete sampler for random variables used for white noise Z
    std::vector<double> results_Z(ncells*(nsteps-1)*(2*nsynapses));
    auto synapse_sampler_Z = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_Z, 2*nsynapses, pm, n, samples);
    };

    // concrete sampler for random variables used for white noise w
    std::vector<double> results_w1(ncells*(nsteps-1)*(ncvs));
    auto density_sampler_w1 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_w1, ncvs, pm, n, samples);
    };
    std::vector<double> results_w2(ncells*(nsteps-1)*(ncvs));
    auto density_sampler_w2 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_w2, ncvs, pm, n, samples);
    };

    // concrete sampler for random variables used for white noise q
    std::vector<double> results_q1(ncells*(nsteps-1)*(ncvs));
    auto density_sampler_q1 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_q1, ncvs, pm, n, samples);
    };
    std::vector<double> results_q2(ncells*(nsteps-1)*(ncvs));
    auto density_sampler_q2 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_q2, ncvs, pm, n, samples);
    };

    // concrete sampler for random variables used for white noise b
    std::vector<double> results_b1(ncells*(nsteps-1)*(ncvs));
    auto density_sampler_b1 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_b1, ncvs, pm, n, samples);
    };
    std::vector<double> results_b2(ncells*(nsteps-1)*(ncvs));
    auto density_sampler_b2 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_b2, ncvs, pm, n, samples);
    };

    // instantiate recipe
    sde_recipe rec(ncells, ncvs, labels, dec);

    // add the probes for W
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_point_prng_state_cell{"mean_reverting_stochastic_process", "W", c});
    }
    // add the probes for Q
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_point_prng_state_cell{"mean_reverting_stochastic_process2", "Q", c});
    }
    // add the probes for Z
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_point_prng_state_cell{"mean_reverting_stochastic_process2", "Z", c});
    }

    // add the probes for w
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_density_prng_state_cell{"mean_reverting_stochastic_density_process", "w", c});
    }
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_density_prng_state_cell{"mean_reverting_stochastic_density_process/sigma=0.2", "w", c});
    }

    // add the probes for q
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_density_prng_state_cell{"mean_reverting_stochastic_density_process2", "q", c});
    }
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_density_prng_state_cell{"mean_reverting_stochastic_density_process2/sigma=0.2", "q", c});
    }

    // add the probes for b
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_density_prng_state_cell{"mean_reverting_stochastic_density_process2", "b", c});
    }
    for (unsigned c=0; c<cache_size; ++c) {
        rec.add_probe(c, cable_probe_density_prng_state_cell{"mean_reverting_stochastic_density_process2/sigma=0.2", "b", c});
    }

    auto context = make_context({arbenv::default_concurrency(), -1});
    //auto context = make_context({1, -1});

    // build a simulation object
    simulation sim = simulation::create(rec)
        .set_context(context)
        .set_seed(42);

    // compute sampling times
    std::vector<std::vector<double>> times(cache_size, std::vector<double>(nsteps+1));
    for (unsigned c=0; c<cache_size; ++c)
        for (unsigned i=0; i<nsteps+1; ++i)
            times[c][i] = dt + c*dt + (i*cache_size)*dt;

    // add sampler for W
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c); },
            explicit_schedule(times[c]), synapse_sampler_W, sampling_policy::exact);
    }

    // add sampler for Q
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+cache_size); },
            explicit_schedule(times[c]), synapse_sampler_Q, sampling_policy::exact);
    }

    // add sampler for Z
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+2*cache_size); },
            explicit_schedule(times[c]), synapse_sampler_Z, sampling_policy::exact);
    }

    // add sampler for w
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+3*cache_size); },
            explicit_schedule(times[c]), density_sampler_w1, sampling_policy::exact);
    }
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+4*cache_size); },
            explicit_schedule(times[c]), density_sampler_w2, sampling_policy::exact);
    }

    // add sampler for q
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+5*cache_size); },
            explicit_schedule(times[c]), density_sampler_q1, sampling_policy::exact);
    }
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+6*cache_size); },
            explicit_schedule(times[c]), density_sampler_q2, sampling_policy::exact);
    }

    // add sampler for b
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+7*cache_size); },
            explicit_schedule(times[c]), density_sampler_b1, sampling_policy::exact);
    }
    for (unsigned c=0; c<cache_size; ++c) {
        sim.add_sampler([c](cell_member_type pid) { return (pid.index==c+8*cache_size); },
            explicit_schedule(times[c]), density_sampler_b2, sampling_policy::exact);
    }

    // run the simulation
    sim.run(nsteps*dt, dt);

    // collect all results
    std::vector<double> results;
    results.reserve(
            results_W.size() + results_Q.size() + results_Z.size() +
            2*results_w1.size() + 2*results_q1.size() + 2*results_b1.size());
    results.insert(results.end(), results_W.begin(), results_W.end());
    results.insert(results.end(), results_Q.begin(), results_Q.end());
    results.insert(results.end(), results_Z.begin(), results_Z.end());
    results.insert(results.end(), results_w1.begin(), results_w1.end());
    results.insert(results.end(), results_w2.begin(), results_w2.end());
    results.insert(results.end(), results_q1.begin(), results_q1.end());
    results.insert(results.end(), results_q2.begin(), results_q2.end());
    results.insert(results.end(), results_b1.begin(), results_b1.end());
    results.insert(results.end(), results_b2.begin(), results_b2.end());

    // uniqueness test
    std::sort(results.begin(), results.end());
    auto it = std::adjacent_find(results.begin(), results.end());
    EXPECT_EQ(it, results.end());

    // t test
    auto [mean, variance] = statistics(results);
    t_test(0.0, mean, variance, results.size());
}

// compute mean and variance online
struct accumulator
{
    double n_ = 0;
    double mean_ = 0;
    double var_ = 0;

    accumulator& operator()(double sample) {
        double const delta = sample - mean_;
        mean_ += delta / (++n_);
        var_ += delta * (sample - mean_);
        return *this;
    }

    double mean() const noexcept { return mean_; }
    double variance() const noexcept { return n_ > 1 ? var_/(n_-1) : 0; }
};


// Test the solver by running the mean reverting process many times and comparing statistics to
// expectation value and variance obtained from theory.
// Every synapse in every cell computes the independent evolution of 4 different stochastic
// processes. The results are evaluated at every time step and accumulated. Moreover, several
// simulations with different seed values are run which multiply the samples obtained.
TEST(sde, solver) {
    // simulation parameters
    unsigned ncells = 10;
    unsigned nsynapses = 10;
    unsigned ncvs = 1;
    double const dt = 1.0/16;
    unsigned nsteps = 200;
    unsigned nsims = 10;

    // make labels (and locations for synapses)
    auto labels = make_label_dict(nsynapses);

    // Decorations with a bunch of stochastic processes
    std::string m1 = "mean_reverting_stochastic_process/kappa=0.1,sigma=0.1";
    std::string m2 = "mean_reverting_stochastic_process/kappa=0.2,sigma=0.1";
    std::string m3 = "mean_reverting_stochastic_process/kappa=0.1,sigma=0.2";
    std::string m4 = "mean_reverting_stochastic_process/kappa=0.2,sigma=0.2";
    decor dec;
    dec.place(*labels.locset("locs"), synapse(m1), "m1");
    dec.place(*labels.locset("locs"), synapse(m2), "m2");
    dec.place(*labels.locset("locs"), synapse(m3), "m3");
    dec.place(*labels.locset("locs"), synapse(m4), "m4");

    // a basic sampler: stores result in a vector
    auto sampler_ = [ncells, nsteps] (std::vector<double>& results, unsigned count,
        probe_metadata pm, std::size_t n, sample_record const * samples) {

        auto* point_info_ptr = arb::util::any_cast<const std::vector<arb::cable_probe_point_info>*>(pm.meta);
        assert(point_info_ptr != nullptr);

        unsigned n_entities = point_info_ptr->size();
        assert(n_entities == count);

        unsigned offset = pm.id.gid*(nsteps)*n_entities;
        unsigned stride = n_entities;
        assert(n == nsteps);
        for (std::size_t i = 0; i<n; ++i) {
            auto* value_range = arb::util::any_cast<const arb::cable_sample_range*>(samples[i].data);
            assert(value_range);
            const auto& [lo, hi] = *value_range;
            assert(n_entities==hi-lo);
            for (unsigned j = 0; j<n_entities; ++j) {
                results[offset + stride*i + j] = lo[j];
            }
        }
    };

    // concrete sampler for process m1
    std::vector<double> results_m1(ncells*(nsteps)*(nsynapses));
    auto sampler_m1 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m1, nsynapses, pm, n, samples);
    };
    // concrete sampler for process m2
    std::vector<double> results_m2(ncells*(nsteps)*(nsynapses));
    auto sampler_m2 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m2, nsynapses, pm, n, samples);
    };
    // concrete sampler for process m3
    std::vector<double> results_m3(ncells*(nsteps)*(nsynapses));
    auto sampler_m3 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m3, nsynapses, pm, n, samples);
    };
    // concrete sampler for process m4
    std::vector<double> results_m4(ncells*(nsteps)*(nsynapses));
    auto sampler_m4 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m4, nsynapses, pm, n, samples);
    };

    // instantiate recipe
    sde_recipe rec(ncells, ncvs, labels, dec);

    // add probes
    rec.add_probe(1, cable_probe_point_state_cell{m1, "S"});
    rec.add_probe(2, cable_probe_point_state_cell{m2, "S"});
    rec.add_probe(3, cable_probe_point_state_cell{m3, "S"});
    rec.add_probe(4, cable_probe_point_state_cell{m4, "S"});

    // results are accumulated for each time step
    std::vector<accumulator> stats_m1(nsteps);
    std::vector<accumulator> stats_m2(nsteps);
    std::vector<accumulator> stats_m3(nsteps);
    std::vector<accumulator> stats_m4(nsteps);


    auto context = make_context({arbenv::default_concurrency(), -1});
    //auto context = make_context({1, -1});

    for (unsigned s=0; s<nsims; ++s)
    {
        // build a simulation object
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_seed(s);

        // add sampler
        sim.add_sampler([](cell_member_type pid) { return (pid.index==0); }, regular_schedule(dt),
            sampler_m1, sampling_policy::exact);
        sim.add_sampler([](cell_member_type pid) { return (pid.index==1); }, regular_schedule(dt),
            sampler_m2, sampling_policy::exact);
        sim.add_sampler([](cell_member_type pid) { return (pid.index==2); }, regular_schedule(dt),
            sampler_m3, sampling_policy::exact);
        sim.add_sampler([](cell_member_type pid) { return (pid.index==3); }, regular_schedule(dt),
            sampler_m4, sampling_policy::exact);
        
        // run the simulation
        sim.run(nsteps*dt, dt);

        for (unsigned int k=0; k<ncells; ++k){
            for (unsigned int i=0; i<nsteps; ++i){
                for (unsigned int j=0; j<(nsynapses); ++j){
                    stats_m1[i](results_m1[k*(nsteps)*(nsynapses) + i*(nsynapses) + j ]);
                    stats_m2[i](results_m2[k*(nsteps)*(nsynapses) + i*(nsynapses) + j ]);
                    stats_m3[i](results_m3[k*(nsteps)*(nsynapses) + i*(nsynapses) + j ]);
                    stats_m4[i](results_m4[k*(nsteps)*(nsynapses) + i*(nsynapses) + j ]);
                }
            }
        }
    }

    // analytical solutions
    auto expected = [](double kappa, double sigma, double t) -> std::pair<double,double> {
        double const mu = 1.0;
        double const S_0 = 2.0;
        return {
            mu - (mu-S_0)*std::exp(-kappa*t),
            (sigma*sigma/(2*kappa))*(1.0 - std::exp(-2*kappa*t))
        };
    };
    auto expected_m1 = [&](double t) { return expected(0.1, 0.1, t); };
    auto expected_m2 = [&](double t) { return expected(0.2, 0.1, t); };
    auto expected_m3 = [&](double t) { return expected(0.1, 0.2, t); };
    auto expected_m4 = [&](double t) { return expected(0.2, 0.2, t); };

    // t test
    auto test = [&] (auto func, auto const& stats) {
        for (unsigned int i=0; i<nsteps; ++i){
            auto [mu, sigma] = func(i*dt);
            double const mean = stats[i].mean();
            double const var = stats[i].variance();
            std::size_t const n = stats[i].n_;
            t_test_1000(mu, mean, var, n);
        }
    };
    test(expected_m1, stats_m1);
    test(expected_m2, stats_m2);
    test(expected_m3, stats_m3);
    test(expected_m4, stats_m4);
}

// compute mean and variance
std::pair<double, double> statistics(std::vector<double> const & samples)
{
    accumulator a;
    for (auto sample : samples) a(sample);
    return {a.mean(), a.variance()};
}
