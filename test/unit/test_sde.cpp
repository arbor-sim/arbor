
#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <cmath>

#include <arborio/label_parse.hpp>

#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>
#include <arbor/schedule.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/util/any_ptr.hpp>
#ifdef ARB_GPU_ENABLED
#include "memory/gpu_wrappers.hpp"
#endif

#include <arborenv/default_env.hpp>

#include "unit_test_catalogue.hpp"
#include "../simple_recipes.hpp"

// ============================
// helper classes and functions
// ============================

// data storage, can be filled concurrently
template<typename T>
struct archive {
    std::vector<T> data_;
    std::atomic<std::size_t> index_ = 0u;

    archive(std::size_t n) : data_(n) {}

    archive(const archive& other)
    : data_{other.data_}
    , index_{other.index_.load()}
    {}

    T* claim(std::size_t n) {
        std::size_t expected, desired;
        do {
            expected = index_.load();
            desired = expected + n;
        }
        while (!index_.compare_exchange_weak(expected, desired));
        return &(data_[expected]);
    }

    // not thread safe
    void reset() {
        std::fill(data_.begin(), data_.end(), T{0});
        index_.store(0u);
    }

    std::size_t size() const noexcept { return data_.size(); }
};

// compute mean and variance online
// uses Welford-Knuth algorithm for the variance
struct accumulator {
    std::size_t n_ = 0;
    double mean_ = 0;
    double var_ = 0;

    accumulator& operator()(double sample) {
        double const delta = sample - mean_;
        mean_ += delta / (++n_);
        var_ += delta * (sample - mean_);
        return *this;
    }

    std::size_t n() const noexcept { return n_; }
    double mean() const noexcept { return mean_; }
    double variance() const noexcept { return n_ > 1 ? var_/(n_-1) : 0; }
};

// Cumulative distribtion function of normal distribution
double cdf_0(double x, double mu, double sigma) {
    return 0.5*(1 + std::erf((x-mu)/(sigma*std::sqrt(2))));
}

// Supremum of distances of cumulative sample distribution function with respect to
// normal cumulative distribution function
template<typename T>
double cdf_distance(const std::vector<T>& ordered_samples, double mu, double sigma) {
    const std::size_t n = ordered_samples.size();
    const double n_inv = 1.0/n;
    double D_sup = 0;
    for (std::size_t i=0; i<n; ++i) {
        const double x = ordered_samples[i];
        const double F_0 = cdf_0(x, mu, sigma);
        const double d_u = std::abs((i+1)*n_inv - F_0);
        const double d_l = std::abs(i*n_inv - F_0);
        D_sup = std::max(D_sup, d_u);
        D_sup = std::max(D_sup, d_l);
    }
    return D_sup;
}

// Kolmogorov-Smirnov test
// Returns true if null hypothesis (samples are normally distributed) can not be rejected at
// significance level alpha = 5%
template<typename T>
bool ks(const std::vector<T>& ordered_samples, double mu, double sigma) {
    const std::size_t n = ordered_samples.size();
    const double D_sup = cdf_distance(ordered_samples, mu, sigma);
    // Kolmogorov statistic K
    double K = std::sqrt(n)*D_sup;
    // Critical value for significance level alpha (approximation for n > 35)
    const double alpha = 0.05;
    const double K_c = std::sqrt(-0.5*std::log(0.5*alpha));
    const bool ret = (K < K_c);
    if (!ret) {
        std::cout << "ks test failed: "
            << K << " is not smaller than critical value " << K_c << "\n";
    }
    return ret;
}

// Anderson-Darling test
// Returns true if null hypothesis (samples are normally distributed) can not be rejected at
// significance level alpha = 5%
template<typename T>
bool ad(const std::vector<T>& ordered_samples, double mu, double sigma) {
    const std::size_t n = ordered_samples.size();
    double a_mean = 0;
    for (std::size_t i=0; i<n; ++i) {
        const double x = ordered_samples[i];
        const double a =
            (2*(i+1)-1)*std::log(cdf_0(x, mu, sigma)) +
            (2*(n-(i+1))+1)*std::log(1-cdf_0(x, mu, sigma));
        double const delta = a - a_mean;
        a_mean += delta / (i+1);
    }
    // Anderson-Darling distance
    const double A2 = -a_mean - n;
    // Critical value for significance level alpha = 5% (n > 5)
    const double A2_c = 2.492;
    const bool ret = (A2 < A2_c);
    if (!ret) {
        std::cout << "ad test failed: "
            << A2 << " is not smaller than critical value " << A2_c << "\n";
    }
    return ret;
}

// Student's t-test
// Returns true if null hypothesis (sample mean is equal to mu) can not be rejected at
// significance level alpha = 5%
bool t_test_mean(double mu, double sample_mean, double sample_variance, std::size_t n) {
    // t statistic
    const double t = std::sqrt(n/sample_variance)*(sample_mean - mu);
    // Critical value for significance level alpha = 5% (n = ∞)
    const double t_c = 1.960;
    const bool ret = (t < t_c);
    if (!ret) {
        std::cout << "t test failed: "
            << t << " is not smaller than critical value " << t_c
            << ", sample_mean = " << sample_mean << ", expected mean " << mu << "\n";
    }
    return ret;
}

// Chi^2 test
// Returns true if null hypothesis (sample variance is equal to sigma_squared) can not be rejected
// at significance level alpha = 5%
bool chi_2_test_variance(double sigma_squared, double sample_mean, double sample_variance, std::size_t n) {
    // compute statistic following chi squared distribution
    const double c = (n-1)*sample_variance/sigma_squared;
    // we assume many samples, so chi squared distribution becomes normal distribution
    // compute standard normally distributed variable
    const double c_n = (c-n)/std::sqrt(2*n);
    // critical value at 5%
    const double c_n_c = 1.959963984540;
    const bool ret = ((c_n < c_n_c) && (c_n > -c_n_c));
    if (!ret) {
        std::cout << "chi^2 test failed: "
            << c_n << " is not between critical values (" << -c_n_c << ", " << c_n_c << ")"
            << ", sample_variance = " << sample_variance
            << ", expected variance = " << sigma_squared 
            << ", number of samples = " << n << "\n";
    }
    return ret;
}

// combined statistical tests (assumes data is sorted)
template<typename T>
void test_statistics(const std::vector<T>& ordered_samples, double mu, double sigma) {
    // uniqueness test
    auto it = std::adjacent_find(ordered_samples.begin(), ordered_samples.end());
    EXPECT_EQ(it, ordered_samples.end());

    // goodness-of-fit tests (checks whether normally distributed)
    EXPECT_TRUE(ks(ordered_samples, mu, sigma));
    EXPECT_TRUE(ad(ordered_samples, mu, sigma));

    // mean and variance tests (assumes normal distribution)
    accumulator acc;
    for (auto x : ordered_samples) acc(x);
    EXPECT_TRUE(t_test_mean(mu, acc.mean(), acc.variance(), acc.n()));
    EXPECT_TRUE(chi_2_test_variance(sigma*sigma, acc.mean(), acc.variance(), acc.n()));
}

// =================================
// Declarations and global variables
// =================================

using namespace arb;
using namespace arborio::literals;

// forward declaration
class sde_recipe;

// global variables used in overriden advance methods
sde_recipe* rec_ptr = nullptr;
archive<arb_value_type>* archive_ptr = nullptr;

// declaration of overriding advance methods
void advance_mean_reverting_stochastic_density_process(arb_mechanism_ppack* pp);
void advance_mean_reverting_stochastic_density_process2(arb_mechanism_ppack* pp);
void advance_mean_reverting_stochastic_process(arb_mechanism_ppack* pp);
void advance_mean_reverting_stochastic_process2(arb_mechanism_ppack* pp);

// ===========================================
// Recipe with multiple mechanism replacements
// ===========================================

// helper macro for replacing a mechanism's advance method
#define REPLACE_IMPLEMENTATION(MECH)                                                               \
{                                                                                                  \
    auto inst = cell_gprop_.catalogue.instance(arb_backend_kind_cpu, #MECH);                       \
    advance_ ## MECH = inst.mech->iface_.advance_state;                                            \
    inst.mech->iface_.advance_state = &::advance_ ## MECH;                                         \
    cell_gprop_.catalogue.register_implementation(#MECH, std::move(inst.mech));                    \
}

// This recipe generates simple 1D cells (consisting of 2 segments) and optionally replaces the
// mechanism's implementation by dispatching to a user defined advance function.
class sde_recipe: public simple_recipe_base {
public:
    sde_recipe(unsigned ncell, unsigned ncvs, label_dict labels, decor dec,
        bool replace_implementation = true)
    : simple_recipe_base()
    , ncell_(ncell) {
        // add unit test catalogue
        cell_gprop_.catalogue.import(make_unit_test_catalogue(), "");

        // replace mechanisms' advance methods
        if (replace_implementation) {
            rec_ptr = this;
            REPLACE_IMPLEMENTATION(mean_reverting_stochastic_density_process)
            REPLACE_IMPLEMENTATION(mean_reverting_stochastic_density_process2)
            REPLACE_IMPLEMENTATION(mean_reverting_stochastic_process)
            REPLACE_IMPLEMENTATION(mean_reverting_stochastic_process2)
        }

        // set cvs explicitly
        double const cv_size = 1.0;
        dec.set_default(cv_policy_max_extent(cv_size));

        // generate cells
        unsigned const n1 = ncvs/2;
        for (unsigned int i=0; i<ncell_; ++i) {
            segment_tree tree;
            tree.append(mnpos, {i*20., 0, 0.0, 4.0}, {i*20., 0, n1*cv_size, 4.0}, 1);
            tree.append(0, {i*20., 0, ncvs*cv_size, 4.0}, 2);
            cells_.push_back(cable_cell(morphology(tree), dec, labels));
        }
    }

    cell_size_type num_cells() const override { return ncell_; }

    util::unique_any get_cell_description(cell_gid_type gid) const override { return cells_[gid]; }

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    std::any get_global_properties(cell_kind) const override { return cell_gprop_; }
    
    sde_recipe& add_probe_all_gids(probe_tag tag, std::any address) {
        for (unsigned i=0; i<cells_.size(); ++i) {
            simple_recipe_base::add_probe(i, tag, address);
        }
        return *this;
    }

private:
    unsigned ncell_;
    std::vector<cable_cell> cells_;

public:
    // pointers to original advance methods
    arb_mechanism_method advance_mean_reverting_stochastic_density_process;
    arb_mechanism_method advance_mean_reverting_stochastic_density_process2;
    arb_mechanism_method advance_mean_reverting_stochastic_process;
    arb_mechanism_method advance_mean_reverting_stochastic_process2;
};

// generic advance method used for all pertinent mechanisms
// first argument indicates the number of random variables
void advance_common(unsigned int n_rv, arb_mechanism_ppack* pp) {
    const auto width = pp->width;
    arb_value_type* ptr = archive_ptr->claim(width * n_rv);
    for (arb_size_type j=0; j<n_rv; ++j) {
        for (arb_size_type i=0; i<width; ++i) {
            ptr[j*width+i] = pp->random_numbers[j][i];
        }
    }
}

// overriden advance methods dispatch to common implementation and then to original method
void advance_mean_reverting_stochastic_density_process(arb_mechanism_ppack* pp) {
    advance_common(1, pp);
    rec_ptr->advance_mean_reverting_stochastic_density_process(pp);
}
void advance_mean_reverting_stochastic_density_process2(arb_mechanism_ppack* pp) {
    advance_common(2, pp);
    rec_ptr->advance_mean_reverting_stochastic_density_process2(pp);
}
void advance_mean_reverting_stochastic_process(arb_mechanism_ppack* pp) {
    advance_common(1, pp);
    rec_ptr->advance_mean_reverting_stochastic_process(pp);
}
void advance_mean_reverting_stochastic_process2(arb_mechanism_ppack* pp) {
    advance_common(2, pp);
    rec_ptr->advance_mean_reverting_stochastic_process2(pp);
}

// =====
// Tests
// =====

// generate a label dictionary with locations for synapses
label_dict make_label_dict(unsigned nsynapse) {
    label_dict labels;
    auto t_1 = "(tag 1)"_reg;
    labels.set("left", t_1);
    auto t_2 = "(tag 2)"_reg;
    labels.set("right", t_2);
    auto locs = *arborio::parse_locset_expression("(uniform (all) 0 " + std::to_string(nsynapse-1) + " 0)");
    labels.set("locs", locs);
    return labels;
}

// Test quality of generated random numbers.
// In order to cover all situations, we have multiple cells running on multiple threads, using
// different load balancing strategies. Furthermore, multiple (and duplicated) stochastic point
// mechanisms and multiple (and duplicated) stochastic density mechanisms are added for each cell.
// The simulations are run for several time steps, and we sample the relevant random values in each
// time step.
// The quality of the random numbers is assesed by checking
// - consistency
//     The same simulations with varying number of threads and different partitioning must give the
//     same random values
// - uniqueness
//     The random values must be unique and cannot repeat within a simulation and with respect to
//     another simulation with a different seed
// - goodness of fit
//     We use Kolmogorov-Smirnoff and Anderson-Darling tests to check that we can not reject the
//     null hypothesis that the values are (standard) normally distributed
// - mean and variance
//     Assuming a normal distribution, we check that the mean and the variance are close to the
//     expected values using a student's t-test and a chi^2 test, respectively.
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

    // instantiate recipe
    sde_recipe rec(ncells, ncvs, labels, dec, true);

    // calculate storage needs
    // - 2 point processes with 1 random variable each
    // - 2 point processes with 2 random variables each
    // - same for density processes
    std::size_t n_rv_synapses = ncells*nsynapses*(1+1+2+2);
    std::size_t n_rv_densities = ncells*ncvs*(1+1+2+2);
    std::size_t n_rv_per_dt = n_rv_synapses + n_rv_densities;
    std::size_t n_rv = n_rv_per_dt*(nsteps+1);

    // setup storage
    std::vector<arb_value_type> data;
    archive<arb_value_type> arch(n_rv);
    archive_ptr = &arch;

    // Run a bunch of different simulations, with different concurrency and load balancing
    // Check that generated random numbers are identical

    // single-threaded, one cell per thread
    {
        auto context = make_context({1, -1});
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_seed(42);
        sim.run(nsteps*dt, dt);

        // sort data and store for comparison
        std::sort(arch.data_.begin(), arch.data_.end());
        data = arch.data_;
    }

    // multi-threaded, one cell per thread
    {
        auto context = make_context({arbenv::default_concurrency(), -1});
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_seed(42);
        arch.reset();
        sim.run(nsteps*dt, dt);

        // sort data
        std::sort(arch.data_.begin(), arch.data_.end());

        // check for equality
        EXPECT_TRUE(std::equal(data.begin(), data.end(), arch.data_.begin()));
    }

    // single-threaded, 2 cells per thread
    {
        auto context = make_context({1, -1});
        partition_hint_map hint_map;
        partition_hint h;
        h.cpu_group_size = 2;
        hint_map[cell_kind::cable] = h;
        auto decomp = partition_load_balance(rec, context, hint_map);
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_decomposition(decomp)
            .set_seed(42);
        arch.reset();
        sim.run(nsteps*dt, dt);

        // sort data
        std::sort(arch.data_.begin(), arch.data_.end());

        // check for equality
        EXPECT_TRUE(std::equal(data.begin(), data.end(), arch.data_.begin()));
    }

    // multi-threaded, 2 cells per thread
    {
        auto context = make_context({arbenv::default_concurrency(), -1});
        partition_hint_map hint_map;
        partition_hint h;
        h.cpu_group_size = 2;
        hint_map[cell_kind::cable] = h;
        auto decomp = partition_load_balance(rec, context, hint_map);
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_decomposition(decomp)
            .set_seed(42);
        arch.reset();
        sim.run(nsteps*dt, dt);

        // sort data
        std::sort(arch.data_.begin(), arch.data_.end());

        // check for equality
        EXPECT_TRUE(std::equal(data.begin(), data.end(), arch.data_.begin()));
    }

    // Run another simulation with different seed and check that values are different
    {
        auto context = make_context({arbenv::default_concurrency(), -1});
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_seed(0);
        arch.reset();
        sim.run(nsteps*dt, dt);

        // sort data
        std::sort(arch.data_.begin(), arch.data_.end());

        // merge sort into combined data
        std::vector<arb_value_type> combined(data.size()*2);
        std::merge(
            arch.data_.begin(), arch.data_.end(),
            data.begin(), data.end(),
            combined.begin());

        // uniqueness test
        auto it = std::adjacent_find(combined.begin(), combined.end());
        EXPECT_EQ(it, combined.end());
    }

    // test statistics
    test_statistics(data, 0.0, 1.0);

    // test statistics with different seed
    test_statistics(arch.data_, 0.0, 1.0);
}

// Test the solver by running the mean reverting process many times.
// Every synapse in every cell computes the independent evolution of 4 different stochastic
// processes. The results are evaluated at every time step and accumulated. Moreover, several
// simulations with different seed values are run which multiply the samples obtained.
// Quality of results is checked by comparing to expected mean and expected standard deviation
// (relative error less than 1%).  We could also test the statistics with t test/ chi^2 test for
// mean and variance - this is possible since the solution of the SDE is normally distributed
// (linear SDE). However, due to the slow convergence of the variance, the sample size needs to be
// very large, and the test would take too long to complete.
TEST(sde, solver) {
    // simulation parameters
    unsigned ncells = 4;
    unsigned nsynapses = 2000;
    unsigned ncvs = 1;
    double const dt = 1.0/512; // need relatively small time steps due to low accuracy
    unsigned nsteps = 100;
    unsigned nsims = 4;

    // make labels (and locations for synapses)
    auto labels = make_label_dict(nsynapses);

    // Decorations with a bunch of stochastic processes
    std::string m1 = "mean_reverting_stochastic_process/kappa=0.1,sigma=0.1";
    std::string m2 = "mean_reverting_stochastic_process/kappa=0.01,sigma=0.1";
    std::string m3 = "mean_reverting_stochastic_process/kappa=0.1,sigma=0.05";
    std::string m4 = "mean_reverting_stochastic_process/kappa=0.01,sigma=0.05";
    decor dec;
    dec.place(*labels.locset("locs"), synapse(m1), "m1");
    dec.place(*labels.locset("locs"), synapse(m2), "m2");
    dec.place(*labels.locset("locs"), synapse(m3), "m3");
    dec.place(*labels.locset("locs"), synapse(m4), "m4");

    // a basic sampler: stores result in a vector
    auto sampler_ = [nsteps] (std::vector<arb_value_type>& results, unsigned count,
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
    std::vector<arb_value_type> results_m1(ncells*(nsteps)*(nsynapses));
    auto sampler_m1 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m1, nsynapses, pm, n, samples);
    };
    // concrete sampler for process m2
    std::vector<arb_value_type> results_m2(ncells*(nsteps)*(nsynapses));
    auto sampler_m2 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m2, nsynapses, pm, n, samples);
    };
    // concrete sampler for process m3
    std::vector<arb_value_type> results_m3(ncells*(nsteps)*(nsynapses));
    auto sampler_m3 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m3, nsynapses, pm, n, samples);
    };
    // concrete sampler for process m4
    std::vector<arb_value_type> results_m4(ncells*(nsteps)*(nsynapses));
    auto sampler_m4 = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_m4, nsynapses, pm, n, samples);
    };

    // instantiate recipe
    sde_recipe rec(ncells, ncvs, labels, dec, false);

    // add probes
    rec.add_probe_all_gids(1, cable_probe_point_state_cell{m1, "S"});
    rec.add_probe_all_gids(2, cable_probe_point_state_cell{m2, "S"});
    rec.add_probe_all_gids(3, cable_probe_point_state_cell{m3, "S"});
    rec.add_probe_all_gids(4, cable_probe_point_state_cell{m4, "S"});

    // results are accumulated for each time step
    std::vector<accumulator> stats_m1(nsteps);
    std::vector<accumulator> stats_m2(nsteps);
    std::vector<accumulator> stats_m3(nsteps);
    std::vector<accumulator> stats_m4(nsteps);

    // context
    auto context = make_context({arbenv::default_concurrency(), -1});

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

        // accumulate statistics for sampled data
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
        const double mu = 1.0;
        const double S_0 = 2.0;
        return {
            mu - (mu-S_0)*std::exp(-kappa*t),
            (sigma*sigma/(2*kappa))*(1.0 - std::exp(-2*kappa*t))
        };
    };
    auto expected_m1 = [&](double t) { return expected(0.1, 0.1, t); };
    auto expected_m2 = [&](double t) { return expected(0.01, 0.1, t); };
    auto expected_m3 = [&](double t) { return expected(0.1, 0.05, t); };
    auto expected_m4 = [&](double t) { return expected(0.01, 0.05, t); };

    auto test = [&] (auto func, const auto& stats) {
        for (unsigned int i=1; i<nsteps; ++i) {
            auto [mu, sigma_squared] = func(i*dt);
            double const mean = stats[i].mean();
            double const var = stats[i].variance();

            auto relative_error = [](double result, double expected) {
                return std::abs(result-expected)/expected;
            };

            EXPECT_LT( relative_error(mean, mu)*100, 1.0 );
            EXPECT_LT( relative_error(std::sqrt(var), std::sqrt(sigma_squared))*100, 2.0 );

            // using statistcal tests:
            //std::size_t const n = stats[i].n();
            //EXPECT_TRUE(t_test_mean(mu, mean, var, n));
            //EXPECT_TRUE(chi_2_test_variance(sigma_squared, mean, var, n));
        }
    };
    test(expected_m1, stats_m1);
    test(expected_m2, stats_m2);
    test(expected_m3, stats_m3);
    test(expected_m4, stats_m4);
}

// coupled linear SDE with 2 white noise sources
TEST(sde, coupled) {
    // simulation parameters
    unsigned ncells = 4;
    unsigned nsynapses = 2000;
    unsigned ncvs = 1;
    double const dt = 1.0/512; // need relatively small time steps due to low accuracy
    unsigned nsteps = 100;
    unsigned nsims = 4;

    // make labels (and locations for synapses)
    auto labels = make_label_dict(nsynapses);

    // Decorations
    std::string m1 = "stochastic_volatility";
    decor dec;
    dec.place(*labels.locset("locs"), synapse(m1), "m1");

    // a basic sampler: stores result in a vector
    auto sampler_ = [nsteps] (std::vector<arb_value_type>& results, unsigned count,
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

    // concrete sampler for P
    std::vector<arb_value_type> results_P(ncells*(nsteps)*(nsynapses));
    auto sampler_P = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_P, nsynapses, pm, n, samples);
    };
    // concrete sampler for sigma
    std::vector<arb_value_type> results_sigma(ncells*(nsteps)*(nsynapses));
    auto sampler_sigma = [&] (probe_metadata pm, std::size_t n, sample_record const * samples) {
        sampler_(results_sigma, nsynapses, pm, n, samples);
    };

    // instantiate recipe
    sde_recipe rec(ncells, ncvs, labels, dec, false);

    // add probes
    rec.add_probe_all_gids(1, cable_probe_point_state_cell{m1, "P"});
    rec.add_probe_all_gids(2, cable_probe_point_state_cell{m1, "sigma"});

    // results are accumulated for each time step
    std::vector<accumulator> stats_P(nsteps);
    std::vector<accumulator> stats_sigma(nsteps);
    std::vector<accumulator> stats_Psigma(nsteps);

    // context
    auto context = make_context({arbenv::default_concurrency(), -1});

    for (unsigned s=0; s<nsims; ++s)
    {
        // build a simulation object
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_seed(s);

        // add sampler
        sim.add_sampler([](cell_member_type pid) { return (pid.index==0); }, regular_schedule(dt),
            sampler_P, sampling_policy::exact);
        sim.add_sampler([](cell_member_type pid) { return (pid.index==1); }, regular_schedule(dt),
            sampler_sigma, sampling_policy::exact);

        // run the simulation
        sim.run(nsteps*dt, dt);

        // accumulate statistics for sampled data
        for (unsigned int k=0; k<ncells; ++k){
            for (unsigned int i=0; i<nsteps; ++i){
                for (unsigned int j=0; j<(nsynapses); ++j){
                    const double P = results_P[k*(nsteps)*(nsynapses) + i*(nsynapses) + j ];
                    const double sigma = results_sigma[k*(nsteps)*(nsynapses) + i*(nsynapses) + j ];
                    stats_P[i](P);
                    stats_sigma[i](sigma);
                    stats_Psigma[i](P*sigma);
                }
            }
        }
    }

    // analytical solutions
    auto expected = [](double t, double mu, double theta, double kappa, double sigma_1,
        double P0, double sigma0) -> std::array<double,4> {
        // E[P]                                        = mu*t + P0
        // E[sigma]                                    = (simga0 - theta)*exp(-kappa*t) + theta
        // Cov[P,P]         = E[P^2]-E[P]^2            = complicated
        // Cov[P,sigma]     = E[P*sigma]-E[P]*E[sigma] = 0
        // Cov[sigma,sigma] = E[sigma^2]-E[sigma]^2    = sigma_1^2*(1-exp(-2*kappa*t)/(2*kappa)
        return {
            mu*t + P0,
            (sigma0 - theta)*std::exp(-kappa*t) + theta,
            sigma_1*sigma_1*(1.0-std::exp(-2*kappa*t))/(2.0*kappa)
        };
    };

    for (unsigned int i=1; i<nsteps; ++i) {
        auto ex = expected(i*dt, 0.1, 0.1, 0.1, 0.1, 1, 0.2);

        const double E_P = ex[0];
        const double E_sigma = ex[1];
        const double Var_sigma = ex[2];

        const double stats_Cov_P_sigma =
            stats_Psigma[i].mean() - stats_P[i].mean()*stats_sigma[i].mean();

        auto relative_error = [](double result, double expected) {
            return std::abs(result-expected)/expected;
        };

        EXPECT_LT( relative_error(stats_P[i].mean(), E_P)*100, 1.0 );
        EXPECT_LT( relative_error(stats_sigma[i].mean(), E_sigma)*100,  1.0 );
        EXPECT_LT( relative_error(std::sqrt(stats_sigma[i].variance()),
            std::sqrt(Var_sigma))*100, 2.0 );
        EXPECT_LT( stats_Cov_P_sigma, 1.0e-4);
    }
}


#ifdef ARB_GPU_ENABLED

// forward declaration
class sde_recipe_gpu;

// global variables used in overriden advance methods
sde_recipe_gpu* rec_gpu_ptr = nullptr;

// declaration of overriding advance methods
void advance_mech_cpu(arb_mechanism_ppack* pp);
void advance_mech_gpu(arb_mechanism_ppack* pp);

class sde_recipe_gpu: public simple_recipe_base {
public:
    sde_recipe_gpu(unsigned ncell, unsigned ncvs, label_dict labels, decor dec)
    : simple_recipe_base()
    , ncell_(ncell) {
        // add unit test catalogue
        cell_gprop_.catalogue.import(make_unit_test_catalogue(), "");

        rec_gpu_ptr = this;

        // replace mechanisms' advance methods
        auto inst_cpu = cell_gprop_.catalogue.instance(arb_backend_kind_cpu, "mean_reverting_stochastic_process");
        advance_mech_cpu = inst_cpu.mech->iface_.advance_state;
        inst_cpu.mech->iface_.advance_state = &::advance_mech_cpu;
        cell_gprop_.catalogue.register_implementation("mean_reverting_stochastic_process", std::move(inst_cpu.mech));

        auto inst_gpu = cell_gprop_.catalogue.instance(arb_backend_kind_gpu, "mean_reverting_stochastic_process");
        advance_mech_gpu = inst_gpu.mech->iface_.advance_state;
        inst_gpu.mech->iface_.advance_state = &::advance_mech_gpu;
        cell_gprop_.catalogue.register_implementation("mean_reverting_stochastic_process", std::move(inst_gpu.mech));

        // set cvs explicitly
        double const cv_size = 1.0;
        dec.set_default(cv_policy_max_extent(cv_size));

        // generate cells
        unsigned const n1 = ncvs/2;
        for (unsigned int i=0; i<ncell_; ++i) {
            segment_tree tree;
            tree.append(mnpos, {i*20., 0, 0.0, 4.0}, {i*20., 0, n1*cv_size, 4.0}, 1);
            tree.append(0, {i*20., 0, ncvs*cv_size, 4.0}, 2);
            cells_.push_back(cable_cell(morphology(tree), dec, labels));
        }
    }

    cell_size_type num_cells() const override { return ncell_; }

    util::unique_any get_cell_description(cell_gid_type gid) const override { return cells_[gid]; }

    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }

    std::any get_global_properties(cell_kind) const override { return cell_gprop_; }

private:
    unsigned ncell_;
    std::vector<cable_cell> cells_;

public:
    // pointers to original advance methods
    arb_mechanism_method advance_mech_cpu;
    arb_mechanism_method advance_mech_gpu;
};

void advance_mech_cpu(arb_mechanism_ppack* pp) {
    const auto width = pp->width;
    arb_value_type* ptr = archive_ptr->claim(width);
    for (arb_size_type i=0; i<width; ++i) {
        ptr[i] = pp->random_numbers[0][i];
    }
    rec_gpu_ptr->advance_mech_cpu(pp);
}

void advance_mech_gpu(arb_mechanism_ppack* pp) {
    const auto width = pp->width;
    arb_value_type* ptr = archive_ptr->claim(width);
    // copy gpu pointer
    arb_value_type const * gpu_ptr;
    memory::gpu_memcpy_d2h(&gpu_ptr, pp->random_numbers, sizeof(arb_value_type const*));
    // copy data
    memory::gpu_memcpy_d2h(ptr, gpu_ptr, width*sizeof(arb_value_type));
    //tmp.resize(width);
    rec_gpu_ptr->advance_mech_gpu(pp);
}

template<class T>
bool almost_equal(T x, T y, unsigned ulp, T abs_tol = std::numeric_limits<T>::min()) {
    if (x == y) return true;
    const auto diff = std::abs(x-y);
    const auto norm = std::min(std::abs(x) + std::abs(y), std::numeric_limits<T>::max());
    return (diff < std::max(abs_tol, std::numeric_limits<T>::epsilon() * norm * ulp));
}

// This test checks that the GPU implementation returns the same random numbers as the CPU version
TEST(sde, gpu) {
    // simulation parameters
    unsigned ncells = 4;
    unsigned nsynapses = 100;
    unsigned ncvs = 100;
    double const dt = 0.5;
    unsigned nsteps = 50;

    // make labels (and locations for synapses)
    auto labels = make_label_dict(nsynapses);

    // instantiate recipe
    decor dec;
    dec.paint("(all)"_reg , density("hh"));
    dec.place(*labels.locset("locs"), synapse("mean_reverting_stochastic_process"), "synapses");
    sde_recipe_gpu rec(ncells, ncvs, labels, dec);

    // calculate storage needs
    std::size_t n_rv_synapses = ncells*nsynapses;
    std::size_t n_rv_per_dt = n_rv_synapses;
    std::size_t n_rv = n_rv_per_dt*(nsteps+1);

    // setup storage
    archive<arb_value_type> arch_cpu(n_rv);
    archive<arb_value_type> arch_gpu(n_rv);

    // on CPU
    {
        archive_ptr = &arch_cpu;
        auto context = make_context({1, -1});
        partition_hint_map hint_map;
        partition_hint h;
        h.cpu_group_size = ncells;
        hint_map[cell_kind::cable] = h;
        auto decomp = partition_load_balance(rec, context, hint_map);
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_decomposition(decomp)
            .set_seed(42);
        sim.run(nsteps*dt, dt);
    }

    // on GPU
    {
        archive_ptr = &arch_gpu;
        auto context = make_context({1, arbenv::default_gpu()});
        simulation sim = simulation::create(rec)
            .set_context(context)
            .set_seed(42);
        sim.run(nsteps*dt, dt);
    }

    // compare values
    for (std::size_t i=0; i<arch_cpu.size(); ++i) {
        EXPECT_TRUE(
            almost_equal(
                arch_gpu.data_[i],
                arch_cpu.data_[i],
                128,
                4*std::numeric_limits<arb_value_type>::epsilon()
            )
        );
    }
}
#endif
