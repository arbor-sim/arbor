#include "../gtest.h"

#include <pi_cell.hpp>
#include <pi_cell_group.hpp>

#include "../simple_recipes.hpp"
#include <random>

#include <iostream>


using namespace arb;
using namespace std;

using pi_recipe = homogeneous_recipe<cell_kind::inhomogeneous_poisson_source, ips_cell>;

TEST(ips_cell_group, basic_usage)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0;
    time_type end = 1000.0;
    time_type rate = 20;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms

    constexpr time_type dt = 0.01; // dt is ignored by ips_cell_group::advance().

    // Create the generator
    std::mt19937 gen(0);
    auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
    auto prop_per_time_step = (rate / 1000.0) * sample_delta;

    // Create the spikes and store in the correct way
    std::vector<spike> spikes;
    for (time_type t = begin; t < end; t += sample_delta) {
        if (distribution(gen) < prop_per_time_step) {
            spikes.push_back({ { 0, 0 }, t });
        }
    }

    // Create the cell_group
    ips_cell desc{ begin, end, rate, sample_delta};
    ips_cell_group sut({0}, pi_recipe(1u, desc));
    std::vector<spike> spikes_from_cell;
    for (int idx = 0; idx < 10; ++idx) {
        epoch ep(100.0 * idx, 100.0 * idx + 100.0);
        sut.advance(ep, dt, {});
        spikes_from_cell.insert(spikes_from_cell.end(), sut.spikes().begin(), sut.spikes().end());
        sut.clear_spikes();

    }

    EXPECT_EQ(spikes.size(), spikes_from_cell.size());


    for (auto s1 = spikes.begin(), s2 = spikes_from_cell.begin(); s1 < spikes.end(); ++s1, ++s2) {
        ASSERT_FLOAT_EQ(s1->time, s2->time);
    }
}


TEST(ips_cell_group, later_start)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0;
    time_type end = 1000.0;
    time_type rate = 20;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms

    constexpr time_type dt = 0.01; // dt is ignored by ips_cell_group::advance().

                                   // Create the generator
    std::mt19937 gen(0);
    auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
    auto prop_per_time_step = (rate / 1000.0) * sample_delta;

    // Create the spikes and store in the correct way
    std::vector<spike> spikes;
    for (time_type t = begin; t < end; t += sample_delta) {
        if (distribution(gen) < prop_per_time_step) {
            spikes.push_back({ { 0, 0 }, t });
        }
    }

    // Create the cell_group
    ips_cell desc{ begin, end, rate, sample_delta };
    ips_cell_group sut({ 0 }, pi_recipe(1u, desc));
    std::vector<spike> spikes_from_cell;
    for (int idx = 0; idx < 10; ++idx) {
        epoch ep(100.0 * idx, 100.0 * idx + 100.0);
        sut.advance(ep, dt, {});
        spikes_from_cell.insert(spikes_from_cell.end(), sut.spikes().begin(), sut.spikes().end());
        sut.clear_spikes();

    }

    EXPECT_EQ(spikes.size(), spikes_from_cell.size());


    for (auto s1 = spikes.begin(), s2 = spikes_from_cell.begin(); s1 < spikes.end(); ++s1, ++s2) {
        ASSERT_FLOAT_EQ(s1->time, s2->time);
    }
}


TEST(ips_cell_group, cell_kind_correct)
{
    ips_cell desc{0.1, 0.01, 0.2};
    ips_cell_group sut({0}, pi_recipe(1u, desc));

    EXPECT_EQ(cell_kind::inhomogeneous_poisson_source, sut.get_cell_kind());
}
