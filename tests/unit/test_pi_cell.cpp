#include "../gtest.h"

#include <pi_cell.hpp>
#include <pi_cell_group.hpp>

#include "../simple_recipes.hpp"
#include <random>

#include <iostream>


using namespace arb;
using namespace std;

using pi_recipe = homogeneous_recipe<cell_kind::regular_spike_source, pi_cell>;

TEST(pi_cell, basic_usage)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0;
    time_type end = 1000.0;
    time_type rate = 20;

    constexpr time_type dt = 0.01; // dt is ignored by pi_cell_group::advance().

    // Create the generator
    std::mt19937 gen(0);
    auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
    auto prop_per_time_step = (rate / 1000.0) * dt;

    // Create the spikes and store in the correct way
    std::vector<spike> spikes;
    int counter = 0;
    for (time_type t = begin; t < end; t += dt) {
        if (distribution(gen) < prop_per_time_step) {
            spikes.push_back({ { 0, 0 }, t });
            cout << counter << ", " << t << endl;
        }
        counter++;
    }



    // Create the cell_group
    pi_cell desc{ begin, end, rate};
    pi_cell_group sut({0}, pi_recipe(1u, desc));
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
//
//TEST(pi_cell, poll_time_after_end_time)
//{
//    constexpr time_type dt = 0.01; // dt is ignored by pi_cell_group::advance().
//
//    pi_cell desc{0.125, 0.03125, 0.5};
//    pi_cell_group sut({0}, pi_recipe(1u, desc));
//
//    // Expect 12 spikes in this time frame.
//    sut.advance(epoch(0, 0.7), dt, {});
//    EXPECT_EQ(12u, sut.spikes().size());
//
//    // Now ask for spikes for a time slot already passed:
//    // It should result in zero spikes because of the internal state!
//    sut.clear_spikes();
//    sut.advance(epoch(0, 0.2), dt, {});
//    EXPECT_EQ(0u, sut.spikes().size());
//
//    sut.reset();
//
//    // Expect 12 excluding the 0.5
//    sut.advance(epoch(0, 0.5), dt, {});
//    EXPECT_EQ(12u, sut.spikes().size());
//}
//
//TEST(pi_cell, cell_kind_correct)
//{
//    pi_cell desc{0.1, 0.01, 0.2};
//    pi_cell_group sut({0}, pi_recipe(1u, desc));
//
//    EXPECT_EQ(cell_kind::regular_spike_source, sut.get_cell_kind());
//}
