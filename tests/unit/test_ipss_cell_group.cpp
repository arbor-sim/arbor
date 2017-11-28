#include "../gtest.h"

#include <ipss_cell_description.hpp>
#include <ipss_cell_group.hpp>

#include "../simple_recipes.hpp"
#include <random>

#include <iostream>


using namespace arb;
using namespace std;

using ipss_recipe = homogeneous_recipe<cell_kind::inhomogeneous_poisson_spike_source,
    ipss_cell_description>;


std::vector<spike> create_poisson_spike_train(time_type begin, time_type end,
     double sample_delta, cell_gid_type gid,
    std::vector<std::pair<time_type, double>> rates_per_time, bool interpolate = false ) {

    // Create the generator
    std::mt19937 gen(gid);
    auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
    double prop_per_time_step;
    double prop_per_next_time_step;
    unsigned steps;
    double delta_prob;
    // Create the spikes and store in the correct way
    std::vector<spike> spikes;

    time_type t = begin;
    // We asume a properly constructed time_rate and start time
    // and that the first entry starts at begin
    for (unsigned idx = 0; idx < rates_per_time.size() - 1; ++idx) {
        prop_per_time_step = (rates_per_time.at(idx).second / 1000.0) * sample_delta;
        if (interpolate) {
            prop_per_next_time_step = (rates_per_time.at(idx + 1).second / 1000.0) * sample_delta;
            steps = (rates_per_time.at(idx + 1).first - rates_per_time.at(idx).first) / sample_delta;
            delta_prob = (prop_per_next_time_step - prop_per_time_step) / steps;

            // When the begin time is after the first sample we need to interpolate
            if (idx == 0 && rates_per_time.size() > 1 && begin > rates_per_time.at(0).first) {
                // How many step from pair time till start time
                unsigned steps = (begin - rates_per_time.at(0).first) / sample_delta;
                prop_per_time_step += steps * delta_prob;
            }
        }

        for (; t < rates_per_time.at(idx + 1).first; t += sample_delta) {
            auto dice_roll = distribution(gen);

            if (dice_roll < prop_per_time_step) {

                spikes.push_back({ { gid, 0 }, t });
            }

            if (interpolate)   {
                prop_per_time_step += delta_prob;
            }
        }
    }

    // Treat the last differently (until end time)
    prop_per_time_step = (rates_per_time.at(rates_per_time.size() - 1).second / 1000.0) * sample_delta;

    for (; t < end; t += sample_delta) {
        if (distribution(gen) < prop_per_time_step) {
            spikes.push_back({ { 0, 0 }, t });
        }
    }

    return spikes;
}

TEST(ipss_cell_group, basic_usage_non_interpolate_constant)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0;
    time_type end = 1000.0;
    time_type rate = 20;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms
    bool interpolate = false;

    // The rate changes we want
    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, rate });

    // Target output
    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
         sample_delta, 0,  rates_per_time);

    // Create the cell group itself
    ipss_cell_group sut({0},
        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));

    // run the cell
    for (std::size_t idx = 0; idx < 10; ++idx) {
        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
    }
    // Check the output of the cell
    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
        ASSERT_FLOAT_EQ(s1->time, s2->time);
    }
}

TEST(ipss_cell_group, differt_rates_non_interpolate)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0;
    time_type end = 10.0;
    time_type rate = 2000;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms
    bool interpolate = false;

    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, 0.0 });
    rates_per_time.push_back({ 3.0, rate * 3.0 });
    rates_per_time.push_back({ 6.0, rate * 4.0});
    rates_per_time.push_back({ 8.0, rate * 0.0 });


    // Create the spikes and store in the correct way
    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
         sample_delta, 0, rates_per_time);

    ipss_cell_group sut({ 0 },
        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));

    for (std::size_t idx = 0; idx < 10; ++idx) {
        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
    }

    // Check the output of the cell
    EXPECT_TRUE(sut.spikes().size() > 0);
    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
        ASSERT_FLOAT_EQ(s1->time, s2->time);
    }
}

TEST(ipss_cell_group, differt_rates_interpolate)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0;
    time_type end = 10.0;
    time_type rate = 2000;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms
    bool interpolate = true;

    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, 0.0 });
    rates_per_time.push_back({ 3.0, rate * 3.0 });
    rates_per_time.push_back({ 6.0, rate * 4.0 });
    rates_per_time.push_back({ 8.0, rate * 0.0 });


    // Create the spikes and store in the correct way
    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
        sample_delta, 0, rates_per_time, interpolate);

    ipss_cell_group sut({ 0 },
        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));

    for (std::size_t idx = 0; idx < 10; ++idx) {
        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
    }

    // Check the output of the cell
    EXPECT_TRUE(sut.spikes().size() > 0);
    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
        ASSERT_FLOAT_EQ(s1->time, s2->time);
    }
}

TEST(ipss_cell_group, test_reset)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0;
    time_type end = 10.0;
    time_type rate = 2000;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms
    bool interpolate = true;

    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, 0.0 });
    rates_per_time.push_back({ 3.0, rate * 3.0 });
    rates_per_time.push_back({ 6.0, rate * 4.0 });
    rates_per_time.push_back({ 8.0, rate * 0.0 });


    // Create the spikes and store in the correct way
    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
        sample_delta,0, rates_per_time, interpolate);

    ipss_cell_group sut({ 0 },
        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));

    for (std::size_t idx = 0; idx < 10; ++idx) {
        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
    }
    sut.reset();
    for (std::size_t idx = 0; idx < 10; ++idx) {
        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
    }



    // Check the output of the cell
    EXPECT_TRUE(sut.spikes().size() > 0);
    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
        ASSERT_FLOAT_EQ(s1->time, s2->time);
    }
}


TEST(ipss_cell_group, start_end_different_then_zero)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 1.5;
    time_type end = 10.0;
    time_type rate = 2000;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms
    bool interpolate = true;

    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, 0.0 });
    rates_per_time.push_back({ 3.0, rate * 3.0 });
    rates_per_time.push_back({ 6.0, rate * 4.0 });
    rates_per_time.push_back({ 8.0, rate * 0.0 });


    // Create the spikes and store in the correct way
    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
        sample_delta, 0, rates_per_time, interpolate);

    ipss_cell_group sut({ 0 },
        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));

    for (std::size_t idx = 0; idx < 10; ++idx) {
        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
    }

    // Check the output of the cell
    EXPECT_TRUE(sut.spikes().size() > 0);
    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
        ASSERT_FLOAT_EQ(s1->time, s2->time);
    }
}

TEST(ipss_cell_group, multiple_cells)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 1.5;
    time_type end = 10.0;
    time_type rate = 2000;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms
    bool interpolate = true;

    unsigned nr_cells = 10;

    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, 0.0 });
    rates_per_time.push_back({ 3.0, rate * 3.0 });
    rates_per_time.push_back({ 6.0, rate * 4.0 });
    rates_per_time.push_back({ 8.0, rate * 0.0 });


    std::vector<spike> target_spikes;

    // Create the spikes and store in the correct way
    std::vector<cell_gid_type> gids;
    for (unsigned idx = 0; idx < nr_cells; ++idx)
    {
        gids.push_back(idx);
       auto generated_spikes = create_poisson_spike_train(begin, end,
            sample_delta, idx, rates_per_time, interpolate);

        target_spikes.insert(target_spikes.end(), generated_spikes.begin(),
            generated_spikes.end());
    }

    ipss_cell_group sut(gids,
        ipss_recipe(nr_cells, { begin, end, sample_delta, rates_per_time, interpolate }));

    for (std::size_t idx = 0; idx < 10; ++idx) {
        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
    }

    // Check the output of the cell
    EXPECT_TRUE(sut.spikes().size() > 0);


    EXPECT_EQ(target_spikes.size(), sut.spikes().size());

    for (std::vector<spike>::const_iterator s2 = sut.spikes().begin();
        s2 < sut.spikes().end(); ++s2) {

        // Now loop over all the entries in the and find the matching spike
        std::vector<spike>::iterator s1 = target_spikes.begin();
        bool match_found = false;
        for (;s1 < target_spikes.end(); ++s1)
        {
            // Check if the current spike is a match
            // TODO: maybe use a better float compare?
            if (s2->source == s1->source && s2->time == s1->time) {
                match_found = true;
                break;
            }

        }
        ASSERT_TRUE(match_found) << "Did not found a matching spike in the target set";

        // Now remove this spike to prevent double matching
        target_spikes.erase(s1);
    }

    // The target vector should now be empty!

    EXPECT_TRUE(target_spikes.size() == 0);
}


TEST(ipss_cell_group, cell_kind_correct)
{
    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, 20 });
    ipss_cell_description desc{0.1, 0.01, 0.2, rates_per_time};
    ipss_cell_group sut({0}, ipss_recipe(1u, desc));

    EXPECT_EQ(cell_kind::inhomogeneous_poisson_spike_source, sut.get_cell_kind());
}


TEST(ipss_cell_group, start_before_first_rate_change)
{
    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.11, 20 });
    ipss_cell_description desc{ 0.1, 0.01, 0.2, rates_per_time };

    // Gtest does not have the expect_exception shorthand
    try {
        ipss_cell_group sut({ 0 }, ipss_recipe(1u, desc));
        FAIL() << "Expected a failure";
    }
    catch (std::logic_error const & err)
    {
        EXPECT_EQ(err.what(), std::string("The start time of the neuron is before the first time/rate pair"));
    }
    catch (...)
    {
        FAIL() << "Expected logic_error but different exception encountered";
    }
}