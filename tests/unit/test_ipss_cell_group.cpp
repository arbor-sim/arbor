#include "../gtest.h"

#include <ipss_cell_description.hpp>
#include <ipss_cell_group.hpp>

#include "../simple_recipes.hpp"
#include <random>
#include <math.h>
#include <cmath>

#include <iostream>



using namespace arb;
using namespace std;

using ipss_recipe = homogeneous_recipe<cell_kind::inhomogeneous_poisson_spike_source,
    ipss_cell_description>;


TEST(ipss_cell_group, time_to_step_idx)
{
    // Some shorthand for typed values
    time_type zero = 0.0;
    time_type one = 1.0;
    time_type time_step = 0.1;

    ASSERT_EQ(0ul, ipss_cell::time_to_step_idx(0.0, time_step));
    ASSERT_EQ(1ul, ipss_cell::time_to_step_idx(nextafter(zero, one), time_step));
    ASSERT_EQ(1ul, ipss_cell::time_to_step_idx(nextafter(time_step, zero), time_step));
    ASSERT_EQ(1ul, ipss_cell::time_to_step_idx(time_step, time_step));
    ASSERT_EQ(2ul, ipss_cell::time_to_step_idx(nextafter(time_step, one), time_step));
}

std::vector<spike> create_poisson_spike_train_new(time_type begin, time_type end,
    time_type sample_delta, cell_gid_type gid,
    std::vector<std::pair<time_type, double>> rates_per_time, bool interpolate = false) {

    // Create the generator
    std::mt19937 gen(gid);
    auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);

    // Create the spikes and store in the correct way
    std::vector<spike> spikes;

    // For mathematical consistency we take time steps starting at t=0.0
    // All times, ratest_per_time, begin and end will be rounded upwards to a
    // whole time steps.
    auto begin_idx = ipss_cell::time_to_step_idx(begin, sample_delta);
    auto end_idx = ipss_cell::time_to_step_idx(end, sample_delta);

    std::vector<std::pair<unsigned long, double>> rate_change_at_step;
    for (auto time_rate_pair : rates_per_time) {
        auto time_idx = ipss_cell::time_to_step_idx(time_rate_pair.first, sample_delta);
        rate_change_at_step.push_back({ time_idx, time_rate_pair.second });
    }

    // Update the step idx to the begin time step
    auto rates_ptr = rate_change_at_step.begin();
    auto next_rates_ptr = rates_ptr;
    next_rates_ptr++;

    // Update the pointers until the next ptr is after begin
    while (next_rates_ptr->first < begin_idx) {
        rates_ptr++;
        next_rates_ptr++;
    }

    double rate_per_dt =  (next_rates_ptr->second - rates_ptr->second) /
        (next_rates_ptr->first - rates_ptr->first);
    double base_rate = interpolate ? rates_ptr->second + 0.5 * rate_per_dt : rates_ptr->second;

    // Normalize to ms!!
    rate_per_dt /= 1000.0;
    base_rate /=  1000.0;

    //cout << "base_rate: " << base_rate << " rate_per_dt: " << rate_per_dt << std::endl;
    unsigned long step = begin_idx;
    do {
        // If we need to update our rates this step
        if (step == next_rates_ptr->first) {
            rates_ptr++;
            next_rates_ptr++;
            // new baserate and dt
            rate_per_dt = (next_rates_ptr->second - rates_ptr->second) /
                (next_rates_ptr->first - rates_ptr->first);
            base_rate = interpolate ? rates_ptr->second + 0.5 * rate_per_dt : rates_ptr->second;

            // Normalize to ms!!
            rate_per_dt /= 1000.0;
            base_rate /= 1000.0;

            //cout << "base_rate: " << base_rate << " rate_per_dt: " << rate_per_dt  << std::endl;
        }
        double spike_rate = interpolate ? base_rate + (step - rates_ptr->first) * rate_per_dt : base_rate;

        auto dice_roll = distribution(gen);

        //cout << "dice_roll: " << dice_roll << " spike_rate: " << spike_rate << std::endl;
        if (dice_roll < spike_rate) {
            //cout << "spike!" << endl;
            spikes.push_back({ { gid, 0 }, step * sample_delta });
        }

    } while (++step < end_idx);

    return spikes;
}


TEST(ipss_cell_group, basic_usage_non_interpolate_constant)
{
    // Create an array of spike times for 1000 ms of time using the same
    // seed for the random number generator (in one go)
    // Then let the cell_group generate the spikes, they should be the same
    time_type begin = 0.0;
    time_type end = 20.0;
    time_type rate = 10.0;  // Hz
    time_type sample_delta = 0.1; // 0.1 ms
    bool interpolate = true;

    // The rate changes we want
    std::vector<std::pair<time_type, double>> rates_per_time;
    rates_per_time.push_back({ 0.0, 0.0 });
    rates_per_time.push_back({ 10.0, rate });
    rates_per_time.push_back({ 20.0, rate *10 });


    std::vector<spike> target_spikes = create_poisson_spike_train_new(begin, end,
        sample_delta, 0, rates_per_time, interpolate);


    // Create the cell group itself
    ipss_cell_group sut({ 0 },
        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));

    // run the cell
    time_type epoch_length = (end - begin) / 10;
    sut.advance({0, time_type(10.0) }, 0.01,  {});
    sut.advance({ 1, time_type(20.0) }, 0.01, {});

    auto sut_spikes = sut.spikes();
    // Check the output of the cell
    std::cout << "Sizes: " << sut_spikes.size() << ", " << target_spikes.size() << std::endl;
    std::vector<spike>::const_iterator s1 = sut_spikes.begin();
    std::vector<spike>::const_iterator s2 = target_spikes.begin();
    for (; s2 < target_spikes.end(); ++s1, ++s2) {
        std::cout << s1->time << "," << s2->time << std::endl;
    }
}
//
//TEST(ipss_cell_group, basic_usage_non_interpolate_constant)
//{
//    // Create an array of spike times for 1000 ms of time using the same
//    // seed for the random number generator (in one go)
//    // Then let the cell_group generate the spikes, they should be the same
//    time_type begin = 0;
//    time_type end = 1000.0;
//    time_type rate = 20;  // Hz
//    time_type sample_delta = 0.1; // 0.1 ms
//    bool interpolate = false;
//
//    // The rate changes we want
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.0, rate });
//
//    // Target output
//    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
//         sample_delta, 0,  rates_per_time);
//
//    std::vector<spike> target_spikes2 = create_poisson_spike_train_new(begin, end,
//        sample_delta, 0, rates_per_time);
//
//
//    // Create the cell group itself
//    ipss_cell_group sut({0},
//        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));
//
//    // run the cell
//    for (std::size_t idx = 0; idx < 10; ++idx) {
//        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
//    }
//    // Check the output of the cell
//    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
//    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
//        ASSERT_FLOAT_EQ(s1->time, s2->time);
//    }
//}
//
//TEST(ipss_cell_group, differt_rates_non_interpolate)
//{
//    // Create an array of spike times for 1000 ms of time using the same
//    // seed for the random number generator (in one go)
//    // Then let the cell_group generate the spikes, they should be the same
//    time_type begin = 0;
//    time_type end = 10.0;
//    time_type rate = 2000;  // Hz
//    time_type sample_delta = 0.1; // 0.1 ms
//    bool interpolate = false;
//
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.0, 0.0 });
//    rates_per_time.push_back({ 3.0, rate * 3.0 });
//    rates_per_time.push_back({ 6.0, rate * 4.0});
//    rates_per_time.push_back({ 8.0, rate * 0.0 });
//
//
//    // Create the spikes and store in the correct way
//    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
//         sample_delta, 0, rates_per_time);
//
//    ipss_cell_group sut({ 0 },
//        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));
//
//    for (std::size_t idx = 0; idx < 10; ++idx) {
//        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
//    }
//
//    // Check the output of the cell
//    EXPECT_TRUE(sut.spikes().size() > 0);
//    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
//    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
//        ASSERT_FLOAT_EQ(s1->time, s2->time);
//    }
//}
//
//TEST(ipss_cell_group, differt_rates_interpolate)
//{
//    // Create an array of spike times for 1000 ms of time using the same
//    // seed for the random number generator (in one go)
//    // Then let the cell_group generate the spikes, they should be the same
//    time_type begin = 0;
//    time_type end = 10.0;
//    time_type rate = 2000;  // Hz
//    time_type sample_delta = 0.1; // 0.1 ms
//    bool interpolate = true;
//
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.0, 0.0 });
//    rates_per_time.push_back({ 3.0, rate * 3.0 });
//    rates_per_time.push_back({ 6.0, rate * 4.0 });
//    rates_per_time.push_back({ 8.0, rate * 0.0 });
//
//
//    // Create the spikes and store in the correct way
//    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
//        sample_delta, 0, rates_per_time, interpolate);
//
//    ipss_cell_group sut({ 0 },
//        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));
//
//    for (std::size_t idx = 0; idx < 10; ++idx) {
//        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
//    }
//
//    // Check the output of the cell
//    EXPECT_TRUE(sut.spikes().size() > 0);
//    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
//    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
//        ASSERT_FLOAT_EQ(s1->time, s2->time);
//    }
//}
//
//TEST(ipss_cell_group, test_reset)
//{
//    // Create an array of spike times for 1000 ms of time using the same
//    // seed for the random number generator (in one go)
//    // Then let the cell_group generate the spikes, they should be the same
//    time_type begin = 0;
//    time_type end = 10.0;
//    time_type rate = 2000;  // Hz
//    time_type sample_delta = 0.1; // 0.1 ms
//    bool interpolate = true;
//
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.0, 0.0 });
//    rates_per_time.push_back({ 3.0, rate * 3.0 });
//    rates_per_time.push_back({ 6.0, rate * 4.0 });
//    rates_per_time.push_back({ 8.0, rate * 0.0 });
//
//
//    // Create the spikes and store in the correct way
//    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
//        sample_delta,0, rates_per_time, interpolate);
//
//    ipss_cell_group sut({ 0 },
//        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));
//
//    for (std::size_t idx = 0; idx < 10; ++idx) {
//        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
//    }
//    sut.reset();
//    for (std::size_t idx = 0; idx < 10; ++idx) {
//        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
//    }
//
//
//
//    // Check the output of the cell
//    EXPECT_TRUE(sut.spikes().size() > 0);
//    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
//    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
//        ASSERT_FLOAT_EQ(s1->time, s2->time);
//    }
//}
//
//
//TEST(ipss_cell_group, start_end_different_then_zero)
//{
//    // Create an array of spike times for 1000 ms of time using the same
//    // seed for the random number generator (in one go)
//    // Then let the cell_group generate the spikes, they should be the same
//    time_type begin = 1.5;
//    time_type end = 10.0;
//    time_type rate = 2000;  // Hz
//    time_type sample_delta = 0.1; // 0.1 ms
//    bool interpolate = true;
//
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.0, 0.0 });
//    rates_per_time.push_back({ 3.0, rate * 3.0 });
//    rates_per_time.push_back({ 6.0, rate * 4.0 });
//    rates_per_time.push_back({ 8.0, rate * 0.0 });
//
//
//    // Create the spikes and store in the correct way
//    std::vector<spike> target_spikes = create_poisson_spike_train(begin, end,
//        sample_delta, 0, rates_per_time, interpolate);
//
//    ipss_cell_group sut({ 0 },
//        ipss_recipe(1u, { begin, end, sample_delta, rates_per_time, interpolate }));
//
//    for (std::size_t idx = 0; idx < 10; ++idx) {
//        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
//    }
//
//    // Check the output of the cell
//    EXPECT_TRUE(sut.spikes().size() > 0);
//    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
//    for (std::vector<spike>::const_iterator s1 = target_spikes.begin(), s2 = sut.spikes().begin(); s1 < target_spikes.end(); ++s1, ++s2) {
//        ASSERT_FLOAT_EQ(s1->time, s2->time);
//    }
//}
//
//TEST(ipss_cell_group, multiple_cells)
//{
//    // Create an array of spike times for 1000 ms of time using the same
//    // seed for the random number generator (in one go)
//    // Then let the cell_group generate the spikes, they should be the same
//    time_type begin = 1.5;
//    time_type end = 10.0;
//    time_type rate = 2000;  // Hz
//    time_type sample_delta = 0.1; // 0.1 ms
//    bool interpolate = true;
//
//    unsigned nr_cells = 10;
//
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.0, 0.0 });
//    rates_per_time.push_back({ 3.0, rate * 3.0 });
//    rates_per_time.push_back({ 6.0, rate * 4.0 });
//    rates_per_time.push_back({ 8.0, rate * 0.0 });
//
//
//    std::vector<spike> target_spikes;
//
//    // Create the spikes and store in the correct way
//    std::vector<cell_gid_type> gids;
//    for (unsigned idx = 0; idx < nr_cells; ++idx)
//    {
//        gids.push_back(idx);
//       auto generated_spikes = create_poisson_spike_train(begin, end,
//            sample_delta, idx, rates_per_time, interpolate);
//
//        target_spikes.insert(target_spikes.end(), generated_spikes.begin(),
//            generated_spikes.end());
//    }
//
//    ipss_cell_group sut(gids,
//        ipss_recipe(nr_cells, { begin, end, sample_delta, rates_per_time, interpolate }));
//
//    for (std::size_t idx = 0; idx < 10; ++idx) {
//        sut.advance({ idx, time_type(100.0) * idx + time_type(100.0) }, 0.01, {});
//    }
//
//    // Check the output of the cell
//    EXPECT_TRUE(sut.spikes().size() > 0);
//
//
//    EXPECT_EQ(target_spikes.size(), sut.spikes().size());
//
//    for (std::vector<spike>::const_iterator s2 = sut.spikes().begin();
//        s2 < sut.spikes().end(); ++s2) {
//
//        // Now loop over all the entries in the and find the matching spike
//        std::vector<spike>::iterator s1 = target_spikes.begin();
//        bool match_found = false;
//        for (;s1 < target_spikes.end(); ++s1)
//        {
//            // Check if the current spike is a match
//            // TODO: maybe use a better float compare?
//            if (s2->source == s1->source && s2->time == s1->time) {
//                match_found = true;
//                break;
//            }
//
//        }
//        ASSERT_TRUE(match_found) << "Did not found a matching spike in the target set";
//
//        // Now remove this spike to prevent double matching
//        target_spikes.erase(s1);
//    }
//
//    // The target vector should now be empty!
//
//    EXPECT_TRUE(target_spikes.size() == 0);
//}
//
//
//TEST(ipss_cell_group, cell_kind_correct)
//{
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.0, 20 });
//    ipss_cell_description desc{0.1, 0.01, 0.2, rates_per_time};
//    ipss_cell_group sut({0}, ipss_recipe(1u, desc));
//
//    EXPECT_EQ(cell_kind::inhomogeneous_poisson_spike_source, sut.get_cell_kind());
//}
//
//
//TEST(ipss_cell_group, start_before_first_rate_change)
//{
//    std::vector<std::pair<time_type, double>> rates_per_time;
//    rates_per_time.push_back({ 0.11, 20 });
//    ipss_cell_description desc{ 0.1, 0.01, 0.2, rates_per_time };
//
//    // Gtest does not have the expect_exception shorthand
//    try {
//        ipss_cell_group sut({ 0 }, ipss_recipe(1u, desc));
//        FAIL() << "Expected a failure";
//    }
//    catch (std::logic_error const & err)
//    {
//        EXPECT_EQ(err.what(), std::string("The start time of the neuron is before the first time/rate pair"));
//    }
//    catch (...)
//    {
//        FAIL() << "Expected logic_error but different exception encountered";
//    }
//}