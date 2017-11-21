#include "../gtest.h"
#include <random_generator.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace arb::random_generator;

TEST(random_generator, serial) {
    double lambda = 1.0 / 5.0;

    unsigned key1 = 0;
    unsigned key2 = 100;

    unsigned counter1 = 0;
    unsigned counter2 = 100;

    unsigned n_rep = 5;

    unsigned size = (key2 - key1) * (counter2 - counter1);
    std::vector<std::vector<double> > sampled(n_rep, std::vector<double>(size));

    for (unsigned rep = 0; rep < n_rep; ++rep) {
        unsigned index = 0;
        std::ofstream output("poisson" + std::to_string(rep) + ".txt");
        for (unsigned key = key1; key < key2; ++key) {
            for (unsigned counter = counter1; counter < counter2; ++counter) {
                sampled[rep][index] = sample_poisson(lambda, counter, key);
                output << counter << " " << key << " " << sampled[rep][index] << std::endl;
                index++;
            }
        }
        output.close();
    }

    // for each sample, check whether they are the same in all repetitions
    for (unsigned i = 0; i < size; ++i) {
        for (unsigned rep = 0; rep < n_rep; ++rep) {
            EXPECT_EQ(sampled[rep][i], sampled[(rep + 1) % n_rep][i]);
        }
    }
}

