#pragma once
#include <vector>

#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

namespace arb {
namespace util {

std::vector<double> uniform(uint64_t seed, unsigned left, unsigned right) {
    typedef r123::Threefry2x64 cbrng;
    std::vector<double> r;

    cbrng::key_type key = {{seed}};
    cbrng::ctr_type ctr = {{0,0}};
    cbrng g;

    unsigned i = left;
    if (i%2 && i<=right) {
        ctr[0] = i/2;
        cbrng::ctr_type rand = g(ctr, key);
        r.push_back(r123::u01<double>(rand[1]));;
        ++i;
    }
    while (i < 2*((right+1)/2)) {
        ctr[0] = i/2;
        cbrng::ctr_type rand = g(ctr, key);
        r.push_back(r123::u01<double>(rand[0]));
        r.push_back(r123::u01<double>(rand[1]));
        i += 2;
    }
    if (i<=right) {
        ctr[0] = i/2;
        cbrng::ctr_type rand = g(ctr, key);
        r.push_back(r123::u01<double>(rand[0]));
    }
    return r;
}

}
}