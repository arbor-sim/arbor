#pragma once
#include <vector>

#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

namespace arb {
namespace util {

struct uniform_t {
    typedef r123::Threefry2x64 rng;
    rng::key_type key_{{0}};
    rng::ctr_type ctr_{{0,0}};
    rng g;
    double cache_ = -1;

    uniform_t(std::uint64_t seed, std::uint64_t discard=0):
        key_{{seed}}, ctr_{{discard/2, 0}} {
        // discard was odd, so we prime the cache by generating one pair, discarding the first
        if (discard % 2) (*this)();
    }

    double operator()() {
        // have one value cached, so return that and invalidate
        if (cache_ > 0) {
            return std::exchange(cache_, -1);
        }
        // nothing in cache, so generate two new numbers caching the second
        else {
            auto rand = g(ctr_, key_);
            ctr_[0] += 1;
            cache_ = r123::u01<double>(rand[1]);
            return r123::u01<double>(rand[0]);
        }
    }
};

inline
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
