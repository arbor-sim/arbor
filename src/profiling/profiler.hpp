#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <threading/threading.hpp>

namespace arb {
namespace util {

using timer_type = arb::threading::timer;
using time_point = timer_type::time_point;

struct stamp {
    std::size_t index;
    time_point time;
};

struct sample {
    std::size_t count=0;
    double time=0.;
    sample() = default;
};

// Records the accumulated time spent in profiler regions on one thread.
// There is one recorder for each thread.
class recorder {
    std::vector<stamp> stamps_;
    std::vector<sample> samples_;

public:
    const std::vector<sample>& samples() const;
    void enter(std::size_t index);
    void leave();
    void clear();
};

// forward declaration of profile output
struct profile;

// Manages the thread-local recorders.
class profiler {
    std::vector<recorder> recorders_;
    std::unordered_map<const char*, std::size_t> name_index_;
    std::vector<std::string> region_names_;
    std::mutex mutex_;
    time_point tstart_;
    time_point tstop_;
    bool running_ = false;

public:
    profiler();
    void start();
    void stop();
    void restart();
    void enter(std::size_t index);
    void enter(const char* name);
    void leave();
    const std::vector<std::string>& regions() const;
    std::size_t index_from_name(const char* name);
    profile results() const;
};

// profile tree node
struct profile_node {
    std::string name;
    double time;
    std::size_t count;
    std::vector<profile_node> children;
    profile_node(std::string n, double t, std::size_t c):
        name(std::move(n)), time(t), count(c) {}
    profile_node() = default;
};

// The results of a profiler run.
struct profile {
    std::vector<std::string> names;
    std::vector<std::vector<sample>> samples;
    double time_taken;
    profile_node tree;
};


namespace data {
    extern profiler profiler;
} // namespace data


void profiler_start();
void profiler_stop();
void profiler_restart();
void profiler_leave();
void profiler_leave(unsigned n);
void profiler_print();

#define PL arb::util::profiler_leave
#define PE(name) \
    static std::size_t name ## _profile_region_tag__ = arb::util::data::profiler.index_from_name(#name); \
    arb::util::data::profiler.enter(name ## _profile_region_tag__);

} // namespace util
} // namespace arb
