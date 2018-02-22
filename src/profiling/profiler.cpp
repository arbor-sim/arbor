#include <cstdio>
#include <iostream>

#include <util/span.hpp>
#include <util/rangeutil.hpp>

#include "profiler.hpp"

namespace arb {
namespace util {

#ifdef ARB_HAVE_PROFILING
namespace {
    bool is_valid_region_string(const std::string& s) {
        if (s.size()==0u || s.front()=='_' || s.back()=='_') return false;
        return s.find("__") == s.npos;
    }

    std::vector<std::string> split(const std::string& str) {
        std::vector<std::string> cont;
        std::size_t first = 0;
        std::size_t last = str.find('_');
        while (last != std::string::npos) {
            cont.push_back(str.substr(first, last - first));
            first = last + 1;
            last = str.find('_', first);
        }
        cont.push_back(str.substr(first, last - first));
        return cont;
    }
}

// Holds the accumulated number of calls and time spent in a region.
struct sample {
    std::size_t count=0;
    double time=0.;

    sample() = default;
};

// Records the accumulated time spent in profiler regions on one thread.
// There is one recorder for each thread.
class recorder {
    // used to mark that the recorder is not currently timing a region.
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    // The index of the region being timed.
    // If set to npos, no region is being timed.
    std::size_t index_ = npos;

    // The time at which the currently profiled region started.
    time_point start_time_;

    // One accumulator for call count and wall time for each region.
    std::vector<sample> samples_;

public:
    // Return a list of the accumulated call count and wall times for each region.
    const std::vector<sample>& samples() const;

    // Start timing the region with index.
    // Throws std::runtime_error if already timing a region.
    void enter(std::size_t index);

    // Stop timing the current region, and add the time taken to the accumulated time.
    // Throws std::runtime_error if not currently timing a region.
    void leave();

    // Reset all of the accumulated call counts and times to zero.
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


// Utility structure used to organise profiler regions into a tree for printing.
struct profile_node {
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    std::string name;
    double time = 0;
    std::size_t count = npos;
    std::vector<profile_node> children;

    profile_node() = default;
    profile_node(std::string n, double t, std::size_t c):
        name(std::move(n)), time(t), count(c) {}
    profile_node(std::string n):
        name(std::move(n)), time(0), count(npos) {}
};

namespace data {
    arb::util::profiler profiler;
}

// recorder implementation

const std::vector<sample>& recorder::samples() const {
    return samples_;
}

void recorder::enter(std::size_t index) {
    if (index_!=npos) {
        throw std::runtime_error("recorder::enter without matching recorder::leave");
    }
    index_ = index;
    start_time_ = timer_type::tic();
    if (index>=samples_.size()) {
        samples_.resize(index+1);
    }
}

void recorder::leave() {
    if (index_==npos) {
        throw std::runtime_error("recorder::leave without matching recorder::enter");
    }
    samples_[index_].count++;
    samples_[index_].time += timer_type::toc(start_time_);
    index_ = npos;
}

void recorder::clear() {
    index_ = npos;
    for (auto& s:samples_) {
        s.time = 0;
        s.count = 0;
    }
}

// profiler implementation

profiler::profiler() {
    recorders_.resize(threading::num_threads());
}

void profiler::enter(std::size_t index) {
    recorders_[threading::thread_id()].enter(index);
}

void profiler::enter(const char* name) {
    const auto index = index_from_name(name);
    recorders_[threading::thread_id()].enter(index);
}

void profiler::leave() {
    recorders_[threading::thread_id()].leave();
}

void profiler::start() {
    if (running_) {
        throw std::runtime_error("Can't start a profiler that is running.");
    }
    running_ = true;
    tstart_ = timer_type::tic();
    tstart_= timer_type::tic();
}

void profiler::stop() {
    if (!running_) {
        throw std::runtime_error("Can't stop a profiler that isn't running.");
    }
    running_ = false;
    tstop_ = timer_type::tic();
}

void profiler::restart() {
    if (running_) {
        throw std::runtime_error("Can't restart a profiler that is running.");
    }
    for (auto& r: recorders_) {
        r.clear();
    }
    tstart_ = timer_type::tic();
}

std::size_t profiler::index_from_name(const char* name) {
    // The name_index_ hash table is shared by all threads, so all access
    // has to be protected by a mutex.
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = name_index_.find(name);
    if (it==name_index_.end()) {
        const auto index = region_names_.size();
        name_index_[name] = index;
        region_names_.emplace_back(name);
        return index;
    }
    return it->second;
}

double sort_profile_tree(profile_node& n) {
    // accumulate all time taken in children
    if (!n.children.empty()) {
        n.time = 0;
        for (auto &c: n.children) {
            sort_profile_tree(c);
            n.time += c.time;
        }
    }

    // sort the children in descending order of time taken
    util::sort_by(n.children, [](const profile_node& n){return -n.time;});

    return n.time;
}

void print(profile_node& n,
           float wall_time,
           unsigned nthreads,
           float thresh,
           std::string indent="")
{
    auto name = indent + n.name;
    float per_thread_time = n.time/nthreads;
    float proportion = per_thread_time/wall_time*100;

    // If the percentage of overall time for this region is below the
    // threashold, stop drawing this branch.
    if (proportion<thresh) return;

    if (n.count==profile_node::npos) {
        printf("_p_ %-20s%12s%12.3f%12.3f%8.1f\n",
               name.c_str(), "-", float(n.time), per_thread_time, proportion);
    }
    else {
        printf("_p_ %-20s%12lu%12.3f%12.3f%8.1f\n",
               name.c_str(), n.count, float(n.time), per_thread_time, proportion);
    }

    // print each of the children in turn
    for (auto& c: n.children) print(c, wall_time, nthreads, thresh, indent+"  ");
};

profile profiler::results() const {
    const auto nregions = region_names_.size();

    profile p;
    p.wall_time = timer_type::difference(tstart_, tstop_);
    p.names = region_names_;

    p.times = std::vector<double>(nregions);
    p.counts = std::vector<std::size_t>(nregions);
    for (auto& r: recorders_) {
        auto& samples = r.samples();
        for (auto i: make_span(0, samples.size())) {
            p.times[i]  += samples[i].time;
            p.counts[i] += samples[i].count;
        }
    }

    p.num_threads = recorders_.size();

    return p;
}

profile_node make_profile_tree(const profile& p) {
    using std::vector;
    using std::size_t;
    using util::make_span;
    using util::assign_from;
    using util::transform_view;

    // Take the name of each region, and split into a sequence of sub-region-strings.
    // e.g. "advance_integrate_state" -> "advance", "integrate", "state"
    vector<vector<std::string>> names = assign_from(transform_view(p.names, split));

    // Build a tree description of the regions and sub-regions in the profile.
    profile_node tree("root");
    for (auto idx: make_span(0, p.names.size())) {
        profile_node* node = &tree;
        const auto depth  = names[idx].size();
        for (auto i: make_span(0, depth-1)) {
            auto& node_name = names[idx][i];
            auto& kids = node->children;

            // Find child of node that matches node_name
            auto child = std::find_if(
                kids.begin(), kids.end(), [&](profile_node& n){return n.name==node_name;});

            if (child==kids.end()) { // Insert an empty node in the tree.
                node->children.emplace_back(node_name);
                node = &node->children.back();
            }
            else { // Node already exists.
                node = &(*child);
            }
        }
        node->children.emplace_back(names[idx].back(), p.times[idx], p.counts[idx]);
    }
    sort_profile_tree(tree);

    return tree;
}

const std::vector<std::string>& profiler::regions() const {
    return region_names_;
}

//
// convenience functions for instrumenting code.
//

void profiler_leave() {
    data::profiler.leave();
}

void profiler_start() {
    data::profiler.start();
}

void profiler_stop() {
    data::profiler.stop();
}

void profiler_restart() {
    data::profiler.restart();
}

std::size_t profiler_region_id(const char* name) {
    if (!is_valid_region_string(name)) {
        throw std::runtime_error(std::string("'")+name+"' is not a valid profiler region name.");
    }
    return data::profiler.index_from_name(name);
}

void profiler_enter(std::size_t region_id) {
    data::profiler.enter(region_id);
}

// Print profiler statistics to stdout.
// All regions that take less than threshold% of total time are not printed.
void profiler_print(const profile& prof, float threshold) {
    using util::make_span;

    auto tree = make_profile_tree(prof);

    printf("_p_ %-20s%12s%12s%12s%8s\n", "REGION", "CALLS", "THREAD", "WALL", "\%");
    print(tree, prof.wall_time, prof.num_threads, threshold, "");
}

profile profiler_summary() {
    return data::profiler.results();
}

#else

void profiler_leave() {}
void profiler_start() {}
void profiler_stop() {}
void profiler_restart() {}
void profiler_enter(std::size_t) {}
profile profiler_summary();
void profiler_print(const profile& prof, float threshold) {};
profile profiler_summary() {return profile();}
std::size_t profiler_region_id(const char*) {return 0;}

#endif // ARB_HAVE_PROFILING

} // namespace util
} // namespace arb
