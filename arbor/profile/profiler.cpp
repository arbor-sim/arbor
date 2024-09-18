#include <cstdio>
#include <mutex>
#include <ostream>

#include <arbor/context.hpp>
#include <arbor/profile/profiler.hpp>

#include "execution_context.hpp"
#include "threading/threading.hpp"
#include "util/span.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace profile {

using timer_type = timer<>;

#ifdef ARB_HAVE_PROFILING
namespace {
    // Check whether a string describes a valid profiler region name.
    bool is_valid_region_string(const std::string& s) {
        return (s.size()!=0u) && (s.front()!=':') && (s.back()!=':');
    }

    //
    // Return a list of the words in the string, using ':' as the delimiter
    // string, e.g.:
    //      "communicator"             -> {"communicator"}
    //      "communicator_events"      -> {"communicator", "events"}
    //      "communicator_events_sort" -> {"communicator", "events", "sort"}
    std::vector<std::string> split(const std::string& str) {
        std::vector<std::string> cont;
        std::size_t first = 0;
        std::size_t last = str.find(':');
        while (last != std::string::npos) {
            cont.push_back(str.substr(first, last - first));
            first = last + 1;
            last = str.find(':', first);
        }
        cont.push_back(str.substr(first, last - first));
        return cont;
    }
}

// Holds the accumulated number of calls and time spent in a region.
struct profile_accumulator {
    std::size_t count=0;
    double time=0.;
};

// Records the accumulated time spent in profiler regions on one thread.
// There is one recorder for each thread.
class recorder {
    // used to mark that the recorder is not currently timing a region.
    static constexpr region_id_type npos = std::numeric_limits<region_id_type>::max();

    // The index of the region being timed.
    // If set to npos, no region is being timed.
    region_id_type index_ = npos;

    tick_type start_time_;

    // One accumulator for call count and wall time for each region.
    std::vector<profile_accumulator> accumulators_;

public:
    // Return a list of the accumulated call count and wall times for each region.
    const std::vector<profile_accumulator>& accumulators() const;

    // Start timing the region with index.
    // Throws std::runtime_error if already timing a region.
    void enter(region_id_type index);

    // Stop timing the current region, and add the time taken to the accumulated time.
    // Throws std::runtime_error if not currently timing a region.
    void leave();

    // Reset all of the accumulated call counts and times to zero.
    void clear();
};

// Manages the thread-local recorders.
class profiler {
    std::vector<recorder> recorders_;

    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    // Hash table that maps region names to a unique index.
    // The regions are assigned consecutive indexes in the order that they are
    // added to the profiler with calls to `region_index()`, with the first
    // region numbered zero.
    std::unordered_map<std::string, region_id_type> name_index_;

    // The name of each region being recorded, with index stored in name_index_
    // is used to index into region_names_.
    std::vector<std::string> region_names_;

    // Used to protect name_index_, which is shared between all threads.
    std::mutex mutex_;

    // Flag to indicate whether the profiler has been initialized with the task_system
    bool init_ = false;

public:
    profiler();

    void initialize(task_system_handle& ts);
    void enter(region_id_type index);
    void enter(const std::string& name);
    void leave();
    const std::vector<std::string>& regions() const;
    region_id_type region_index(const std::string& name);
    profile results() const;

    static profiler& get_global_profiler() {
        static profiler p;
        return p;
    }
};


// Utility structure used to organise profiler regions into a tree for printing.
struct profile_node {
    static constexpr region_id_type npos = std::numeric_limits<region_id_type>::max();

    std::string name;
    double time = 0;
    region_id_type count = npos;
    std::vector<profile_node> children;

    profile_node() = default;
    profile_node(std::string n, double t, region_id_type c):
        name(std::move(n)), time(t), count(c) {}
    profile_node(std::string n):
        name(std::move(n)), time(0), count(npos) {}
};

// recorder implementation

const std::vector<profile_accumulator>& recorder::accumulators() const {
    return accumulators_;
}

void recorder::enter(region_id_type index) {
    if (index_!=npos) {
        throw std::runtime_error("recorder::enter without matching recorder::leave");
    }
    if (index>=accumulators_.size()) {
        accumulators_.resize(index+1);
    }
    index_ = index;
    start_time_ = timer_type::tic();
}

void recorder::leave() {
    // calculate the elapsed time before any other steps, to increase accuracy.
    auto delta = timer_type::toc(start_time_);

    if (index_==npos) {
        throw std::runtime_error("recorder::leave without matching recorder::enter");
    }
    accumulators_[index_].count++;
    accumulators_[index_].time += delta;
    index_ = npos;
}

void recorder::clear() {
    index_ = npos;
    accumulators_.resize(0);
}

// profiler implementation

profiler::profiler() {}

void profiler::initialize(task_system_handle& ts) {
    recorders_.resize(ts.get()->get_num_threads());
    thread_ids_ = ts.get()->get_thread_ids();
    init_ = true;
}

void profiler::enter(region_id_type index) {
    if (!init_) return;
    recorders_[thread_ids_.at(std::this_thread::get_id())].enter(index);
}

void profiler::enter(const std::string& name) {
    if (!init_) return;
    const auto index = region_index(name);
    recorders_[thread_ids_.at(std::this_thread::get_id())].enter(index);
}

void profiler::leave() {
    if (!init_) return;
    recorders_[thread_ids_.at(std::this_thread::get_id())].leave();
}

region_id_type profiler::region_index(const std::string& name) {
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

// Used to prepare the profiler output for printing.
// Perform a depth first traversal of a profile tree that:
// - sorts the children of each node in ascending order of time taken;
// - sets the time taken for each non-leaf node to the sum of its children.
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

profile profiler::results() const {
    const auto nregions = region_names_.size();

    profile p;
    p.names = region_names_;

    p.times = std::vector<double>(nregions);
    p.counts = std::vector<region_id_type>(nregions);
    for (auto& r: recorders_) {
        auto& accumulators = r.accumulators();
        for (auto i: util::make_span(0, accumulators.size())) {
            p.times[i]  += accumulators[i].time;
            p.counts[i] += accumulators[i].count;
        }
    }

    p.num_threads = recorders_.size();

    // Remove elements with count == 0
    for(unsigned i=0; i<p.counts.size();) {
        if (p.counts[i] != 0) {
            ++i;
            continue;
        }
        std::swap(p.counts[i], p.counts.back());
        std::swap(p.times[i],  p.times.back());
        std::swap(p.names[i],  p.names.back());
        p.counts.pop_back();
        p.times.pop_back();
        p.names.pop_back();
    }

    return p;
}

profile_node make_profile_tree(const profile& p) {
    using std::vector;
    using util::assign_from;
    using util::transform_view;

    // Take the name of each region, and split into a sequence of sub-region-strings.
    // e.g. "advance_integrate_state" -> "advance", "integrate", "state"
    vector<vector<std::string>> names = assign_from(transform_view(p.names, split));

    // Build a tree description of the regions and sub-regions in the profile.
    profile_node tree("root");
    for (auto idx: util::make_span(0, p.names.size())) {
        profile_node* node = &tree;
        const auto depth  = names[idx].size();
        for (auto i: util::make_span(0, depth-1)) {
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

struct prof_line {
    std::string name;
    std::string count;
    std::string time;
    std::string thread;
    std::string percent;
};

void print_lines(std::vector<prof_line>& lines,
                 profile_node& n,
                 float wall_time,
                 unsigned nthreads,
                 float thresh,
                 std::string indent) {
    static char buf[80];

    float per_thread_time = n.time/nthreads;
    float proportion = n.time/wall_time*100;

    // If the percentage of overall time for this region is below the
    // threashold, stop drawing this branch.
    if (proportion < thresh) return;

    prof_line res;
    res.name =  indent + n.name;
    res.count = (n.count==profile_node::npos) ? "-" : std::to_string(n.count);
    snprintf(buf, std::size(buf), "%.3f", float(n.time));
    res.time = buf;
    snprintf(buf, std::size(buf), "%.3f", float(per_thread_time));
    res.thread = buf;
    snprintf(buf, std::size(buf), "%.1f", float(proportion));
    res.percent = buf;
    lines.push_back(res);
    // print each of the children in turn
    for (auto& c: n.children) print_lines(lines, c, wall_time, nthreads, thresh, indent + "  ");
};

void print(std::ostream& os,
           profile_node& n,
           float wall_time,
           unsigned nthreads,
           float thresh) {
    std::vector<prof_line> lines{{"REGION", "CALLS", "THREAD", "WALL", "\%"}};
    print_lines(lines, n, wall_time, nthreads, thresh, "");
    // fixing up lengths here
    std::size_t max_len_name = 0;
    std::size_t max_len_count = 0;
    std::size_t max_len_thread = 0;
    std::size_t max_len_time = 0;
    std::size_t max_len_percent = 0;
    for (const auto& line: lines) {
        max_len_name = std::max(max_len_name, line.name.size());
        max_len_count = std::max(max_len_count, line.count.size());
        max_len_thread = std::max(max_len_thread, line.thread.size());
        max_len_time = std::max(max_len_time, line.time.size());
        max_len_percent = std::max(max_len_percent, line.percent.size());
    }

    auto lpad = [](const std::string& s, std::size_t n) { return std::string(n - s.size(), ' ') + s + "    "; };
    auto rpad = [](const std::string& s, std::size_t n) { return s + std::string(n - s.size(), ' ') + "    "; };

    for (const auto& line: lines) os << rpad(line.name, max_len_name)
                                     << lpad(line.count, max_len_count)
                                     << lpad(line.thread, max_len_thread)
                                     << lpad(line.time, max_len_time)
                                     << lpad(line.percent, max_len_percent)
                                     << '\n';
};

//
// convenience functions for instrumenting code.
//

ARB_ARBOR_API void profiler_leave() {
    profiler::get_global_profiler().leave();
}

ARB_ARBOR_API region_id_type profiler_region_id(const std::string& name) {
    if (!is_valid_region_string(name)) {
        throw std::runtime_error(std::string("'")+name+"' is not a valid profiler region name.");
    }
    return profiler::get_global_profiler().region_index(name);
}

ARB_ARBOR_API void profiler_enter(region_id_type region_id) {
    profiler::get_global_profiler().enter(region_id);
}

ARB_ARBOR_API void profiler_initialize(context ctx) {
    profiler::get_global_profiler().initialize(ctx->thread_pool);
}

// Print profiler statistics to an ostream
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const profile& prof) {
    auto tree = make_profile_tree(prof);
    print(o, tree, tree.time, prof.num_threads, 0);
    return o;
}

ARB_ARBOR_API profile profiler_summary() {
    return profiler::get_global_profiler().results();
}

ARB_ARBOR_API std::ostream& print_profiler_summary(std::ostream& os, double limit) {
    auto prof = profiler_summary();
    auto tree = make_profile_tree(prof);
    print(os, tree, tree.time, prof.num_threads, limit);
    return os;
}
    
#else

ARB_ARBOR_API void profiler_leave() {}
ARB_ARBOR_API void profiler_enter(region_id_type) {}
ARB_ARBOR_API profile profiler_summary();
ARB_ARBOR_API profile profiler_summary() {return {};}
ARB_ARBOR_API region_id_type profiler_region_id(const std::string&) {return 0;}
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const profile&) {return o;}
ARB_ARBOR_API std::ostream& profiler_print_summary(std::ostream& os, double) { return os; }


#endif // ARB_HAVE_PROFILING

} // namespace profile
} // namespace arb
