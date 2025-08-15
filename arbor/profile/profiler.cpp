#include <cstdio>
#include <mutex>
#include <ostream>
#include <utility>

#include <arbor/context.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/profile/timer.hpp>

#include "execution_context.hpp"
#include "threading/threading.hpp"
#include "util/span.hpp"
#include "util/rangeutil.hpp"


namespace arb {
namespace profile {

using util::make_span;

#ifdef ARB_HAVE_PROFILING
namespace {
    // Check whether a string describes a valid profiler region name.
    bool is_valid_region_string(const std::string& s) {
        return (s.size()!=0u) && (s.front()!=':') && (s.back()!=':');
    }
}

// Holds the accumulated number of calls and time spent in a region.
struct profile_accumulator {
    std::size_t count=0;
    double time=0.;
    tick_type start_time{};
    bool running = false;
};

struct VectorHash {
    std::size_t operator()(const std::vector<std::uint32_t>& v) const {
        std::size_t seed = v.size();
        for (auto& i : v) {
            seed ^= std::hash<std::uint32_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Records the accumulated time spent in profiler regions on one thread.
// There is one recorder for each thread.
class recorder {
    // One accumulator for call count and wall time for each region.
    std::unordered_map<std::vector<std::uint32_t>, profile_accumulator, VectorHash> accumulators_{};

    std::vector<std::uint32_t> current_timer_stack{};

public:
    // Return a list of the accumulated call count and wall times for each region.
    [[nodiscard]] const std::unordered_map<std::vector<std::uint32_t>, profile_accumulator, VectorHash>& accumulators() const;

    // Start timing the region with index.
    // Throws std::runtime_error if already timing a region.
    void enter(region_id_type index,  const std::vector<std::string>& names);

    // Stop timing the current region, and add the time taken to the accumulated time.
    // Throws std::runtime_error if not currently timing a region.
    void leave(region_id_type index, const std::vector<std::string>& names);

    void thread_started(std::vector<std::uint32_t> timer_stack);

    const std::vector<std::uint32_t>& get_call_stack() const;

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
    void thread_started(const std::vector<std::uint32_t>& timer_stack);
    const std::vector<std::uint32_t>& get_current_timer_stack();
    void leave(region_id_type index);
    void leave(const std::string& name);
    const std::vector<std::string>& regions() const;
    region_id_type region_index(const std::string& name);
    profile results() const;

    static profiler& get_global_profiler() {
        static profiler p;
        return p;
    }

    void clear() {
        for (auto& r: recorders_) r.clear();
        name_index_.clear();
        region_names_.clear();
    }
};


// Utility structure used to organise profiler regions into a tree for printing.
struct profile_node {
    static constexpr region_id_type npos = std::numeric_limits<region_id_type>::max();

    std::string name;
    double time = 0;
    double time_childs = 0;
    region_id_type count = npos;
    std::vector<profile_node> children;

    profile_node() = default;
    profile_node(std::string n, double t, region_id_type c):
            name(std::move(n)), time(t), count(c) {}
    profile_node(std::string n):
            name(std::move(n)), time(0), count(npos) {}
};

// recorder implementation

const std::unordered_map<std::vector<std::uint32_t>, profile_accumulator, VectorHash>& recorder::accumulators() const {
    return accumulators_;
}

void recorder::enter(region_id_type index, const std::vector<std::string>& names) {
    current_timer_stack.push_back(index);
    auto& cur_acc = accumulators_[current_timer_stack];
    if (cur_acc.running) {
        throw std::runtime_error("recorder::enter you entered the timer twice "+names[index]);
    }

    cur_acc.start_time = timer::tic();
    cur_acc.running = true;
}

void recorder::leave(region_id_type index, const std::vector<std::string>& names) {
    if(current_timer_stack[current_timer_stack.size()-1] != index) {
        throw std::runtime_error("recorder::leave without matching recorder::enter "+names[index] + " / "+names[current_timer_stack[current_timer_stack.size()-1]] );
    }
    auto& cur_acc = accumulators_[current_timer_stack];

    if (!cur_acc.running) {
        throw std::runtime_error("recorder::leave without matching recorder::enter. That shouldn't be possible");
    }

    // calculate the elapsed time before any other steps, to increase accuracy.
    auto delta = timer::toc(cur_acc.start_time);

    cur_acc.count++;
    cur_acc.time += delta;
    cur_acc.running = false;

    current_timer_stack.erase(std::next(current_timer_stack.begin(), current_timer_stack.size()-1));
}

void recorder::clear() {
    accumulators_.clear();
    current_timer_stack.clear();
}

void recorder::thread_started(std::vector<std::uint32_t> timer_stack) {
    current_timer_stack = std::move(timer_stack);
}
const std::vector<std::uint32_t> &recorder::get_call_stack() const {
    return current_timer_stack;
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
    recorders_[thread_ids_.at(std::this_thread::get_id())].enter(index, region_names_);
}

void profiler::enter(const std::string& name) {
    if (!init_) return;
    const auto index = region_index(name);
    recorders_[thread_ids_.at(std::this_thread::get_id())].enter(index, region_names_);
}

void profiler::leave(const std::string& name) {
    if (!init_) return;
    const auto index = region_index(name);
    recorders_[thread_ids_.at(std::this_thread::get_id())].leave(index, region_names_);
}

void profiler::leave(region_id_type index) {
    if (!init_) return;
    recorders_[thread_ids_.at(std::this_thread::get_id())].leave(index, region_names_);
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
        for (auto &c: n.children) {
            sort_profile_tree(c);
            n.time_childs += c.time;
        }
    }

    // sort the children in descending order of time taken
    util::sort_by(n.children, [](const profile_node& n){return -n.time;});

    return n.time;
}

profile profiler::results() const {
    profile p;
    p.names = region_names_;

    p.times = {};
    p.counts = {};
    for (auto& r: recorders_) {
        auto& accumulators = r.accumulators();
        for (auto &[timer_stack, acc] : accumulators) {
            auto it = std::find(p.stacks.begin(), p.stacks.end(), timer_stack);
            auto i=0U;
            if(p.stacks.end() == it) {
                i = p.times.size();
                p.stacks.push_back(timer_stack);
                p.times.push_back(0);
                p.counts.push_back(0);
            } else {
                i = std::distance(p.stacks.begin(), it);
            }
            p.times[i]  += acc.time;
            p.counts[i] += acc.count;
        }
    }

    p.num_threads = recorders_.size();

    return p;
}

profile_node make_profile_tree(const profile& p) {
    using std::vector;
    using util::assign_from;
    using util::transform_view;

    const auto& region_names = p.names;
    // Build a tree description of the regions and sub-regions in the profile.
    profile_node tree("r");



    for (const auto i: make_span(0,p.stacks.size())) {
        const auto & ids = p.stacks[i];
        profile_node* node = &tree;
        const auto depth  = ids.size();
        for (auto j: make_span(0, depth)) {
            auto& node_name = region_names[ids[j]];
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
        if(node->name == region_names[ids.back()]) {
            node->time += p.times[i];
            node->count += p.counts[i];
        } else {
            node->children.emplace_back(region_names[ids.back()], p.times[i], p.counts[i]);
        }
    }
    if(tree.children.size() != 1) {
        throw std::invalid_argument("More than one root timer");
    }
    tree = tree.children[0];
    sort_profile_tree(tree);

    return tree;
}

const std::vector<std::string>& profiler::regions() const {
    return region_names_;
}

void profiler::thread_started(const std::vector<std::uint32_t>& timer_stack) {
    recorders_[thread_ids_.at(std::this_thread::get_id())].thread_started(timer_stack);
}

const std::vector<std::uint32_t> &profiler::get_current_timer_stack() {
    return recorders_[thread_ids_.at(std::this_thread::get_id())].get_call_stack();
}

struct prof_line {
    std::string name;
    std::string count;
    std::string time;
    std::string time_childs;
    std::string ratio_captured;
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
    snprintf(buf, std::size(buf), "%.3f", float(n.time_childs));
    res.time_childs = buf;
    snprintf(buf, std::size(buf), "%.1f", float(n.time > 0 ? n.time_childs / n.time * 100.0 : 0));
    res.ratio_captured = buf;
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
    std::vector<prof_line> lines{{"REGION", "CALLS", "WALL", "WALLCHILDS", "WALLCAPTURED\%", "THREAD", "\%"}};
    print_lines(lines, n, wall_time, nthreads, thresh, "");
    // fixing up lengths here
    std::size_t max_len_name = 0;
    std::size_t max_len_count = 0;
    std::size_t max_len_thread = 0;
    std::size_t max_len_time = 0;
    std::size_t max_len_time_childs = 0;
    std::size_t max_ratio = 0;
    std::size_t max_len_percent = 0;
    for (const auto& line: lines) {
        max_len_name = std::max(max_len_name, line.name.size());
        max_len_count = std::max(max_len_count, line.count.size());
        max_len_thread = std::max(max_len_thread, line.thread.size());
        max_len_time = std::max(max_len_time, line.time.size());
        max_len_time_childs = std::max(max_len_time_childs, line.time_childs.size());
        max_ratio = std::max(max_ratio, line.ratio_captured.size());
        max_len_percent = std::max(max_len_percent, line.percent.size());
    }

    auto lpad = [](const std::string& s, std::size_t n) { return std::string(n - s.size(), ' ') + s + "    "; };
    auto rpad = [](const std::string& s, std::size_t n) { return s + std::string(n - s.size(), ' ') + "    "; };

    for (const auto& line: lines) os << rpad(line.name, max_len_name)
                                     << lpad(line.count, max_len_count)
                                     << lpad(line.time, max_len_time)
                                     << lpad(line.time_childs, max_len_time_childs)
                                     << lpad(line.ratio_captured, max_ratio)
                                     << lpad(line.thread, max_len_thread)
                                     << lpad(line.percent, max_len_percent)
                                     << '\n';
};

//
// convenience functions for instrumenting code.
//

ARB_ARBOR_API void profiler_leave(region_id_type id) {
    profiler::get_global_profiler().leave(id);
}

ARB_ARBOR_API void profiler_clear() {
    profiler::get_global_profiler().clear();
}
ARB_ARBOR_API void thread_started(const std::vector<std::uint32_t>& timer_stack) {
    profiler::get_global_profiler().thread_started(timer_stack);
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
    print(os, tree, tree.time_childs, prof.num_threads, limit);
    return os;
}

ARB_ARBOR_API const std::vector<std::uint32_t>& get_current_timer_stack() {
    return profiler::get_global_profiler().get_current_timer_stack();
}

#else

ARB_ARBOR_API void profiler_clear() {}
ARB_ARBOR_API void profiler_leave() {}
ARB_ARBOR_API void profiler_enter(region_id_type) {}
ARB_ARBOR_API void thread_started() {}
ARB_ARBOR_API const std::vector<std::uint32_t>& get_current_timer_stack() { return {}; }
ARB_ARBOR_API profile profiler_summary();
ARB_ARBOR_API profile profiler_summary() {return profile();}
ARB_ARBOR_API region_id_type profiler_region_id(const std::string&) {return 0;}
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const profile&) {return o;}
ARB_ARBOR_API std::ostream& profiler_print_summary(std::ostream& os, double limit) { return os; }


#endif // ARB_HAVE_PROFILING

} // namespace profile
} // namespace arb
