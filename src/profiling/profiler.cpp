#include <numeric>

#ifdef NMC_HAVE_GPU
    #include <cuda_profiler_api.h>
#endif

#include <common_types.hpp>
#include <communication/global_policy.hpp>
#include <profiling/profiler.hpp>
#include <util/make_unique.hpp>
#include <util/debug.hpp>

namespace nest {
namespace mc {
namespace util {

// Here we provide functionality that the profiler can use to control the CUDA
// profiler nvprof. The cudaStartProfiler and cudaStopProfiler API calls are
// provided to let a program control which parts of the program are to be
// profiled.
// Here are some wrappers that the NestMC profiler restrict nvprof to recording
// only the time intervals that the user requests when they start and stop the
// profiler.
// It is a simple wrapper around the API calls with a mutex to ensure correct
// behaviour when multiple threads attempt to start or stop the profiler.
#ifdef NMC_HAVE_GPU
namespace gpu {
    bool is_running_nvprof = false;
    std::mutex gpu_profiler_mutex;

    void start_nvprof() {
        std::lock_guard<std::mutex> guard(gpu_profiler_mutex);
        if (!is_running_nvprof) {
            cudaProfilerStart();
        }
        is_running_nvprof = true;
    }

    void stop_nvprof() {
        std::lock_guard<std::mutex> guard(gpu_profiler_mutex);
        if (is_running_nvprof) {
            cudaProfilerStop();
        }
        is_running_nvprof = false;
    }
}
#else
namespace gpu {
    void start_nvprof() {}
    void stop_nvprof()  {}
}
#endif

/////////////////////////////////////////////////////////
// profiler_node
/////////////////////////////////////////////////////////
void profiler_node::print(int indent) {
    std::string s = std::string(indent, ' ') + name;
    std::cout << s
              << std::string(60-s.size(), '.')
              << value
              << "\n";
    for (auto& n : children) {
        n.print(indent+2);
    }
}

void profiler_node::print(std::ostream& stream, double threshold) {
    // convert threshold from proportion to time
    threshold *= value;
    print_sub(stream, 0, threshold, value);
}

void profiler_node::print_sub(
    std::ostream& stream,
    int indent,
    double threshold,
    double total)
{
    char buffer[512];

    if (value < threshold) {
        std::cout << green("not printing ") << name << std::endl;
        return;
    }

    auto max_contribution =
        std::accumulate(
                children.begin(), children.end(), -1.,
                [] (double lhs, const profiler_node& rhs) {
                    return lhs > rhs.value ? lhs : rhs.value;
                }
        );

    // print the table row
    auto const indent_str = std::string(indent, ' ');
    auto label = indent_str + name;
    float percentage = 100.*value/total;
    snprintf(buffer, sizeof(buffer), "%-25s%10.3f%10.1f",
                    label.c_str(),
                    float(value),
                    float(percentage));
    bool print_children =
        threshold==0. ? children.size()>0
                      : max_contribution >= threshold;

    stream << (print_children ? white(buffer) : buffer) << "\n";

    if (print_children) {
        auto other = 0.;
        for (auto& n : children) {
            if (n.value<threshold || n.name=="other") {
                other += n.value;
            }
            else {
                n.print_sub(stream, indent + 2, threshold, total);
            }
        }
        if (other>=std::max(threshold, 0.001) && children.size()) {
            label = indent_str + "  other";
            percentage = 100.*other/total;
            snprintf(buffer, sizeof(buffer), "%-25s%10.3f%10.1f",
                            label.c_str(), float(other), percentage);
            stream << buffer << std::endl;
        }
    }
}

void profiler_node::fuse(const profiler_node& other) {
    for (auto& n : other.children) {
        auto it = std::find(children.begin(), children.end(), n);
        if (it!=children.end()) {
            (*it).fuse(n);
        }
        else {
            children.push_back(n);
        }
    }

    value += other.value;
}

double profiler_node::time_in_other() const {
    auto o = std::find_if(
        children.begin(), children.end(),
        [](const profiler_node& n) {
            return n.name == std::string("other");
        }
    );
    return o==children.end() ? 0. : o->value;
}

void profiler_node::scale(double factor) {
    value *= factor;
    for (auto& n : children) {
        n.scale(factor);
    }
}

profiler_node::json profiler_node::as_json() const {
    json node;
    node["name"] = name;
    node["time"] = value;
    for (const auto& n : children) {
        node["regions"].push_back(n.as_json());
    }
    return node;
}

profiler_node operator+ (const profiler_node& lhs, const profiler_node& rhs) {
    assert(lhs.name == rhs.name);
    auto node = lhs;
    node.fuse(rhs);
    return node;
}

bool operator== (const profiler_node& lhs, const profiler_node& rhs) {
    return lhs.name == rhs.name;
}

/////////////////////////////////////////////////////////
// region_type
/////////////////////////////////////////////////////////
region_type* region_type::subregion(const char* n) {
    size_t hsh = impl::hash(n);
    auto s = subregions_.find(hsh);
    if (s == subregions_.end()) {
        subregions_[hsh] = util::make_unique<region_type>(n, this);
        return subregions_[hsh].get();
    }
    return s->second.get();
}

double region_type::subregion_contributions() const {
    return
        std::accumulate(
            subregions_.begin(), subregions_.end(), 0.,
            [](double l, decltype(*(subregions_.begin())) r) {
                return l+r.second->total();
            }
        );
}

profiler_node region_type::populate_performance_tree() const {
    profiler_node tree(total(), name());

    for (auto& it : subregions_) {
        tree.children.push_back(it.second->populate_performance_tree());
    }

    // sort the contributions in descending order
    std::stable_sort(
        tree.children.begin(), tree.children.end(),
        [](const profiler_node& lhs, const profiler_node& rhs) {
            return lhs.value>rhs.value;
        }
    );

    if (tree.children.size()) {
        // find the contribution of parts of the code that were not explicitly profiled
        auto contributions =
            std::accumulate(
                tree.children.begin(), tree.children.end(), 0.,
                [](double v, profiler_node& n) {
                    return v+n.value;
                }
            );
        auto other = total() - contributions;

        // add the "other" category
        tree.children.emplace_back(other, std::string("other"));
    }

    return tree;
}

/////////////////////////////////////////////////////////
// region_type
/////////////////////////////////////////////////////////
void profiler::enter(const char* name) {
    if (!is_activated()) return;
    current_region_ = current_region_->subregion(name);
    current_region_->start_time();
}

void profiler::leave() {
    if (!is_activated()) return;
    if (current_region_->parent()==nullptr) {
        throw std::out_of_range("attempt to leave root memory tracing region");
    }
    current_region_->end_time();
    current_region_ = current_region_->parent();
}

void profiler::leave(int n) {
    EXPECTS(n>=1);

    while(n--) {
        leave();
    }
}

void profiler::start() {
    gpu::start_nvprof();
    if (is_activated()) {
        throw std::out_of_range(
                "attempt to start an already running profiler"
              );
    }
    activate();
    start_time_ = timer_type::tic();
    root_region_.start_time();
}

void profiler::stop() {
    if (!is_in_root()) {
        throw std::out_of_range(
                "profiler must be in root region when stopped"
              );
    }
    root_region_.end_time();
    stop_time_ = timer_type::tic();

    deactivate();
}

void profiler::restart() {
    if (!is_activated()) {
        start();
        return;
    }
    deactivate();
    root_region_.clear();
    start();
}


profiler_node profiler::performance_tree() {
    if (is_activated()) {
        stop();
    }
    return root_region_.populate_performance_tree();
}


#ifdef NMC_HAVE_PROFILING
namespace data {
    profiler_wrapper profilers_(profiler("root"));
}

profiler& get_profiler() {
    auto& p = data::profilers_.local();
    if (!p.is_activated()) {
        p.start();
    }
    return p;
}

// this will throw an exception if the profler has already been started
void profiler_start() {
    data::profilers_.local().start();
}
void profiler_stop() {
    get_profiler().stop();
}
void profiler_enter(const char* n) {
    get_profiler().enter(n);
}

void profiler_leave() {
    get_profiler().leave();
}
void profiler_leave(int nlevels) {
    get_profiler().leave(nlevels);
}

/// iterate over all profilers and ensure that they have the same start stop times
void profilers_stop() {
    gpu::stop_nvprof();
    for (auto& p : data::profilers_) {
        p.stop();
    }
}

/// iterate over all profilers and reset
void profilers_restart() {
    for (auto& p : data::profilers_) {
        p.restart();
    }
}

void profiler_output(double threshold, std::size_t num_local_work_items) {
    profilers_stop();

    // Find the earliest start time and latest stop time over all profilers
    // This can be used to calculate the wall time for this communicator.
    // The min-max values are used because, for example, the individual
    // profilers might start at different times. In this case, the time stamp
    // when the first profiler started is taken as the start time of the whole
    // measurement period. Likewise for the last profiler to stop.
    auto start_time = data::profilers_.begin()->start_time();
    auto stop_time = data::profilers_.begin()->stop_time();
    for(auto& p : data::profilers_) {
        start_time = std::min(start_time, p.start_time());
        stop_time  = std::max(stop_time,  p.stop_time());
    }
    // calculate the wall time
    auto wall_time = timer_type::difference(start_time, stop_time);
    // calculate the accumulated wall time over all threads
    auto nthreads = data::profilers_.size();
    auto thread_wall = wall_time * nthreads;

    // gather the profilers into one accumulated profile over all threads
    auto thread_measured = 0.; // accumulator for the time measured in each thread
    auto p = profiler_node(0, "total");
    for(auto& thread_profiler : data::profilers_) {
        auto tree = thread_profiler.performance_tree();
        thread_measured += tree.value - tree.time_in_other();
        p.fuse(thread_profiler.performance_tree());
    }
    auto efficiency = 100. * thread_measured / thread_wall;

    p.scale(1./nthreads);

    auto ncomms = communication::global_policy::size();
    auto comm_rank = communication::global_policy::id();
    bool print = comm_rank==0 ? true : false;

    // calculate the throughput in terms of work items per second
    auto local_throughput = num_local_work_items / wall_time;
    auto global_throughput = communication::global_policy::sum(local_throughput);

    if(print) {
        std::cout << " ---------------------------------------------------- \n";
        std::cout << "|                      profiler                      |\n";
        std::cout << " ---------------------------------------------------- \n";
        char line[128];
        std::snprintf(
            line, sizeof(line), "%-18s%10.3f s\n",
            "wall time", float(wall_time));
        std::cout << line;
        std::snprintf(
            line, sizeof(line), "%-18s%10d\n",
            "communicators", int(ncomms));
        std::snprintf(
            line, sizeof(line), "%-18s%10d\n",
            "threads", int(nthreads));
        std::cout << line;
        std::snprintf(
            line, sizeof(line), "%-18s%10.2f %%\n",
            "thread efficiency", float(efficiency));
        std::cout << line << "\n";
        p.print(std::cout, threshold);
        std::cout << "\n";
        std::snprintf(
            line, sizeof(line), "%-18s%10s%10s\n",
            "", "local", "global");
        std::cout << line;
        std::snprintf(
            line, sizeof(line), "%-18s%10d%10d\n",
            "throughput", int(local_throughput), int(global_throughput));
        std::cout << line;

        std::cout << "\n\n";
    }

    nlohmann::json as_json;
    as_json["wall time"] = wall_time;
    as_json["threads"] = nthreads;
    as_json["efficiency"] = efficiency;
    as_json["communicators"] = ncomms;
    as_json["throughput"] = unsigned(local_throughput);
    as_json["rank"] = comm_rank;
    as_json["regions"] = p.as_json();

    auto fname = std::string("profile_" + std::to_string(comm_rank));
    std::ofstream fid(fname);
    fid << std::setw(1) << as_json;
}

#else
void profiler_start() {}
void profiler_stop() {}
void profiler_enter(const char*) {}
void profiler_leave() {}
void profiler_leave(int) {}
void profilers_stop() {}
void profiler_output(double threshold, std::size_t num_local_work_items) {}
void profilers_restart() {};
#endif

} // namespace util
} // namespace mc
} // namespace nest
