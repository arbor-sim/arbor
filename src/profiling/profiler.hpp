#pragma once

#include <algorithm>
#include <unordered_map>
#include <map>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>

#include <cassert>
#include <cstdlib>

#include <threading/threading.hpp>

namespace nest {
namespace mc {
namespace util {

static inline std::string green(std::string s)  { return s; }
static inline std::string yellow(std::string s) { return s; }
static inline std::string white(std::string s)  { return s; }
static inline std::string red(std::string s)    { return s; }
static inline std::string cyan(std::string s)   { return s; }

namespace impl {

    /// simple hashing function for strings
    ///     - for easy comparison of strings over MPI
    ///     - for fast searching of regions named with strings
    static inline
    size_t hash(char* s) {
        size_t h = 5381;

        while (*s) {
            h = ((h << 5) + h) + int(*s);
            ++s;
        }
        return h;
    }

    /// std::string overload for hash
    static inline
    size_t hash(const std::string& s) {
        return hash(s.c_str());
    }

    struct profiler_node {
        double value;
        std::string name;
        std::vector<profiler_node> children;

        profiler_node() :
            value(0.), name("")
        {}

        profiler_node(double v, const std::string& n) :
            value(v), name(n)
        {}

        void print(int indent=0) {
            std::string s = std::string(indent, ' ') + name;
            std::cout << s
                      << std::string(60-s.size(), '.')
                      << value
                      << "\n";
            for (auto& n : children) {
                n.print(indent+2);
            }
        }

        friend profiler_node operator+ (
            const profiler_node& lhs,
            const profiler_node& rhs)
        {
            assert(lhs.name == rhs.name);
            auto node = lhs;
            node.fuse(rhs);
            return node;
        }

        friend bool operator== (
            const profiler_node& lhs,
            const profiler_node& rhs)
        {
            return lhs.name == rhs.name;
        }

        void print(std::ostream& stream, double threshold) {
            // convert threshold from proportion to time
            threshold *= value;
            print_sub(stream, 0, threshold, value);
        }

        void print_sub(
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

            if (print_children) {
                stream << white(buffer) << std::endl;
            }
            else {
                stream << buffer << std::endl;
            }

            if (print_children) {
                auto other = 0.;
                for (auto &n : children) {
                    if (n.value<threshold || n.name=="other") {
                        other += n.value;
                    }
                    else {
                        n.print_sub(stream, indent + 2, threshold, total);
                    }
                }
                if (other >= threshold && children.size()) {
                    label = indent_str + "  other";
                    percentage = 100.*other/total;
                    snprintf(buffer, sizeof(buffer), "%-25s%10.3f%10.1f",
                                    label.c_str(), float(other), percentage);
                    stream << buffer << std::endl;
                }
            }
        }

        void fuse(const profiler_node& other) {
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
    };
} // namespace impl

using timer_type = nest::mc::threading::timer;

// a region in the profiler, has
// - name
// - accumulated timer
// - nested sub-regions
class region_type {
    region_type *parent_ = nullptr;
    std::string name_;
    size_t hash_;
    std::unordered_map<size_t, std::unique_ptr<region_type>> subregions_;
    timer_type::time_point start_time_;
    double total_time_ = 0;

public:

    using profiler_node = impl::profiler_node;

    explicit region_type(std::string n) :
        //name_(std::move(n))
        name_(n)
    {
        std::cout << "creating region " << name_ << "\n";
        start_time_ = timer_type::tic();
        std::cout << "  started timer " << name_ << "\n";
        hash_ = impl::hash(n);
        std::cout << "  hashed " << name_ << " -> " << hash_ << "\n";
    }

    explicit region_type(const char* n) :
        region_type(std::string(n))
    {}

    region_type(std::string n, region_type* p) :
        region_type(std::move(n))
    {
        parent_ = p;
    }

    const std::string& name() const {
        return name_;
    }

    void name(std::string n) {
        name_ = std::move(n);
    }

    region_type* parent() {
        return parent_;
    }

    void start_time() { start_time_ = timer_type::tic(); }
    void end_time  () { total_time_ += timer_type::toc(start_time_); }

    bool has_subregions() const {
        return subregions_.size() > 0;
    }

    size_t hash() const {
        return hash_;
    }

    region_type* subregion(const char* n) {
        size_t hsh = impl::hash(n);
        auto s = subregions_.find(hsh);
        if (s == subregions_.end()) {
            subregions_[hsh] = util::make_unique<region_type>(n, this);
            return subregions_[hsh].get();
        }
        return s->second.get();
    }

    double subregion_contributions() const {
        return
            std::accumulate(
                subregions_.begin(), subregions_.end(), 0.,
                [](double l, decltype(*(subregions_.begin())) r) {
                    return l+r.second->total();
                }
            );
    }

    double total() const {
        return total_time_;
    }

    profiler_node populate_performance_tree() const {
        profiler_node tree(total(), name());

        for (auto &it : subregions_) {
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
};

class profiler {
public:
    profiler(std::string name) :
        root_region_(std::move(name))
    {}

    // the copy constructor doesn't do a "deep copy"
    // it simply creates a new profiler with the same name
    // This is needed for tbb to create a list of thread local profilers
    profiler(const profiler& other) :
        profiler(other.root_region_.name())
    {}

    void enter(const char* name) {
        if (!is_activated()) return;
        auto start = timer_type::tic();
        current_region_ = current_region_->subregion(name);
        current_region_->start_time();
        self_time_ += timer_type::toc(start);
    }

    void leave() {
        if (!is_activated()) return;
        auto start = timer_type::tic();
        if (current_region_->parent()==nullptr) {
            throw std::out_of_range("attempt to leave root memory tracing region");
        }
        current_region_->end_time();
        current_region_ = current_region_->parent();
        self_time_ += timer_type::toc(start);
    }

    region_type& regions() {
        return root_region_;
    }

    region_type* current_region() {
        return current_region_;
    }

    double self_time() const {
        return self_time_;
    }

    bool is_in_root() const {
        return &root_region_ == current_region_;
    }

    bool is_activated() const {
        return activated_;
    }

    void start() {
        if (is_activated()) {
            throw std::out_of_range(
                    "attempt to start an already running profiler"
                  );
        }
        activate();
        root_region_.start_time();
    }

    void stop() {
        if (!is_in_root()) {
            throw std::out_of_range(
                    "attempt to profiler that is not in the root region"
                  );
        }
        root_region_.end_time();
        deactivate();
    }

    region_type::profiler_node performance_tree() {
        if (is_activated()) {
            stop();
        }
        return root_region_.populate_performance_tree();
    }

private:
    void activate()   { activated_ = true;  }
    void deactivate() { activated_ = false; }

    bool activated_ = false;
    region_type root_region_;
    region_type* current_region_ = &root_region_;
    double self_time_ = 0.;
};

namespace data {
    using profiler_wrapper = nest::mc::threading::enumerable_thread_specific<profiler>;
    profiler_wrapper profilers_(profiler("root"));
}

/*
#ifdef WITH_PROFILING
namespace data {
    using profiler_wrapper = nest::mc::threading::enumerable_thread_specific<profiler>;
    profiler_wrapper profilers_(profiler("root"));
}

inline profiler& get_profiler() {
    auto& p = data::profilers_.local();
    if (!p.is_activated()) {
        p.start();
    }
    return p;
}

// this will throw an exception if the profler has already been started
inline void profiler_start() {
    data::profilers_.local().start();
}
inline void profiler_stop() {
    get_profiler().stop();
}
inline void profiler_enter(const char* n) {
    get_profiler().enter(n);
}
inline void profiler_leave() {
    get_profiler().leave();
}

// iterate over all profilers and ensure that they have the same start stop times
inline void stop_profilers() {
    std::cout << "::profiler : stopping " << data::profilers_.size() << " profilers\n";
    for (auto& p : data::profilers_) {
        p.stop();
    }
}

#else
*/
inline void profiler_start() {}
inline void profiler_stop() {}
inline void profiler_enter(const char* n) {}
inline void profiler_leave() {}
inline void stop_profilers() {}
//#endif

} // namespace util
} // namespace mc
} // namespace nest
