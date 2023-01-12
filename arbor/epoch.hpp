#pragma once

#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// An epoch describes an integration interval within an ongoing simulation.
// Epochs within a simulation are sequentially numbered by id, and each
// describe a half open time interval [t0, t1), where t1 for one epoch
// is t0 for the next.
//
// At the end of an epoch the simulation state corresponds to the time t1,
// save for any pending events, which will be delivered at the beginning of
// the next epoch.

struct epoch {
    std::ptrdiff_t id = -1;
    time_type t0 = 0, t1 = 0;

    epoch() = default;
    epoch(std::ptrdiff_t id, time_type t0, time_type t1):
        id(id), t0(t0), t1(t1) {}

    operator bool() const {
        arb_assert(id>=-1);
        return id>=0;
    }

    bool empty() const {
        return t1<=t0;
    }

    time_type duration() const {
        return t1-t0;
    }

    void advance_to(time_type next_t1) {
        t0 = t1;
        t1 = next_t1;
        ++id;
    }

    void reset() {
        *this = epoch();
    }

};

class epoch_interval {
public:
    struct interval {
        time_type t0;
        time_type t1;

        time_type dt() const noexcept { return t1 - t0; }
        time_type midpoint() const noexcept { return t0 + 0.5*dt(); }
    };

    using vec = std::vector<interval>;
    using value_type = vec::value_type;
    using const_iterator = vec::const_iterator;
    using size_type = vec::size_type;

    static constexpr size_type npos = std::numeric_limits<size_type>::max();

public:
    epoch_interval(const epoch& ep, time_type dt) {
        t0_ = ep.t0;
        t1_ = ep.t1;
        dt = dt < 0 ? ep.duration() : dt;
        dt_ = std::max(std::numeric_limits<time_type>::min(), dt);
        const time_type delta = ep.empty()? 0: ep.duration();
        const size_type m = static_cast<size_type>(delta/dt_);
        const size_type n = m*dt_ + 1e-8*dt_ >= delta ? m : m + 1;
        data_.reserve(n);
        for(std::size_t i=0; i<n; ++i) {
            data_.push_back(interval{t0_ + i*dt_, t0_ + (i+1)*dt_});
        }
        if (!empty()) data_.back().t1 = t1_;
    }

    epoch_interval(epoch_interval&&) noexcept = default;
    epoch_interval(const epoch_interval&) = default;

    epoch_interval& operator=(epoch_interval&&) noexcept = default;
    epoch_interval& operator=(const epoch_interval&) = default;

    bool empty() const noexcept { return data_.empty(); }
    size_type size() const noexcept { return data_.size(); }
    const_iterator begin() const noexcept { return data_.cbegin(); }
    const_iterator end() const noexcept { return data_.cend(); }
    interval operator[](size_type i) const noexcept { return data_[i]; }

    size_type index(time_type t) const noexcept {
        if (empty() || t<t0_ || t>t1_) return npos;
        return std::min(static_cast<size_type>((t-t0_)/dt_),size()-1);
    }

    size_type midpoint_index(time_type t) const noexcept {
        if (empty()) return npos;
        if (t > data_.back().midpoint()) return npos;
        if (t < data_.front().midpoint()) return 0;
        if (size()==1) return 0;
        auto left = std::min(static_cast<size_type>((t-t0_)/dt_),size()-1);
        if (left >= size()-1) return size()-1;
        //auto right = left + 1;
        return t <= data_[left].midpoint()? left : left+1;
    }

    time_type last_midpoint() const noexcept {
        if (empty()) return npos;
        return data_.back().midpoint();
    }

    template<class CharT, class Traits = std::char_traits<CharT>>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out, const epoch_interval& ei) {
        out << "{";
        for (const auto& x : ei) {
            out << " [ " << x.t0 << ", " << x.t1 << " ]";
        }
        out << " }\n";
        out << "{";
        for (const auto& x : ei) {
            out << " " << x.midpoint();
        }
        out << " }\n";
        return out;
    }

private:
    time_type t0_;
    time_type t1_;
    time_type dt_;
    vec data_;
};

} // namespace arb
