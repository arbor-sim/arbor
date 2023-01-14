#pragma once

#include <algorithm>
#include <iosfwd>
#include <limits>
#include <vector>

#include "epoch.hpp"

namespace arb {

// A timestep_range splits the time interval between t0 and t1 into equally sized timesteps of
// length dt. The last timestep is adjusted to match t1 and is either shorter or minimally longer
// than the specified dt (max length = (1+1e-8)*dt). The range can be iterated over and the
// iterators point to objects of type `timestep` which can be queried for start and end times, as
// well as dt and midpoint.

class timestep_range {
public:
    struct timestep {
        time_type t0_;
        time_type t1_;
        time_type t0() const noexcept { return t0_; }
        time_type t1() const noexcept { return t1_; }
        time_type dt() const noexcept { return t1_ - t0_; }
        time_type midpoint() const noexcept { return t0_ + 0.5*dt(); }
    };

    using vec = std::vector<timestep>;
    using value_type = vec::value_type;
    using const_iterator = vec::const_iterator;
    using size_type = vec::size_type;

    static constexpr size_type npos = std::numeric_limits<size_type>::max();

public:
    timestep_range() noexcept { reset(); }
    timestep_range(time_type t1, time_type dt) { reset(t1, dt); }
    timestep_range(const epoch& ep, time_type dt) { reset(ep, dt); }
    timestep_range(time_type t0, time_type t1, time_type dt) { reset(t0, t1, dt); }

    timestep_range(timestep_range&&) noexcept = default;
    timestep_range(const timestep_range&) = default;

    timestep_range& operator=(timestep_range&&) noexcept = default;
    timestep_range& operator=(const timestep_range&) = default;

    timestep_range& reset() noexcept {
        t0_ = 0;
        t1_ = 0;
        dt_ = 1;
        data_.clear();
        return *this;
    }

    timestep_range& reset(time_type t1, time_type dt) {
        return reset(0, t1, dt);
    }

    timestep_range& reset(const epoch& ep, time_type dt) {
        return reset(ep.t0, ep.t1, dt);
    }

    timestep_range& reset(time_type t0, time_type t1, time_type dt) {
        data_.clear();
        t0_ = t0;
        t1_ = t1;
        dt = dt < 0 ? (t1-t0) : dt;
        dt_ = std::max(std::numeric_limits<time_type>::min(), dt);
        const time_type delta = (t1<=t0)? 0: t1-t0;
        const size_type m = static_cast<size_type>(delta/dt_);
        const size_type n = m*dt_ + 1e-8*dt_ >= delta ? m : m + 1;
        data_.reserve(n);
        for(std::size_t i=0; i<n; ++i) {
            data_.push_back(timestep{t0_ + i*dt_, t0_ + (i+1)*dt_});
        }
        if (!empty()) data_.back().t1_ = t1_;
        return *this;
    }

    time_type t0() const noexcept { return t0_; }
    time_type t1() const noexcept { return t1_; }

    bool empty() const noexcept { return data_.empty(); }
    size_type size() const noexcept { return data_.size(); }

    const_iterator begin() const noexcept { return data_.cbegin(); }
    const_iterator end() const noexcept { return data_.cend(); }

    timestep operator[](size_type i) const noexcept { return data_[i]; }

    size_type index(time_type t) const noexcept {
        if (empty() || t<t0_ || t>=t1_) return npos;
        const auto idx = static_cast<size_type>((t-t0_)/dt_);
        // check if really in interval to rule out roundoff errors
        if (t>= data_[idx].t0() && t < data_[idx].t1()) return idx;
        else if (t < data_[idx].t0()) return idx-1;
        else return idx+1;
    }

    template<class CharT, class Traits = std::char_traits<CharT>>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out,
        const timestep_range& r) {
        out << "{";
        for (const auto& x : r) {
            out << " [ " << x.t0() << ", " << x.t1() << " ]";
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
