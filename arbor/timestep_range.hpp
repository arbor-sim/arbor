#pragma once

#include <algorithm>
#include <iosfwd>
#include <limits>

#include <arbor/assert.hpp>
#include <util/iterutil.hpp>

#include "epoch.hpp"

namespace arb {

// A timestep_range splits the time interval between t0 and t1 into equally sized timesteps of
// length dt. The last timestep is adjusted to match t1 and is either shorter or minimally longer
// than the specified dt. The range can be iterated over and the iterators derefernce to objects of
// type `timestep` which can be queried for start and end times, as well as dt and midpoint.
class timestep_range {
public: // member types
    // Representation of a time step
    struct timestep {
        time_type t0_;
        time_type t1_;
        time_type t_begin() const noexcept { return t0_; }
        time_type t_end() const noexcept { return t1_; }
        time_type dt() const noexcept { return t1_ - t0_; }
        time_type midpoint() const noexcept { return t0_ + 0.5*dt(); }
    };

    using value_type = timestep;
    using size_type = unsigned;

private: // members
    time_type t0_;
    time_type t1_;
    time_type dt_;
    size_type n_;

public: // access
    timestep operator[](size_type i) const noexcept {
        arb_assert(i < n_);
        return { t0_+ i*dt_, i+1 >= n_ ? t1_ : t0_ + (i+1)*dt_};
    }

public: // iterator
    class const_iterator:
        public util::generating_view_iterator_adaptor<const_iterator, timestep_range> {
    private:
        using base = util::generating_view_iterator_adaptor<const_iterator, timestep_range>;
        friend class timestep_range;

        const timestep_range * r_ = nullptr;

    public: // ctor, assignment
        const_iterator() noexcept = default;
        const_iterator(const const_iterator&) noexcept = default;
        const_iterator& operator=(const const_iterator&) noexcept = default;

        const timestep_range& view() const noexcept {
            arb_assert(r_);
            return *r_;
        }

    private:
        const_iterator(const timestep_range* r, std::size_t i) noexcept : base(i), r_{r} {}
    };

public: // ctors, assignment, reset
    timestep_range() noexcept { reset(); }
    timestep_range(time_type t1, time_type dt) { reset(t1, dt); }
    timestep_range(const epoch& ep, time_type dt) { reset(ep, dt); }
    timestep_range(time_type t0, time_type t1, time_type dt) { reset(t0, t1, dt); }

    timestep_range(timestep_range&&) noexcept = default;
    timestep_range(const timestep_range&) = default;

    timestep_range& operator=(timestep_range&&) noexcept = default;
    timestep_range& operator=(const timestep_range&) = default;

    timestep_range& reset() noexcept {
        return reset(0, 0, 1);
    }

    timestep_range& reset(time_type t1, time_type dt) {
        return reset(0, t1, dt);
    }

    timestep_range& reset(const epoch& ep, time_type dt) {
        return reset(ep.t0, ep.t1, dt);
    }

    timestep_range& reset(time_type t0, time_type t1, time_type dt) {
        t0_ = t0;
        t1_ = t1;
        dt = dt < 0 ? (t1-t0) : dt;
        dt_ = std::max(std::numeric_limits<time_type>::min(), dt);
        const time_type delta = (t1<=t0)? 0: t1-t0;
        const auto m = static_cast<size_type>(delta/dt_);
        // Allow slightly larger time steps at the end of the epoch in order to avoid tiny time
        // steps, if the the last time step m*dt_ is at most m floating point representations
        // smaller than t1_. The tolerable floating point range is approximated by the scaled
        // machine epsilon times m.
        n_ = m*dt_ + std::numeric_limits<time_type>::epsilon()*t1_*m >= delta ? m : m + 1;
        return *this;
    }

public: // access and queries
    time_type t_begin() const noexcept { return t0_; }
    time_type t_end() const noexcept { return t1_; }

    bool empty() const noexcept { return !n_; }
    size_type size() const noexcept { return n_; }

    const_iterator begin() const noexcept { return {this, 0u}; }
    const_iterator end() const noexcept { return {this, n_}; }

    const_iterator find(time_type t) const noexcept {
        if (!n_ || t < t0_ || t >= t1_) return end();
        const auto n = std::min((size_type)((t-t0_)/dt_), n_-1);
        const auto [t0,t1] = this->operator[](n);
        if (t>=t0 && t<t1) return {this, n};
        else if (t<t0) return {this, n-1};
        else return {this, n+1};
    }

    template<class CharT, class Traits = std::char_traits<CharT>>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out,
        const timestep_range& r) {
        out << "{";
        for (const auto& x : r) {
            out << " [ " << x.t_begin() << ", " << x.t_end() << " ),";
        }
        out << " }\n";
        return out;
    }
};

} // namespace arb
