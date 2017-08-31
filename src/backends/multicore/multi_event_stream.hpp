#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include <limits>
#include <ostream>
#include <utility>

#include <common_types.hpp>
#include <backends/event.hpp>
#include <generic_event.hpp>
#include <util/debug.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>
#include <util/strprintf.hpp>

namespace nest {
namespace mc {
namespace multicore {

template <typename Event>
class multi_event_stream {
public:
    using size_type = cell_size_type;
    using event_type = Event;

    using event_time_type = ::nest::mc::event_time_type<Event>;
    using event_data_type = ::nest::mc::event_time_type<Event>;
    using event_index_type = ::nest::mc::event_index_type<Event>;

    multi_event_stream() {}

    explicit multi_event_stream(size_type n_stream):
       span_(n_stream), mark_(n_stream) {}

    size_type n_streams() const { return span_.size(); }

    bool empty() const { return remaining_==0; }

    void clear() {
        ev_data_.clear();
        remaining_ = 0;

        util::fill(span_, span_type(0u, 0u));
        util::fill(mark_, 0u);
    }

    // Initialize event streams from a vector of events, sorted first by index
    // and then by time.
    void init(const std::vector<Event>& staged) {
        using ::nest::mc::event_time;
        using ::nest::mc::event_index;
        using ::nest::mc::event_data;

        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw std::range_error("too many events");
        }

        // Staged events should already be sorted by index.
        EXPECT(algorithm::is_sorted_by(staged, [](const Event& ev) { return event_index(ev); }));

        std::size_t n_ev = staged.size();

        util::assign_by(ev_data_, staged, [](const Event& ev) { return event_data(ev); });
        util::assign_by(ev_time_, staged, [](const Event& ev) { return event_time(ev); });

        // Determine divisions by `event_index` in ev list.
        span_.clear();
        span_.reserve(n_stream_);

        mark_.clear();
        span_.reserve(n_stream_);

        size_type ev_begin_i = 0;
        size_type ev_i = 0;
        for (size_type s = 0; s<n_stream_; ++s) {
            while (ev_i<n_ev && event_index(staged[ev_i])<s+1) ++ev_i;

            // Within a subrange of events with the same index, events should
            // be sorted by time.
            EXPECT(std::is_sorted(&ev_time_[ev_begin], &ev_time_[ev_i]));
            span_.push_back({ev_begin, ev_i});
            mark_.push_back(ev_begin);
            ev_begin = ev_i;
        }

        remaining_ = n_ev;
    }

    // Designate for processing events `ev` at head of each event stream `i`
    // until `event_time(ev)` > `t_until[i]`.
    template <typename TimeSeq>
    void mark_until_after(const TimeSeq& t_until) {
        using ::nest::mc::event_time;

        EXPECTS(n_streams()==util::size(t_until));

        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            auto& span = span_[i];
            auto& mark = mark_[i];
            auto t = t_until[i];

            mark = span.first;
            while (mark!=span.second && !(event_time(ev_[mark])>t)) {
                ++mark;
            }
        }
    }

    // Remove marked events from front of each event stream.
    void drop_marked_events() {
        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            remaining_ -= (mark_[i]-span_[i].first);
            span_[i].first = mark_[i];
        }
    }

    // Return range of marked events on stream `i`.
    util::range<event_data_type*> marked_events(size_type i) {
        return {&ev_data_[span_[i].first], &ev_data_[mark_[i]]};
    }

    // If the head of `i`th event stream exists and has time less than `t_until[i]`, set
    // `t_until[i]` to the event time.
    template <typename TimeSeq>
    void event_time_if_before(TimeSeq& t_until) {
        using ::nest::mc::event_time;

        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            const auto& span = span_[i];
            if (span.second==span.first) {
               continue;
            }

            auto ev_t = event_time(ev_[span.first]);
            if (t_until[i]>ev_t) {
                t_until[i] = ev_t;
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const multi_event_stream& m);

private:
    using span_type = std::pair<size_type, size_type>;

    std::vector<event_time_type> ev_time_;
    std::vector<event_data_type> ev_;
    std::vector<span_type> span_;
    std::vector<size_type> mark_;
    size_type remaining_ = 0;
};


template <typename Event>
inline std::ostream& operator<<(std::ostream& out, const multi_event_stream<Event>& m) {
    using ::nest::mc::event_time;

    auto n = m.n_streams();

    out << "\n[";
    unsigned i = 0;
    for (unsigned ev_i = 0; ev_i<m.ev_.size(); ++ev_i) {
        while (m.span_[i].second<=ev_i && i<n) ++i;
        if (i<n) {
            out << util::strprintf(" % 7d ", i);
        }
        else {
            out << "      ?";
        }
    }
    out << "\n[";

    i = 0;
    for (unsigned ev_i = 0; ev_i<m.ev_.size(); ++ev_i) {
        while (m.span_[i].second<=ev_i && i<n) ++i;

        bool discarded = i<n && m.span_[i].first>ev_i;
        bool marked = i<n && m.mark_[i]>ev_i;

        if (discarded) {
            out << "        x";
        }
        else {
            out << util::strprintf(" % 7.3f%c", event_time(m.ev_[ev_i]), marked?'*':' ');
        }
    }
    out << "]\n";
    return out;
}

} // namespace multicore
} // namespace nest
} // namespace mc
