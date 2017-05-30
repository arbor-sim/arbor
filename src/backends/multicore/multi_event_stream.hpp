#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include <limits>
#include <ostream>
#include <utility>

#include <common_types.hpp>
#include <backends/event.hpp>
#include <util/debug.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>
#include <util/strprintf.hpp>

namespace nest {
namespace mc {
namespace multicore {

class multi_event_stream {
public:
    using size_type = cell_size_type;
    using value_type = double;

    multi_event_stream() {}

    explicit multi_event_stream(size_type n_stream):
       span_(n_stream), mark_(n_stream) {}

    size_type n_streams() const { return span_.size(); }

    bool empty() const { return remaining_==0; }

    void clear() {
        ev_.clear();
        remaining_ = 0;

        util::fill(span_, span_type(0u, 0u));
        util::fill(mark_, 0u);
    }

    // Initialize event streams from a vector of `deliverable_event`.
    void init(std::vector<deliverable_event> staged) {
        if (staged.size()>std::numeric_limits<size_type>::max()) {
            throw std::range_error("too many events");
        }

        ev_ = std::move(staged);
        util::stable_sort_by(ev_, [](const deliverable_event& e) { return e.handle.cell_index; });

        util::fill(span_, span_type(0u, 0u));
        util::fill(mark_, 0u);

        size_type si = 0;
        for (size_type ev_i = 0; ev_i<ev_.size(); ++ev_i) {
            size_type i = ev_[ev_i].handle.cell_index;
            EXPECTS(i<n_streams());

            if (si<i) {
                // Moved on to a new cell: make empty spans for any skipped cells.
                for (size_type j = si+1; j<i; ++j) {
                    span_[j].second = span_[si].second;
                }
                si = i;
            }
            // Include event in this cell's span.
            span_[si].second = ev_i+1;
        }

        while (++si<n_streams()) {
            span_[si].second = span_[si-1].second;
        }

        for (size_type i = 1u; i<n_streams(); ++i) {
            mark_[i] = span_[i].first = span_[i-1].second;
        }

        remaining_ = ev_.size();
    }

    // Designate for processing events `ev` at head of each event stream `i`
    // until `event_time(ev)` > `t_until[i]`.
    template <typename TimeSeq>
    void mark_until_after(const TimeSeq& t_until) {
        EXPECTS(n_streams()==util::size(t_until));

        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            auto& span = span_[i];
            auto& mark = mark_[i];
            auto t = t_until[i]; 

            mark = span.first;
            while (mark!=span.second && !(ev_[mark].time>t)) {
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
    util::range<deliverable_event*> marked_events(size_type i) {
        return {&ev_[span_[i].first], &ev_[mark_[i]]};
    }

    // If the head of `i`th event stream exists and has time less than `t_until[i]`, set
    // `t_until[i]` to the event time.
    template <typename TimeSeq>
    void event_time_if_before(TimeSeq& t_until) {
        // note: operation on each `i` is independent.
        for (size_type i = 0; i<n_streams(); ++i) {
            const auto& span = span_[i];
            if (span.second==span.first) {
               continue;
            }

            auto ev_t = ev_[span.first].time;
            if (t_until[i]>ev_t) {
                t_until[i] = ev_t;
            }
        }
    }

private:
    using span_type = std::pair<size_type, size_type>;

    std::vector<deliverable_event> ev_;
    std::vector<span_type> span_;
    std::vector<size_type> mark_;
    size_type remaining_ = 0;
};

} // namespace multicore
} // namespace nest
} // namespace mc
