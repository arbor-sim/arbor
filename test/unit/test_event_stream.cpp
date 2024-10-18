#include "backends/multicore/event_stream.hpp"
#include "./test_event_stream.hpp"

namespace {

template<typename Result>
void check(Result result) {
    for (std::size_t step=0; step<result.steps.size(); ++step) {
        for (auto& [mech_id, stream] :  result.streams) {
            stream.mark();
            auto marked = stream.marked_events();
            check_result(marked.begin, result.expected[mech_id][step]);
        }
    }
}

}

TEST(event_stream, single_step) {
    check(single_step<multicore::spike_event_stream>());
}

TEST(event_stream, multi_step) {
    check(multi_step<multicore::spike_event_stream>());
}
