#pragma once

/*
* Simple(st?) implementation of a recorder of scalar
* trace data from a cell probe, with some metadata.
*/

#include <stdexcept>
#include <type_traits>
#include <vector>
#include <iostream>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_ptr.hpp>

#include <contra/relay.hpp>
#include <contra/zmq/zeromq_transport.hpp>
#include <nesci/producer/arbor_multimeter.hpp>

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;
using arb::mc_cell;
using arb::section_kind;


using traces_type = std::vector<std::tuple< arb::cell_gid_type, arb::cell_lid_type,
    std::vector<std::tuple<arb::time_type, double>> >>;


// sampler that inserts trace information into a vector
// data insertion is guarded by a mutex, multiple threads might call this sampler function
// After data insertion a signal is forwarded to the receiver side that
// information is available.
class thread_forwarding_sampler {
public:
    explicit thread_forwarding_sampler(traces_type &traces, std::mutex& mutex,
        std::condition_variable& wake_up) : traces_(traces) , mutex_(mutex), wake_up_(wake_up)
    {}

    void operator()(cell_member_type probe_id, arb::probe_tag tag, std::size_t n,
        const arb::sample_record* recs) {

        // Local data structure for storing the trace. Filled outside of the mutex
        std::vector<std::tuple<arb::time_type, double>> trace;

        // For all samples n in the current batch
        for (std::size_t i = 0; i < n; ++i) {
            // TODO: Do we need to check this every single time?
            if (auto p = arb::util::any_cast<const double*>(recs[i].data)) {
                trace.push_back({ recs[i].time, *p });
            }
            else {
                throw std::runtime_error("unexpected sample type in printing_sampler");
            }
        }

        { // take the lock
            std::lock_guard<std::mutex> guard(mutex_);
            traces_.push_back({ probe_id.gid, probe_id.index, std::move(trace) });
        }
        //Tell the other side to wake up outside of the lock
        wake_up_.notify_one();
    }

private:
    traces_type & traces_;
    std::mutex& mutex_;
    std::condition_variable& wake_up_;
};


// publisher, to be used as a thread that consumes data generated in a different thread
//
// Waits for the wake_up signal guarded by a lock that is released on the other side
// While having the lock, the trace vector data is swapped and
// the quit flag are copied. With this done the lock is released and the other
// side notified that processing can continue.

void publisher(
    traces_type &traces,
    std::mutex & queue_mutex, std::condition_variable &wake_up, bool& quit)
{
    contra::Relay<contra::ZMQTransport> relay { contra::ZMQTransport::Type::SERVER, "tcp://*:5555", false };
    nesci::producer::ArborMultimeter multimeter { "some_name" };

    traces_type traces_local;
    bool quit_local;
    while (true) {
        // Wait on the wake_up signal,
        // TODO: WHy a unique lock here and a lock gaurd on the other side
        std::unique_lock<std::mutex> lock(queue_mutex);
        wake_up.wait(lock, [] {return true; });
        // We now have the mutex

        // Copy / swap the mutex guarded variables
        traces_local.swap(traces);
        quit_local = quit;

        // Release our mutex and signal the other thread we are done
        lock.unlock();
        // TODO: Should this be all? Because only on receiver can continue
        wake_up.notify_one();

        // Simple plotting
        for (auto& entry : traces_local) {
            auto gid = std::get<0>(entry);
            auto lid = std::get<1>(entry);
            auto trace = std::get<2>(entry);

            const std::string nesci_id = std::to_string(gid) + " " + std::to_string(lid);

            // std::cout << gid << ", " << lid << " \n";
            for (auto& value : trace) {
                auto time = std::get<0>(value);
                auto voltage = std::get<1>(value);

                // std::cout << time << ", " << voltage << "\n";

                nesci::producer::ArborMultimeter::Datum datum {
                    time, "voltage", nesci_id, voltage};
                multimeter.Record(datum);
            }
        }
        traces_local.clear();
        relay.Send(multimeter.node());
        multimeter.node().reset();

        if (quit_local) {
            break;
        }
    }
}
