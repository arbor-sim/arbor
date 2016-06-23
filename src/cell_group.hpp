#pragma once

#include <vector>

#include <cell.hpp>
#include <event_queue.hpp>
#include <communication/spike.hpp>
#include <communication/spike_source.hpp>

namespace nest {
namespace mc {

template <typename Cell>
class cell_group {
    public :

    using index_type = uint32_t;
    using cell_type = Cell;
    using value_type = typename cell_type::value_type;
    using size_type  = typename cell_type::value_type;
    using spike_detector_type = spike_detector<Cell>;

    struct spike_source_type {
        index_type index;
        spike_detector_type source;
    };

    cell_group() = default;

    cell_group(const cell& c) :
        cell_{c}
    {
        cell_.voltage()(memory::all) = -65.;
        cell_.initialize();

        for(auto& d : c.detectors()) {
            spike_sources_.push_back( {
                0u, spike_detector_type(cell_, d.first, d.second, 0.f)
            });
        }
    }

    void set_source_gids(index_type gid) {
        for(auto& s : spike_sources_) {
            s.index = gid++;
        }
    }

    void set_target_lids(index_type lid) {
        first_target_lid_ = lid;
    }

    void splat(std::string fname) {
        char buffer[128];
        std::ofstream fid(fname);
        for(auto i=0u; i<tt.size(); ++i) {
            sprintf(buffer, "%8.4f %16.8f %16.8f\n", tt[i], vs[i], vd[i]);
            fid << buffer;
        }
    }

    void advance(double tfinal, double dt) {

        while (cell_.time()<tfinal) {
            tt.push_back(cell_.time());
            vs.push_back(cell_.voltage({0,0.0}));
            vd.push_back(cell_.voltage({1,0.5}));

            // look for events in the next time step
            auto tstep = std::min(tfinal, cell_.time()+dt);
            auto next = events_.pop_if_before(tstep);
            auto tnext = next ? next->time: tstep;

            // integrate cell state
            cell_.advance(tnext - cell_.time());

            // check for new spikes
            for (auto& s : spike_sources_) {
                auto spike = s.source.test(cell_, cell_.time());
                if(spike) {
                    spikes_.push_back({s.index, spike.get()});
                }
            }

            // apply events
            if (next) {
                cell_.apply_event(next.get());
            }
        }

    }

    template <typename R>
    void enqueue_events(R events) {
        for(auto e : events) {
            e.target -= first_target_lid_;
            events_.push(e);
        }
    }

    const std::vector<communication::spike<index_type>>&
    spikes() const {
        return spikes_;
    }

    cell_type&       cell()       { return cell_; }
    const cell_type& cell() const { return cell_; }

    const std::vector<spike_source_type>&
    spike_sources() const
    {
        return spike_sources_;
    }

    void clear_spikes() {
        spikes_.clear();
    }

    private :

    // TEMPORARY...
    std::vector<float> tt;
    std::vector<float> vs;
    std::vector<float> vd;

    cell_type cell_;
    std::vector<spike_source_type> spike_sources_;

    // spikes that are generated
    std::vector<communication::spike<index_type>> spikes_;
    event_queue events_;

    index_type first_target_lid_;
};

} // namespace mc
} // namespace nest
