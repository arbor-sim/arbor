#include <functional>
#include <iostream>

#include <arbor/spike.hpp>
#include <aux/spike_emitter.hpp>

namespace aux {

spike_emitter::spike_emitter(std::ostream& out): out(out) {}

void spike_emitter::operator()(const std::vector<arb::spike>& spikes) {
    char line[45];
    for (auto& s: spikes) {
        int n = std::snprintf(line, sizeof(line), "%u %.4f",  s.source.gid, s.time);
        if (n<0) {
            throw std::system_error(errno, std::generic_category());
        }

        out.get().write(line, n).put('\n');
    }
};

} // namespace aux
