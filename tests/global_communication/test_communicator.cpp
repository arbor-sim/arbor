#include "../gtest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>

using namespace nest::mc;

using time_type = float;
using communicator_type = communication::communicator<time_type, communication::global_policy>;

bool is_dry_run() {
    return communication::global_policy::kind() ==
        communication::global_policy_kind::dryrun;
}

TEST(communicator, policy_basics) {
    using policy = communication::global_policy;

    const auto num_domains = policy::size();
    const auto rank = policy::id();

    EXPECT_EQ(policy::min(rank), 0);
    if (!is_dry_run()) {
        EXPECT_EQ(policy::max(rank), num_domains-1);
    }
}

// some proxy types for communication testing
struct source_proxy {
    source_proxy() = default;
    source_proxy(int g): gid(g) {}

    int gid = 0;
};

bool operator==(int other, source_proxy s) {return s.gid==other;};
bool operator==(source_proxy s, int other) {return s.gid==other;};

struct spike_proxy {
    spike_proxy() = default;
    spike_proxy(int s, int d): source(s), domain(d) {}
    source_proxy source = 0;
    int domain = 0;
};

TEST(communicator, gather_spikes) {
    using policy = communication::global_policy;

    const auto num_domains = policy::size();
    const auto rank = policy::id();

    // dry run mode is a special case
    if (is_dry_run()) {
        const auto n_local_spikes = 10;
        const auto n_local_cells = n_local_spikes;

        // Important: set up meta-data in dry run back end.
        policy::set_sizes(policy::size(), n_local_cells);

        // create local spikes for communication
        std::vector<spike_proxy> local_spikes;
        for (auto i=0; i<n_local_spikes; ++i) {
            local_spikes.push_back(spike_proxy{i, rank});
        }

        // perform exchange
        const auto global_spikes = policy::gather_spikes(local_spikes);

        // test that partition information is correct
        const auto& part = global_spikes.partition();
        EXPECT_EQ(num_domains+1u, part.size());
        for (auto i=0u; i<part.size(); ++i) {
            EXPECT_EQ(part[i], n_local_spikes*i);
        }

        // test that spikes were correctly exchanged
        //
        // The local spikes had sources numbered 0:n_local_spikes-1
        // The global exchange should replicate the local spikes and
        // shift their sources to make them local to the "dummy" source
        // domain.
        // We set the model up with n_local_cells==n_local_spikes with
        // one spike per local cell, so the result of the global exchange
        // is a list of num_domains*n_local_spikes spikes that have
        // contiguous source gid
        const auto& spikes = global_spikes.values();
        for (auto i=0u; i<spikes.size(); ++i) {
            const auto s = spikes[i];
            EXPECT_EQ(i, unsigned(s.source.gid));
            EXPECT_EQ(0, s.domain);
        }
    }
    else {
        const auto scale = 10;
        auto sumn = [scale](int n) {return scale*n*(n+1)/2;};
        const auto n_local_spikes = scale*rank;

        // create local spikes for communication
        // the ranks generate different numbers of spikes, with the ranks
        // generating the following number of spikes
        //      [ 0, scale, 2*scale, 3*scale, ..., (num_domains-1)*scale ]
        // i.e. 0 spikes on the first rank, scale spikes on the second, and so on.
        std::vector<spike_proxy> local_spikes;
        const auto local_start_id = sumn(rank-1);
        for (auto i=0; i<n_local_spikes; ++i) {
            local_spikes.push_back(spike_proxy{local_start_id+i, rank});
        }

        // perform exchange
        const auto global_spikes = policy::gather_spikes(local_spikes);

        // test that partition information is correct
        const auto& part =global_spikes.partition();
        EXPECT_EQ(unsigned(num_domains+1), part.size());
        EXPECT_EQ(0, (int)part[0]);
        for (auto i=1u; i<part.size(); ++i) {
            EXPECT_EQ(sumn(i-1), (int)part[i]);
        }

        // test that spikes were correctly exchanged
        for (auto domain=0; domain<num_domains; ++domain) {
            auto source = sumn(domain-1);
            const auto first_spike = global_spikes.values().begin() + sumn(domain-1);
            const auto last_spike  = global_spikes.values().begin() + sumn(domain);
            const auto spikes = util::make_range(first_spike, last_spike);
            for (auto s: spikes) {
                EXPECT_EQ(s.domain, domain);
                EXPECT_EQ(s.source, source++);
            }
        }
    }
}

