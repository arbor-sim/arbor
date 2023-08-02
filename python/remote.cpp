#include <arbor/communication/remote.hpp>
#include <arbor/version.hpp>

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "conversion.hpp"
#include "context.hpp"
#include "mpi.hpp"
#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {
    using namespace pybind11::literals;
#ifdef ARB_MPI_ENABLED
void register_remote(pybind11::module& m) {
    std::cout << "Registering remote with MPI\n";
    auto s = m.def_submodule("remote", "Wrappers for remote communication.");

    pybind11::class_<arb::remote::msg_null> msg_null(s, "msg_null", "Empty message.");
    msg_null
        .def(pybind11::init<>([]() { return arb::remote::msg_null{};}))
        .def("__repr__", [](const arb::remote::msg_null&){return "(arb::remote::msg_null)";})
        .def("__str__",  [](const arb::remote::msg_null&){return "(msg_null)";});

    pybind11::class_<arb::remote::msg_abort> msg_abort(s, "msg_abort", "Aborting with error.");
    msg_abort
        .def(pybind11::init<>([](const std::string& s) {
            auto res = arb::remote::msg_abort{};
            std::memset(res.reason, 0x0, sizeof(res.reason));
            std::strncpy(res.reason, s.c_str(), 512);
            return res;
        }),
        "reason"_a,
        "Signal abort with a reason.")
        .def(pybind11::init<>([]() {
            auto res = arb::remote::msg_abort{};
            std::memset(res.reason, 0x0, sizeof(res.reason));
            return res;
        }),
        "Signal abort without a reason.")
        .def("__repr__", [](const arb::remote::msg_abort& s){return util::pprintf("(arb::remote::msg_abort reason={})", s.reason);})
        .def("__str__", [](const arb::remote::msg_abort& s){return util::pprintf("(abort reason={})", s.reason);});

    pybind11::class_<arb::remote::msg_epoch> msg_epoch(s, "msg_epoch", "Commencing epoch.");
    msg_epoch
        .def(pybind11::init<>([](double f, double t) { return arb::remote::msg_epoch{f, t}; }),
             "from"_a, "to"_a,
             "Signal commencing of epoch [from, to).")
        .def("__repr__", [](const arb::remote::msg_epoch& s){return util::pprintf("(arb::remote::msg_epoch from={} to={})", s.t_start, s.t_end);})
        .def("__str__", [](const arb::remote::msg_epoch& s){return util::pprintf("(epoch from={} to={})", s.t_start, s.t_end);});

    pybind11::class_<arb::remote::msg_done> msg_done(s, "msg_done", "Concluded simulation period with final time.");
    msg_done
        .def(pybind11::init<>([](float t) { return arb::remote::msg_done{t}; }),
        "final"_a,
        "Signal conclusion of simulation at time `final``.")
        .def("__repr__", [](const arb::remote::msg_done& s){return util::pprintf("(arb::remote::msg_done to={})", s.time);})
        .def("__str__", [](const arb::remote::msg_done& s){return util::pprintf("(done to={})", s.time);});

    s.def("exchange_ctrl",
          [](arb::remote::ctrl_message msg, pybind11::object mpi) {
              auto err = "Invalid MPI Communicator.";
              if (can_convert_to_mpi_comm(mpi)) {
                  return arb::remote::exchange_ctrl(msg, convert_to_mpi_comm(mpi));
              }
              else if (auto c = py2optional<mpi_comm_shim>(mpi, err)) {
                  return arb::remote::exchange_ctrl(msg, c->comm);
              } else {
                  throw pyarb_error(err);
              }
          },
          "msg"_a, "mpi_comm"_a,
          "Send given control message to all peers and receive their (unanimous) answer.");

    pybind11::class_<arb::remote::arb_spike> arb_spike(s, "arb_spike", "Empty message.");
    arb_spike.def(pybind11::init<>([](std::uint32_t gid, std::uint32_t lid, double t) { return arb::remote::arb_spike{{gid, lid}, t};}),
                  "gid"_a, "lid"_a, "time"_a,
                  "Spike caused by cell `gid` on location `lid` at time `time`.")
        .def("__repr__", [](const arb::remote::arb_spike& s){return util::pprintf("(arb::remote::arb_spike gid={} lid={} time={})", s.source.gid, s.source.lid, s.time);})
        .def("__str__", [](const arb::remote::arb_spike& s){return util::pprintf("(spike gid={} lid={} time={})", s.source.gid, s.source.lid, s.time);});

    s.def("gather_spikes",
          [](const std::vector<arb::remote::arb_spike>& msg, pybind11::object mpi) {
              auto err = "Invalid MPI Communicator.";
              if (can_convert_to_mpi_comm(mpi)) {
                  return arb::remote::gather_spikes(msg, convert_to_mpi_comm(mpi));
              }
              else if (auto c = py2optional<mpi_comm_shim>(mpi, err)) {
                  return arb::remote::gather_spikes(msg, c->comm);
              } else {
                  throw pyarb_error(err);
              }
          },
          "msg"_a, "mpi_comm"_a,
          "Send list of spikes to all peers and receive their collected answer.");
}
#else
void register_remote(pybind11::module& m) {}
#endif
}
