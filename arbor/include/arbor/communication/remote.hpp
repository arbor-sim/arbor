#pragma once

#include <cstring>
#include <memory>
#include <cstdint>
#include <exception>
#include <variant>

#include <mpi.h>

namespace arb {
namespace remote {

// Magic protocol tag
constexpr std::uint8_t ARB_REMOTE_MAGIC = 0xAB;

// Remote protocol version
constexpr std::uint8_t ARB_REMOTE_VERSION_MAJOR = 0x00;
constexpr std::uint8_t ARB_REMOTE_VERSION_MINOR = 0x01;
constexpr std::uint8_t ARB_REMOTE_VERSION_PATCH = 0x00;

// Message buffer length
constexpr std::size_t ARB_REMOTE_MESSAGE_LENGTH = 1024;

// Who sends the control message?
constexpr int ARB_REMOTE_ROOT = 0;

// Messages
struct msg_null {
    static constexpr std::uint8_t tag = 0x00;
    std::uint8_t null = 0x0;
};

struct msg_abort {
    static constexpr std::uint8_t tag = 0x01;
    char reason[512];
};

struct msg_epoch {
    static constexpr std::uint8_t tag = 0x02;
    float t_start;
    float t_end;
};

struct msg_done {
    static constexpr std::uint8_t tag = 0x03;
    float time = 0.0f;
};

using ctrl_message = std::variant<msg_null,
                                  msg_abort,
                                  msg_epoch,
                                  msg_done>;

// Exceptions
struct remote_error: std::runtime_error {
    remote_error(const std::string& msg): std::runtime_error{msg} {}
};

struct unexpected_version: remote_error {
    unexpected_version(): remote_error{"Arbor remote: Magic or Version mismatch."} {}
};

struct unexpected_message: remote_error {
    unexpected_message(): remote_error{"Arbor remote: Received unknown tag."} {}
};

struct mpi_error: remote_error {
    mpi_error(const std::string& where, int rc):
        remote_error{"MPI failed in " + where + " with error=" + std::to_string(rc)} {}
};

inline
void mpi_checked(int rc, const std::string& where) {
    if (rc != MPI_SUCCESS) throw mpi_error{where, rc};
}

// Prepare byte buffer for sending.
inline
std::vector<char> make_send_buffer(const ctrl_message& msg) {
    std::vector<char> send(ARB_REMOTE_MESSAGE_LENGTH, 0x0);
    std::size_t ptr = 0;
    send[ptr++] = ARB_REMOTE_MAGIC;
    send[ptr++] = ARB_REMOTE_VERSION_MAJOR;
    send[ptr++] = ARB_REMOTE_VERSION_MINOR;
    send[ptr++] = ARB_REMOTE_VERSION_PATCH;
    auto visitor = [&send, &ptr] (auto&& m) {
        using T = std::decay_t<decltype(m)>;
        send[ptr++] = T::tag;
        memcpy(send.data() + ptr, &m, sizeof(m

));
    };
    std::visit(visitor, msg);
    return send;
}

// Consume receive buffer and generate a message
inline
ctrl_message make_ctrl_message(const std::vector<char>& recv) {
    std::size_t ptr = 0;
    std::uint8_t mag = recv[ptr++];
    std::uint8_t maj = recv[ptr++];
    std::uint8_t min = recv[ptr++];
    std::uint8_t pat = recv[ptr++];
    if ((mag != ARB_REMOTE_MAGIC) ||
        (maj != ARB_REMOTE_VERSION_MAJOR) ||
        (min != ARB_REMOTE_VERSION_MINOR) ||
        (pat != ARB_REMOTE_VERSION_PATCH)) throw unexpected_message{};
    std::uint8_t tag = recv[ptr++];
    auto payload = recv.data() + ptr;
    ctrl_message result = msg_null{};
    switch(tag) {
        case msg_null::tag: {
            result = *reinterpret_cast<const msg_null*>(payload);
            break;
        }
        case msg_abort::tag: {
            result = *reinterpret_cast<const msg_abort*>(payload);
            break;
        }
        case msg_epoch::tag: {
            result = *reinterpret_cast<const msg_epoch*>(payload);
            break;
        }
        case msg_done::tag: {
            result = *reinterpret_cast<const msg_done*>(payload);
            break;
        }
        default: throw unexpected_message{};
    }
    return result;
}

// Exchange control message.
inline
ctrl_message exchange_ctrl(MPI_Comm comm, const ctrl_message& msg) {
    ctrl_message result = msg_null{};
    int rank = -1;
    mpi_checked(MPI_Comm_rank(comm, &rank), "echange ctrl block: comm rank");
    if (rank != ARB_REMOTE_ROOT) return result;

    auto send = make_send_buffer(msg);
    auto recv = std::vector<char>(ARB_REMOTE_MESSAGE_LENGTH, 0x0);
    MPI_Status ignored;
    mpi_checked(MPI_Sendrecv((const void*) send.data(), ARB_REMOTE_MESSAGE_LENGTH, MPI_CHAR, ARB_REMOTE_ROOT, ARB_REMOTE_MAGIC,
                             (void*)       recv.data(), ARB_REMOTE_MESSAGE_LENGTH, MPI_CHAR, MPI_ANY_SOURCE,  ARB_REMOTE_MAGIC,
                             comm,
                             &ignored), "exchange control block: Sendrecv");
    return make_ctrl_message(recv);
}

template <typename T>
std::vector<T> gather_all(const std::vector<T>& send, MPI_Comm comm) {
    int size = -1, rank = -1;
    mpi_checked(MPI_Comm_size(comm, &size), "gather_all: comm size");
    mpi_checked(MPI_Comm_rank(comm, &rank), "gather_all: comm rank");
    int send_count = send.size();
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    mpi_checked(MPI_Allgather(&send_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm),
                "gather_all exchanging counts: Allgather");
    int recv_bytes = 0;
    int recv_count = 0;
    for (int ix = 0; ix < size; ++ix) {
        recv_count += counts[ix]; // Number of items to receive.
        counts[ix] *= sizeof(T);  // Number of Bytes for rank ``ix`
        displs[ix]  = recv_bytes; // Offset for rank `ix` in Bytes
        recv_bytes += counts[ix]; // Total number of items so far

    }
    std::vector<T> recv(recv_count);
    auto send_bytes = send_count*sizeof(T);
    mpi_checked(MPI_Allgatherv(send.data(), send_bytes,                   MPI_BYTE, // send buffer
                               recv.data(), counts.data(), displs.data(), MPI_BYTE, // recv buffer
                               comm),
                "gather_all exchanging payload: Allgatherv");
    return recv;
}

} // remote
} // arb
