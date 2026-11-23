#pragma once

#include "sdma.hpp"
#include <common/pto_message.hpp> // Linx-TileOP-API PR 315

namespace deepseek_v41::wse {

template <class T> bool send_packet(uint32_t destination, const T &packet) {
    linx_msgb_result status;
    do {
        status = pto::PTOSendMessage(destination, packet);
    } while (status == LINX_MSGB_BUSY);
    return status == LINX_MSGB_OK || status == LINX_MSGB_DUPLICATE;
}

template <class T> bool receive_packet(T &packet) {
    linx_msgb_result status;
    do {
        status = pto::PTORecvMessage(packet);
    } while (status == LINX_MSGB_EMPTY);
    return status == LINX_MSGB_OK;
}

// Copy tickets start at one. Ticket zero with an empty payload ends the session.
inline constexpr CopyRequest EndOfCopies{};

// A dedicated, pre-provisioned endpoint pair supports one outstanding copy.
class MessageSdma {
  public:
    explicit MessageSdma(uint32_t endpoint) : endpoint_(endpoint) {}

    bool exchange(const CopyRequest &request, CopyDone &done) const {
        __atomic_thread_fence(__ATOMIC_RELEASE);
        if (!send_packet(endpoint_, request) || !receive_packet(done))
            return false;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        return true;
    }

    bool finish() const {
        CopyDone done{};
        return exchange(EndOfCopies, done) && done.matches(EndOfCopies);
    }

  private:
    uint32_t endpoint_;
};

// A platform engine must finish the destination write in the negotiated
// visibility domain before returning a successful CopyDone.
template <class Engine> bool serve_copies(uint32_t client, Engine &engine) {
    for (;;) {
        CopyRequest request;
        if (!receive_packet(request))
            return false;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        const bool finished = request.ticket == EndOfCopies.ticket && request.bytes == 0;
        CopyDone done{};
        if (!finished && !engine.exchange(request, done))
            done = {request.ticket, 0, -1};
        __atomic_thread_fence(__ATOMIC_RELEASE);
        if (!send_packet(client, done))
            return false;
        if (finished)
            return true;
    }
}

} // namespace deepseek_v41::wse
