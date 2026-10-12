#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core::conversation_detail {

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
/// private_stream: this thread's stream on the current device, created on first use.
inline cudaError_t private_stream(cudaStream_t& s) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    s = streams[(size_t) dev];
    return cudaSuccess;
}

/// `queued`: the copy is only queued on the private stream, and the caller synchronizes the device (or orders a
/// later launch after it with order_after_queued) before anything reads the target or writes the source.
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes, bool queued = false) {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess || queued) return e;
    return cudaStreamSynchronize(s);
}

/// Work queued on the legacy stream after this call starts after every copy this thread queued on the current
/// device's private stream (an event the legacy stream waits for; no host wait).
inline cudaError_t order_after_queued() {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    int dev = 0;
    e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaEvent_t> events;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= events.size()) events.resize((size_t) dev + 1, nullptr);    // never destroyed: process lifetime
    if (!events[(size_t) dev]) {
        e = cudaEventCreateWithFlags(&events[(size_t) dev], cudaEventDisableTiming);
        if (e != cudaSuccess) { events[(size_t) dev] = nullptr; return e; }
    }
    e = cudaEventRecord(events[(size_t) dev], s);
    if (e != cudaSuccess) return e;
    return cudaStreamWaitEvent(nullptr, events[(size_t) dev], 0);
}

/// staging_events: this thread's two events on the current device for a staged capture (StagedCopy), created on first
/// use and kept for the process like the private stream.
inline cudaError_t staging_events(std::array<cudaEvent_t, 2>& out) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<std::array<cudaEvent_t, 2>> events;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= events.size()) events.resize((size_t) dev + 1, {nullptr, nullptr});   // never destroyed
    for (cudaEvent_t& ev : events[(size_t) dev])
        if (!ev) {
            e = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (e != cudaSuccess) { ev = nullptr; return e; }
        }
    out = events[(size_t) dev];
    return cudaSuccess;
}

/// A capture's copies into host memory as ConversationCapture asks: without staging buffers each copy goes straight
/// into its target (copy_nonblocking, complete on return); with them, in blocks of at most `block` bytes through the
/// two pinned buffers on this thread's private stream - block i into buffer i % 2, then the host copy of block i - 2
/// out of that buffer into its target while block i - 1 is in flight.  Every target holds its bytes once finish()
/// returned true.  `cancel` (set by another thread) stops it between blocks.  The targets must stay where they are
/// until finish(); the destructor waits for the blocks in flight (a buffer is never left with a copy into it running).
class StagedCopy {
public:
    StagedCopy(void* const staging[2], size_t block, const std::atomic<bool>* cancel)
        : buf_{static_cast<uint8_t*>(staging[0]), static_cast<uint8_t*>(staging[1])}, block_(block), cancel_(cancel) {
        if (!buf_[0] || !buf_[1] || block_ == 0) buf_ = {nullptr, nullptr};
    }
    StagedCopy(const StagedCopy&) = delete;
    StagedCopy& operator=(const StagedCopy&) = delete;
    ~StagedCopy() {
        for (int b = 0; b < 2; ++b)
            if (pending_[b].dst) (void) cudaEventSynchronize(ev_[b]);
    }
    bool copy(void* dst, const void* src, size_t n, std::string& error) {
        if (!n) return true;
        if (cancelled(error)) return false;
        if (!buf_[0]) {
            const cudaError_t e = copy_nonblocking(dst, src, n);
            return e == cudaSuccess || fail(error, e);
        }
        if (!stream_) {
            cudaError_t e = private_stream(stream_);
            if (e == cudaSuccess) e = staging_events(ev_);
            if (e != cudaSuccess) { stream_ = nullptr; return fail(error, e); }
        }
        uint8_t* d = static_cast<uint8_t*>(dst);
        const uint8_t* s = static_cast<const uint8_t*>(src);
        while (n) {
            if (cancelled(error)) return false;
            const size_t m = n < block_ ? n : block_;
            const int b = next_;
            if (!land(b, error)) return false;   // the buffer's last block goes to its target first
            cudaError_t e = cudaMemcpyAsync(buf_[b], s, m, cudaMemcpyDefault, stream_);
            if (e == cudaSuccess) e = cudaEventRecord(ev_[b], stream_);
            if (e != cudaSuccess) {
                (void) cudaStreamSynchronize(stream_);
                return fail(error, e);
            }
            pending_[b] = {d, m};
            next_ ^= 1;
            d += m; s += m; n -= m;
        }
        return true;
    }
    /// every block lands, the older one first
    bool finish(std::string& error) {
        if (!buf_[0]) return true;
        const int older = next_;   // the buffer written longest ago (if pending)
        return land(older, error) && land(older ^ 1, error);
    }

private:
    struct Pending { uint8_t* dst = nullptr; size_t n = 0; };
    bool land(int b, std::string& error) {
        if (!pending_[b].dst) return true;
        const cudaError_t e = cudaEventSynchronize(ev_[b]);
        if (e != cudaSuccess) { pending_[b] = {}; return fail(error, e); }
        std::memcpy(pending_[b].dst, buf_[b], pending_[b].n);
        pending_[b] = {};
        return true;
    }
    bool cancelled(std::string& error) {
        if (!cancel_ || !cancel_->load(std::memory_order_relaxed)) return false;
        error = "conversation snapshot: cancelled";
        return true;
    }
    static bool fail(std::string& error, cudaError_t e) {
        error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
        return false;
    }
    std::array<uint8_t*, 2> buf_;
    size_t block_;
    const std::atomic<bool>* cancel_;
    cudaStream_t stream_ = nullptr;
    std::array<cudaEvent_t, 2> ev_{nullptr, nullptr};
    std::array<Pending, 2> pending_{};
    int next_ = 0;
};

} // namespace strata::core::conversation_detail
