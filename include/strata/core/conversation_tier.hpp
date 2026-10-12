// The conversation cache's disk tier (STRATA_CONV_TIER_DIR): an emergency overflow for parked conversations.  When the
// RAM cache has to evict a long conversation (ConversationCache::make_room), the image goes here instead of being
// dropped, and a later request that continues it reads it back instead of reading its prompt again.  Pure host code:
// no CUDA.
//
// What it writes: one session file (format v1, conversation_file.hpp: O_DIRECT, temporary file, flush, rename) per
// stage of a layer split - each stage's part of a parked image is an ordinary single-carve image - and then a small
// `.meta` file, the commit record: the tokens, the checkpoints' lengths and the run-time marks the v1 format does not
// hold (pinned, slot_turn, draftless).  A set without its meta file is never used and is removed at the next start.
//
// What it keeps apart: a conversation is written once.  An image whose tokens a file already holds, or holds all but
// fewer than `regrow_tokens` of (the same conversation a few turns on), is not written again: the file stays, and the
// request that continues it reads the few new tokens on top of it.  Grown by more, it is written in full and the older
// file of it goes.  A daily write budget, a disk cap (bytes and images), a time to live and an in-flight bound decide
// the rest; what does not fit is dropped as without the tier, and counted.
//
// Exactness: a read gives the image the RAM cache evicted, byte for byte (the v1 round trip, plus the marks from the
// meta file).  `verify` reads every file back after the write and compares it with the image, and stores a digest of
// the whole image that every restore is checked against.  Identity: every file carries the session identity (model
// and configuration fingerprints, conversation_file.hpp); the engine mixes into it a fingerprint of its binary, its
// arguments and its STRATA_* environment, so a file of another build, model or setting is never restored.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

struct ConversationTierConfig {
    std::string dir;                              ///< where the files go (created, mode 0700 on POSIX)
    int64_t min_tokens = 16384;                   ///< a shorter conversation is dropped, not written
    uint64_t budget_bytes = 200ull * 1000 * 1000 * 1000;   ///< bytes written per day (local calendar day); 0: none
    uint64_t cap_bytes = 600ull * 1000 * 1000 * 1000;      ///< bytes on disk at most
    uint64_t max_images = 2000;                   ///< conversations on disk at most (each is parts + 1 files)
    int64_t ttl_s = 168 * 3600;                   ///< an image unused this long is removed
    uint64_t inflight_bytes = 2048ull << 20;      ///< RAM evicted images may hold while they wait for the writer
    int64_t regrow_tokens = 8192;                 ///< a conversation on disk is written again once it grew this much
    uint64_t min_free_bytes = 64ull << 30;        ///< the disk keeps this much free (each file's preflight)
    bool warm = false;                            ///< keep (and index) the files of an earlier run; else remove them
    bool verify = false;                          ///< read every write back and check every restore's digest
    bool durable = true;                          ///< flush files and the folder (the crash-safety of the rename)
    SessionFileIdentity id;                       ///< written into every file; a read with another is refused
    std::function<void()> thread_start;           ///< run first on every thread the tier starts (CPU placement)
    std::function<int64_t()> clock;               ///< unix seconds (tests); default: the system clock
    std::function<void(const std::string&)> log;  ///< one event line, the counters appended (default: none)
    /// tests: an errno value makes a step fail.  The session file steps ("write", "file_flush", "rename",
    /// "dir_flush") and the tier's own: "meta" (the commit record's write), "read" (a restore's read)
    std::function<int(const char* step)> fault;
};

struct ConversationTierStats {
    uint64_t files = 0, parts = 0, disk_bytes = 0;            ///< conversations on disk, their files, their bytes
    uint64_t written_today = 0, budget = 0, budget_left = 0;
    uint64_t spills = 0, written_bytes = 0;                   ///< conversations written (this run) and their bytes
    uint64_t restores = 0, restores_queue = 0, read_bytes = 0; ///< read back (from disk / taken from the write queue)
    uint64_t misses = 0;                                      ///< a restore that failed: the prompt is read instead
    uint64_t drops_small = 0, drops_budget = 0, drops_busy = 0, drops_ram = 0, drops_cap = 0, drops_io = 0;
    uint64_t kept = 0;                                        ///< evictions a file already held (nothing written)
    uint64_t superseded = 0, expired = 0, evicted = 0;        ///< files removed: grown on, too old, over the cap
    uint64_t pending = 0, pending_bytes = 0;                  ///< evicted images waiting for the writer
    uint64_t foreign = 0;                                     ///< files of another identity left in the folder
    uint64_t verified = 0;                                    ///< writes and restores `verify` compared
};

/// A digest of every byte, id and mark of a parked image (its stage images too), for `verify`.
uint64_t conversation_image_digest(const SavedConversation& image);

/// The tier's fingerprint of what the engine runs: the configuration fingerprint, the binary at `exe` (its size, first
/// and last MiB), the arguments and every STRATA_* environment variable but STRATA_CONV_TIER_*, sorted.
bool conversation_tier_fingerprint(uint64_t config, const std::string& exe, const std::vector<std::string>& args,
                                   uint64_t& out, std::string& error);

class ConversationTier {
public:
    struct Match {
        uint64_t key = 0;
        int64_t tokens = 0;
        bool live = false;       ///< the conversation's end (else one of its checkpoints)
        bool draftless = false;  ///< its last stage's image has no draft-layer K/V (conversation_cache.hpp)
        size_t parts = 0;        ///< images: 1 + the later stages'
    };
    struct LoadInfo {
        bool from_queue = false; ///< still waiting for the writer: handed back without a read
        uint64_t bytes = 0;      ///< read from disk
        double ms = 0;
    };

    explicit ConversationTier(ConversationTierConfig config);
    ~ConversationTier();   // what waits for the writer is dropped; a write under way finishes first
    ConversationTier(const ConversationTier&) = delete;
    ConversationTier& operator=(const ConversationTier&) = delete;

    /// Creates the folder, reads the day's budget, removes what an earlier run left (or indexes it, `warm`) and starts
    /// the writer.  False: the tier stays off (`error` says why).
    bool start(std::string& error);
    bool on() const { return started_; }

    /// The RAM cache let `image` go (ConversationCache's eviction sink).  `spill`: it may be written (the cache made
    /// room for another image); false: it is freed now (the RAM floor or a save asked for the memory) and counted.
    void evicted(SavedConversation&& image, bool spill);

    /// The image on disk (or waiting for the writer) that holds the longest start of `prompt`, as
    /// ConversationCache::best matches (tokens, pictures, control vector mode).  tokens == 0: none.
    /// (instantiated for int32_t and int64_t tokens)
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const;

    /// Reads the image `key` names: one session file per part at once, each with its `limits` (part 0: the first
    /// stage, k: stage k), then the marks of the meta file.  An image still waiting for the writer is handed back
    /// as it is.  False: `why` says why, nothing usable came back, the caller reads the prompt; a file that is
    /// unreadable or not this engine's goes (a RAM refusal leaves it).
    bool load(uint64_t key, const std::vector<SessionReadLimits>& limits, SavedConversation& out, std::string& why,
              LoadInfo* info = nullptr);
    /// The caller refused an image `load` gave (its validation): the files go and it counts as a miss.
    void reject(uint64_t key, const std::string& why);

    /// #879: every image goes, the ones waiting for the writer and the files.
    void clear();
    /// Waits until the writer has nothing left (tests, and the engine's teardown never needs it).
    void drain();

    ConversationTierStats stats() const;
    /// "files=F parts=P disk_bytes=D written_today=W budget=B budget_left=L spills=S ..." (every stats field)
    std::string status() const;

private:
    enum class State { queued, writing, disk };
    struct Entry {
        uint64_t key = 0;
        State state = State::queued;
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        bool cvec = true;
        struct Check { size_t len = 0; std::vector<int32_t> own; std::vector<ConversationImageKey> imgs; };
        std::vector<Check> checks;   ///< own ids only when not a start of `ids`
        struct Marks { bool draftless = false, live_pinned = false, live_slot_turn = false;
                       std::vector<uint8_t> pinned, slot_turn; };
        std::vector<Marks> parts;    ///< per part
        std::vector<uint64_t> part_bytes;
        uint64_t bytes = 0;          ///< the files (parts and meta); queued: the image's RAM
        uint64_t digest = 0;
        int64_t created = 0, used = 0;
        int readers = 0;
        bool cancelled = false;      ///< clear() while it was written: its files go when the write ends
        std::unique_ptr<SavedConversation> image;   ///< queued / writing only
    };
    using EntryPtr = std::shared_ptr<Entry>;

    int64_t now() const;
    void log_event(const std::string& what);
    std::string status_locked() const;
    std::string path(uint64_t key, int part) const;   ///< part < 0: the meta file
    void writer_loop();
    bool write_set(Entry& e, uint64_t& bytes, std::string& why);   ///< writer thread, without the lock
    void remove_files(uint64_t key, size_t parts);
    void remove_entry_locked(size_t i);               ///< and its files (an image on disk)
    bool sweep_locked(uint64_t incoming);             ///< false: the cap leaves no room for `incoming` bytes
    void roll_day_locked();
    void save_budget_locked();
    void scan(std::string& error);
    size_t find_locked(uint64_t key) const;          ///< SIZE_MAX: none
    size_t cover_locked(const SavedConversation& image) const;   ///< an entry index that holds it, or SIZE_MAX
    static int64_t day_of(int64_t unix_s);

    ConversationTierConfig cfg_;
    bool started_ = false;
    mutable std::mutex mu_;
    std::condition_variable work_, idle_, changed_;
    std::vector<EntryPtr> entries_;   ///< everything known, oldest first: queued, writing, on disk
    std::thread writer_;
    bool stop_ = false, busy_ = false;
    uint64_t next_key_ = 0;
    int64_t day_ = 0;
    ConversationTierStats st_;
};

} // namespace strata::core
